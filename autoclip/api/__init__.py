"""API HTTP cục bộ (127.0.0.1:8787) cho panel AutoClip trong ReiCut và dashboard web.

Hai giao diện cùng gọi một API nên thấy chung hàng đợi, kho clip và cài đặt. Tác vụ dài
(batch, ingest, tải video) chạy trong một luồng nền, mỗi lúc một tác vụ. Chỉ bind localhost.

    GET  /api/health                  kiểm tra sống
    GET  /api/recipes                 recipe trong recipes/*.json
    GET  /api/pool                    số clip/segment trong kho, số segment vừa dùng
    GET  /api/clips                   các clip nguồn trong kho
    GET  /api/status                  tác vụ đang chạy (hoặc lần gần nhất)
    POST /api/batches                 {recipe, count, seed?, llm} → batch video
    POST /api/ingest                  {folder, whisper} → nhập một thư mục clip
    POST /api/fetch                   {urls, page?, whisper} → tải video rồi nhập
    GET  /api/jobs?limit=50           job mới nhất kèm render, QA, thumbnail
    GET  /api/renders/{id}/video      file MP4
    GET  /api/settings                cài đặt LLM/Whisper/Facebook (không trả key)
    PUT  /api/settings                lưu cài đặt; token Facebook vào DPAPI
    GET  /api/settings/models         model mà endpoint LLM liệt kê
    POST /api/settings/test           gọi thử LLM
    POST /api/settings/facebook/test  kiểm tra Page id + token
    POST /api/renders/{id}/publish    đăng ngay hoặc hẹn giờ lên Page
    GET  /api/publications            lịch sử đăng bài
    POST /api/metrics/import          nhập CSV lượt xem vào bảng metrics
    GET  /api/metrics                 metrics gần đây + hiệu quả theo kiểu video
    GET  /api/schedules               lịch chạy batch
    POST /api/schedules               thêm lịch chạy (một lần hoặc lặp)
    GET  /api/system                  dung lượng ổ đĩa + cấu hình dọn dẹp
    POST /api/system/cleanup          dọn proxy/project/video cũ
    GET  /                            dashboard web
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from autoclip.batch import VoiceOptions
from autoclip.brand import BrandKit
from autoclip.config import ENV_FILE, Settings, get_settings
from autoclip.contracts import RecipeSpec
from autoclip.db import make_engine
from autoclip.db.migrate import upgrade
from autoclip.db.models import Clip, Job, Recipe, Render, Segment
from autoclip.diversity import recent_segment_ids
from autoclip.metrics import best_recipe, import_metrics, list_metrics, recipe_scores
from autoclip.styles import (
    TRANSITIONS,
    StyleForm,
    Styles,
    estimate_videos,
    slugify,
    spec_to_form,
)

log = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"


class BatchRequest(BaseModel):
    recipe: str = Field(
        ...,
        description="Tên file trong recipes/, hoặc 'auto' để ưu tiên kiểu có views cao",
    )
    count: int = Field(5, ge=1, le=100)
    seed: int | None = None
    llm: bool = True
    voice: VoiceOptions | None = None
    """Có thì làm video lồng tiếng (kịch bản AI theo chủ đề hoặc gõ tay)."""
    workers: int | None = Field(None, ge=1, le=16)
    """Số job render song song; trống = RENDER_WORKERS trong .env."""


class IngestRequest(BaseModel):
    folder: str
    whisper: bool = True


class FetchRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1)
    page: str | None = None
    whisper: bool = True


class ReviewRequest(BaseModel):
    review: Literal["pending", "approved", "rejected"]


class EditRequest(BaseModel):
    headline: str | None = None
    title: str | None = None


class ExportRequest(BaseModel):
    folder: str = ""
    include_exported: bool = False


class SettingsUpdate(BaseModel):
    llm_provider: Literal["none", "openai", "anthropic", "openai_compat"] | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = Field(None, description="Chỉ ghi; bỏ trống = giữ key cũ")
    llm_timeout_sec: float | None = Field(None, gt=0, le=1800)
    vision_model: str | None = None
    whisper_model: str | None = None
    fb_page_id: str | None = None
    fb_graph_version: str | None = None
    fb_token: str | None = Field(None, description="Chỉ ghi; bỏ trống = giữ token cũ")
    render_workers: int | None = Field(None, ge=1, le=16)
    maintenance_interval_min: int | None = Field(None, ge=1, le=1440)
    cleanup_proxies_days: int | None = Field(None, ge=0, le=3650)
    cleanup_projects_days: int | None = Field(None, ge=0, le=3650)
    cleanup_videos_days: int | None = Field(None, ge=0, le=3650)
    disk_warn_free_gb: float | None = Field(None, ge=0, le=1000)


class PublishRequest(BaseModel):
    at: datetime | None = Field(None, description="Giờ hẹn (ISO); trống = đăng ngay")
    caption: str | None = None


class MetricsImportRequest(BaseModel):
    csv: str = Field(..., min_length=1)
    platform: str = "facebook"
    render_id: int | None = None


class ScheduleRequest(BaseModel):
    recipe: str = "auto"
    count: int = Field(5, ge=1, le=100)
    at: datetime = Field(..., description="Giờ chạy (ISO); quá khứ = chạy ngay")
    repeat_minutes: int = Field(0, ge=0, le=10080)
    seed: int | None = None
    llm: bool = True
    voice: VoiceOptions | None = None
    workers: int | None = Field(None, ge=1, le=16)


class ToggleRequest(BaseModel):
    enabled: bool


class CleanupRequest(BaseModel):
    dry_run: bool = False
    proxies_days: int | None = Field(None, ge=0, le=3650)
    projects_days: int | None = Field(None, ge=0, le=3650)
    videos_days: int | None = Field(None, ge=0, le=3650)


class PublishScheduler:
    """Vòng nền gửi các lịch đăng đã tới giờ; chỉ dòng người dùng tạo mới có trong DB."""

    def __init__(self, engine, settings_fn: Callable[[], Settings]) -> None:
        self.engine = engine
        self.settings_fn = settings_fn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        from autoclip.publish import recover_interrupted

        if self._thread and self._thread.is_alive():
            return
        recover_interrupted(self.engine)
        self._thread = threading.Thread(target=self._loop, name="autoclip-publish", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        from autoclip.publish import publish_due

        while not self._stop.wait(max(5, self.settings_fn().publish_check_sec)):
            try:
                publish_due(self.engine, self.settings_fn())
            except Exception:
                log.exception("scheduler đăng bài lỗi")


class MaintenanceScheduler:
    """Chạy lịch batch, dọn dữ liệu định kỳ và cảnh báo ổ đĩa khi service đang bật."""

    def __init__(
        self,
        engine,
        runner: BatchRunner,
        resolve_spec: Callable[[str], RecipeSpec],
        settings_fn: Callable[[], Settings],
    ) -> None:
        self.engine = engine
        self.runner = runner
        self.resolve_spec = resolve_spec
        self.settings_fn = settings_fn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_cleanup = time.monotonic()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="autoclip-maintenance", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.wait(30):
            try:
                self._maintain()
            except Exception:
                log.exception("bảo trì định kỳ lỗi")

    def _maintain(self) -> None:
        from autoclip.ops import cleanup, disk_status, protected_paths
        from autoclip.schedules import defer, due_schedules, mark_run

        s = self.settings_fn()
        if time.monotonic() - self._last_cleanup >= s.maintenance_interval_min * 60:
            self._last_cleanup = time.monotonic()
            report = cleanup(s, keep=protected_paths(self.engine))
            if report.deleted:
                log.info("bảo trì: xoá %d mục, %.1f MB", report.deleted, report.mb_freed)
            for disk in disk_status(s):
                if disk.low:
                    log.warning(
                        "ổ %s chỉ còn %.1f GB (ngưỡng %.1f GB)",
                        disk.path,
                        disk.free_gb,
                        s.disk_warn_free_gb,
                    )
        for row in due_schedules(self.engine):
            if self.runner.busy:
                return
            opts = row.options or {}
            req = BatchRequest(
                recipe=row.recipe,
                count=row.count,
                seed=row.seed,
                llm=bool(opts.get("llm", True)),
                voice=opts.get("voice"),
                workers=opts.get("workers"),
            )
            try:
                self.runner.start(self.resolve_spec(row.recipe), req)
            except HTTPException as exc:
                if exc.status_code == 409:
                    return
                log.warning("lịch %s chưa chạy được: %s", row.id, exc.detail)
                defer(self.engine, row.id, 5)
                continue
            mark_run(self.engine, row.id)
            log.info("lịch %s: đã bắt đầu %s x%d", row.id, row.recipe, row.count)


class BatchRunner:
    """Một tác vụ dài tại một thời điểm, chạy nền; trạng thái đọc qua /api/status."""

    def __init__(self, run=None) -> None:
        self._run = run
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state: dict[str, Any] = {"running": False}

    @property
    def busy(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _start(self, kind: str, title: str, requested: int, work: Callable[[], str | None]):
        with self._lock:
            if self.busy:
                raise HTTPException(
                    409, f"đang chạy: {self.state.get('title')} — đợi xong rồi làm tiếp"
                )
            self.state = {
                "running": True,
                "kind": kind,
                "title": title,
                "recipe": title if kind == "batch" else None,
                "requested": requested,
                "done": 0,
                "failed": 0,
                "message": None,
                "started_at": datetime.now(UTC).isoformat(),
                "finished_at": None,
                "stopped_reason": None,
                "error": None,
            }
            self._thread = threading.Thread(
                target=self._work, args=(work,), name=f"autoclip-{kind}", daemon=True
            )
            self._thread.start()
            return dict(self.state)

    def _work(self, work: Callable[[], str | None]) -> None:
        try:
            self.state["message"] = work()
        except Exception as exc:  # lỗi bất ngờ phải hiện lên giao diện, không chết lặng lẽ
            log.exception("tác vụ lỗi")
            self.state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            self.state["running"] = False
            self.state["finished_at"] = datetime.now(UTC).isoformat()

    def _progress(self, item) -> None:
        self.state["done" if item.status == "done" else "failed"] += 1

    def start(self, spec, req: BatchRequest) -> dict[str, Any]:
        def work() -> str | None:
            run = self._run
            if run is None:
                from autoclip.batch import run_batch as run
            extra = {"voice": req.voice} if req.voice else {}
            result = run(
                spec,
                req.count,
                seed=req.seed,
                use_llm=req.llm,
                progress=self._progress,
                workers=req.workers,
                **extra,
            )
            self.state["stopped_reason"] = result.stopped_reason
            return None

        return self._start("batch", spec.name, req.count, work)

    def start_ingest(self, folder: Path, whisper: bool) -> dict[str, Any]:
        def work() -> str:
            from autoclip.db import SqlClipRepository
            from autoclip.ingest import IngestOptions, ingest_folder

            engine = make_engine()
            options = IngestOptions.from_settings(get_settings())
            options.whisper_enabled = whisper
            report = ingest_folder(
                folder,
                SqlClipRepository(engine),
                options,
                # Panel hiện "title": cho người dùng thấy clip đang làm (clip dài mất vài phút)
                progress=lambda i, n, p: self.state.update(
                    done=i - 1, requested=n, title=f"Nhập {folder.name} · clip {i}/{n}: {p.name}"
                ),
            )
            engine.dispose()
            self.state.update(
                done=report.clips_added, failed=len(report.errors), title=f"Nhập {folder.name}"
            )
            return (
                f"Nhập {report.clips_added} clip mới ({report.segments_added} đoạn), "
                f"bỏ qua {report.clips_skipped} clip đã có, {len(report.errors)} lỗi"
            )

        return self._start("ingest", f"Nhập {folder.name}", 0, work)

    def start_rerender(self, render_id: int, headline: str | None, title: str | None):
        """Dựng lại đúng video đó (cùng cảnh) với headline mới; kết quả là render mới của job."""

        def work() -> str:
            from autoclip.batch import _qa, render_fields
            from autoclip.contracts import RenderPlan, RenderText
            from autoclip.render import render

            engine = make_engine()
            with Session(engine, expire_on_commit=False) as db:
                old = db.get(Render, render_id)
                job = db.get(Job, old.job_id)
                spec = RecipeSpec.model_validate(db.get(Recipe, job.recipe_id).spec)
                plan = RenderPlan.model_validate(old.plan)
                if headline is not None:
                    plan.texts = [t for t in plan.texts if t.kind == "captions"]
                    heads = [t for t in spec.texts if t.kind != "captions"]
                    if headline.strip() and heads:
                        plan.texts.append(
                            RenderText(
                                kind=heads[0].kind,
                                text=headline.strip().upper(),
                                slot=heads[0].slot,
                                style=heads[0].style,
                                pos=heads[0].pos,
                                start=0.0,
                                end=2.8,
                            )
                        )
                if title is not None:
                    plan.params["title"] = title
                job.status = "running"
                db.commit()
                rendered = render(spec, plan)
                qa = _qa(rendered.output, spec, plan)
                db.add(
                    Render(
                        job_id=job.id,
                        output_path=str(rendered.output),
                        qa_score=qa.score,
                        qa_notes="; ".join(qa.notes) or None,
                        duration=rendered.video_sec,
                        size_bytes=rendered.output.stat().st_size,
                        **render_fields(rendered.plan, rendered),
                    )
                )
                job.status = "done"
                db.commit()
            engine.dispose()
            self.state.update(done=1)
            return "Đã dựng lại video với chữ mới."

        return self._start("rerender", f"Dựng lại video #{render_id}", 1, work)

    def start_vision(self, limit: int | None) -> dict[str, Any]:
        def work() -> str:
            from autoclip.llm import LLMClient
            from autoclip.vision_tag import tag_segments

            llm = LLMClient.from_settings(get_settings(), vision=True)
            if not llm.available:
                raise RuntimeError("chưa cấu hình model nhìn ảnh (Cài đặt > Model nhìn ảnh)")
            engine = make_engine()
            done, failed = tag_segments(
                engine,
                llm,
                limit=limit,
                progress=lambda i, n: self.state.update(done=i, requested=n),
            )
            engine.dispose()
            return f"Đã gắn nhãn cảnh cho {done} đoạn, {failed} lỗi"

        return self._start("vision", "AI gắn nhãn cảnh", 0, work)

    def start_fetch(self, req: FetchRequest) -> dict[str, Any]:
        def work() -> str:
            from autoclip.fetch import expand_urls, fetch

            urls = expand_urls(req.urls)
            self.state["requested"] = len(urls)
            r = fetch(urls, get_settings().data_dir / "source", name=req.page or None)
            self.state.update(done=len(r.downloaded), failed=len(r.errors))
            msg = f"Tải mới {len(r.downloaded)} video, {r.skipped} đã có, {len(r.errors)} lỗi"
            if r.downloaded or r.skipped:
                self.state.update(kind="ingest", title=f"Nhập {r.dest.name}")
                from autoclip.db import SqlClipRepository
                from autoclip.ingest import IngestOptions, ingest_folder

                engine = make_engine()
                options = IngestOptions.from_settings(get_settings())
                options.whisper_enabled = req.whisper
                report = ingest_folder(r.dest, SqlClipRepository(engine), options)
                engine.dispose()
                msg += f" · nhập {report.clips_added} clip ({report.segments_added} đoạn)"
            return msg

        return self._start("fetch", "Tải video nguồn", len(req.urls), work)


def update_env(path: Path, values: dict[str, str]) -> None:
    """Ghi các khoá vào file .env, giữ nguyên dòng/chú thích khác."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    left = dict(values)
    out = []
    for line in lines:
        key = (
            line.split("=", 1)[0].strip()
            if "=" in line and not line.lstrip().startswith("#")
            else None
        )
        if key in left:
            out.append(f"{key}={left.pop(key)}")
        else:
            out.append(line)
    out.extend(f"{k}={v}" for k, v in left.items())
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _render_thumb(render: Render, thumbs_dir: Path) -> str | None:
    """Ảnh nhỏ giữa video (cache data/thumbs/renders/<id>.jpg) cho danh sách job."""
    out = thumbs_dir / "renders" / f"{render.id}.jpg"
    if out.is_file():
        return str(out)
    src = Path(render.output_path)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not src.is_file():
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    at = max(0.0, (render.duration or 2.0) / 2)
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-y",
        "-ss",
        f"{at:.2f}",
        "-i",
        str(src),
        "-frames:v",
        "1",
        "-vf",
        "scale=180:-2",
        str(out),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return str(out) if out.is_file() else None


def create_app(
    settings: Settings | None = None,
    runner: BatchRunner | None = None,
    env_file: Path | None = None,
) -> FastAPI:
    s = settings or get_settings()
    engine = make_engine()
    upgrade(engine)
    runner = runner or BatchRunner()
    env_path = env_file or ENV_FILE
    publish_scheduler = PublishScheduler(engine, get_settings)
    maintenance_scheduler: MaintenanceScheduler | None = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        publish_scheduler.start()
        if maintenance_scheduler is not None:
            maintenance_scheduler.start()
        try:
            yield
        finally:
            if maintenance_scheduler is not None:
                maintenance_scheduler.stop()
            publish_scheduler.stop()

    app = FastAPI(title="AutoClip", version="0.4", lifespan=lifespan)
    app.state.runner = runner
    app.state.publish_scheduler = publish_scheduler

    def styles() -> Styles:
        return Styles(get_settings().data_dir / "recipes")

    def load_spec(name: str):
        spec = styles().load(name)
        if spec is None:
            raise HTTPException(404, f"không có kiểu video {name}")
        return spec

    def resolve_spec(name: str) -> RecipeSpec:
        """'auto' = kiểu có views trung bình cao nhất; chưa có metrics thì kiểu đầu tiên."""
        if name != "auto":
            return load_spec(name)
        readable = [
            spec
            for _file, spec, _builtin, error in styles().list()
            if spec is not None and not error
        ]
        if not readable:
            raise HTTPException(404, "chưa có kiểu video nào đọc được")
        winner = best_recipe(engine, [spec.name for spec in readable])
        return next((spec for spec in readable if spec.name == winner), readable[0])

    maintenance_scheduler = MaintenanceScheduler(engine, runner, resolve_spec, get_settings)
    app.state.maintenance_scheduler = maintenance_scheduler

    def pool_stats() -> tuple[int, float | None]:
        """(số đoạn còn dùng được, độ dài trung bình của chúng)."""
        recent = recent_segment_ids(engine, get_settings().no_repeat_last_n_renders)
        with Session(engine) as db:
            rows = db.execute(select(Segment.id, Segment.start, Segment.end)).all()
        free = [e - st for sid, st, e in rows if sid not in recent and e - st >= 0.8]
        return len(free), (sum(free) / len(free) if free else None)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "busy": runner.busy}

    @app.get("/api/recipes")
    def recipes() -> list[dict[str, Any]]:
        available, avg = pool_stats()
        scores = recipe_scores(engine)
        ranked = sorted(
            scores,
            key=lambda name: (
                -float(scores[name].get("avg_views", 0)),
                -int(scores[name].get("samples", 0)),
            ),
        )
        out = []
        for file, spec, builtin, error in styles().list():
            if spec is None:
                out.append({"file": file, "builtin": builtin, "error": error})
                continue
            perf = scores.get(spec.name, {})
            out.append(
                {
                    "file": file,
                    "name": spec.name,
                    "label": spec.label or spec.name,
                    "description": spec.description or "",
                    "builtin": builtin,
                    "aspect": spec.aspect,
                    "duration_target": list(spec.duration_target),
                    "slots": [sl.role for sl in spec.slots],
                    "est_videos": estimate_videos(spec, available, avg),
                    "avg_views": perf.get("avg_views"),
                    "total_views": perf.get("total_views"),
                    "metric_samples": perf.get("samples", 0),
                    "priority": ranked.index(spec.name) + 1 if spec.name in ranked else None,
                    "form": spec_to_form(spec).model_dump(),
                }
            )
        return out

    @app.get("/api/recipes/options")
    def recipe_options() -> dict[str, Any]:
        return {"transitions": TRANSITIONS}

    @app.post("/api/recipes", status_code=201)
    def save_recipe(form: StyleForm, file: str | None = None) -> dict[str, Any]:
        """Lưu kiểu tự tạo. Có `file` của kiểu tự tạo → cập nhật; kiểu có sẵn → tạo bản sao."""
        saved = styles().save(form, file)
        return next(r for r in recipes() if r["file"] == saved)

    @app.delete("/api/recipes/{file}")
    def delete_recipe(file: str) -> dict[str, Any]:
        if styles().is_builtin(file):
            raise HTTPException(400, "kiểu có sẵn không xoá được")
        if not styles().delete(file):
            raise HTTPException(404, f"không có kiểu {file}")
        return {"deleted": file}

    @app.get("/api/pool")
    def pool() -> dict[str, Any]:
        with Session(engine) as db:
            clips = db.scalar(select(func.count()).select_from(Clip)) or 0
            segments = db.scalar(select(func.count()).select_from(Segment)) or 0
        recent = recent_segment_ids(engine, get_settings().no_repeat_last_n_renders)
        return {
            "clips": clips,
            "segments": segments,
            "recently_used": len(recent),
            "available": max(0, segments - len(recent)),
        }

    @app.get("/api/clips")
    def clips() -> list[dict[str, Any]]:
        with Session(engine) as db:
            rows = db.execute(
                select(Clip, func.count(Segment.id), func.min(Segment.thumb_path))
                .outerjoin(Segment, Segment.clip_id == Clip.id)
                .group_by(Clip.id)
                .order_by(Clip.id.desc())
            ).all()
        return [
            {
                "id": c.id,
                "name": Path(c.path).name,
                "path": c.path,
                "duration": c.duration,
                "size": f"{c.width}x{c.height}",
                "segments": n,
                "thumb_path": thumb if thumb and Path(thumb).is_file() else None,
                "exists": Path(c.path).is_file(),
            }
            for c, n, thumb in rows
        ]

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return dict(runner.state)

    @app.post("/api/batches", status_code=202)
    def create_batch(req: BatchRequest) -> dict[str, Any]:
        return runner.start(resolve_spec(req.recipe), req)

    @app.get("/api/schedules")
    def schedules() -> list[dict[str, Any]]:
        from autoclip import schedules as store

        return [store.serialize(row) for row in store.list_schedules(engine)]

    @app.post("/api/schedules", status_code=201)
    def add_schedule(req: ScheduleRequest) -> dict[str, Any]:
        from autoclip import schedules as store

        resolve_spec(req.recipe)
        options: dict[str, Any] = {"llm": req.llm, "workers": req.workers}
        if req.voice is not None:
            options["voice"] = req.voice.model_dump(mode="json")
        row = store.create_schedule(
            engine,
            recipe=req.recipe,
            count=req.count,
            run_at=req.at,
            repeat_minutes=req.repeat_minutes,
            seed=req.seed,
            options=options,
        )
        return store.serialize(row)

    @app.delete("/api/schedules/{schedule_id}")
    def remove_schedule(schedule_id: int) -> dict[str, Any]:
        from autoclip import schedules as store

        if not store.delete_schedule(engine, schedule_id):
            raise HTTPException(404, "không có lịch này")
        return {"deleted": schedule_id}

    @app.post("/api/schedules/{schedule_id}/toggle")
    def toggle_schedule(schedule_id: int, req: ToggleRequest) -> dict[str, Any]:
        from autoclip import schedules as store

        if not store.set_enabled(engine, schedule_id, req.enabled):
            raise HTTPException(404, "không có lịch này")
        return {"id": schedule_id, "enabled": req.enabled}

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        from autoclip.ops import disk_status

        cur = get_settings()
        return {
            "disk": [asdict(d) for d in disk_status(cur)],
            "cleanup": {
                "proxies_days": cur.cleanup_proxies_days,
                "projects_days": cur.cleanup_projects_days,
                "videos_days": cur.cleanup_videos_days,
            },
            "warn_free_gb": cur.disk_warn_free_gb,
        }

    @app.post("/api/system/cleanup")
    def system_cleanup(req: CleanupRequest) -> dict[str, Any]:
        from autoclip.ops import cleanup, protected_paths

        report = cleanup(
            get_settings(),
            keep=protected_paths(engine),
            dry_run=req.dry_run,
            proxies_days=req.proxies_days,
            projects_days=req.projects_days,
            videos_days=req.videos_days,
        )
        out = asdict(report)
        out["mb_freed"] = report.mb_freed
        return out

    @app.post("/api/ingest", status_code=202)
    def ingest(req: IngestRequest) -> dict[str, Any]:
        if not req.folder.strip():
            raise HTTPException(400, "chưa chọn thư mục")
        folder = Path(req.folder.strip()).expanduser()
        if not folder.is_dir():
            raise HTTPException(404, f"không có thư mục {folder}")
        return runner.start_ingest(folder, req.whisper)

    @app.post("/api/fetch", status_code=202)
    def fetch(req: FetchRequest) -> dict[str, Any]:
        return runner.start_fetch(req)

    @app.get("/api/jobs")
    def jobs(limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        thumbs = get_settings().thumbs_dir
        with Session(engine) as db:
            rows = db.execute(
                select(Job, Recipe.name)
                .join(Recipe, Recipe.id == Job.recipe_id)
                .order_by(Job.id.desc())
                .limit(limit)
            ).all()
            out = []
            for job, recipe_name in rows:
                render = db.scalar(
                    select(Render).where(Render.job_id == job.id).order_by(Render.id.desc())
                )
                project = (
                    s.projects_dir / Path(render.output_path).stem if render is not None else None
                )
                out.append(
                    {
                        "id": job.id,
                        "recipe": recipe_name,
                        "seed": job.seed,
                        "status": job.status,
                        "attempts": job.attempts,
                        "error": job.error,
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                        "render": None
                        if render is None
                        else {
                            "id": render.id,
                            "qa_score": render.qa_score,
                            "qa_notes": render.qa_notes,
                            "duration": render.duration,
                            "size_bytes": render.size_bytes,
                            "file": Path(render.output_path).name,
                            "path": render.output_path,
                            "video_url": f"/api/renders/{render.id}/video",
                            "thumb_path": _render_thumb(render, thumbs)
                            if job.status == "done"
                            else None,
                            "project_path": render.project_path
                            or (str(project) if project is not None and project.is_dir() else None),
                            "headline": render.headline,
                            "title": render.title,
                            "review": render.review,
                            "exported": render.exported_at is not None,
                        },
                    }
                )
        return out

    @app.get("/api/renders/{render_id}/video")
    def video(render_id: int) -> FileResponse:
        with Session(engine) as db:
            render = db.get(Render, render_id)
        if render is None:
            raise HTTPException(404, "không có render này")
        path = Path(render.output_path).resolve()
        if s.out_dir.resolve() not in path.parents or not path.is_file():
            raise HTTPException(404, "file video không còn")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.get("/api/settings")
    def read_settings() -> dict[str, Any]:
        from autoclip.publish import facebook_token

        cur = get_settings()
        return {
            "llm_provider": cur.llm_provider,
            "llm_base_url": cur.llm_base_url or "",
            "llm_model": cur.llm_model or "",
            "llm_timeout_sec": cur.llm_timeout_sec,
            "vision_model": cur.vision_model or "",
            "whisper_model": cur.whisper_model,
            "has_api_key": bool(cur.llm_api_key or cur.openai_api_key or cur.anthropic_api_key),
            "fb_page_id": cur.fb_page_id,
            "fb_graph_version": cur.fb_graph_version,
            "has_fb_token": bool(facebook_token(cur)),
            "render_workers": cur.render_workers,
            "maintenance_interval_min": cur.maintenance_interval_min,
            "cleanup_proxies_days": cur.cleanup_proxies_days,
            "cleanup_projects_days": cur.cleanup_projects_days,
            "cleanup_videos_days": cur.cleanup_videos_days,
            "disk_warn_free_gb": cur.disk_warn_free_gb,
        }

    @app.put("/api/settings")
    def write_settings(req: SettingsUpdate) -> dict[str, Any]:
        from autoclip.publish import clear_facebook_token, save_facebook_token

        data = req.model_dump(exclude_none=True)
        token = data.pop("fb_token", None)
        values: dict[str, str] = {}
        for field, value in data.items():
            if field == "llm_api_key" and not str(value).strip():
                continue  # bỏ trống = giữ key cũ
            values[field.upper()] = str(value).strip()
        if values:
            update_env(env_path, values)
            get_settings.cache_clear()
        if token is not None:
            if str(token).strip():
                save_facebook_token(get_settings(), str(token))
            elif str(token) == "":
                clear_facebook_token(get_settings())
        return read_settings()

    @app.post("/api/settings/facebook/test")
    def test_facebook() -> dict[str, Any]:
        from autoclip.publish import PublishError, check_page

        try:
            name = check_page(get_settings())
        except PublishError as exc:
            return {"ok": False, "message": str(exc)}
        return {"ok": True, "message": name}

    @app.get("/api/settings/models")
    def list_models() -> list[str]:
        cur = get_settings()
        if cur.llm_provider != "openai_compat" or not cur.llm_base_url or not cur.llm_api_key:
            return []
        req = urllib.request.Request(
            cur.llm_base_url.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {cur.llm_api_key.get_secret_value()}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise HTTPException(502, f"không lấy được danh sách model: {exc}") from exc
        return sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))

    @app.post("/api/settings/test")
    def test_llm() -> dict[str, Any]:
        from autoclip.llm import LLMClient, LLMError

        client = LLMClient.from_settings(get_settings())
        if not client.available:
            return {"ok": False, "message": "Chưa cấu hình đủ provider / URL / key / model"}
        t0 = time.perf_counter()
        try:
            reply = client.chat("Trả lời đúng một từ: OK", max_tokens=4000)
        except LLMError as exc:
            return {"ok": False, "message": str(exc), "seconds": round(time.perf_counter() - t0, 1)}
        return {
            "ok": True,
            "message": reply.strip()[:200],
            "seconds": round(time.perf_counter() - t0, 1),
        }

    def get_render(db: Session, render_id: int) -> Render:
        render = db.get(Render, render_id)
        if render is None:
            raise HTTPException(404, "không có video này")
        return render

    @app.post("/api/renders/{render_id}/review")
    def review(render_id: int, req: ReviewRequest) -> dict[str, Any]:
        with Session(engine) as db:
            get_render(db, render_id).review = req.review
            db.commit()
        return {"id": render_id, "review": req.review}

    @app.post("/api/renders/{render_id}/edit", status_code=202)
    def edit_render(render_id: int, req: EditRequest) -> dict[str, Any]:
        """Sửa tiêu đề bài đăng (lưu ngay); sửa headline thì dựng lại video."""
        with Session(engine) as db:
            render = get_render(db, render_id)
            if req.title is not None:
                render.title = req.title.strip() or None
                db.commit()
            changed = req.headline is not None and req.headline.strip().upper() != (
                render.headline or ""
            )
            if changed and render.plan is None:
                raise HTTPException(400, "video cũ chưa lưu plan, không dựng lại được")
        if not changed:
            return {"id": render_id, "rerender": False}
        runner.start_rerender(render_id, req.headline, req.title)
        return {"id": render_id, "rerender": True}

    @app.post("/api/jobs/{job_id}/redo", status_code=202)
    def redo(job_id: int, llm: bool = True) -> dict[str, Any]:
        """Làm một video mới cùng kiểu với job này (seed khác, cảnh khác)."""
        import random

        with Session(engine) as db:
            job = db.get(Job, job_id)
            if job is None:
                raise HTTPException(404, "không có job này")
            spec = RecipeSpec.model_validate(db.get(Recipe, job.recipe_id).spec)
        req = BatchRequest(recipe="-", count=1, seed=random.randrange(10**6), llm=llm)
        return runner.start(spec, req)

    @app.post("/api/export")
    def export(req: ExportRequest) -> dict[str, Any]:
        """Chép video đã duyệt (chưa xuất) ra thư mục, mỗi video kèm .txt caption để đăng."""
        dest = Path(req.folder.strip() or str(get_settings().data_dir / "export")).expanduser()
        dest.mkdir(parents=True, exist_ok=True)
        done = []
        with Session(engine) as db:
            latest = select(func.max(Render.id)).group_by(Render.job_id).scalar_subquery()
            q = select(Render).where(Render.id.in_(latest), Render.review == "approved")
            if not req.include_exported:
                q = q.where(Render.exported_at.is_(None))
            for render in db.scalars(q.order_by(Render.id)):
                src = Path(render.output_path)
                if not src.is_file():
                    continue
                stem = f"{render.id:04d}-" + slugify(render.headline or src.stem)[:50]
                shutil.copy2(src, dest / f"{stem}.mp4")
                caption = render.title or render.headline or ""
                (dest / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
                render.exported_at = datetime.now(UTC)
                done.append(stem)
            db.commit()
        return {"folder": str(dest), "count": len(done), "files": done}

    @app.post("/api/renders/{render_id}/publish", status_code=202)
    def publish(render_id: int, req: PublishRequest) -> dict[str, Any]:
        from autoclip.publish import PublishError, schedule_publication

        with Session(engine) as db:
            get_render(db, render_id)
        try:
            row = schedule_publication(
                engine, render_id, settings=get_settings(), at=req.at, caption=req.caption
            )
        except PublishError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "id": row.id,
            "render_id": row.render_id,
            "scheduled_at": row.scheduled_at.isoformat() if row.scheduled_at else None,
            "status": row.status,
        }

    @app.get("/api/publications")
    def publications(limit: int = 100) -> list[dict[str, Any]]:
        from autoclip.publish import list_publications

        return list_publications(engine, limit)

    @app.post("/api/publications/{publication_id}/cancel")
    def cancel_publication(publication_id: int) -> dict[str, Any]:
        from autoclip.publish import cancel_publication as cancel

        if not cancel(engine, publication_id):
            raise HTTPException(409, "chỉ huỷ được lịch đang chờ")
        return {"id": publication_id, "status": "cancelled"}

    @app.post("/api/publications/{publication_id}/retry")
    def retry_publication(publication_id: int) -> dict[str, Any]:
        from autoclip.publish import retry_publication as retry

        if not retry(engine, publication_id):
            raise HTTPException(409, "chỉ thử lại được lịch lỗi/đã huỷ")
        return {"id": publication_id, "status": "scheduled"}

    @app.post("/api/metrics/import")
    def metrics_import(req: MetricsImportRequest) -> dict[str, Any]:
        from autoclip.metrics import MetricsError

        try:
            report = import_metrics(
                engine, req.csv, platform=req.platform, default_render_id=req.render_id
            )
        except MetricsError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "imported": report.imported,
            "skipped": report.skipped,
            "errors": report.errors,
            "recipes": report.recipes,
        }

    @app.get("/api/metrics")
    def metrics(limit: int = 100) -> dict[str, Any]:
        return {
            "rows": list_metrics(engine, limit),
            "recipes": recipe_scores(engine),
        }

    @app.post("/api/vision-tag", status_code=202)
    def vision_tag(limit: int | None = None) -> dict[str, Any]:
        return runner.start_vision(limit)

    @app.get("/api/brand")
    def read_brand() -> dict[str, Any]:
        from autoclip.brand import load_brand, music_tracks

        kit = load_brand(get_settings().data_dir)
        return {**kit.model_dump(), "music_count": len(music_tracks(kit.music_folder))}

    @app.put("/api/brand")
    def write_brand(kit: BrandKit) -> dict[str, Any]:
        from autoclip.brand import save_brand

        if kit.logo_path and not Path(kit.logo_path).is_file():
            raise HTTPException(400, f"không thấy file logo {kit.logo_path}")
        if kit.music_folder and not Path(kit.music_folder).is_dir():
            raise HTTPException(400, f"không thấy thư mục nhạc {kit.music_folder}")
        save_brand(get_settings().data_dir, kit)
        return read_brand()

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    return app
