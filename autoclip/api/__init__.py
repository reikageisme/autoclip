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
    GET  /api/settings                cài đặt LLM/Whisper (không trả key)
    PUT  /api/settings                lưu cài đặt vào .env
    GET  /api/settings/models         model mà endpoint LLM liệt kê
    POST /api/settings/test           gọi thử LLM
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from autoclip.config import Settings, get_settings
from autoclip.db import make_engine
from autoclip.db.migrate import upgrade
from autoclip.db.models import Clip, Job, Recipe, Render, Segment
from autoclip.diversity import recent_segment_ids

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
ENV_FILE = REPO_ROOT / ".env"


class BatchRequest(BaseModel):
    recipe: str = Field(..., description="Tên file trong recipes/, ví dụ hook-3cut-caption.json")
    count: int = Field(5, ge=1, le=100)
    seed: int | None = None
    llm: bool = True


class IngestRequest(BaseModel):
    folder: str
    whisper: bool = True


class FetchRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1)
    page: str | None = None
    whisper: bool = True


class SettingsUpdate(BaseModel):
    llm_provider: Literal["none", "openai", "anthropic", "openai_compat"] | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = Field(None, description="Chỉ ghi; bỏ trống = giữ key cũ")
    llm_timeout_sec: float | None = Field(None, gt=0, le=1800)
    vision_model: str | None = None
    whisper_model: str | None = None


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
            result = run(spec, req.count, seed=req.seed, use_llm=req.llm, progress=self._progress)
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
                progress=lambda i, n, p: self.state.update(done=i - 1, requested=n),
            )
            engine.dispose()
            self.state.update(done=report.clips_added, failed=len(report.errors))
            return (
                f"Nhập {report.clips_added} clip mới ({report.segments_added} đoạn), "
                f"bỏ qua {report.clips_skipped} clip đã có, {len(report.errors)} lỗi"
            )

        return self._start("ingest", f"Nhập {folder.name}", 0, work)

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
    recipes_dir = REPO_ROOT / "recipes"
    env_path = env_file or ENV_FILE
    app = FastAPI(title="AutoClip", version="0.2")
    app.state.runner = runner

    def load_spec(name: str):
        from autoclip.render import load_recipe

        path = (recipes_dir / name).resolve()
        if path.parent != recipes_dir.resolve() or path.suffix != ".json" or not path.is_file():
            raise HTTPException(404, f"không có recipe {name}")
        return load_recipe(path)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "busy": runner.busy}

    @app.get("/api/recipes")
    def recipes() -> list[dict[str, Any]]:
        out = []
        for f in sorted(recipes_dir.glob("*.json")):
            try:
                spec = load_spec(f.name)
            except Exception as exc:  # recipe hỏng vẫn liệt kê để người dùng thấy
                out.append({"file": f.name, "error": str(exc)})
                continue
            out.append(
                {
                    "file": f.name,
                    "name": spec.name,
                    "aspect": spec.aspect,
                    "duration_target": list(spec.duration_target),
                    "slots": [sl.role for sl in spec.slots],
                }
            )
        return out

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
        return runner.start(load_spec(req.recipe), req)

    @app.post("/api/ingest", status_code=202)
    def ingest(req: IngestRequest) -> dict[str, Any]:
        folder = Path(req.folder).expanduser()
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
                            "project_path": str(project)
                            if project is not None and project.is_dir()
                            else None,
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
        cur = get_settings()
        return {
            "llm_provider": cur.llm_provider,
            "llm_base_url": cur.llm_base_url or "",
            "llm_model": cur.llm_model or "",
            "llm_timeout_sec": cur.llm_timeout_sec,
            "vision_model": cur.vision_model or "",
            "whisper_model": cur.whisper_model,
            "has_api_key": bool(cur.llm_api_key or cur.openai_api_key or cur.anthropic_api_key),
        }

    @app.put("/api/settings")
    def write_settings(req: SettingsUpdate) -> dict[str, Any]:
        values: dict[str, str] = {}
        for field, value in req.model_dump(exclude_none=True).items():
            if field == "llm_api_key" and not str(value).strip():
                continue  # bỏ trống = giữ key cũ
            values[field.upper()] = str(value).strip()
        if values:
            update_env(env_path, values)
            get_settings.cache_clear()
        return read_settings()

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

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    return app
