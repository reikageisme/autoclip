"""API HTTP cục bộ (127.0.0.1:8787) cho dashboard web và panel AutoClip trong ReiCut.

Hai giao diện cùng gọi một API nên thấy chung hàng đợi job và video. Batch chạy trong
một luồng nền, mỗi lúc một batch (worker song song là P4). Chỉ bind localhost.

    GET  /api/health               kiểm tra sống
    GET  /api/recipes              recipe trong recipes/*.json
    GET  /api/pool                 số clip/segment trong kho, số segment vừa dùng
    GET  /api/status               batch đang chạy (hoặc lần chạy gần nhất)
    POST /api/batches              {recipe, count, seed?, llm} → bắt đầu batch (409 nếu đang bận)
    GET  /api/jobs?limit=50        job mới nhất kèm render + QA
    GET  /api/renders/{id}/video   file MP4
    GET  /                         dashboard web
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


class BatchRequest(BaseModel):
    recipe: str = Field(..., description="Tên file trong recipes/, ví dụ hook-3cut-caption.json")
    count: int = Field(5, ge=1, le=100)
    seed: int | None = None
    llm: bool = True


class BatchRunner:
    """Một batch tại một thời điểm, chạy trong luồng nền; trạng thái đọc được qua API."""

    def __init__(self, run=None) -> None:
        self._run = run
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state: dict[str, Any] = {"running": False}

    @property
    def busy(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self, spec, req: BatchRequest) -> dict[str, Any]:
        with self._lock:
            if self.busy:
                raise HTTPException(409, "đang có batch chạy, đợi xong rồi tạo tiếp")
            self.state = {
                "running": True,
                "recipe": spec.name,
                "requested": req.count,
                "done": 0,
                "failed": 0,
                "llm": req.llm,
                "started_at": datetime.now(UTC).isoformat(),
                "finished_at": None,
                "stopped_reason": None,
                "error": None,
            }
            self._thread = threading.Thread(
                target=self._work, args=(spec, req), name="autoclip-batch", daemon=True
            )
            self._thread.start()
            return dict(self.state)

    def _progress(self, item) -> None:
        self.state["done" if item.status == "done" else "failed"] += 1

    def _work(self, spec, req: BatchRequest) -> None:
        run = self._run
        if run is None:
            from autoclip.batch import run_batch as run
        try:
            result = run(spec, req.count, seed=req.seed, use_llm=req.llm, progress=self._progress)
            self.state["stopped_reason"] = result.stopped_reason
        except Exception as exc:  # lỗi bất ngờ phải hiện lên giao diện, không chết lặng lẽ
            log.exception("batch lỗi")
            self.state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            self.state["running"] = False
            self.state["finished_at"] = datetime.now(UTC).isoformat()


def create_app(settings: Settings | None = None, runner: BatchRunner | None = None) -> FastAPI:
    s = settings or get_settings()
    engine = make_engine()
    upgrade(engine)
    runner = runner or BatchRunner()
    recipes_dir = REPO_ROOT / "recipes"
    app = FastAPI(title="AutoClip", version="0.1")
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
        recent = recent_segment_ids(engine, s.no_repeat_last_n_renders)
        return {
            "clips": clips,
            "segments": segments,
            "recently_used": len(recent),
            "available": max(0, segments - len(recent)),
        }

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return dict(runner.state)

    @app.post("/api/batches", status_code=202)
    def create_batch(req: BatchRequest) -> dict[str, Any]:
        return runner.start(load_spec(req.recipe), req)

    @app.get("/api/jobs")
    def jobs(limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
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

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    return app
