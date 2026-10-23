"""`autoclip batch`: sinh N video từ một recipe, mỗi video một job trong DB.

Mỗi video: planner (LLM hoặc rules) → render qua Concat → QA → ghi renders + usage_log.
Không segment nào dùng lại trong batch, và loại các segment của N render gần nhất
(`no_repeat_last_n_renders`). QA trượt hoặc lỗi → thử lại seed khác (tối đa
`attempts`). Hết segment → dừng sớm, báo đã làm được bao nhiêu.
"""

from __future__ import annotations

import logging
import random
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from autoclip.config import Settings, get_settings
from autoclip.contracts import QAResult, RecipeSpec, RenderPlan
from autoclip.db import SqlClipRepository, make_engine
from autoclip.db.migrate import upgrade
from autoclip.db.models import Job, Recipe, Render, UsageLog
from autoclip.diversity import recent_segment_ids
from autoclip.planner import PlanError, load_pool, make_planner
from autoclip.render import RenderResult, render

log = logging.getLogger(__name__)

RETRY_SEED_STEP = 100_000
BLACK_NOTE = "black_spans"


@dataclass
class BatchItem:
    job_id: int
    seed: int
    status: str
    output: Path | None = None
    plan: RenderPlan | None = None
    qa: QAResult | None = None
    render_sec: float = 0.0
    video_sec: float = 0.0
    error: str | None = None


@dataclass
class BatchResult:
    recipe: str
    requested: int
    items: list[BatchItem] = field(default_factory=list)
    stopped_reason: str | None = None

    @property
    def done(self) -> list[BatchItem]:
        return [i for i in self.items if i.status == "done"]


def headline_of(plan: RenderPlan) -> str | None:
    return next((t.text for t in plan.texts if t.kind != "captions"), None)


def render_fields(plan: RenderPlan, rendered: RenderResult) -> dict:
    """Phần của dòng renders để duyệt, sửa chữ và mở lại trong ReiCut."""
    return {
        "plan": plan.model_dump(mode="json"),
        "headline": headline_of(plan),
        "title": str(plan.params.get("title") or "") or None,
        "project_path": str(rendered.project) if rendered.project else None,
    }


def _qa(path: Path, spec: RecipeSpec, plan: RenderPlan) -> QAResult:
    from autoclip.qa import evaluate
    from autoclip.qa.checks import check_black_spans

    result = evaluate(path, spec, frames_count=5)
    notes = list(result.notes)
    passed = result.passed
    black = check_black_spans(path)
    if not black.passed:
        passed = False
        notes.append(f"{BLACK_NOTE}: {black.detail}")
    ids = [i.segment_id for i in plan.items]
    if len(ids) != len(set(ids)):
        passed = False
        notes.append("plan dùng lại segment")
    return QAResult(score=result.score, passed=passed, notes=notes)


def _recipe_id(s: Session, spec: RecipeSpec) -> int:
    row = s.scalar(select(Recipe).where(Recipe.name == spec.name))
    data = spec.model_dump(mode="json")
    if row is None:
        row = Recipe(
            name=spec.name,
            source_project=spec.source_project,
            aspect=spec.aspect,
            duration_target=list(spec.duration_target),
            spec=data,
        )
        s.add(row)
    else:
        row.spec, row.aspect, row.duration_target = data, spec.aspect, list(spec.duration_target)
    s.flush()
    return row.id


class VoiceOptions(BaseModel):
    """Chế độ lồng tiếng cho batch: AI viết theo `topic`, hoặc đọc `script` người gõ."""

    topic: str = ""
    script: str = ""
    """Mỗi dòng một câu. Có AI + topic thì đây là kịch bản dự phòng khi AI lỗi."""
    speed: float = Field(1.0, ge=0.7, le=1.5)
    lines: int = Field(0, ge=0, le=20)
    """Số câu AI viết; 0 = theo độ dài kiểu video (~3.5 giây/câu)."""


def voice_plan(
    spec: RecipeSpec,
    pool,
    seed: int,
    exclude: set[int],
    voice: VoiceOptions,
    settings: Settings,
    use_llm: bool,
) -> RenderPlan:
    """Kịch bản → giọng đọc → plan lồng tiếng cho một video."""
    from autoclip.diversity import make_variant
    from autoclip.llm import LLMClient
    from autoclip.planner.voiceover import Line, plan_voiceover
    from autoclip.script import clean_lines, write_script
    from autoclip.voice import synthesize

    lines = clean_lines(voice.script)
    headline, title = "", ""
    llm = LLMClient.from_settings(settings)
    if use_llm and voice.topic.strip() and llm.available:
        target = sum(spec.duration_target) / 2
        n = voice.lines or max(3, min(12, round(target / 3.5)))
        try:
            lines, headline, title = write_script(
                llm, voice.topic, lines=n, angle=make_variant(spec, seed).angle
            )
        except Exception as exc:  # AI lỗi: dùng kịch bản người gõ nếu có
            log.warning("AI viết kịch bản lỗi (seed %s): %s", seed, exc)
    if not lines:
        raise ValueError("chưa có kịch bản: nhập chủ đề (cần AI) hoặc gõ kịch bản")
    folder = settings.data_dir / "voice" / str(seed)
    spoken = []
    for i, text in enumerate(lines):
        wav = folder / f"{i:02d}.wav"
        spoken.append(
            Line(text, wav, synthesize(text, wav, settings.data_dir / "models", voice.speed))
        )
    return plan_voiceover(
        spec, pool, seed, exclude, spoken, headline=headline or lines[0], title=title
    )


def _qa_spec(spec: RecipeSpec, plan: RenderPlan) -> RecipeSpec:
    """Lồng tiếng: thời lượng do lời đọc quyết định, không theo khoảng của kiểu video."""
    if plan.params.get("planner") != "voiceover":
        return spec
    total = sum((i.src_end - i.src_start) / i.speed for i in plan.items)
    outro = (plan.params.get("brand") or {}).get("outro") or {}
    return spec.model_copy(
        update={"duration_target": (total * 0.9, total * 1.1 + float(outro.get("seconds", 0)) + 1)}
    )


def run_batch(
    spec: RecipeSpec,
    count: int,
    *,
    seed: int | None = None,
    use_llm: bool = True,
    voice: VoiceOptions | None = None,
    attempts: int = 3,
    settings: Settings | None = None,
    workers: int | None = None,
    render_fn: Callable[[RecipeSpec, RenderPlan], RenderResult] = render,
    qa_fn: Callable[[Path, RecipeSpec, RenderPlan], QAResult] = _qa,
    progress: Callable[[BatchItem], None] | None = None,
) -> BatchResult:
    s = settings or get_settings()
    engine = make_engine()
    upgrade(engine)
    repo = SqlClipRepository(engine)
    pool = load_pool(repo)
    planner = make_planner(use_llm)
    base_seed = seed if seed is not None else random.randrange(1_000_000)
    exclude = recent_segment_ids(engine, s.no_repeat_last_n_renders)
    result = BatchResult(recipe=spec.name, requested=count)
    worker_count = max(1, min(int(workers or s.render_workers or 1), max(1, count), 16))
    lock = threading.Lock()
    stop = threading.Event()

    with Session(engine, expire_on_commit=False) as db:
        recipe_id = _recipe_id(db, spec)
        db.commit()

    log.info(
        "batch %s x%d: planner=%s, %d worker, kho %d segment, loại %d segment đã dùng gần đây",
        spec.name,
        count,
        planner.name,
        worker_count,
        len(pool.segments),
        len(exclude),
    )

    def set_job(job_id: int, **fields: object) -> None:
        with Session(engine, expire_on_commit=False) as db:
            job = db.get(Job, job_id)
            if job is not None:
                for key, value in fields.items():
                    setattr(job, key, value)
                db.commit()

    def delete_job(job_id: int) -> None:
        with Session(engine, expire_on_commit=False) as db:
            job = db.get(Job, job_id)
            if job is not None:
                db.delete(job)
                db.commit()

    def release(ids: set[int]) -> None:
        with lock:
            exclude.difference_update(ids)

    def process(n: int) -> BatchItem | None:
        if stop.is_set():
            return None
        job_seed = base_seed + n
        with Session(engine, expire_on_commit=False) as db:
            job = Job(recipe_id=recipe_id, seed=job_seed, status="queued")
            db.add(job)
            db.commit()
            job_id = job.id
        item = BatchItem(job_id=job_id, seed=job_seed, status="queued")
        reserved: set[int] = set()

        for attempt in range(attempts):
            if stop.is_set() and attempt == 0:
                delete_job(job_id)
                return None
            plan_seed = job_seed + attempt * RETRY_SEED_STEP
            set_job(job_id, attempts=attempt + 1, status="running")
            with lock:
                if stop.is_set() and attempt == 0:
                    delete_job(job_id)
                    return None
                try:
                    plan = (
                        voice_plan(spec, pool, plan_seed, exclude, voice, s, use_llm)
                        if voice
                        else planner.plan(spec, pool, plan_seed, exclude)
                    )
                except PlanError as exc:
                    result.stopped_reason = (
                        f"Kho clip đã dùng hết sau {len(result.done)} video — nhập thêm clip nguồn"
                    )
                    stop.set()
                    if attempt == 0:
                        delete_job(job_id)
                        return None
                    set_job(job_id, status="failed", error=str(exc))
                    item.status, item.error = "failed", str(exc)
                    return item
                ids = {i.segment_id for i in plan.items if i.segment_id is not None}
                reserved |= ids - exclude
                exclude.update(ids)

            try:
                rendered = render_fn(spec, plan)
                set_job(job_id, status="qa")
                qa = qa_fn(rendered.output, _qa_spec(spec, rendered.plan), rendered.plan)
            except Exception as exc:  # render/QA lỗi → thử seed khác
                log.exception("job %s lần %d lỗi", job_id, attempt + 1)
                error = f"{type(exc).__name__}: {exc}"
                item.status, item.error = "failed", error
                set_job(job_id, error=error)
                release(reserved)
                reserved.clear()
                continue

            item.plan, item.qa, item.output = plan, qa, rendered.output
            item.render_sec, item.video_sec = rendered.render_sec, rendered.video_sec
            with Session(engine, expire_on_commit=False) as db:
                row = Render(
                    job_id=job_id,
                    output_path=str(rendered.output),
                    qa_score=qa.score,
                    qa_notes="; ".join(qa.notes) or None,
                    duration=rendered.video_sec,
                    size_bytes=rendered.output.stat().st_size,
                    **render_fields(rendered.plan, rendered),
                )
                db.add(row)
                db.flush()
                job = db.get(Job, job_id)
                if qa.passed:
                    ids = {i.segment_id for i in plan.items if i.segment_id is not None}
                    db.add_all(
                        UsageLog(segment_id=sid, render_id=row.id, used_at=datetime.now(UTC))
                        for sid in ids
                    )
                    if job is not None:
                        job.status, job.error = "done", None
                    item.status, item.error = "done", None
                    db.commit()
                    return item
                if job is not None:
                    job.error = "QA trượt: " + "; ".join(qa.notes)
                item.status = "failed"
                item.error = "QA trượt: " + "; ".join(qa.notes)
                db.commit()
            # Đoạn đen: lỗi render của Concat phụ thuộc chuỗi cảnh liền trước (không phải
            # transition) — lần thử sau lập plan mới bằng seed khác, chuỗi cảnh khác.
            release(reserved)
            reserved.clear()

        set_job(job_id, status="failed")
        return item

    try:
        if count > 0:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(process, n) for n in range(count)]
                for future in as_completed(futures):
                    item = future.result()
                    if item is not None:
                        result.items.append(item)
                        if progress:
                            progress(item)
    finally:
        result.items.sort(key=lambda i: i.seed)
        engine.dispose()
    return result
