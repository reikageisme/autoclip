"""`autoclip batch`: sinh N video từ một recipe, mỗi video một job trong DB.

Mỗi video: planner (LLM hoặc rules) → render qua Concat → QA → ghi renders + usage_log.
Không segment nào dùng lại trong batch, và loại các segment của N render gần nhất
(`no_repeat_last_n_renders`). QA trượt hoặc lỗi → thử lại seed khác (tối đa
`attempts`). Hết segment → dừng sớm, báo đã làm được bao nhiêu.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

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


def run_batch(
    spec: RecipeSpec,
    count: int,
    *,
    seed: int | None = None,
    use_llm: bool = True,
    attempts: int = 2,
    settings: Settings | None = None,
    render_fn: Callable[[RecipeSpec, RenderPlan], RenderResult] = render,
    qa_fn: Callable[[Path, RecipeSpec, RenderPlan], QAResult] = _qa,
    progress: Callable[[BatchItem], None] | None = None,
) -> BatchResult:
    s = settings or get_settings()
    engine = make_engine()
    upgrade(engine)
    sessions = Session(engine, expire_on_commit=False)
    repo = SqlClipRepository(engine)
    pool = load_pool(repo)
    planner = make_planner(use_llm)
    base_seed = seed if seed is not None else random.randrange(1_000_000)
    exclude = recent_segment_ids(engine, s.no_repeat_last_n_renders)
    result = BatchResult(recipe=spec.name, requested=count)
    log.info(
        "batch %s x%d: planner=%s, kho %d segment, loại %d segment đã dùng gần đây",
        spec.name,
        count,
        planner.name,
        len(pool.segments),
        len(exclude),
    )

    with sessions as db:
        recipe_id = _recipe_id(db, spec)
        db.commit()

        for n in range(count):
            job_seed = base_seed + n
            job = Job(recipe_id=recipe_id, seed=job_seed, status="queued")
            db.add(job)
            db.commit()
            item = BatchItem(job_id=job.id, seed=job_seed, status="queued")

            retry_plan: RenderPlan | None = None
            for attempt in range(attempts):
                plan_seed = job_seed + attempt * RETRY_SEED_STEP
                job.attempts = attempt + 1
                job.status = "running"
                db.commit()
                try:
                    if retry_plan is not None:
                        plan, retry_plan = retry_plan, None
                    else:
                        plan = planner.plan(spec, pool, plan_seed, exclude)
                except PlanError as exc:
                    result.stopped_reason = (
                        f"Kho clip đã dùng hết sau {len(result.done)} video — nhập thêm clip nguồn"
                    )
                    if attempt == 0:
                        # Chưa dựng gì: không để lại một job "lỗi" trong hàng đợi.
                        db.delete(job)
                        db.commit()
                        item = None
                    else:
                        job.status, job.error = "failed", str(exc)
                        db.commit()
                        item.status, item.error = "failed", str(exc)
                    break
                try:
                    rendered = render_fn(spec, plan)
                    job.status = "qa"
                    db.commit()
                    qa = qa_fn(rendered.output, spec, plan)
                except Exception as exc:  # render/QA lỗi → thử seed khác
                    log.exception("job %s lần %d lỗi", job.id, attempt + 1)
                    job.error = f"{type(exc).__name__}: {exc}"
                    item.status, item.error = "failed", job.error
                    continue

                row = Render(
                    job_id=job.id,
                    output_path=str(rendered.output),
                    qa_score=qa.score,
                    qa_notes="; ".join(qa.notes) or None,
                    duration=rendered.video_sec,
                    size_bytes=rendered.output.stat().st_size,
                    **render_fields(rendered.plan, rendered),
                )
                db.add(row)
                db.flush()
                item.plan, item.qa, item.output = plan, qa, rendered.output
                item.render_sec, item.video_sec = rendered.render_sec, rendered.video_sec
                if qa.passed:
                    ids = {i.segment_id for i in plan.items if i.segment_id is not None}
                    db.add_all(
                        UsageLog(segment_id=sid, render_id=row.id, used_at=datetime.now(UTC))
                        for sid in ids
                    )
                    exclude |= ids
                    job.status, job.error = "done", None
                    item.status, item.error = "done", None
                    db.commit()
                    break
                job.error = "QA trượt: " + "; ".join(qa.notes)
                item.status, item.error = "failed", job.error
                db.commit()
                if any(n.startswith(BLACK_NOTE) for n in qa.notes):
                    # Transition của Concat đôi khi render ra đoạn đen: giữ plan, bỏ transition.
                    retry_plan = plan.model_copy(deep=True)
                    for it in retry_plan.items:
                        it.transition = None
                    retry_plan.params["transitions"] = "off (black_spans)"
            else:
                job.status = "failed"
                db.commit()

            if item is not None:
                result.items.append(item)
                if progress:
                    progress(item)
            if result.stopped_reason:
                break
    engine.dispose()
    return result
