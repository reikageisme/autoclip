from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from autoclip.batch import run_batch
from autoclip.config import get_settings
from autoclip.contracts import ClipInfo, QAResult, SegmentInfo
from autoclip.db import SqlClipRepository, make_engine
from autoclip.db.migrate import upgrade
from autoclip.db.models import Job, Render, UsageLog
from autoclip.diversity import recent_segment_ids
from autoclip.render import RenderResult, load_recipe

RECIPE = Path(__file__).resolve().parents[1] / "recipes" / "hook-3cut-caption.json"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    get_settings.cache_clear()
    engine = make_engine()
    upgrade(engine)
    repo = SqlClipRepository(engine)
    for cid in range(1, 4):
        f = tmp_path / f"c{cid}.mp4"
        f.write_bytes(b"x")
        clip = repo.add_clip(
            ClipInfo(
                path=str(f),
                sha1=f"{cid:040d}",
                duration=40.0,
                width=1080,
                height=1920,
                fps=30.0,
                has_audio=True,
            )
        )
        repo.add_segments(
            clip.id,
            [
                SegmentInfo(start=k * 4.0, end=k * 4.0 + 4.0, tags=["action"], hook_score=0.5)
                for k in range(8)
            ],
        )
    yield engine, tmp_path
    engine.dispose()
    get_settings.cache_clear()


def fake_render(tmp_path):
    def _render(spec, plan):
        out = tmp_path / f"out-{plan.seed}.mp4"
        out.write_bytes(b"video")
        return RenderResult(output=out, plan=plan, video_sec=16.0, render_sec=30.0, commands=40)

    return _render


def ok_qa(path, spec, plan):
    return QAResult(score=0.9, passed=True, notes=[])


def test_batch_no_repeat_and_db_rows(env):
    engine, tmp_path = env
    spec = load_recipe(RECIPE)
    r = run_batch(spec, 3, seed=1, use_llm=False, render_fn=fake_render(tmp_path), qa_fn=ok_qa)
    assert len(r.done) == 3
    sets = [{i.segment_id for i in d.plan.items} for d in r.done]
    assert all(not (a & b) for k, a in enumerate(sets) for b in sets[k + 1 :])
    with Session(engine) as s:
        assert s.scalar(select(func.count()).select_from(Job).where(Job.status == "done")) == 3
        assert s.scalar(select(func.count()).select_from(Render)) == 3
        assert s.scalar(select(func.count()).select_from(UsageLog)) == sum(len(x) for x in sets)
    # batch sau tránh segment của các render gần đây
    assert recent_segment_ids(engine, 50) == set().union(*sets)


def test_batch_stops_when_pool_exhausted(env):
    _, tmp_path = env
    spec = load_recipe(RECIPE)
    r = run_batch(spec, 20, seed=1, use_llm=False, render_fn=fake_render(tmp_path), qa_fn=ok_qa)
    assert 0 < len(r.done) < 20
    assert r.stopped_reason and "dùng hết" in r.stopped_reason
    assert all(i.status == "done" for i in r.items)  # không có job "lỗi" vì hết kho


def test_batch_retries_when_qa_fails(env):
    _, tmp_path = env
    calls = []

    def flaky_qa(path, spec, plan):
        calls.append(plan.seed)
        return QAResult(score=0.2, passed=len(calls) > 1, notes=["khung đen"])

    r = run_batch(
        load_recipe(RECIPE),
        1,
        seed=5,
        use_llm=False,
        render_fn=fake_render(tmp_path),
        qa_fn=flaky_qa,
    )
    assert len(r.done) == 1 and calls == [5, 100_005]


def test_black_spans_retry_with_new_plan(env):
    _, tmp_path = env
    seen = []

    def black_then_ok(path, spec, plan):
        seen.append(plan.seed)
        if len(seen) == 1:
            return QAResult(score=0.5, passed=False, notes=["black_spans: black 6.30-7.07s"])
        return QAResult(score=1.0, passed=True, notes=[])

    r = run_batch(
        load_recipe(RECIPE),
        1,
        seed=9,
        use_llm=False,
        render_fn=fake_render(tmp_path),
        qa_fn=black_then_ok,
    )
    assert len(r.done) == 1
    assert seen == [9, 100_009]  # lần 2 là plan mới, seed khác
