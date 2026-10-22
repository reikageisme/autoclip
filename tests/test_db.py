from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from autoclip.contracts import ClipInfo, ClipRepository, SegmentFilter, SegmentInfo
from autoclip.db import SqlClipRepository, make_engine
from autoclip.db.migrate import upgrade
from autoclip.db.models import Base


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    upgrade(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def repo(engine) -> SqlClipRepository:
    return SqlClipRepository(engine)


def _clip(sha1: str = "a" * 40) -> ClipInfo:
    return ClipInfo(
        path="C:/clips/x.mp4",
        sha1=sha1,
        duration=10.0,
        width=1080,
        height=1920,
        fps=30.0,
        has_audio=True,
    )


def test_migration_matches_models(engine):
    tables = set(inspect(engine).get_table_names()) - {"alembic_version"}
    assert tables == set(Base.metadata.tables)
    for name, table in Base.metadata.tables.items():
        cols = {c["name"] for c in inspect(engine).get_columns(name)}
        assert cols == {c.name for c in table.columns}, name


def test_wal_enabled(engine):
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"


def test_repo_satisfies_protocol(repo):
    assert isinstance(repo, ClipRepository)


def test_add_clip_is_idempotent_by_sha1(repo):
    first = repo.add_clip(_clip())
    second = repo.add_clip(_clip())
    assert first.id is not None
    assert second.id == first.id
    assert repo.get_clip_by_sha1("a" * 40) == first
    assert repo.get_clip_by_sha1("b" * 40) is None


def test_segments_roundtrip_and_filter(repo):
    clip = repo.add_clip(_clip())
    other = repo.add_clip(_clip("c" * 40))
    saved = repo.add_segments(
        clip.id,
        [
            SegmentInfo(start=0, end=2, tags=["face", "action"], hook_score=0.9, phash="ff00"),
            SegmentInfo(start=2, end=6, tags=["calm"], hook_score=0.2, transcript="xin chào"),
            SegmentInfo(start=6, end=7, tags=[], hook_score=0.5),
        ],
    )
    repo.add_segments(other.id, [SegmentInfo(start=0, end=3, tags=["face"])])

    assert [s.clip_id for s in saved] == [clip.id] * 3
    assert all(s.id is not None for s in saved)
    assert saved[1].transcript == "xin chào"

    assert len(repo.list_segments()) == 4
    assert len(repo.list_segments(SegmentFilter(clip_id=clip.id))) == 3
    assert {s.start for s in repo.list_segments(SegmentFilter(tags_any=["face"]))} == {0}
    assert len(repo.list_segments(SegmentFilter(tags_any=["face"]))) == 2
    assert [s.id for s in repo.list_segments(SegmentFilter(min_hook_score=0.5))] == [
        saved[0].id,
        saved[2].id,
    ]
    assert [
        s.end
        for s in repo.list_segments(
            SegmentFilter(clip_id=clip.id, min_duration=1.5, max_duration=3)
        )
    ] == [2]
    excluded = repo.list_segments(SegmentFilter(clip_id=clip.id, exclude_ids=[saved[0].id]))
    assert [s.id for s in excluded] == [saved[1].id, saved[2].id]
    assert len(repo.list_segments(SegmentFilter(limit=2))) == 2
