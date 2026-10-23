from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from autoclip.config import get_settings
from autoclip.db import make_engine
from autoclip.db.migrate import upgrade
from autoclip.db.models import Job, Publication, Recipe, Render
from autoclip.publish import (
    PublishError,
    cancel_publication,
    publish_due,
    retry_publication,
    save_facebook_token,
    schedule_publication,
)
from autoclip.secrets import SecretStore


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FB_PAGE_ID", "123456")
    get_settings.cache_clear()
    engine = make_engine()
    upgrade(engine)
    settings = get_settings()
    save_facebook_token(settings, "page-token")
    out = settings.out_dir / "v1.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"mp4")
    with Session(engine, expire_on_commit=False) as db:
        recipe = Recipe(name="r", aspect="9:16", duration_target=[15, 30], spec={})
        db.add(recipe)
        db.flush()
        job = Job(recipe_id=recipe.id, seed=1, status="done")
        db.add(job)
        db.flush()
        render = Render(job_id=job.id, output_path=str(out), headline="HÀNG VỀ", title="Mô tả")
        db.add(render)
        db.commit()
        rid = render.id
    yield engine, settings, rid
    engine.dispose()
    get_settings.cache_clear()


def test_token_encrypted_on_disk(env):
    _, settings, _ = env
    raw = SecretStore(settings.data_dir)._path("facebook_page_token").read_bytes()
    assert raw.startswith(b"DPAPI1")
    assert b"page-token" not in raw


def test_schedule_and_publish_due(env):
    engine, settings, rid = env
    row = schedule_publication(
        engine, rid, settings=settings, at=datetime.now(UTC) - timedelta(seconds=1)
    )
    assert row.status == "scheduled"
    with Session(engine) as db:
        db.get(Publication, row.id).scheduled_at = datetime.now(UTC) - timedelta(seconds=10)
        db.commit()
    seen = []

    def fake(render, settings, *, token, caption):
        seen.append((render.id, token, caption))
        return "page_123_post_9"

    result = publish_due(engine, settings, publisher=fake)
    assert [r.status for r in result] == ["published"]
    assert result[0].post_id == "page_123_post_9"
    assert seen == [(rid, "page-token", "Mô tả")]
    with Session(engine) as db:
        saved = db.get(Publication, result[0].publication_id)
        assert saved.status == "published" and saved.fb_post_id == "page_123_post_9"


def test_failed_publish_can_retry_or_cancel(env):
    engine, settings, rid = env
    row = schedule_publication(
        engine, rid, settings=settings, at=datetime.now(UTC) - timedelta(seconds=1)
    )
    with Session(engine) as db:
        db.get(Publication, row.id).scheduled_at = datetime.now(UTC) - timedelta(seconds=10)
        db.commit()

    def boom(*args, **kwargs):
        raise PublishError("token hết hạn")

    result = publish_due(engine, settings, publisher=boom)
    assert result[0].status == "failed" and "token" in (result[0].error or "")
    assert retry_publication(engine, row.id)
    assert cancel_publication(engine, row.id)
    assert not cancel_publication(engine, row.id)


def test_missing_page_or_token_is_rejected(env, monkeypatch):
    engine, settings, rid = env
    monkeypatch.setenv("FB_PAGE_ID", "")
    get_settings.cache_clear()
    with pytest.raises(PublishError):
        schedule_publication(engine, rid, settings=get_settings())


def test_interrupted_publication_marked_failed(env):
    from autoclip.publish import recover_interrupted

    engine, settings, rid = env
    row = schedule_publication(engine, rid, settings=settings)
    with Session(engine) as db:
        db.get(Publication, row.id).status = "publishing"
        db.commit()
    assert recover_interrupted(engine) == 1
    with Session(engine) as db:
        assert db.get(Publication, row.id).status == "failed"


def test_cleanup_keeps_library_and_pending_videos(env):
    import os

    from autoclip.db.models import Clip
    from autoclip.ops import cleanup, protected_paths

    engine, settings, rid = env
    proxies = settings.data_dir / "proxies"
    proxies.mkdir(parents=True)
    used, stale = proxies / "used.mp4", proxies / "stale.mp4"
    old_video = settings.out_dir / "old.mp4"
    for f in (used, stale, old_video):
        f.write_bytes(b"x")
    pending = settings.out_dir / "v1.mp4"
    old = (datetime.now(UTC) - timedelta(days=90)).timestamp()
    for f in (used, stale, old_video, pending):
        os.utime(f, (old, old))
    with Session(engine) as db:
        db.add(
            Clip(
                path=str(used),
                sha1="a" * 40,
                duration=1,
                width=1,
                height=1,
                fps=30,
                has_audio=False,
            )
        )
        db.commit()
    schedule_publication(engine, rid, settings=settings)

    keep = protected_paths(engine)
    dry = cleanup(settings, dry_run=True, keep=keep)
    assert dry.details["proxies"] == 1 and dry.details["videos"] == 1
    assert stale.exists()
    cleanup(settings, keep=keep)
    assert used.exists() and pending.exists()
    assert not stale.exists() and not old_video.exists()
