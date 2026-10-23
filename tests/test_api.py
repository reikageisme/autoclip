from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autoclip.api import BatchRunner, create_app
from autoclip.config import get_settings


class FakeResult:
    stopped_reason = None


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FB_PAGE_ID", "123456")
    get_settings.cache_clear()
    calls = []

    def fake_run(spec, count, *, seed, use_llm, progress, workers=None):
        calls.append((spec.name, count, seed, use_llm))
        time.sleep(0.2)
        return FakeResult()

    app = create_app(runner=BatchRunner(run=fake_run))
    with TestClient(app) as c:
        c.calls = calls
        yield c
    get_settings.cache_clear()


def test_health_recipes_pool(client):
    assert client.get("/api/health").json()["ok"] is True
    recipes = client.get("/api/recipes").json()
    assert any(r["file"] == "hook-3cut-caption.json" for r in recipes)
    assert client.get("/api/pool").json() == {
        "clips": 0,
        "segments": 0,
        "recently_used": 0,
        "available": 0,
    }
    assert client.get("/api/jobs").json() == []
    assert "AutoClip" in client.get("/").text


def test_start_batch_then_busy_then_done(client):
    r = client.post(
        "/api/batches", json={"recipe": "hook-3cut-caption.json", "count": 3, "llm": False}
    )
    assert r.status_code == 202 and r.json()["running"] is True
    busy = client.post("/api/batches", json={"recipe": "hook-3cut-caption.json", "count": 1})
    assert busy.status_code == 409
    for _ in range(50):
        if not client.get("/api/status").json()["running"]:
            break
        time.sleep(0.05)
    st = client.get("/api/status").json()
    assert st["running"] is False and st["finished_at"]
    assert client.calls == [("hook-3cut-caption", 3, None, False)]


def test_rejects_unknown_or_escaping_recipe(client):
    assert client.post("/api/batches", json={"recipe": "nope.json"}).status_code == 404
    assert client.post("/api/batches", json={"recipe": "../pyproject.toml"}).status_code == 404
    assert (
        client.post(
            "/api/batches", json={"recipe": "hook-3cut-caption.json", "count": 0}
        ).status_code
        == 422
    )


def test_missing_render_video_404(client):
    assert client.get("/api/renders/999/video").status_code == 404


def test_settings_roundtrip_keeps_key_and_comments(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# giữ chú thích\nLLM_PROVIDER=none\nLLM_API_KEY=sk-old\n", encoding="utf-8")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    for k in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    get_settings.cache_clear()
    with TestClient(create_app(env_file=env)) as c:
        r = c.put(
            "/api/settings",
            json={
                "llm_provider": "openai_compat",
                "llm_model": "m1",
                "llm_api_key": "",
                "llm_base_url": "http://x/v1",
            },
        )
        assert r.status_code == 200
        text = env.read_text(encoding="utf-8")
        assert "# giữ chú thích" in text and "LLM_API_KEY=sk-old" in text
        assert "LLM_PROVIDER=openai_compat" in text and "LLM_MODEL=m1" in text
        assert "sk-old" not in r.text  # API không bao giờ trả key
    get_settings.cache_clear()


def test_ingest_rejects_missing_folder(client, tmp_path):
    assert client.post("/api/ingest", json={"folder": str(tmp_path / "nope")}).status_code == 404
    assert client.post("/api/ingest", json={"folder": "  "}).status_code == 400
    assert client.get("/api/clips").json() == []


def test_custom_style_crud(client):
    form = {
        "label": "Thử nghiệm",
        "duration_min": 12,
        "duration_max": 20,
        "transitions": ["cut", "fade"],
    }
    r = client.post("/api/recipes", json=form)
    assert r.status_code == 201, r.text
    made = r.json()
    assert made["builtin"] is False and made["label"] == "Thử nghiệm" and "est_videos" in made
    files = {x["file"]: x for x in client.get("/api/recipes").json()}
    assert made["file"] in files and files["hook-3cut-caption.json"]["builtin"] is True
    assert client.delete("/api/recipes/hook-3cut-caption.json").status_code == 400
    assert client.delete(f"/api/recipes/{made['file']}").status_code == 200
    assert "fade" in client.get("/api/recipes/options").json()["transitions"]


def test_facebook_settings_are_secret(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# giữ chú thích\nLLM_PROVIDER=none\n", encoding="utf-8")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FB_PAGE_ID", "123456")
    get_settings.cache_clear()
    with TestClient(create_app(env_file=env)) as c:
        r = c.put(
            "/api/settings",
            json={"fb_page_id": "123456", "fb_token": "EAAB-bi-mat"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["fb_page_id"] == "123456" and body["has_fb_token"] is True
        assert "EAAB-bi-mat" not in r.text
        text = env.read_text(encoding="utf-8")
        assert "FB_PAGE_ID=123456" in text and "EAAB-bi-mat" not in text
        from autoclip.publish import facebook_token

        assert facebook_token(get_settings()) == "EAAB-bi-mat"
        assert c.get("/api/settings").json()["has_fb_token"] is True
    get_settings.cache_clear()


def test_publish_and_metrics_api(client):
    from sqlalchemy.orm import Session

    from autoclip.db import make_engine
    from autoclip.db.models import Job, Recipe, Render

    settings = get_settings()
    out = settings.out_dir / "v1.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"mp4")
    engine = make_engine()
    with Session(engine, expire_on_commit=False) as db:
        recipe = Recipe(
            name="hook-3cut-caption", aspect="9:16", duration_target=[15, 30], spec={}
        )
        db.add(recipe)
        db.flush()
        job = Job(recipe_id=recipe.id, seed=1, status="done")
        db.add(job)
        db.flush()
        render = Render(job_id=job.id, output_path=str(out), title="Bài thử")
        db.add(render)
        db.commit()
        rid = render.id
    engine.dispose()

    assert (
        client.post(f"/api/renders/{rid}/publish", json={}).status_code == 400
    )  # chưa có token
    from autoclip.publish import save_facebook_token

    save_facebook_token(settings, "token-thu")
    r = client.post(f"/api/renders/{rid}/publish", json={})
    assert r.status_code == 202 and r.json()["status"] == "scheduled"
    pubs = client.get("/api/publications").json()
    assert pubs[0]["render_id"] == rid and pubs[0]["status"] == "scheduled"

    r = client.post("/api/metrics/import", json={"csv": "video,views\nv1.mp4,1234\n"})
    assert r.status_code == 200 and r.json()["imported"] == 1
    recipes = client.get("/api/recipes").json()
    row = next(x for x in recipes if x["file"] == "hook-3cut-caption.json")
    assert row["avg_views"] == 1234 and row["priority"] == 1

    r = client.post(
        "/api/batches",
        json={"recipe": "auto", "count": 1, "llm": False},
    )
    assert r.status_code == 202
    assert client.calls[-1][0] == "hook-3cut-caption"


def test_schedules_and_system_api(client):
    at = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    r = client.post(
        "/api/schedules",
        json={"recipe": "hook-3cut-caption.json", "count": 2, "at": at, "repeat_minutes": 60},
    )
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    assert r.json()["enabled"] is True and r.json()["repeat_minutes"] == 60
    rows = client.get("/api/schedules").json()
    assert len(rows) == 1 and rows[0]["id"] == sid
    assert client.post(f"/api/schedules/{sid}/toggle", json={"enabled": False}).status_code == 200
    assert client.get("/api/schedules").json()[0]["enabled"] is False
    assert client.delete(f"/api/schedules/{sid}").status_code == 200
    assert client.get("/api/schedules").json() == []
    assert client.post("/api/schedules", json={"recipe": "nope", "at": at}).status_code == 404

    system = client.get("/api/system").json()
    assert system["disk"] and "free_gb" in system["disk"][0]
    assert system["cleanup"]["videos_days"] == 30


def test_cleanup_api_keeps_recent_files(client):
    data = Path(get_settings().data_dir)
    old = data / "proxies" / "old.mp4"
    new = data / "proxies" / "new.mp4"
    for path in (old, new):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    old_time = (datetime.now(UTC) - timedelta(days=40)).timestamp()
    os.utime(old, (old_time, old_time))

    preview = client.post("/api/system/cleanup", json={"dry_run": True}).json()
    assert preview["details"]["proxies"] == 1 and old.exists()
    done = client.post("/api/system/cleanup", json={}).json()
    assert done["deleted"] == 1 and not old.exists() and new.exists()


def test_review_edit_title_and_export(client, tmp_path):
    from sqlalchemy.orm import Session

    from autoclip.db import make_engine
    from autoclip.db.models import Job, Recipe, Render

    video = get_settings().out_dir / "v1.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"mp4")
    engine = make_engine()
    with Session(engine) as db:
        recipe = Recipe(name="r", aspect="9:16", duration_target=[15, 30], spec={})
        db.add(recipe)
        db.flush()
        job = Job(recipe_id=recipe.id, seed=1, status="done")
        db.add(job)
        db.flush()
        db.add(Render(job_id=job.id, output_path=str(video), headline="HÀNG VỀ", title="cũ"))
        db.commit()
        rid = db.query(Render).one().id
    engine.dispose()

    assert (
        client.post(f"/api/renders/{rid}/review", json={"review": "approved"}).json()["review"]
        == "approved"
    )
    r = client.post(
        f"/api/renders/{rid}/edit",
        json={"title": "Nhập hàng nhanh #logistics", "headline": "hàng về"},
    )
    assert r.json() == {
        "id": rid,
        "rerender": False,
    }  # headline không đổi (khác hoa/thường) -> không dựng lại
    job_row = client.get("/api/jobs").json()[0]["render"]
    assert job_row["title"] == "Nhập hàng nhanh #logistics" and job_row["review"] == "approved"

    out = tmp_path / "xuat"
    res = client.post("/api/export", json={"folder": str(out)}).json()
    assert res["count"] == 1
    names = sorted(p.name for p in out.iterdir())
    assert names[0].endswith(".mp4") and names[1].endswith(".txt")
    assert "Nhập hàng nhanh" in (out / names[1]).read_text(encoding="utf-8")
    assert (
        client.post("/api/export", json={"folder": str(out)}).json()["count"] == 0
    )  # không xuất lại
