from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from autoclip.api import BatchRunner, create_app
from autoclip.config import get_settings


class FakeResult:
    stopped_reason = None


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    calls = []

    def fake_run(spec, count, *, seed, use_llm, progress):
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
