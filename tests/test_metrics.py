from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from autoclip.config import get_settings
from autoclip.db import make_engine
from autoclip.db.migrate import upgrade
from autoclip.db.models import Job, Recipe, Render
from autoclip.metrics import best_recipe, import_metrics, list_metrics, recipe_scores


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    engine = make_engine()
    upgrade(engine)

    def video(recipe_name: str, seed: int) -> Render:
        out = get_settings().out_dir / f"{recipe_name}-{seed}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")
        spec = {"name": recipe_name, "aspect": "9:16", "duration_target": [15, 30]}
        recipe = Recipe(
            name=recipe_name, aspect="9:16", duration_target=[15, 30], spec=spec
        )
        with Session(engine, expire_on_commit=False) as db:
            db.add(recipe)
            db.flush()
            job = Job(recipe_id=recipe.id, seed=seed, status="done")
            db.add(job)
            db.flush()
            row = Render(job_id=job.id, output_path=str(out))
            db.add(row)
            db.commit()
            return row
    a = video("kieu-a", 1)
    b = video("kieu-b", 2)
    yield engine, a.id, b.id
    engine.dispose()
    get_settings.cache_clear()


def test_import_csv_views_and_rank_recipes(env):
    engine, a, b = env
    report = import_metrics(
        engine,
        "render_id,views,avg_watch,likes\n"
        f"{a},1000,3.5,10\n"
        f"{b},5000,4.0,20\n"
        "bad,9,1,1\n",
    )
    assert report.imported == 2 and report.skipped == 1
    assert report.errors and "không tìm thấy" in report.errors[0]
    assert recipe_scores(engine)["kieu-b"]["avg_views"] == 5000
    assert best_recipe(engine, ["kieu-a", "kieu-b"]) == "kieu-b"

    # CSV kiểu Việt Nam: dấu ;, tên file, số 1.2K; lần đo mới hơn thay lần cũ
    report = import_metrics(
        engine,
        "video;luot_xem;thoi_luong_xem\n"
        "kieu-a-1.mp4;1.2K;4,5\n",
        platform="facebook",
    )
    assert report.imported == 1
    assert recipe_scores(engine)["kieu-a"]["avg_views"] == 1200
    assert len(list_metrics(engine)) == 3


def test_views_with_dot_thousands(env):
    engine, a, _ = env
    report = import_metrics(engine, f"render_id,views\n{a},1.234\n")
    assert report.imported == 1
    assert recipe_scores(engine)["kieu-a"]["avg_views"] == 1234


def test_import_rejects_empty_or_bad_csv(env):
    engine, _, _ = env
    with pytest.raises(ValueError):
        import_metrics(engine, "   ")
    with pytest.raises(ValueError):
        import_metrics(engine, "views\n1\n")  # không có video
