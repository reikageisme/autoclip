from __future__ import annotations

import json
import shutil
from pathlib import Path

from autoclip.recipes.capcut_reader import load
from autoclip.recipes.extract import extract_recipe, report, write_report

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "capcut"
SNAPSHOTS = Path(__file__).resolve().parent / "snapshots"


def _dump(spec: object) -> dict[str, object]:
    return spec.model_dump(mode="json", exclude_none=True)  # type: ignore[attr-defined]


def test_extract_basic_matches_snapshot() -> None:
    spec = extract_recipe(load(FIXTURES / "basic"))
    snapshot = json.loads((SNAPSHOTS / "basic_recipe.json").read_text(encoding="utf-8"))
    assert _dump(spec) == snapshot
    assert spec.name == "basic-hook-recipe"
    assert spec.aspect == "9:16"
    assert spec.duration_target == (6.3, 7.7)
    assert spec.music.mode == "source"
    assert [slot.role for slot in spec.slots] == ["hook", "body", "outro"]
    assert spec.slots[1].repeat == (1, 1)
    assert spec.slots[1].transition == ["concat.zoom-transition"]
    assert [text.kind for text in spec.texts] == ["headline", "headline"]
    assert spec.variation.palette == ["#ffd400", "#ffffff"]
    assert spec.variation.fonts == ["Inter", "Roboto"]
    assert {(item.kind, item.name) for item in spec.unmapped} == {
        ("effect", "Hologram"),
        ("sticker", "Arrow"),
        ("transition", "Glitch Pro"),
    }


def test_extract_captions() -> None:
    spec = extract_recipe(load(FIXTURES / "captions"))
    assert spec.aspect == "16:9"
    assert spec.music.mode == "none"
    assert len(spec.texts) == 4
    assert all(text.kind == "captions" for text in spec.texts)
    assert all(text.source == "draft" for text in spec.texts)
    assert [slot.role for slot in spec.slots] == ["hook", "outro"]
    assert spec.slots[0].speed == (1.25, 1.25)
    assert spec.slots[1].transition == ["concat.dissolve"]
    assert spec.unmapped == []


def test_extract_minimal() -> None:
    spec = extract_recipe(load(FIXTURES / "minimal"))
    assert [slot.role for slot in spec.slots] == ["hook"]
    assert spec.texts == []
    assert spec.music.mode == "none"
    assert spec.variation.fonts == ["Hanken Grotesk"]
    assert spec.unmapped == []


def test_report_counts_all_statuses() -> None:
    markdown = report(FIXTURES)
    assert "# Báo cáo draft CapCut" in markdown
    assert "| basic | ok |" in markdown
    assert "| captions | ok |" in markdown
    assert "| minimal | ok |" in markdown
    assert "| encrypted | encrypted |" in markdown
    assert "| corrupt | corrupt |" in markdown
    assert "Tìm thấy: **5** draft" in markdown
    assert "Tổng thời lượng draft đọc được: **13.5s**" in markdown


def test_report_empty_directory(tmp_path: Path) -> None:
    markdown = report(tmp_path)
    assert "Tìm thấy: **0** draft" in markdown
    assert "không có draft" in markdown


def test_write_report(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    shutil.copytree(FIXTURES / "basic", samples / "basic")
    output = write_report(samples, tmp_path / "out" / "report.md")
    assert output.is_file()
    assert "| basic | ok |" in output.read_text(encoding="utf-8")
