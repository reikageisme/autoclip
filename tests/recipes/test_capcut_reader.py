from __future__ import annotations

from pathlib import Path

import pytest

from autoclip.contracts import DraftStatus
from autoclip.recipes.capcut_reader import DraftError, detect, load

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "capcut"


def test_detect_statuses() -> None:
    assert detect(FIXTURES / "basic") is DraftStatus.OK
    assert detect(FIXTURES / "captions") is DraftStatus.OK
    assert detect(FIXTURES / "encrypted") is DraftStatus.ENCRYPTED
    assert detect(FIXTURES / "corrupt") is DraftStatus.CORRUPT
    assert detect(FIXTURES / "missing") is DraftStatus.MISSING
    assert detect(FIXTURES / "does-not-exist") is DraftStatus.MISSING


def test_load_basic_draft() -> None:
    draft = load(FIXTURES / "basic")
    assert draft.name == "Basic Hook"
    assert draft.aspect == "9:16"
    assert (draft.canvas_width, draft.canvas_height) == (1080, 1920)
    assert draft.fps == 30
    assert draft.duration == pytest.approx(7.0)
    assert len(draft.tracks) == 3
    assert len(draft.visual_segments) == 3
    assert len(draft.text_segments) == 2
    assert len(draft.audio_segments) == 1

    hook, body, outro = draft.visual_segments
    assert hook.start == pytest.approx(0.0)
    assert hook.duration == pytest.approx(2.0)
    assert body.source_start == pytest.approx(0.5)
    assert body.transition is not None
    assert body.transition.name == "Zoom"
    assert body.transition.duration == pytest.approx(0.5)
    assert body.sticker is not None
    assert body.sticker.name == "Arrow"
    assert [effect.name for effect in hook.effects] == ["Vignette", "Hologram"]
    assert outro.transition is not None
    assert outro.transition.name == "Glitch Pro"

    first_text = draft.text_segments[0]
    assert first_text.text is not None
    assert first_text.text.text == "WOW!"
    assert first_text.text.font_family == "Inter"
    assert first_text.text.font_size == pytest.approx(14.0)
    assert first_text.text.bold is True
    assert first_text.text.has_stroke is True
    assert first_text.text.color == "#ffffff"
    assert first_text.transform == {"x": 0.0, "y": -0.7}
    assert first_text.end == pytest.approx(2.0)


def test_load_accepts_draft_info_filename() -> None:
    draft = load(FIXTURES / "captions")
    assert draft.draft_file.endswith("draft_info.json")
    assert draft.aspect == "16:9"
    assert draft.fps == 25
    assert draft.duration == pytest.approx(5.0)
    assert [segment.speed for segment in draft.visual_segments] == [1.25, 1.0]


def test_load_corrupt_raises_with_status() -> None:
    with pytest.raises(DraftError) as caught:
        load(FIXTURES / "corrupt")
    assert caught.value.status is DraftStatus.CORRUPT


def test_load_encrypted_raises_with_status() -> None:
    with pytest.raises(DraftError) as caught:
        load(FIXTURES / "encrypted")
    assert caught.value.status is DraftStatus.ENCRYPTED


def test_load_missing_raises_with_status() -> None:
    with pytest.raises(DraftError) as caught:
        load(FIXTURES / "missing")
    assert caught.value.status is DraftStatus.MISSING


def test_text_style_parse_error_is_isolated() -> None:
    draft = load(FIXTURES / "basic")
    assert all(not segment.text.parse_error for segment in draft.text_segments if segment.text)
