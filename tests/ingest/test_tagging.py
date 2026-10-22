from __future__ import annotations

import numpy as np
import pytest

from autoclip.ingest.tagging import (
    HeuristicTagger,
    TaggerUnavailable,
    TagResult,
    VisionLLMTagger,
    combine_hook_score,
    frame_statistics,
)


def _frame(value: int, size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_frame_statistics_brightness_and_motion() -> None:
    brightness, motion = frame_statistics([_frame(0), _frame(255)])
    assert brightness == pytest.approx(0.5, abs=0.02)
    assert motion == pytest.approx(1.0, abs=0.02)


def test_frame_statistics_empty() -> None:
    assert frame_statistics([]) == (0.0, 0.0)


def test_combine_hook_score_bounds() -> None:
    assert combine_hook_score(0.15, True, True) == pytest.approx(1.0)
    assert combine_hook_score(0.0, False, False) == 0.0
    assert combine_hook_score(0.0, True, False) == pytest.approx(0.30)
    assert combine_hook_score(0.0, False, True) == pytest.approx(0.25)
    assert combine_hook_score(1.0, False, False) == pytest.approx(0.45)


def test_heuristic_tagger_tags_and_scores() -> None:
    tagger = HeuristicTagger(face_detector=lambda frames: True)
    result = tagger.tag([_frame(200)], "  hello there ")
    assert isinstance(result, TagResult)
    assert "bright" in result.tags
    assert "static" in result.tags
    assert "face" in result.tags
    assert "speech" in result.tags
    assert 0.0 <= result.hook_score <= 1.0


def test_heuristic_tagger_dark_silent_frames() -> None:
    tagger = HeuristicTagger(face_detector=lambda frames: False)
    result = tagger.tag([_frame(0)], "")
    assert "dark" in result.tags
    assert "face" not in result.tags
    assert "speech" not in result.tags
    assert result.hook_score == 0.0


def test_heuristic_tagger_without_frames() -> None:
    result = HeuristicTagger(face_detector=lambda frames: False).tag([], "")
    assert result.hook_score == 0.0
    assert "dark" in result.tags


def test_vision_tagger_is_skipped_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    tagger = VisionLLMTagger(provider="openai")
    assert tagger.available is False
    with pytest.raises(TaggerUnavailable):
        tagger.tag([_frame(10)], "text")


def test_vision_tagger_uses_injected_client() -> None:
    tagger = VisionLLMTagger(client=lambda frames, transcript: TagResult(["face"], 0.9))
    assert tagger.available is True
    result = tagger.tag([_frame(10)], "text")
    assert result.tags == ["face"]
    assert result.hook_score == 0.9
