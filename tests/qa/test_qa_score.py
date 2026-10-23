from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from autoclip.llm import LLMClient
from autoclip.qa.frames import grab_frame
from autoclip.qa.score import VisionLLMScorer, evaluate


def _spec(aspect: str = "9:16", target=(1.5, 2.5)):
    return SimpleNamespace(aspect=aspect, duration_target=target)


def test_evaluate_good_render(qa_media: dict[str, Path]) -> None:
    result = evaluate(qa_media["testsrc"], _spec())
    assert result.passed is True
    assert result.score >= 0.9
    assert result.notes == []


def test_evaluate_black_render_fails(qa_media: dict[str, Path]) -> None:
    result = evaluate(qa_media["black"], None)
    assert result.passed is False
    assert result.score < 0.6
    assert any(note.startswith("frames:") for note in result.notes)
    assert any(note.startswith("audio:") for note in result.notes)


def test_evaluate_wrong_aspect_fails(qa_media: dict[str, Path]) -> None:
    result = evaluate(qa_media["testsrc"], _spec(aspect="16:9"))
    assert result.passed is False
    assert any(note.startswith("aspect:") for note in result.notes)


def test_evaluate_wrong_duration_fails(qa_media: dict[str, Path]) -> None:
    result = evaluate(qa_media["testsrc"], _spec(target=(20.0, 30.0)))
    assert result.passed is False
    assert any(note.startswith("duration:") for note in result.notes)


def test_evaluate_with_grab_frame_callable(qa_media: dict[str, Path]) -> None:
    path = qa_media["testsrc"]

    def grab(timestamp: float):
        return grab_frame(path, timestamp)

    result = evaluate(path, _spec(), grab_frame_fn=grab)
    assert result.passed is True


def test_evaluate_merges_vision_notes(qa_media: dict[str, Path]) -> None:
    result = evaluate(
        qa_media["testsrc"],
        _spec(),
        vision_scorer=lambda frames: (1.0, ["nice composition"]),
    )
    assert result.passed is True
    assert "vision: nice composition" in result.notes


def test_vision_failure_can_reject_render(qa_media: dict[str, Path]) -> None:
    result = evaluate(
        qa_media["testsrc"],
        _spec(),
        vision_scorer=lambda frames: {"score": 0.1, "passed": False, "notes": ["bad"]},
    )
    assert result.passed is False
    assert "vision: bad" in result.notes


def test_vision_scorer_unavailable_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    scorer = VisionLLMScorer(llm=LLMClient("none", None))
    assert scorer.available is False
    score, notes = scorer.score([])
    assert score == 0.0
    assert notes
