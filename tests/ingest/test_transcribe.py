from __future__ import annotations

from types import SimpleNamespace

import pytest

from autoclip.ingest.transcribe import (
    Transcriber,
    TranscriptSegment,
    assign_transcript,
    normalize_text,
)


class FakeModel:
    def transcribe(self, path: str, **kwargs: object):
        assert path
        assert kwargs
        return [
            SimpleNamespace(start=0.2, end=0.9, text="  hello   world "),
            SimpleNamespace(start=1.1, end=1.8, text="again"),
        ], None


class ExplodingModel:
    def transcribe(self, path: str, **kwargs: object):
        raise RuntimeError("whisper exploded")


def test_normalize_text() -> None:
    assert normalize_text("  a \n b\tc ") == "a b c"


def test_assign_transcript_by_overlap() -> None:
    words = [
        TranscriptSegment(0.2, 0.9, "hello"),
        TranscriptSegment(1.1, 1.8, "world"),
        TranscriptSegment(2.2, 2.9, "next"),
    ]
    assert assign_transcript([(0.0, 2.0), (2.0, 3.0)], words) == ["hello world", "next"]


def test_assign_transcript_empty_words() -> None:
    assert assign_transcript([(0.0, 1.0), (1.0, 2.0)], []) == ["", ""]


def test_disabled_transcriber_does_not_load_model() -> None:
    transcriber = Transcriber(enabled=False)
    assert transcriber.transcribe("anything.mp4") == []
    assert transcriber.transcripts_for("anything.mp4", [(0.0, 1.0)]) == [""]
    assert transcriber._model is None


def test_transcriber_with_injected_model(tmp_path) -> None:
    transcriber = Transcriber(enabled=True)
    transcriber._model = FakeModel()
    assert transcriber.transcripts_for("x.mp4", [(0.0, 1.0), (1.0, 2.0)]) == [
        "hello world",
        "again",
    ]


def test_failed_transcription_is_non_strict_by_default() -> None:
    transcriber = Transcriber(enabled=True)
    transcriber._model = ExplodingModel()
    assert transcriber.transcripts_for("x.mp4", [(0.0, 1.0)]) == [""]


def test_strict_transcription_raises() -> None:
    transcriber = Transcriber(enabled=True, strict=True)
    transcriber._model = ExplodingModel()
    with pytest.raises(RuntimeError):
        transcriber.transcripts_for("x.mp4", [(0.0, 1.0)])


class WordModel:
    """Một câu dài 0.5–3.5s vắt qua 2 cảnh, có word timestamps như faster-whisper."""

    def transcribe(self, path: str, **kwargs: object):
        assert kwargs.get("word_timestamps") is True
        words = [
            SimpleNamespace(start=0.5, end=1.0, word=" hàng"),
            SimpleNamespace(start=1.0, end=1.6, word=" về"),
            SimpleNamespace(start=2.2, end=2.8, word=" đúng"),
            SimpleNamespace(start=2.8, end=3.5, word=" hẹn"),
        ]
        return [SimpleNamespace(start=0.5, end=3.5, text="hàng về đúng hẹn", words=words)], None


def test_long_sentence_is_split_across_scenes_by_words() -> None:
    transcriber = Transcriber(enabled=True)
    transcriber._model = WordModel()
    assert transcriber.transcripts_for("x.mp4", [(0.0, 2.0), (2.0, 4.0)]) == [
        "hàng về",
        "đúng hẹn",
    ]


def test_failure_is_recorded_not_silent() -> None:
    transcriber = Transcriber(enabled=True)
    transcriber._model = ExplodingModel()
    assert transcriber.transcripts_for("x.mp4", [(0.0, 1.0)]) == [""]
    assert transcriber.last_error and "whisper exploded" in transcriber.last_error
