"""Speech-to-text with faster-whisper, aligned to detected scenes."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TranscriptSegment:
    """A whisper segment expressed in seconds."""

    start: float
    end: float
    text: str


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def assign_transcript(
    scene_bounds: Sequence[tuple[float, float]],
    whisper_segments: Iterable[TranscriptSegment],
) -> list[str]:
    """Assign each whisper segment to the scene with the largest overlap.

    Every whisper segment is used at most once, so text does not leak across
    scene boundaries.  Segments without overlap go to the scene containing
    their midpoint.  Text per scene is joined in chronological order.
    """
    bounds = [(float(start), float(end)) for start, end in scene_bounds]
    pieces: list[list[tuple[float, str]]] = [[] for _ in bounds]
    for word in whisper_segments:
        best_index = -1
        best_overlap = 0.0
        for index, (start, end) in enumerate(bounds):
            overlap = min(end, word.end) - max(start, word.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index < 0:
            midpoint = (word.start + word.end) / 2
            for index, (start, end) in enumerate(bounds):
                if start <= midpoint <= end:
                    best_index = index
                    break
        if best_index >= 0:
            pieces[best_index].append((word.start, word.text))

    result: list[str] = []
    for scene_pieces in pieces:
        scene_pieces.sort(key=lambda item: item[0])
        result.append(normalize_text(" ".join(text for _, text in scene_pieces)))
    return result


class Transcriber:
    """Lazy faster-whisper wrapper.

    The model is only loaded on the first transcription so that
    ``whisper_enabled=False`` keeps ingest fully offline and fast.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        enabled: bool = True,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        strict: bool = False,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.enabled = enabled
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.strict = strict
        self._model: Any | None = None

    @classmethod
    def from_options(cls, options: Any) -> Transcriber:
        return cls(
            model_size=getattr(options, "whisper_model", "small"),
            device=getattr(options, "whisper_device", "cpu"),
            compute_type=getattr(options, "whisper_compute_type", "int8"),
            enabled=getattr(options, "whisper_enabled", True),
            language=getattr(options, "whisper_language", None),
            beam_size=getattr(options, "whisper_beam_size", 5),
            vad_filter=getattr(options, "whisper_vad_filter", True),
            strict=getattr(options, "whisper_strict", False),
        )

    def _load(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - optional heavy dependency
                raise RuntimeError(
                    "faster-whisper is not installed; disable whisper or install it"
                ) from exc
            logger.info(
                "loading faster-whisper model=%s device=%s compute_type=%s",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, path: str | Path) -> list[TranscriptSegment]:
        if not self.enabled:
            return []
        model = self._load()
        segments, _info = model.transcribe(
            str(path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        return [
            TranscriptSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=normalize_text(segment.text),
            )
            for segment in segments
        ]

    def transcripts_for(
        self,
        path: str | Path,
        scene_bounds: Sequence[tuple[float, float]],
    ) -> list[str]:
        if not self.enabled:
            return ["" for _ in scene_bounds]
        try:
            segments = self.transcribe(path)
        except Exception:
            if self.strict:
                raise
            logger.exception("transcription failed for %s; continuing without text", path)
            return ["" for _ in scene_bounds]
        return assign_transcript(scene_bounds, segments)
