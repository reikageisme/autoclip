"""Speech-to-text with faster-whisper, aligned to detected scenes."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
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
    """Assign each piece (ideally a single word) to the scene with the largest overlap.

    Feed word-level pieces: a whole whisper sentence often spans several scenes
    and would otherwise land in only one of them.  Every piece is used at most
    once; pieces without overlap go to the scene containing their midpoint.
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
        beam_size: int = 1,
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
        self.last_error: str | None = None
        """Lỗi của lần transcripts_for gần nhất (khi strict=False), để pipeline ghi báo cáo."""
        self._model: Any | None = None
        self._pending: tuple[str, Future[list[TranscriptSegment]]] | None = None
        self._pool: ThreadPoolExecutor | None = None

    def prefetch(self, path: str | Path) -> None:
        """Bắt đầu chép lời `path` ở luồng nền (chạy song song với quét hình).

        `transcripts_for` cùng file sau đó sẽ dùng kết quả này thay vì chạy lại.
        """
        if not self.enabled:
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="whisper")
        self._pending = (str(path), self._pool.submit(self.transcribe, path))

    @classmethod
    def from_options(cls, options: Any) -> Transcriber:
        return cls(
            model_size=getattr(options, "whisper_model", "small"),
            device=getattr(options, "whisper_device", "cpu"),
            compute_type=getattr(options, "whisper_compute_type", "int8"),
            enabled=getattr(options, "whisper_enabled", True),
            language=getattr(options, "whisper_language", None),
            beam_size=getattr(options, "whisper_beam_size", 1),
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
                # faster-whisper mặc định chỉ 4 luồng; ffmpeg quét hình chạy song song
                # nên chừa lại một nửa số nhân cho nó
                cpu_threads=max(4, (os.cpu_count() or 8) // 2),
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
            word_timestamps=True,
        )
        pieces: list[TranscriptSegment] = []
        for segment in segments:
            words = getattr(segment, "words", None) or []
            if words:
                pieces.extend(
                    TranscriptSegment(float(w.start), float(w.end), normalize_text(w.word))
                    for w in words
                    # từ Whisper tự thấy không chắc (< 30%) thường là đoán bừa trên tiếng ồn
                    if normalize_text(w.word) and getattr(w, "probability", 1.0) >= 0.3
                )
            else:
                pieces.append(
                    TranscriptSegment(
                        float(segment.start), float(segment.end), normalize_text(segment.text)
                    )
                )
        return pieces

    def transcripts_for(
        self,
        path: str | Path,
        scene_bounds: Sequence[tuple[float, float]],
    ) -> list[str]:
        self.last_error = None
        if not self.enabled:
            return ["" for _ in scene_bounds]
        pending, self._pending = self._pending, None
        try:
            if pending is not None and pending[0] == str(path):
                segments = pending[1].result()
            else:
                segments = self.transcribe(path)
        except Exception as exc:
            if self.strict:
                raise
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("transcription failed for %s; continuing without text: %s", path, exc)
            return ["" for _ in scene_bounds]
        return assign_transcript(scene_bounds, segments)
