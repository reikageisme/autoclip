"""Scene detection built on PySceneDetect's ContentDetector."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


class SceneDetectUnavailable(RuntimeError):
    """Raised when PySceneDetect is not installed."""


def merge_short_scenes(
    scenes: Iterable[tuple[float, float]],
    min_scene_sec: float = 0.8,
    duration: float | None = None,
) -> list[tuple[float, float]]:
    """Merge scenes shorter than ``min_scene_sec`` into an adjacent scene.

    The first scene absorbs the next one when it is too short; every later
    short scene is appended to the previous scene.  Adjacent/overlapping
    boundaries are coalesced at the end so the result is always contiguous.
    """
    cleaned: list[list[float]] = []
    for start, end in scenes:
        start = max(0.0, float(start))
        end = float(end)
        if duration is not None:
            end = min(end, float(duration))
        if end > start:
            cleaned.append([start, end])

    cleaned.sort(key=lambda item: (item[0], item[1]))

    if not cleaned:
        if duration and duration > 0:
            return [(0.0, float(duration))]
        return []

    min_length = max(0.0, float(min_scene_sec))
    changed = True
    while changed:
        changed = False
        for index, (start, end) in enumerate(cleaned):
            if end - start >= min_length:
                continue
            if index > 0:
                cleaned[index - 1][1] = max(cleaned[index - 1][1], end)
                cleaned.pop(index)
                changed = True
                break
            if len(cleaned) > 1:
                cleaned[index + 1][0] = min(cleaned[index + 1][0], start)
                cleaned.pop(index)
                changed = True
                break

    merged: list[tuple[float, float]] = []
    for start, end in cleaned:
        if merged and start < merged[-1][1] - 1e-6:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _seconds(timecode: object) -> float:
    value = getattr(timecode, "seconds", None)
    if value is None:
        value = timecode.get_seconds()  # type: ignore[attr-defined]
    return float(value)


def _scene_seconds(scenes: Sequence[object]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for pair in scenes:
        start, end = pair  # type: ignore[misc]
        result.append((_seconds(start), _seconds(end)))
    return result


def detect_scenes(
    path: str | Path,
    min_scene_sec: float = 0.8,
    threshold: float = 27.0,
    duration: float | None = None,
    show_progress: bool = False,
) -> list[tuple[float, float]]:
    """Detect scenes and merge every scene shorter than ``min_scene_sec``."""
    try:
        from scenedetect import ContentDetector, detect
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SceneDetectUnavailable(
            "PySceneDetect is required for scene detection; install scenedetect[opencv]"
        ) from exc

    detected = detect(
        str(path),
        ContentDetector(threshold=threshold),
        show_progress=show_progress,
        start_in_scene=True,
    )
    raw = _scene_seconds(detected)
    merged = merge_short_scenes(raw, min_scene_sec=min_scene_sec, duration=duration)
    logger.debug("detected %d scenes in %s -> %d after merge", len(raw), path, len(merged))
    return merged
