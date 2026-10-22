"""Perceptual-hash deduplication for scene segments."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HASH_SIDE = 32
HASH_BLOCK = 8
HASH_HEX_DIGITS = 16


@lru_cache(maxsize=4)
def _dct_matrix(size: int) -> Any:
    import numpy as np

    index = np.arange(size).reshape(-1, 1)
    position = np.arange(size).reshape(1, -1)
    matrix = np.cos(np.pi * (2 * position + 1) * index / (2 * size)) * np.sqrt(2.0 / size)
    matrix[0, :] *= np.sqrt(0.5)
    return matrix


def _as_gray_32(frame: Any) -> Any:
    import numpy as np
    from PIL import Image

    array = np.asarray(frame)
    if array.ndim == 3:
        if array.shape[-1] >= 3:
            array = (
                array[..., 0].astype(np.float64) * 0.299
                + array[..., 1].astype(np.float64) * 0.587
                + array[..., 2].astype(np.float64) * 0.114
            )
        else:
            array = array[..., 0]
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    image = image.resize((HASH_SIDE, HASH_SIDE), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float64)


def phash_from_frame(frame: Any) -> str:
    """Compute a 64-bit DCT perceptual hash as 16 hex characters."""
    import numpy as np

    gray = _as_gray_32(frame)
    matrix = _dct_matrix(HASH_SIDE)
    coefficients = matrix @ gray @ matrix.T
    block = coefficients[:HASH_BLOCK, :HASH_BLOCK].reshape(-1)
    median = float(np.median(block[1:])) if block.size > 1 else float(np.median(block))
    value = 0
    for coefficient in block:
        value = (value << 1) | int(coefficient > median)
    return f"{value:0{HASH_HEX_DIGITS}x}"


def phash_from_frames(frames: Sequence[Any], index: int | None = None) -> str | None:
    if not frames:
        return None
    position = len(frames) // 2 if index is None else max(0, min(index, len(frames) - 1))
    try:
        return phash_from_frame(frames[position])
    except Exception:
        logger.exception("could not compute perceptual hash")
        return None


def phash_keyframe(path: str | Path, timestamp: float) -> str | None:
    from .frames import FrameExtractError, grab_frame

    try:
        return phash_from_frame(grab_frame(path, timestamp))
    except FrameExtractError:
        logger.warning("could not extract keyframe from %s at %.3fs", path, timestamp)
        return None


def _hash_value(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    try:
        return int(text, 16)
    except ValueError:
        return None


def hamming(hash_a: Any, hash_b: Any) -> int:
    """Hamming distance between two hex hashes (``-1`` when invalid)."""
    left = _hash_value(hash_a)
    right = _hash_value(hash_b)
    if left is None or right is None:
        return -1
    return (left ^ right).bit_count()


def is_duplicate(hash_a: Any, hash_b: Any, threshold: int = 6) -> bool:
    distance = hamming(hash_a, hash_b)
    return distance >= 0 and distance <= threshold


def segment_phash(segment: Any) -> Any:
    if isinstance(segment, dict):
        return segment.get("phash")
    return getattr(segment, "phash", None)


def dedupe_segments(segments: Iterable[Any], threshold: int = 6) -> list[Any]:
    """Drop segments whose keyframe is within ``threshold`` of a kept segment."""
    kept: list[Any] = []
    kept_hashes: list[Any] = []
    for segment in segments:
        phash = segment_phash(segment)
        if phash and any(is_duplicate(phash, other, threshold) for other in kept_hashes):
            logger.debug("dropping duplicate segment phash=%s", phash)
            continue
        kept.append(segment)
        if phash:
            kept_hashes.append(phash)
    return kept
