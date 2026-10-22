from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from autoclip.ingest.dedupe import (
    dedupe_segments,
    hamming,
    is_duplicate,
    phash_from_frame,
    phash_from_frames,
)


def _noise(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (32, 32, 3), dtype=np.uint8)


def test_phash_is_deterministic() -> None:
    frame = _noise(1)
    first = phash_from_frame(frame)
    second = phash_from_frame(frame.copy())
    assert first == second
    assert len(first) == 16
    int(first, 16)


def test_phash_differs_for_different_images() -> None:
    assert phash_from_frame(_noise(1)) != phash_from_frame(_noise(2))


def test_hamming_distance() -> None:
    assert hamming("0" * 16, "0" * 16) == 0
    assert hamming("0" * 16, "f" * 16) == 64
    assert hamming("abc", "abd") == 1
    assert hamming(None, "abc") == -1


def test_is_duplicate_threshold() -> None:
    assert is_duplicate("0000000000000000", "0000000000000001", threshold=1)
    assert not is_duplicate("0000000000000000", "000000000000000f", threshold=1)


def test_phash_from_frames_empty() -> None:
    assert phash_from_frames([]) is None


def test_dedupe_segments_keeps_first_and_unique() -> None:
    duplicate = phash_from_frame(_noise(3))
    other = phash_from_frame(_noise(4))
    segments = [
        SimpleNamespace(phash=duplicate),
        SimpleNamespace(phash=duplicate),
        SimpleNamespace(phash=other),
        SimpleNamespace(phash=None),
    ]
    kept = dedupe_segments(segments, threshold=0)
    assert kept == [segments[0], segments[2], segments[3]]
