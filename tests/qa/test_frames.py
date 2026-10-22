from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autoclip.qa.frames import (
    frame_luma,
    frame_means,
    grab_frames,
    grab_frames_from_callable,
    just_times,
    probe_media,
    to_gray,
)


def test_probe_media(qa_media: dict[str, Path]) -> None:
    info = probe_media(qa_media["testsrc"])
    assert info.width == 360
    assert info.height == 640
    assert info.duration == pytest.approx(2.0, abs=0.25)
    assert info.fps == pytest.approx(25.0, abs=0.5)
    assert info.has_audio is True
    assert probe_media(qa_media["silent"]).has_audio is False


def test_just_times() -> None:
    assert just_times(0.0, 3) == [0.0, 0.0, 0.0]
    assert just_times(4.0, 3) == pytest.approx([1.0, 2.0, 3.0])
    assert just_times(4.0, 0) == []


def test_grab_frames_from_video(qa_media: dict[str, Path]) -> None:
    frames = grab_frames(qa_media["testsrc"], 3)
    assert len(frames) == 3
    assert frames[0].shape == (640, 360, 3)
    assert frames[0].dtype == np.uint8


def test_grab_frames_from_callable() -> None:
    frame = np.full((64, 36, 3), 100, dtype=np.uint8)
    seen: list[float] = []

    def grab(timestamp: float):
        seen.append(timestamp)
        return frame

    frames = grab_frames_from_callable(grab, 4, duration=8.0)
    assert len(frames) == 4
    assert seen == pytest.approx([1.6, 3.2, 4.8, 6.4])
    assert frames[0].shape == (64, 36, 3)


def test_gray_helpers() -> None:
    red = np.zeros((8, 8, 3), dtype=np.uint8)
    red[..., 0] = 255
    gray = to_gray(red)
    assert gray.shape == (8, 8)
    assert frame_luma(red) == pytest.approx(0.299 * 255, abs=0.5)
    assert frame_means([red])[0][0] == pytest.approx(255.0)
    assert frame_means([np.full((4, 4), 50, dtype=np.uint8)])[0] == (50.0, 50.0, 50.0)
