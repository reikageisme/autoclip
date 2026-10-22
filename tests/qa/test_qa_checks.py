from __future__ import annotations

from pathlib import Path

import pytest

from autoclip.qa.checks import (
    check_aspect,
    check_audio,
    check_black_or_red_frames,
    check_duration,
    check_resolution,
    parse_aspect,
)
from autoclip.qa.frames import grab_frames


def test_parse_aspect() -> None:
    assert parse_aspect("9:16") == pytest.approx(9 / 16)
    assert parse_aspect("16/9") == pytest.approx(16 / 9)
    assert parse_aspect("1.5") == pytest.approx(1.5)
    assert parse_aspect((1080, 1920)) == pytest.approx(0.5625)
    assert parse_aspect(None) is None
    assert parse_aspect("nonsense") is None


def test_black_frames_fail(qa_media: dict[str, Path]) -> None:
    result = check_black_or_red_frames(grab_frames(qa_media["black"], 3))
    assert result.passed is False
    assert result.score == 0.0


def test_red_frames_fail(qa_media: dict[str, Path]) -> None:
    result = check_black_or_red_frames(grab_frames(qa_media["red"], 2))
    assert result.passed is False


def test_colorful_frames_pass(qa_media: dict[str, Path]) -> None:
    result = check_black_or_red_frames(grab_frames(qa_media["testsrc"], 3))
    assert result.passed is True
    assert result.score > 0.8


def test_check_frames_without_samples() -> None:
    result = check_black_or_red_frames([])
    assert result.passed is False
    assert result.score == 0.0


def test_check_duration() -> None:
    assert check_duration(10.0, (8.0, 12.0)).passed is True
    assert check_duration(10.0, [8.0, 12.0]).passed is True
    assert check_duration(10.0, (20.0, 30.0)).passed is False
    assert check_duration(10.0, None).passed is True
    assert check_duration(11.0, (8.0, 10.0), tolerance=1.0).passed is True


def test_check_audio(qa_media: dict[str, Path]) -> None:
    assert check_audio(qa_media["testsrc"]).passed is True
    assert check_audio(qa_media["silent"]).passed is False
    assert check_audio(qa_media["black"]).passed is False


def test_check_aspect() -> None:
    assert check_aspect(360, 640, "9:16").passed is True
    assert check_aspect(640, 360, "9:16").passed is False
    assert check_aspect(360, 640, None).passed is True
    assert check_aspect(0, 0, "9:16").passed is False


def test_check_resolution() -> None:
    assert check_resolution(360, 640, expected=(360, 640)).passed is True
    assert check_resolution(720, 1280, expected=(360, 640)).passed is False
    assert check_resolution(360, 640, aspect="9:16").passed is True
