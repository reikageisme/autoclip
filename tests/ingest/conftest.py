"""Fixtures for media-ingest tests (tiny ffmpeg-generated videos, < 2 MB)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def _ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        pytest.skip("ffmpeg is required for media fixtures")
    return binary


def _run(args: list[str]) -> None:
    completed = subprocess.run(args, capture_output=True, text=True, timeout=180)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg fixture failed: {completed.stderr[-2000:]}")


def _encode(output: Path, *inputs: str, audio: bool = False) -> None:
    command = [_ffmpeg(), "-y", "-loglevel", "error"]
    for item in inputs:
        command.extend(["-f", "lavfi", "-i", item])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if audio:
        command.extend(["-c:a", "aac", "-b:a", "64k", "-shortest"])
    command.append(str(output))
    _run(command)


@pytest.fixture(scope="session")
def media_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("media")
    files = {
        "testsrc": root / "testsrc.mp4",
        "silent": root / "silent.mp4",
        "black": root / "black.mp4",
        "red": root / "red.mp4",
        "scenes": root / "scenes.mp4",
    }
    _encode(
        files["testsrc"],
        "testsrc2=size=360x640:rate=25:duration=2",
        "sine=frequency=440:duration=2",
        audio=True,
    )
    _encode(files["silent"], "testsrc2=size=360x640:rate=25:duration=2")
    _encode(files["black"], "color=c=black:s=360x640:r=25:d=2")
    _encode(files["red"], "color=c=red:s=360x640:r=25:d=1")
    _run(
        [
            _ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=360x640:r=25:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=360x640:r=25:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=360x640:r=25:d=1",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            str(files["scenes"]),
        ]
    )
    return files
