"""Frame sampling helpers shared by ingest and tagging."""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FrameExtractError(RuntimeError):
    """Raised when a frame cannot be decoded from a video."""


def _binary(env_name: str, default: str) -> str:
    return os.environ.get(env_name) or shutil.which(default) or default


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a project dependency
        raise FrameExtractError("numpy is required for frame analysis") from exc
    return np


def sample_times(start: float, end: float, count: int) -> list[float]:
    """Return ``count`` evenly spaced timestamps inside ``[start, end]``."""
    if count <= 0:
        return []
    if end <= start:
        return [max(0.0, float(start))] * count
    step = (float(end) - float(start)) / (count + 1)
    return [float(start) + step * (index + 1) for index in range(count)]


def grab_frame(path: str | Path, timestamp: float, timeout: float = 30.0) -> Any:
    """Decode a single RGB frame at ``timestamp`` seconds as a numpy array."""
    command = [
        _binary("FFMPEG_BIN", "ffmpeg"),
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, float(timestamp)):.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FrameExtractError(f"ffmpeg could not grab frame from {path}: {exc}") from exc
    if completed.returncode != 0 or not completed.stdout:
        detail = (completed.stderr or b"").decode("utf-8", "replace").strip()
        raise FrameExtractError(f"ffmpeg could not grab frame from {path}: {detail}")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - pillow is a project dependency
        raise FrameExtractError("pillow is required for frame analysis") from exc

    with Image.open(io.BytesIO(completed.stdout)) as image:
        rgb = image.convert("RGB")
        return _numpy().asarray(rgb)


def sample_frames(
    path: str | Path,
    start: float,
    end: float,
    count: int = 3,
) -> list[Any]:
    """Grab ``count`` evenly spaced frames between two scene boundaries."""
    frames: list[Any] = []
    for timestamp in sample_times(start, end, count):
        try:
            frames.append(grab_frame(path, timestamp))
        except FrameExtractError:
            logger.warning("skipping frame at %.3fs for %s", timestamp, path)
    return frames


def to_gray(frame: Any) -> Any:
    """Convert an RGB/RGBA frame to a float32 grayscale array (0-255)."""
    np = _numpy()
    array = np.asarray(frame)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.shape[-1] >= 3:
        channels = array[..., :3].astype(np.float32)
        return channels[..., 0] * 0.299 + channels[..., 1] * 0.587 + channels[..., 2] * 0.114
    return array[..., 0].astype(np.float32)
