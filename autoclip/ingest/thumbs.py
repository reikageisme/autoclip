"""Thumbnail generation for scene segments."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ThumbnailError(RuntimeError):
    """Raised when a thumbnail cannot be produced."""


def _binary(env_name: str, default: str) -> str:
    return os.environ.get(env_name) or shutil.which(default) or default


def thumbnail_filename(sha1: str, index: int) -> str:
    return f"{sha1}_{index}.jpg"


def thumbnail_path(thumbs_dir: str | Path, sha1: str, index: int) -> Path:
    return Path(thumbs_dir) / thumbnail_filename(sha1, index)


def save_frame_jpg(frame: Any, destination: str | Path, quality: int = 85) -> Path:
    """Write an already-decoded RGB frame as JPEG."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - pillow is a project dependency
        raise ThumbnailError("pillow is required to write thumbnails") from exc
    import numpy as np

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(frame).astype("uint8")).convert("RGB")
    image.save(path, format="JPEG", quality=quality)
    return path


def extract_thumbnail(
    video_path: str | Path,
    timestamp: float,
    destination: str | Path,
    width: int = 360,
) -> Path:
    """Extract one frame with ffmpeg and save it as a JPEG thumbnail."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _binary("FFMPEG_BIN", "ffmpeg"),
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, float(timestamp)):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={int(width)}:-2",
        "-q:v",
        "4",
        "-y",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ThumbnailError(f"ffmpeg could not create thumbnail: {exc}") from exc
    if completed.returncode != 0 or not path.is_file():
        detail = (completed.stderr or b"").decode("utf-8", "replace").strip()
        raise ThumbnailError(f"ffmpeg could not create thumbnail: {detail}")
    return path


def make_thumbnail(
    video_path: str | Path,
    timestamp: float,
    thumbs_dir: str | Path,
    sha1: str,
    index: int,
    width: int = 360,
    frame: Any | None = None,
) -> Path:
    """Create ``data/thumbs/<sha1>_<idx>.jpg``.

    When a decoded frame is supplied no extra ffmpeg pass is needed.
    """
    destination = thumbnail_path(thumbs_dir, sha1, index)
    if frame is not None:
        return save_frame_jpg(frame, destination)
    return extract_thumbnail(video_path, timestamp, destination, width=width)
