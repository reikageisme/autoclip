"""Media probing with ffprobe plus content hashing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from autoclip.contracts import ClipInfo

from ._compat import construct

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1024 * 1024
PROBE_TIMEOUT_SEC = 60


class ProbeError(RuntimeError):
    """Raised when ffprobe cannot read a media file."""


def _binary(env_name: str, default: str) -> str:
    return os.environ.get(env_name) or shutil.which(default) or default


def sha1_file(path: str | Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Compute SHA-1 by streaming the file in 1 MiB chunks (memory friendly)."""
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_frame_rate(value: str | None) -> float:
    """Parse ffprobe rational frame rates such as ``30000/1001``."""
    if not value:
        return 0.0
    if "/" in value:
        numerator, _, denominator = value.partition("/")
        top = _as_float(numerator)
        bottom = _as_float(denominator)
        if top is None or not bottom:
            return 0.0
        return top / bottom
    return _as_float(value) or 0.0


def _rotation(video: dict[str, Any]) -> int:
    for side_data in video.get("side_data_list", []) or []:
        if "rotation" in side_data:
            return int(_as_float(side_data["rotation"]) or 0)
    tags = video.get("tags") or {}
    return int(_as_float(tags.get("rotate")) or 0)


def probe(path: str | Path) -> ClipInfo:
    """Probe a clip and return the shared :class:`ClipInfo` contract object."""
    clip_path = Path(path)
    if not clip_path.is_file():
        raise ProbeError(f"file not found: {clip_path}")

    command = [
        _binary("FFPROBE_BIN", "ffprobe"),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(clip_path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PROBE_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProbeError(f"ffprobe failed to run for {clip_path}: {exc}") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        raise ProbeError(f"ffprobe failed for {clip_path}: {detail or 'unknown error'}")

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe returned invalid JSON for {clip_path}") from exc

    streams = payload.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video is None:
        raise ProbeError(f"no video stream in {clip_path}")

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if abs(_rotation(video)) % 180 == 90:
        width, height = height, width

    duration = _as_float((payload.get("format") or {}).get("duration"))
    if duration is None:
        duration = _as_float(video.get("duration")) or 0.0

    fps = parse_frame_rate(video.get("avg_frame_rate")) or parse_frame_rate(
        video.get("r_frame_rate")
    )
    has_audio = any(item.get("codec_type") == "audio" for item in streams)
    sha1 = sha1_file(clip_path)

    logger.debug(
        "probed %s: %sx%s %.3fs %.3ffps audio=%s sha1=%s",
        clip_path,
        width,
        height,
        duration,
        fps,
        has_audio,
        sha1,
    )
    return construct(
        ClipInfo,
        path=str(clip_path.resolve()),
        sha1=sha1,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio=has_audio,
    )
