"""Frame sampling for rendered-video QA."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FrameExtractError(RuntimeError):
    """Raised when ffmpeg/ffprobe cannot read a render."""


@dataclass(slots=True)
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def _binary(env_name: str, default: str) -> str:
    return os.environ.get(env_name) or shutil.which(default) or default


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is a project dependency
        raise FrameExtractError("numpy is required for QA") from exc
    return np


def _ratio(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        top, _, bottom = value.partition("/")
        try:
            return float(top) / float(bottom) if float(bottom) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe_media(path: str | Path) -> MediaInfo:
    """Lightweight ffprobe read (no hashing) used by QA checks."""
    command = [
        _binary("FFPROBE_BIN", "ffprobe"),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FrameExtractError(f"ffprobe failed for {path}: {exc}") from exc
    if completed.returncode != 0:
        raise FrameExtractError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FrameExtractError(f"ffprobe returned invalid JSON for {path}") from exc

    streams = payload.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video is None:
        raise FrameExtractError(f"no video stream in {path}")
    try:
        duration = float((payload.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    for side_data in video.get("side_data_list") or []:
        rotation = side_data.get("rotation")
        if rotation is not None:
            try:
                if abs(int(float(rotation))) % 180 == 90:
                    width, height = height, width
            except (TypeError, ValueError):
                pass
    fps = _ratio(video.get("avg_frame_rate")) or _ratio(video.get("r_frame_rate"))
    has_audio = any(item.get("codec_type") == "audio" for item in streams)
    return MediaInfo(duration=duration, width=width, height=height, fps=fps, has_audio=has_audio)


def grab_frame(path: str | Path, timestamp: float, timeout: float = 30.0) -> Any:
    """Decode one RGB frame at ``timestamp`` seconds as a numpy array."""
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
        completed = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise FrameExtractError(f"ffmpeg could not grab frame from {path}: {exc}") from exc
    if completed.returncode != 0 or not completed.stdout:
        detail = (completed.stderr or b"").decode("utf-8", "replace").strip()
        raise FrameExtractError(f"ffmpeg could not grab frame from {path}: {detail}")
    return _decode_image(completed.stdout)


def _decode_image(payload: Any) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        return _numpy().asarray(image.convert("RGB"))


def _normalize_frame(value: Any) -> Any:
    """Accept a numpy array, PIL image, bytes or a path from a grab callable."""
    if isinstance(value, (bytes, bytearray)):
        return _decode_image(bytes(value))
    if isinstance(value, (str, Path)):
        with open(value, "rb") as handle:
            return _decode_image(handle.read())
    from PIL import Image

    if isinstance(value, Image.Image):
        return _numpy().asarray(value.convert("RGB"))
    return _numpy().asarray(value)


def just_times(duration: float, count: int) -> list[float]:
    if count <= 0:
        return []
    duration = max(0.0, float(duration))
    if duration == 0:
        return [0.0] * count
    step = duration / (count + 1)
    return [step * (index + 1) for index in range(count)]


def grab_frames(
    path: str | Path,
    count: int = 5,
    duration: float | None = None,
    info: MediaInfo | None = None,
) -> list[Any]:
    """Grab ``count`` evenly spaced frames from a video."""
    media = info or probe_media(path)
    total = media.duration if duration is None else float(duration)
    frames: list[Any] = []
    for timestamp in just_times(total, count):
        try:
            frames.append(grab_frame(path, timestamp))
        except FrameExtractError:
            logger.warning("could not grab QA frame at %.3fs from %s", timestamp, path)
    return frames


def grab_frames_from_callable(
    grab_frame_fn: Callable[[float], Any],
    count: int = 5,
    duration: float = 0.0,
) -> list[Any]:
    """Grab frames through an injected callable (e.g. Concat preview_frame)."""
    frames: list[Any] = []
    for timestamp in just_times(duration, count):
        try:
            frames.append(_normalize_frame(grab_frame_fn(timestamp)))
        except Exception:
            logger.warning("injected grab_frame failed at %.3fs", timestamp)
    return frames


def to_gray(frame: Any) -> Any:
    np = _numpy()
    array = np.asarray(frame)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.shape[-1] >= 3:
        channels = array[..., :3].astype(np.float32)
        return channels[..., 0] * 0.299 + channels[..., 1] * 0.587 + channels[..., 2] * 0.114
    return array[..., 0].astype(np.float32)


def frame_luma(frame: Any) -> float:
    gray = to_gray(frame)
    if gray.size == 0:
        return 0.0
    return float(gray.mean())


def frame_means(frames: Sequence[Any]) -> list[tuple[float, float, float]]:
    """Return mean ``(R, G, B)`` per frame."""
    result: list[tuple[float, float, float]] = []
    for frame in frames:
        np = _numpy()
        array = np.asarray(frame)
        if array.ndim == 2:
            mean = float(array.mean())
            result.append((mean, mean, mean))
            continue
        channels = array[..., :3].astype(np.float32)
        means = channels.reshape(-1, 3).mean(axis=0)
        result.append((float(means[0]), float(means[1]), float(means[2])))
    return result
