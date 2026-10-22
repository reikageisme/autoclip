"""Deterministic QA checks for rendered videos."""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frames import frame_luma, frame_means

logger = logging.getLogger(__name__)

VOLUME_RE = re.compile(r"((?:mean|max)_volume):\s*(-?(?:\d+(?:\.\d+)?|inf))\s*dB")


@dataclass(slots=True)
class CheckResult:
    name: str
    passed: bool
    score: float
    detail: str = ""


def parse_aspect(value: Any) -> float | None:
    """Parse ``9:16`` / ``16/9`` / ``1.777`` into a width/height ratio."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        width, height = float(value[0]), float(value[1])
        return width / height if height else None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    text = str(value).strip().lower().replace("x", ":").replace("/", ":")
    if ":" in text:
        top, _, bottom = text.partition(":")
        try:
            height = float(bottom)
            return float(top) / height if height else None
        except ValueError:
            return None
    try:
        ratio = float(text)
    except ValueError:
        return None
    return ratio if ratio > 0 else None


def _is_black(luma: float, luma_threshold: float) -> bool:
    return luma <= luma_threshold


def _is_red(red: float, green: float, blue: float) -> bool:
    return red >= 70.0 and red >= 1.8 * max(green, 1.0) and red >= 1.8 * max(blue, 1.0)


def check_black_or_red_frames(
    frames: Sequence[Any],
    max_bad_ratio: float = 0.15,
    luma_threshold: float = 12.0,
) -> CheckResult:
    """Fail when more than ``max_bad_ratio`` of frames are near-black or solid red."""
    if not frames:
        return CheckResult("frames", False, 0.0, "no frames sampled")
    means = frame_means(frames)
    lumas = [frame_luma(frame) for frame in frames]
    bad = sum(
        1
        for (red, green, blue), luma in zip(means, lumas, strict=True)
        if _is_black(luma, luma_threshold) or _is_red(red, green, blue)
    )
    ratio = bad / len(frames)
    passed = ratio <= max_bad_ratio
    detail = f"{bad}/{len(frames)} black/red frames (ratio={ratio:.2f})"
    return CheckResult("frames", passed, max(0.0, 1.0 - ratio), detail)


def _target_range(target: Any) -> tuple[float, float] | None:
    if target is None:
        return None
    if isinstance(target, dict):
        low = target.get("min", target.get("duration_min"))
        high = target.get("max", target.get("duration_max"))
        if low is None or high is None:
            return None
        return float(low), float(high)
    if isinstance(target, (tuple, list)) and len(target) == 2:
        return float(target[0]), float(target[1])
    if isinstance(target, (int, float)):
        return float(target), float(target)
    return None


def duration_target(spec: Any) -> tuple[float, float] | None:
    if spec is None:
        return None
    if isinstance(spec, dict):
        return _target_range(spec.get("duration_target"))
    return _target_range(getattr(spec, "duration_target", None))


def check_duration(
    duration: float,
    target: Any,
    tolerance: float = 0.0,
) -> CheckResult:
    """Check render duration is inside the recipe target range."""
    bounds = _target_range(target)
    if bounds is None:
        return CheckResult("duration", True, 1.0, "no duration target")
    low, high = bounds
    low -= max(0.0, tolerance)
    high += max(0.0, tolerance)
    passed = low <= duration <= high
    span = max(high - low, 1.0)
    if passed:
        score = 1.0
    else:
        distance = low - duration if duration < low else duration - high
        score = max(0.0, 1.0 - distance / span)
    detail = f"duration={duration:.2f}s target=[{low:.2f}, {high:.2f}]"
    return CheckResult("duration", passed, score, detail)


def check_audio(path: str | Path, silence_db: float = -50.0) -> CheckResult:
    """Fail when the render has no audio stream or is silent end-to-end."""
    command = [
        os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-i",
        str(path),
        "-vn",
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult("audio", False, 0.0, f"ffmpeg audio check failed: {exc}")

    output = f"{completed.stdout}\n{completed.stderr}"
    values = {name: float(value) for name, value in VOLUME_RE.findall(output)}
    if "max_volume" not in values:
        detail = "no audio stream detected" if "audio" in output.lower() else "no volume data"
        return CheckResult("audio", False, 0.0, detail)
    maximum = values["max_volume"]
    passed = maximum > silence_db
    detail = f"max_volume={maximum:.1f}dB mean={values.get('mean_volume', 0.0):.1f}dB"
    return CheckResult("audio", passed, 1.0 if passed else 0.0, detail)


def check_aspect(
    width: int,
    height: int,
    aspect: Any,
    tolerance: float = 0.02,
) -> CheckResult:
    """Check the render width/height ratio against the recipe aspect."""
    expected = parse_aspect(aspect)
    if expected is None:
        return CheckResult("aspect", True, 1.0, "no aspect target")
    if width <= 0 or height <= 0:
        return CheckResult("aspect", False, 0.0, f"invalid size {width}x{height}")
    actual = width / height
    relative = abs(actual - expected) / expected
    passed = relative <= max(0.0, tolerance)
    score = max(0.0, 1.0 - relative)
    if passed:
        score = max(score, 0.7)
    detail = f"{width}x{height} ratio={actual:.4f} expected={expected:.4f}"
    return CheckResult("aspect", passed, score, detail)


def check_resolution(
    width: int,
    height: int,
    expected: tuple[int, int] | None = None,
    aspect: Any = None,
    tolerance: float = 0.02,
) -> CheckResult:
    """Check exact output size when known, otherwise fall back to aspect only."""
    if expected is not None:
        expected_width, expected_height = expected
        width_ok = math.isclose(width, expected_width, rel_tol=tolerance)
        height_ok = math.isclose(height, expected_height, rel_tol=tolerance)
        passed = width_ok and height_ok
        detail = f"{width}x{height} expected={expected_width}x{expected_height}"
        return CheckResult("resolution", passed, 1.0 if passed else 0.0, detail)
    result = check_aspect(width, height, aspect, tolerance=tolerance)
    return CheckResult("resolution", result.passed, result.score, result.detail)
