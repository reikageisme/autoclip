"""Rendered-video QA: frame sampling, deterministic checks and scoring."""

from __future__ import annotations

from .checks import (
    CheckResult,
    check_aspect,
    check_audio,
    check_black_or_red_frames,
    check_duration,
)
from .frames import grab_frames, grab_frames_from_callable, probe_media
from .score import VisionLLMScorer, evaluate

__all__ = [
    "CheckResult",
    "VisionLLMScorer",
    "check_aspect",
    "check_audio",
    "check_black_or_red_frames",
    "check_duration",
    "evaluate",
    "grab_frames",
    "grab_frames_from_callable",
    "probe_media",
]
