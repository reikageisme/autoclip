"""Test-only stand-in for the lead-owned ``autoclip.contracts`` module.

The real contract is not part of the media-ingest ownership scope.  Tests inject
this stub only when the real module is absent, so the production modules always
import the shared contract once the lead lands it.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class DraftStatus(StrEnum):
    OK = "ok"
    ENCRYPTED = "encrypted"
    CORRUPT = "corrupt"
    MISSING = "missing"


@dataclass(slots=True)
class ClipInfo:
    path: str
    sha1: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


@dataclass(slots=True)
class SegmentInfo:
    start: float
    end: float
    transcript: str = ""
    tags: list[str] = field(default_factory=list)
    hook_score: float = 0.0
    phash: str | None = None
    thumb_path: str | None = None
    clip_id: int | None = None
    embedding: bytes | None = None


@dataclass(slots=True)
class IngestError:
    path: str
    message: str


@dataclass(slots=True)
class IngestReport:
    folder: str = ""
    clips_seen: int = 0
    clips_added: int = 0
    clips_skipped: int = 0
    segments_added: int = 0
    segments_deduped: int = 0
    errors: list[IngestError] = field(default_factory=list)
    elapsed_sec: float = 0.0


@dataclass(slots=True)
class RecipeSpec:
    name: str = "test"
    aspect: str = "9:16"
    duration_target: tuple[float, float] = (20.0, 35.0)
    music: dict[str, Any] = field(default_factory=dict)
    slots: list[dict[str, Any]] = field(default_factory=list)
    texts: list[dict[str, Any]] = field(default_factory=list)
    variation: dict[str, Any] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QAResult:
    score: float
    passed: bool
    notes: list[str] = field(default_factory=list)


class ClipRepository(Protocol):
    def get_clip_by_sha1(self, sha1: str) -> Any | None: ...

    def add_clip(self, clip: ClipInfo) -> ClipInfo: ...

    def add_segments(
        self, clip_id: int, segments: list[SegmentInfo]
    ) -> list[SegmentInfo]: ...


def install_if_missing() -> bool:
    """Install the stub in ``sys.modules`` when the real contract is missing."""
    try:
        import autoclip.contracts  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "autoclip.contracts":
            raise
    else:
        return False

    import autoclip

    module = types.ModuleType("autoclip.contracts")
    for name in (
        "ClipInfo",
        "ClipRepository",
        "DraftStatus",
        "IngestError",
        "IngestReport",
        "QAResult",
        "RecipeSpec",
        "SegmentInfo",
    ):
        setattr(module, name, globals()[name])
    sys.modules["autoclip.contracts"] = module
    autoclip.contracts = module
    return True
