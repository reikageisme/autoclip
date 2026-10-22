"""Ingest pipeline: probe, scene split, transcript, tags, phash and thumbnails."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoclip.contracts import ClipInfo, ClipRepository, IngestReport, SegmentInfo

from ._compat import REPORT_COUNTERS, add_error, bump, construct
from .dedupe import dedupe_segments, phash_from_frames
from .frames import sample_frames
from .probe import ProbeError, probe
from .scenes import detect_scenes
from .tagging import HeuristicTagger, Tagger, TagResult, VisionLLMTagger
from .thumbs import ThumbnailError, make_thumbnail
from .transcribe import Transcriber

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".flv",
    ".wmv",
    ".mpg",
    ".mpeg",
)

ProgressCallback = Callable[[int, int, Path], None]

DEFAULT_TAG_WEIGHTS = {"motion": 0.45, "face": 0.30, "speech": 0.25}


@dataclass
class IngestOptions:
    """Runtime knobs for :func:`ingest_folder` (all have safe defaults)."""

    min_scene_sec: float = 0.8
    scene_threshold: float = 27.0
    whisper_enabled: bool = True
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str | None = None
    whisper_beam_size: int = 5
    whisper_vad_filter: bool = True
    whisper_strict: bool = False
    phash_threshold: int = 6
    compute_phash: bool = True
    thumbs_dir: Path = field(default_factory=lambda: Path("data") / "thumbs")
    write_thumbs: bool = True
    thumb_width: int = 360
    frames_per_segment: int = 3
    vision_tagging: bool = False
    vision_provider: str = "openai"
    tag_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TAG_WEIGHTS))
    tagger: Tagger | None = None
    extensions: tuple[str, ...] = VIDEO_EXTENSIONS

    @classmethod
    def from_settings(cls, settings: Any) -> IngestOptions:
        """Build options from a pydantic-settings object without importing it."""
        data_dir = Path(getattr(settings, "data_dir", "data"))
        thumbs_dir = getattr(settings, "thumbs_dir", None) or (data_dir / "thumbs")
        return cls(
            min_scene_sec=float(getattr(settings, "min_scene_sec", 0.8)),
            scene_threshold=float(getattr(settings, "scene_threshold", 27.0)),
            whisper_enabled=bool(getattr(settings, "whisper_enabled", True)),
            whisper_model=str(getattr(settings, "whisper_model", "small")),
            whisper_device=str(getattr(settings, "whisper_device", "cpu")),
            whisper_compute_type=str(getattr(settings, "whisper_compute_type", "int8")),
            whisper_language=getattr(settings, "whisper_language", None),
            phash_threshold=int(getattr(settings, "phash_threshold", 6)),
            thumbs_dir=Path(thumbs_dir),
            frames_per_segment=int(getattr(settings, "frames_per_segment", 3)),
            vision_tagging=bool(getattr(settings, "vision_tagging", False)),
            tag_weights=dict(getattr(settings, "hook_weights", DEFAULT_TAG_WEIGHTS)),
        )


def _new_report(folder: Path) -> IngestReport:
    values: dict[str, Any] = {
        "folder": str(folder),
        "root": str(folder),
        "elapsed_sec": 0.0,
        "errors": [],
    }
    for names in REPORT_COUNTERS.values():
        for name in names:
            values[name] = 0
    return construct(IngestReport, **values)


def discover_videos(folder: str | Path, extensions: Sequence[str] = VIDEO_EXTENSIONS) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []
    allowed = {extension.lower() for extension in extensions}
    videos = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    ]
    return sorted(videos, key=lambda item: str(item).lower())


def _resolve_tagger(options: IngestOptions) -> Tagger:
    if options.tagger is not None:
        return options.tagger
    if options.vision_tagging:
        vision = VisionLLMTagger(provider=options.vision_provider)
        if vision.available:
            return vision
        logger.info("vision tagging requested but no API key is configured; using heuristics")
    return HeuristicTagger(weights=options.tag_weights)


def _add_segments(repository: ClipRepository, clip_id: Any, segments: Sequence[Any]) -> None:
    """Call ``add_segments`` with either ``(clip_id, segments)`` or ``(segments)``."""
    method = repository.add_segments
    try:
        parameters = [
            parameter
            for parameter in inspect.signature(method).parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        parameters = []
    if parameters and len(parameters) < 2:
        method(segments)  # type: ignore[call-arg]
        return
    method(clip_id, segments)  # type: ignore[call-arg]


def _build_segments(
    path: Path,
    info: ClipInfo,
    bounds: Sequence[tuple[float, float]],
    transcripts: Sequence[str],
    tagger: Tagger,
    options: IngestOptions,
) -> list[SegmentInfo]:
    segments: list[SegmentInfo] = []
    for index, (start, end) in enumerate(bounds):
        transcript = transcripts[index] if index < len(transcripts) else ""
        frames = (
            sample_frames(path, start, end, options.frames_per_segment)
            if options.frames_per_segment > 0
            else []
        )
        phash = phash_from_frames(frames) if options.compute_phash else None

        thumb: Path | None = None
        if options.write_thumbs:
            midpoint = (start + end) / 2
            middle_frame = frames[len(frames) // 2] if frames else None
            try:
                thumb = make_thumbnail(
                    path,
                    midpoint,
                    options.thumbs_dir,
                    info.sha1,
                    index,
                    width=options.thumb_width,
                    frame=middle_frame,
                )
            except ThumbnailError:
                logger.warning("thumbnail failed for %s scene %d", path, index)

        try:
            tag_result = tagger.tag(frames, transcript)
        except Exception:
            logger.exception("tagging failed for %s scene %d", path, index)
            tag_result = TagResult()
        tags = list(getattr(tag_result, "tags", []) or [])
        hook_score = float(getattr(tag_result, "hook_score", 0.0) or 0.0)

        segments.append(
            construct(
                SegmentInfo,
                start=float(start),
                end=float(end),
                transcript=transcript,
                tags=tags,
                hook_score=hook_score,
                phash=phash,
                thumb_path=str(thumb) if thumb is not None else None,
                clip_id=None,
                embedding=None,
            )
        )
    return segments


def ingest_folder(
    folder: str | Path,
    repo: ClipRepository,
    options: IngestOptions | None = None,
    *,
    progress: ProgressCallback | None = None,
    probe_fn: Callable[[str | Path], ClipInfo] = probe,
    scenes_fn: Callable[..., list[tuple[float, float]]] = detect_scenes,
    transcriber: Transcriber | None = None,
) -> IngestReport:
    """Ingest every video below ``folder``; DB writes go only through ``repo``.

    Clips are idempotent on SHA-1: an already known clip is skipped before any
    expensive scene/whisper work.  A broken clip does not abort the folder; the
    error is recorded on the report instead.
    """
    opts = options or IngestOptions()
    root = Path(folder)
    report = _new_report(root)
    started = time.perf_counter()
    videos = discover_videos(root, opts.extensions)
    speech = transcriber or Transcriber.from_options(opts)
    tagger = _resolve_tagger(opts)

    for position, path in enumerate(videos, start=1):
        bump(report, "clips_seen")
        if progress is not None:
            progress(position, len(videos), path)
        try:
            info = probe_fn(path)
            existing = repo.get_clip_by_sha1(info.sha1)
            if existing is not None:
                bump(report, "clips_skipped")
                logger.info("skipping already ingested clip %s", path)
                continue

            try:
                bounds = scenes_fn(
                    path,
                    min_scene_sec=opts.min_scene_sec,
                    threshold=opts.scene_threshold,
                    duration=info.duration,
                )
            except Exception:
                logger.exception("scene detection failed for %s; using whole clip", path)
                bounds = [(0.0, float(info.duration))] if info.duration > 0 else []

            transcripts = speech.transcripts_for(path, bounds)
            segments = _build_segments(path, info, bounds, transcripts, tagger, opts)
            deduped = dedupe_segments(segments, opts.phash_threshold)

            saved_clip = repo.add_clip(info)
            clip_id = _clip_id(saved_clip, info)
            if clip_id is None:
                raise RuntimeError("repository did not assign a clip id")
            _add_segments(repo, clip_id, deduped)

            bump(report, "clips_added")
            bump(report, "segments_added", len(deduped))
            bump(report, "segments_deduped", len(segments) - len(deduped))
            logger.info("ingested %s with %d segments", path, len(deduped))
        except ProbeError as exc:
            add_error(report, str(path), exc)
            logger.error("probe failed for %s: %s", path, exc)
        except Exception as exc:
            add_error(report, str(path), exc)
            logger.exception("ingest failed for %s", path)

    if hasattr(report, "elapsed_sec"):
        report.elapsed_sec = round(time.perf_counter() - started, 3)
    logger.info(
        "ingest finished for %s: seen=%s added=%s skipped=%s errors=%s",
        root,
        _counter(report, "clips_seen"),
        _counter(report, "clips_added"),
        _counter(report, "clips_skipped"),
        len(getattr(report, "errors", []) or []),
    )
    return report


def _clip_id(saved: Any, source: Any) -> int | None:
    """Extract the DB id whether the repo returns a ClipInfo or a bare int."""
    if isinstance(saved, int):
        return saved
    clip_id = getattr(saved, "id", None)
    if clip_id is None:
        clip_id = getattr(source, "id", None)
    return clip_id


def _counter(report: IngestReport, canonical: str) -> int:
    for name in REPORT_COUNTERS.get(canonical, (canonical,)):
        if hasattr(report, name):
            return int(getattr(report, name) or 0)
    return 0


def iter_video_files(folder: str | Path) -> Iterable[Path]:
    """Convenience alias used by the CLI."""
    return iter(discover_videos(folder))

