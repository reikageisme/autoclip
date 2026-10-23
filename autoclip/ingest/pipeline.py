"""Ingest pipeline: probe, scene split, transcript, tags, phash and thumbnails."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoclip.contracts import ClipInfo, ClipRepository, IngestError, IngestReport, SegmentInfo

from .dedupe import dedupe_segments, phash_from_frames
from .frames import sample_frames
from .probe import ProbeError, probe
from .proxy import make_proxy, needs_proxy
from .scenes import detect_scenes, split_long_scenes
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

DEFAULT_TAG_WEIGHTS = {"motion": 0.4, "face": 0.35, "speech": 0.25}


@dataclass
class IngestOptions:
    """Runtime knobs for :func:`ingest_folder` (all have safe defaults)."""

    min_scene_sec: float = 0.8
    max_scene_sec: float = 6.0
    scene_threshold: float = 27.0
    whisper_enabled: bool = True
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    whisper_language: str | None = "vi"
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
    tag_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TAG_WEIGHTS))
    tagger: Tagger | None = None
    extensions: tuple[str, ...] = VIDEO_EXTENSIONS
    proxy_dir: Path | None = None
    """Có thì nguồn lớn hơn 1080p được hạ về bản làm việc 1080p ở đây (xem proxy.py)."""

    @classmethod
    def from_settings(cls, settings: Any) -> IngestOptions:
        """Options từ ``autoclip.config.Settings`` — nguồn cấu hình duy nhất."""
        return cls(
            min_scene_sec=settings.min_scene_sec,
            max_scene_sec=settings.max_scene_sec,
            scene_threshold=settings.scene_threshold,
            whisper_enabled=settings.whisper_enabled,
            whisper_model=settings.whisper_model,
            whisper_device=settings.whisper_device,
            whisper_compute_type=settings.whisper_compute_type,
            whisper_language=settings.whisper_language or None,
            phash_threshold=settings.phash_threshold,
            thumbs_dir=Path(settings.thumbs_dir),
            frames_per_segment=settings.frames_per_segment,
            vision_tagging=bool(settings.vision_model),
            tag_weights=dict(settings.hook_weights),
            proxy_dir=Path(settings.data_dir) / "proxies",
        )


def discover_videos(folder: str | Path, extensions: Sequence[str] = VIDEO_EXTENSIONS) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []
    allowed = {extension.lower() for extension in extensions}
    videos = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed]
    return sorted(videos, key=lambda item: str(item).lower())


def _resolve_tagger(options: IngestOptions) -> Tagger:
    if options.tagger is not None:
        return options.tagger
    if options.vision_tagging:
        vision = VisionLLMTagger()
        if vision.available:
            return vision
        logger.info("vision tagging requested but no API key is configured; using heuristics")
    return HeuristicTagger(weights=options.tag_weights)


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
            SegmentInfo(
                start=float(start),
                end=float(end),
                transcript=transcript,
                tags=tags,
                hook_score=hook_score,
                phash=phash,
                thumb_path=str(thumb) if thumb is not None else None,
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
    report = IngestReport(folder=str(root))
    started = time.perf_counter()
    videos = discover_videos(root, opts.extensions)
    speech = transcriber or Transcriber.from_options(opts)
    tagger = _resolve_tagger(opts)

    for position, path in enumerate(videos, start=1):
        report.clips_seen += 1
        if progress is not None:
            progress(position, len(videos), path)
        try:
            info = probe_fn(path)
            existing = repo.get_clip_by_sha1(info.sha1)
            if existing is not None:
                report.clips_skipped += 1
                logger.info("skipping already ingested clip %s", path)
                continue

            if opts.proxy_dir is not None and needs_proxy(info.width, info.height):
                proxy = make_proxy(
                    path, opts.proxy_dir / f"{info.sha1}.mp4", info.width, info.height
                )
                small = probe_fn(proxy)
                # sha1 vẫn của file gốc: nhập lại file gốc thì nhận ra là đã có
                info = small.model_copy(update={"sha1": info.sha1})
                path = proxy

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

            bounds = split_long_scenes(bounds, opts.max_scene_sec)
            transcripts = speech.transcripts_for(path, bounds)
            if speech.last_error:
                report.errors.append(
                    IngestError(path=str(path), message=f"whisper: {speech.last_error}")
                )
            segments = _build_segments(path, info, bounds, transcripts, tagger, opts)
            deduped = dedupe_segments(segments, opts.phash_threshold)

            saved_clip = repo.add_clip(info)
            if saved_clip.id is None:
                raise RuntimeError("repository did not assign a clip id")
            repo.add_segments(saved_clip.id, deduped)

            report.clips_added += 1
            report.segments_added += len(deduped)
            report.segments_deduped += len(segments) - len(deduped)
            logger.info("ingested %s with %d segments", path, len(deduped))
        except ProbeError as exc:
            report.errors.append(IngestError(path=str(path), message=str(exc)))
            logger.error("probe failed for %s: %s", path, exc)
        except Exception as exc:
            report.errors.append(IngestError(path=str(path), message=str(exc)))
            logger.exception("ingest failed for %s", path)

    report.elapsed_sec = round(time.perf_counter() - started, 3)
    logger.info(
        "ingest finished for %s: seen=%s added=%s skipped=%s errors=%s",
        root,
        report.clips_seen,
        report.clips_added,
        report.clips_skipped,
        len(report.errors),
    )
    return report


def iter_video_files(folder: str | Path) -> Iterable[Path]:
    """Convenience alias used by the CLI."""
    return iter(discover_videos(folder))
