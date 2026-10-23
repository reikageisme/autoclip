from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from autoclip.contracts import ClipInfo, SegmentInfo
from autoclip.ingest.pipeline import IngestOptions, discover_videos, ingest_folder


class InMemoryRepo:
    def __init__(self) -> None:
        self.clips: list[tuple[int, ClipInfo]] = []
        self.segments: dict[int, list[SegmentInfo]] = {}

    def get_clip_by_sha1(self, sha1: str) -> ClipInfo | None:
        for _clip_id, clip in self.clips:
            if clip.sha1 == sha1:
                return clip
        return None

    def get_clip(self, clip_id: int) -> ClipInfo | None:
        return next((c for i, c in self.clips if i == clip_id), None)

    def add_clip(self, clip: ClipInfo) -> ClipInfo:
        clip_id = len(self.clips) + 1
        saved = clip.model_copy(update={"id": clip_id})
        self.clips.append((clip_id, saved))
        return saved

    def add_segments(self, clip_id: int, segments: list[SegmentInfo]) -> list[SegmentInfo]:
        stored = list(segments)
        self.segments[clip_id] = stored
        return stored


def _options(tmp_path: Path, **overrides: Any) -> IngestOptions:
    values: dict[str, Any] = {
        "min_scene_sec": 0.5,
        "whisper_enabled": False,
        "frames_per_segment": 1,
        "thumbs_dir": tmp_path / "thumbs",
        "compute_phash": False,
        "phash_threshold": 6,
    }
    values.update(overrides)
    return IngestOptions(**values)


def _fake_scenes(path, min_scene_sec, threshold, duration):
    return [(0.0, min(1.0, duration)), (1.0, duration)]


def _copy(src: Path, destination: Path) -> Path:
    shutil.copyfile(src, destination)
    return destination


def test_discover_videos_filters_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "b.MP4").write_bytes(b"x")
    (tmp_path / "a.mov").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.mkv").write_bytes(b"x")
    found = discover_videos(tmp_path)
    assert [item.name for item in found] == ["a.mov", "b.MP4", "c.mkv"]


def test_ingest_folder_writes_through_repo_and_is_idempotent(
    tmp_path: Path, media_files: dict[str, Path]
) -> None:
    folder = tmp_path / "clips"
    folder.mkdir()
    _copy(media_files["testsrc"], folder / "one.mp4")
    _copy(media_files["silent"], folder / "two.mp4")
    repo = InMemoryRepo()
    seen: list[tuple[int, int, Path]] = []

    report = ingest_folder(
        folder,
        repo,
        _options(tmp_path),
        progress=lambda done, total, path: seen.append((done, total, path)),
        scenes_fn=_fake_scenes,
    )

    assert report.clips_seen == 2
    assert report.clips_added == 2
    assert report.clips_skipped == 0
    assert report.segments_added == 4
    assert report.segments_deduped == 0
    assert report.elapsed_sec >= 0.0
    assert report.errors == []
    assert len(repo.clips) == 2
    assert all(len(items) == 2 for items in repo.segments.values())
    assert len(seen) == 2
    for _clip_id, clip in repo.clips:
        assert Path(clip.path).is_file()
    for segments in repo.segments.values():
        for segment in segments:
            assert segment.transcript == ""
            assert segment.tags
            assert 0.0 <= segment.hook_score <= 1.0
            assert segment.thumb_path is not None
            assert Path(segment.thumb_path).is_file()

    second = ingest_folder(folder, repo, _options(tmp_path), scenes_fn=_fake_scenes)
    assert second.clips_added == 0
    assert second.clips_skipped == 2
    assert len(repo.clips) == 2


def test_ingest_skips_duplicate_content(tmp_path: Path, media_files: dict[str, Path]) -> None:
    folder = tmp_path / "dupes"
    folder.mkdir()
    _copy(media_files["testsrc"], folder / "a.mp4")
    _copy(media_files["testsrc"], folder / "b.mp4")
    repo = InMemoryRepo()
    report = ingest_folder(folder, repo, _options(tmp_path), scenes_fn=_fake_scenes)
    assert report.clips_seen == 2
    assert report.clips_added == 1
    assert report.clips_skipped == 1
    assert len(repo.clips) == 1


def test_ingest_records_broken_file_and_continues(
    tmp_path: Path, media_files: dict[str, Path]
) -> None:
    folder = tmp_path / "mixed"
    folder.mkdir()
    _copy(media_files["testsrc"], folder / "good.mp4")
    (folder / "broken.mp4").write_bytes(b"this is not a video")
    repo = InMemoryRepo()
    report = ingest_folder(folder, repo, _options(tmp_path), scenes_fn=_fake_scenes)
    assert report.clips_seen == 2
    assert report.clips_added == 1
    assert len(report.errors) == 1
    assert report.errors[0].path.endswith("broken.mp4")
    assert report.errors[0].message


def test_ingest_falls_back_to_whole_clip_when_scene_detection_fails(
    tmp_path: Path, media_files: dict[str, Path]
) -> None:
    folder = tmp_path / "fallback"
    folder.mkdir()
    _copy(media_files["black"], folder / "clip.mp4")

    def exploding_scenes(*args: Any, **kwargs: Any):
        raise RuntimeError("no detector")

    repo = InMemoryRepo()
    report = ingest_folder(folder, repo, _options(tmp_path), scenes_fn=exploding_scenes)
    assert report.clips_added == 1
    assert report.segments_added == 1
    assert len(repo.segments[1]) == 1


def test_ingest_computes_phash_when_enabled(tmp_path: Path, media_files: dict[str, Path]) -> None:
    folder = tmp_path / "hash"
    folder.mkdir()
    _copy(media_files["testsrc"], folder / "clip.mp4")
    repo = InMemoryRepo()
    report = ingest_folder(
        folder,
        repo,
        _options(tmp_path, compute_phash=True),
        scenes_fn=_fake_scenes,
    )
    assert report.segments_added >= 1
    assert all(segment.phash for segments in repo.segments.values() for segment in segments)
