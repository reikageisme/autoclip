from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from autoclip.ingest.probe import ProbeError, parse_frame_rate, probe, sha1_file


def test_sha1_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"auto-clip" * 5000
    target = tmp_path / "blob.bin"
    target.write_bytes(payload)
    assert sha1_file(target) == hashlib.sha1(payload).hexdigest()
    assert sha1_file(target, chunk_size=7) == hashlib.sha1(payload).hexdigest()


def test_parse_frame_rate() -> None:
    assert parse_frame_rate("25/1") == pytest.approx(25.0)
    assert parse_frame_rate("30000/1001") == pytest.approx(29.97, rel=1e-3)
    assert parse_frame_rate("0/0") == 0.0
    assert parse_frame_rate(None) == 0.0


def test_probe_video_with_audio(media_files: dict[str, Path]) -> None:
    info = probe(media_files["testsrc"])
    assert info.width == 360
    assert info.height == 640
    assert info.duration == pytest.approx(2.0, abs=0.25)
    assert info.fps == pytest.approx(25.0, abs=0.5)
    assert info.has_audio is True
    assert len(info.sha1) == 40
    assert Path(info.path).is_file()


def test_probe_without_audio(media_files: dict[str, Path]) -> None:
    info = probe(media_files["silent"])
    assert info.has_audio is False


def test_probe_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ProbeError):
        probe(tmp_path / "missing.mp4")


def test_probe_rejects_non_video(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("not a video", encoding="utf-8")
    with pytest.raises(ProbeError):
        probe(target)
