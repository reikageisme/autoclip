from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from autoclip.ingest.thumbs import (
    ThumbnailError,
    extract_thumbnail,
    make_thumbnail,
    save_frame_jpg,
    thumbnail_path,
)


def test_thumbnail_path_name(tmp_path: Path) -> None:
    path = thumbnail_path(tmp_path, "abc123", 4)
    assert path == tmp_path / "abc123_4.jpg"


def test_save_frame_jpg(tmp_path: Path) -> None:
    frame = np.full((64, 36, 3), 128, dtype=np.uint8)
    destination = tmp_path / "nested" / "frame.jpg"
    save_frame_jpg(frame, destination)
    assert destination.is_file()
    with Image.open(destination) as image:
        assert image.size == (36, 64)
        assert image.format == "JPEG"


def test_extract_thumbnail_from_video(tmp_path: Path, media_files: dict[str, Path]) -> None:
    destination = tmp_path / "thumb.jpg"
    result = extract_thumbnail(media_files["testsrc"], 0.5, destination, width=180)
    assert result == destination
    with Image.open(result) as image:
        assert image.size[0] == 180


def test_extract_thumbnail_missing_video(tmp_path: Path) -> None:
    with pytest.raises(ThumbnailError):
        extract_thumbnail(tmp_path / "missing.mp4", 0.0, tmp_path / "out.jpg")


def test_make_thumbnail_uses_decoded_frame(tmp_path: Path, media_files: dict[str, Path]) -> None:
    frame = np.full((32, 18, 3), 200, dtype=np.uint8)
    result = make_thumbnail(
        media_files["testsrc"],
        0.5,
        tmp_path,
        "sha",
        7,
        frame=frame,
    )
    assert result == tmp_path / "sha_7.jpg"
    assert result.is_file()
