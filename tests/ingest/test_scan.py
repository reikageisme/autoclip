from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from autoclip.ingest.scan import scan_video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="cần ffmpeg")


@pytest.fixture(scope="module")
def two_shots(tmp_path_factory) -> Path:
    """2s màu đỏ rồi 2s testsrc2: đúng một điểm cắt ở giây thứ 2."""
    out = tmp_path_factory.mktemp("scan") / "cut.mp4"
    graph = (
        "color=c=red:s=360x640:r=25:d=2[a];testsrc2=s=360x640:r=25:d=2[b];[a][b]concat=n=2:v=1[v]"
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-filter_complex", graph, "-map", "[v]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )  # fmt: skip
    return out


def test_scan_finds_cut_and_samples_frames(two_shots: Path, tmp_path: Path) -> None:
    scan = scan_video(
        two_shots, width=360, height=640, fps=25, duration=4.0, min_scene_sec=0.5,
        workdir=tmp_path,
    )  # fmt: skip
    try:
        assert len(scan.bounds) == 2
        assert scan.bounds[0][0] == 0.0 and scan.bounds[-1][1] == 4.0
        assert abs(scan.bounds[0][1] - 2.0) < 0.1
        assert scan.frame_count >= 7  # 2 khung/giây x 4 giây

        red = scan.frames(0.0, 2.0, 3)
        assert len(red) == 3 and red[0].shape == (640, 360, 3)  # không phóng to quá gốc
        assert red[1][..., 0].mean() > 200 and red[1][..., 1].mean() < 60
        other = scan.frames(2.0, 4.0, 1)
        assert other[0][..., 1].mean() > 60  # không còn là nền đỏ
    finally:
        scan.close()
    assert not scan.frame_dir.exists()


def test_scan_raises_on_bad_file(tmp_path: Path) -> None:
    from autoclip.ingest.scan import ScanError

    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ScanError):
        scan_video(bad, width=360, height=640, fps=25, duration=1.0, workdir=tmp_path)
    assert list(tmp_path.glob("scan-*")) == []
