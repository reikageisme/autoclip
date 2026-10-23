"""Bản làm việc 1080p cho nguồn lớn (footage 4K, HEVC 10-bit…).

Video thành phẩm chỉ 1080x1920, nên giữ nguồn 4K chỉ làm mọi bước chậm đi: dò cảnh phải
giải mã cả 4K, mỗi lần render Concat lại giải mã 4K. Ingest tạo bản H.264 8-bit một lần
(`data/proxies/<sha1>.mp4`) rồi các bước sau dùng bản đó; sha1 vẫn tính trên file gốc nên
nhập lại không tạo trùng.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

MAX_SHORT_SIDE = 1080
MAX_LONG_SIDE = 1920


def needs_proxy(width: int, height: int) -> bool:
    return min(width, height) > MAX_SHORT_SIDE or max(width, height) > MAX_LONG_SIDE


def _ffmpeg() -> str:
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"


def _scale(width: int, height: int) -> str:
    # giữ tỉ lệ, cạnh ngắn về 1080, số chẵn cho H.264
    return "scale=1080:-2" if width <= height else "scale=-2:1080"


def make_proxy(
    src: Path, dest: Path, width: int, height: int, timeout: float = 4 * 3600, force: bool = False
) -> Path:
    """Tạo `dest` nếu chưa có. Thử Intel QSV trước (nhanh hơn), không được thì x264."""
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.mp4")
    vf = _scale(width, height)
    common = ["-hide_banner", "-v", "error", "-y", "-i", str(src)]
    tail = ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp)]
    # Keyframe mỗi giây: Concat tua tới giữa file dài nhanh và đúng. GOP mặc định của QSV
    # (256 frame, ~8.5s) làm cảnh lấy giữa hai keyframe xa bị render ra khung đen.
    gop = ["-g", "30", "-keyint_min", "30"]
    attempts = [
        (
            "qsv",
            [*gop, "-vf", f"{vf},format=nv12", "-c:v", "h264_qsv", "-global_quality", "22"],
        ),
        (
            "x264",
            [
                *gop,
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
            ],
        ),
    ]
    last = ""
    for name, video in attempts:
        cmd = [_ffmpeg(), *common, *video, *tail]
        log.info("proxy %s (%s): %s", src.name, name, dest.name)
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            last = str(exc)
            continue
        if done.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            tmp.replace(dest)
            return dest
        last = (done.stderr or "").strip()[-300:]
        log.warning("proxy %s bằng %s lỗi: %s", src.name, name, last)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"không tạo được bản 1080p cho {src.name}: {last}")
