"""Quét video MỘT lượt giải mã: vừa cắt cảnh, vừa lấy khung mẫu.

Trước đây cắt cảnh (OpenCV đọc từng khung full-res) và lấy mẫu (mỗi khung một lần gọi
ffmpeg, phải tua trong HEVC GOP dài) giải mã video nhiều lần — chiếm ~2/3 thời gian nhập.
Ở đây một tiến trình ffmpeg tách luồng hình làm hai:

- khung 256px ở fps gốc → ContentDetector của PySceneDetect (cùng ngưỡng, cùng độ thu nhỏ
  mà `scenedetect.detect` tự dùng, nên kết quả cắt cảnh như cũ);
- JPEG 480px, `SAMPLE_FPS` khung/giây → ảnh mẫu cho tag/phash/thumbnail.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .frames import sample_times
from .scenes import merge_short_scenes

logger = logging.getLogger(__name__)

SCENE_WIDTH = 256
SAMPLE_WIDTH = 480
SAMPLE_FPS = 2.0


class ScanError(RuntimeError):
    pass


def _even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


@dataclass
class VideoScan:
    bounds: list[tuple[float, float]]
    frame_dir: Path
    frame_count: int
    sample_fps: float = SAMPLE_FPS
    _cache: dict[int, Any] = field(default_factory=dict, repr=False)

    def _load(self, index: int) -> Any:
        if index not in self._cache:
            import numpy as np
            from PIL import Image

            with Image.open(self.frame_dir / f"{index + 1:07d}.jpg") as image:
                self._cache[index] = np.asarray(image.convert("RGB"))
        return self._cache[index]

    def frames(self, start: float, end: float, count: int) -> list[Any]:
        """Khung mẫu gần nhất với `count` thời điểm chia đều trong [start, end]."""
        if self.frame_count <= 0 or count <= 0:
            return []
        lo = min(self.frame_count - 1, max(0, math.ceil(start * self.sample_fps)))
        hi = min(self.frame_count - 1, max(lo, math.floor(end * self.sample_fps)))
        out = []
        for t in sample_times(start, end, count):
            index = min(hi, max(lo, round(t * self.sample_fps)))
            try:
                out.append(self._load(index))
            except OSError:
                logger.warning("thiếu khung mẫu %d trong %s", index, self.frame_dir)
        return out

    def close(self) -> None:
        self._cache.clear()
        shutil.rmtree(self.frame_dir, ignore_errors=True)


def scan_video(
    path: str | Path,
    *,
    width: int,
    height: int,
    fps: float,
    duration: float,
    threshold: float = 27.0,
    min_scene_sec: float = 0.8,
    workdir: str | Path | None = None,
    timeout: float = 4 * 3600,
) -> VideoScan:
    """Cắt cảnh + ghi khung mẫu trong một lượt ffmpeg. Lỗi → `ScanError`."""
    import numpy as np
    from scenedetect import ContentDetector, FrameTimecode

    if width <= 0 or height <= 0:
        raise ScanError(f"kích thước video không hợp lệ: {width}x{height}")
    fps = float(fps) if fps and fps > 0 else 30.0
    sw = _even(min(width, SCENE_WIDTH))
    sh = _even(height * sw / width)
    frame_dir = Path(tempfile.mkdtemp(prefix="scan-", dir=workdir))
    graph = (
        f"[0:v]split=2[a][b];"
        f"[a]fps={fps:.6f},scale={sw}:{sh},format=bgr24[s];"
        f"[b]fps={SAMPLE_FPS},scale={_even(min(width, SAMPLE_WIDTH))}:-2[f]"
    )
    cmd = [
        os.environ.get("FFMPEG_BIN") or "ffmpeg",
        "-nostdin", "-v", "error", "-i", str(path),
        "-filter_complex", graph,
        "-map", "[s]", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        "-map", "[f]", "-q:v", "4", str(frame_dir / "%07d.jpg"),
    ]  # fmt: skip
    detector = ContentDetector(threshold=threshold, min_scene_len=15)
    frame_bytes = sw * sh * 3
    cuts: list[float] = []
    n = 0
    # stderr ra file: PIPE không ai đọc có thể làm ffmpeg kẹt khi in nhiều cảnh báo
    errfile = tempfile.TemporaryFile()  # noqa: SIM115 - đóng ở finally bên dưới
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errfile)
    except OSError as exc:
        errfile.close()
        shutil.rmtree(frame_dir, ignore_errors=True)
        raise ScanError(f"không chạy được ffmpeg: {exc}") from exc
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(sh, sw, 3)
            for cut in detector.process_frame(FrameTimecode(n, fps), frame):
                cuts.append(cut.seconds)
            n += 1
        if n:
            cuts.extend(c.seconds for c in detector.post_process(FrameTimecode(n - 1, fps)))
        proc.stdout.close()
        proc.wait(timeout=timeout)
        errfile.seek(0)
        err = errfile.read()
    except BaseException:
        proc.kill()
        proc.wait()
        shutil.rmtree(frame_dir, ignore_errors=True)
        raise
    finally:
        errfile.close()
    if proc.returncode != 0 or n == 0:
        shutil.rmtree(frame_dir, ignore_errors=True)
        detail = (err or b"").decode("utf-8", "replace").strip()[-500:]
        raise ScanError(f"ffmpeg quét video lỗi ({proc.returncode}): {detail}")

    end = float(duration) if duration and duration > 0 else n / fps
    edges = [0.0, *sorted(c for c in set(cuts) if 0.0 < c < end), end]
    raw = list(zip(edges, edges[1:], strict=False))
    bounds = merge_short_scenes(raw, min_scene_sec=min_scene_sec, duration=end)
    count = len(list(frame_dir.glob("*.jpg")))
    logger.debug("quét %s: %d khung, %d cảnh, %d khung mẫu", path, n, len(bounds), count)
    return VideoScan(bounds=bounds, frame_dir=frame_dir, frame_count=count)
