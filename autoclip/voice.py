"""Giọng đọc tiếng Việt offline: sherpa-onnx + giọng Piper vi_VN-vais1000 (CC BY 4.0).

Model (~65MB) tải về `<DATA_DIR>/models/tts/` ở lần dùng đầu. Trên Windows phải nạp đúng
onnxruntime.dll của gói Python trước sherpa-onnx: Windows 11 có sẵn
System32\\onnxruntime.dll bản cũ (Windows ML), nạp nhầm là sập tiến trình.
"""

from __future__ import annotations

import logging
import tarfile
import threading
import urllib.request
import wave
from pathlib import Path

log = logging.getLogger(__name__)

VOICE = "vits-piper-vi_VN-vais1000-medium"
VOICE_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{VOICE}.tar.bz2"
CREDIT = "Giọng đọc: Piper vi_VN-vais1000 (VAIS-1000 corpus, CC BY 4.0)"

_lock = threading.Lock()
_tts = None


def _preload_onnxruntime() -> None:
    import sys

    if sys.platform != "win32":
        return
    import ctypes

    import onnxruntime

    dll = Path(onnxruntime.__file__).parent / "capi" / "onnxruntime.dll"
    if dll.is_file():
        ctypes.WinDLL(str(dll))


def ensure_model(models_dir: Path) -> Path:
    """Thư mục model giọng đọc, tải + giải nén nếu chưa có."""
    folder = models_dir / "tts" / VOICE
    if (folder / "tokens.txt").is_file():
        return folder
    folder.parent.mkdir(parents=True, exist_ok=True)
    archive = folder.parent / f"{VOICE}.tar.bz2"
    log.info("tải giọng đọc %s", VOICE_URL)
    urllib.request.urlretrieve(VOICE_URL, archive)
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(folder.parent, filter="data")
    archive.unlink(missing_ok=True)
    return folder


def _engine(models_dir: Path):
    global _tts
    with _lock:
        if _tts is None:
            _preload_onnxruntime()
            import sherpa_onnx

            d = ensure_model(models_dir)
            cfg = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=str(d / f"{VOICE.removeprefix('vits-piper-')}.onnx"),
                        tokens=str(d / "tokens.txt"),
                        data_dir=str(d / "espeak-ng-data"),
                    ),
                    num_threads=4,
                )
            )
            _tts = sherpa_onnx.OfflineTts(cfg)
        return _tts


def synthesize(text: str, out: Path, models_dir: Path, speed: float = 1.0) -> float:
    """Đọc `text` ra file WAV `out`; trả về số giây."""
    import numpy as np

    tts = _engine(models_dir)
    with _lock:
        audio = tts.generate(text, sid=0, speed=speed)
    samples = np.clip(np.asarray(audio.samples, dtype=np.float32), -1.0, 1.0)
    # nghỉ 0.15s cuối câu cho tự nhiên khi ghép nhiều câu liền nhau
    pad = np.zeros(int(audio.sample_rate * 0.15), dtype=np.float32)
    samples = np.concatenate([samples, pad])
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(audio.sample_rate)
        w.writeframes((samples * 32767).astype(np.int16).tobytes())
    return len(samples) / audio.sample_rate
