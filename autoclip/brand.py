"""Bộ thương hiệu: logo, cảnh kết CTA, nhạc nền — áp vào mọi video khi dựng.

Lưu ở `<DATA_DIR>/brand.json`. `apply_brand` ghi vào `plan.params["brand"]` những gì
compiler cần (đường dẫn, thời lượng nhạc đã đo, màu…) để compiler vẫn là hàm thuần và
plan lưu trong DB dựng lại được đúng video đó.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from autoclip.contracts import RenderPlan

MUSIC_EXTENSIONS = (".mp3", ".m4a", ".aac", ".wav", ".ogg", ".flac")


class BrandKit(BaseModel):
    logo_path: str = ""
    logo_corner: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = "top-right"
    logo_size: float = Field(0.18, ge=0.05, le=0.5)
    """Bề ngang logo, tỉ lệ bề ngang khung hình."""
    logo_opacity: float = Field(0.9, ge=0.1, le=1.0)

    outro_enabled: bool = False
    outro_text: str = ""
    """Các dòng chữ cảnh kết, ví dụ tên công ty + hotline (xuống dòng bằng \\n)."""
    outro_seconds: float = Field(2.5, ge=1.0, le=8.0)
    outro_background: str = "#111111"
    outro_color: str = "#FFD400"

    music_folder: str = ""
    music_volume: float = Field(0.35, ge=0.0, le=1.0)
    """Âm lượng nhạc khi video không có lời nói."""
    music_volume_speech: float = Field(0.08, ge=0.0, le=1.0)
    """Âm lượng nhạc khi video có lời nói (để nghe rõ lời)."""


def brand_path(data_dir: Path) -> Path:
    return Path(data_dir) / "brand.json"


def load_brand(data_dir: Path) -> BrandKit:
    path = brand_path(data_dir)
    if path.is_file():
        try:
            return BrandKit.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return BrandKit()


def save_brand(data_dir: Path, kit: BrandKit) -> BrandKit:
    path = brand_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(kit.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return kit


def music_tracks(folder: str) -> list[Path]:
    root = Path(folder) if folder else None
    if not root or not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.suffix.lower() in MUSIC_EXTENSIONS)


def audio_duration(path: Path) -> float:
    """Thời lượng file âm thanh (ffprobe); 0 nếu đọc không được — khi đó bỏ nhạc."""
    ffprobe = os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe") or "ffprobe"
    cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        return float((out.stdout or "0").strip() or 0)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


def apply_brand(plan: RenderPlan, kit: BrandKit, *, has_speech: bool | None = None) -> RenderPlan:
    """Bản plan có `params["brand"]`; nhạc chọn theo seed (cùng seed → cùng bài)."""
    brand: dict = {}
    if kit.logo_path and Path(kit.logo_path).is_file():
        brand["logo"] = {
            "path": str(Path(kit.logo_path).resolve()),
            "corner": kit.logo_corner,
            "size": kit.logo_size,
            "opacity": kit.logo_opacity,
        }
    if kit.outro_enabled and kit.outro_text.strip():
        brand["outro"] = {
            "text": kit.outro_text.strip(),
            "seconds": kit.outro_seconds,
            "background": kit.outro_background,
            "color": kit.outro_color,
        }
    tracks = music_tracks(kit.music_folder)
    if tracks:
        track = random.Random(f"music:{plan.seed}").choice(tracks)
        duration = audio_duration(track)
        if duration > 0:
            speech = (
                has_speech
                if has_speech is not None
                else any(t.kind == "captions" for t in plan.texts)
            )
            brand["music"] = {
                "path": str(track.resolve()),
                "duration": round(duration, 3),
                "volume": kit.music_volume_speech if speech else kit.music_volume,
            }
    out = plan.model_copy(deep=True)
    if brand:
        out.params["brand"] = brand
    else:
        out.params.pop("brand", None)
    return out
