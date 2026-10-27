"""Cấu hình toàn cục, đọc từ biến môi trường và file `.env` (pydantic-settings)."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent

FROZEN = bool(getattr(sys, "frozen", False))
"""Đang chạy bản đóng gói (PyInstaller) đi kèm ReiCut, không phải từ mã nguồn."""

if FROZEN:
    # <thư mục cài>/autoclip-service/autoclip-service.exe, concat.exe ở <thư mục cài>
    INSTALL_DIR = Path(sys.executable).resolve().parent.parent
    APP_HOME = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "AutoClip"
    DEFAULT_CONCAT_BIN = INSTALL_DIR / "concat.exe"
else:
    INSTALL_DIR = _REPO_ROOT
    APP_HOME = _REPO_ROOT
    DEFAULT_CONCAT_BIN = Path(r"D:\work\reicut\src\target\quick\concat.exe")

ENV_FILE = APP_HOME / ".env"
"""Cài đặt người dùng (panel ghi vào đây). Bản đóng gói: %LOCALAPPDATA%\\AutoClip\\.env."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Concat / ReiCut ---
    concat_bin: Path = DEFAULT_CONCAT_BIN
    # Thư mục ngoài home được phép ghi; chuỗi ngăn cách bằng os.pathsep, truyền nguyên cho Concat.
    concat_mcp_roots: str = ""
    concat_mcp_mode: Literal["headless", "window", "auto"] = "headless"
    concat_timeout_sec: float = 600.0

    # --- Dữ liệu ---
    data_dir: Path = APP_HOME / "data"

    # --- LLM (P3) ---
    # "openai_compat": endpoint kiểu OpenAI (/v1/chat/completions) như router riêng (9router).
    llm_provider: Literal["none", "openai", "anthropic", "openai_compat"] = "none"
    llm_base_url: str | None = None
    """URL gốc kèm /v1 (ví dụ http://127.0.0.1:<port>/v1) — bắt buộc khi openai_compat."""
    llm_api_key: SecretStr | None = None
    """Key cho openai_compat; openai/anthropic dùng OPENAI_API_KEY/ANTHROPIC_API_KEY."""
    llm_model: str | None = None
    """Tên model đúng như router liệt kê ở /v1/models."""
    vision_model: str | None = None
    """Model nhận ảnh cho vision tagging/QA; None = tắt vision, dùng heuristic."""
    llm_timeout_sec: float = 60.0
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # --- Ingest ---
    min_scene_sec: float = 0.8
    max_scene_sec: float = 6.0
    """Cảnh dài hơn bị chia đều; 0 = giữ nguyên."""
    scene_threshold: float = 27.0
    frames_per_segment: int = 3
    whisper_enabled: bool = True
    whisper_model: str = "small"
    """small | medium | large-v3: lớn hơn thì chép tiếng Việt chuẩn hơn, chậm hơn trên CPU."""
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    whisper_language: str | None = "vi"
    """Ép ngôn ngữ; để trống = tự nhận (hay nhầm với video chỉ có nhạc)."""
    phash_threshold: int = 6
    hook_weights: dict[str, float] = Field(
        default_factory=lambda: {"motion": 0.4, "face": 0.35, "speech": 0.25}
    )

    # --- Diversity (P3) ---
    no_repeat_last_n_renders: int = 50

    # --- Vận hành v2 ---
    render_workers: int = Field(1, ge=1, le=16)
    """Số tiến trình Concat render song song (mỗi job một thư mục project riêng)."""
    maintenance_interval_min: int = Field(30, ge=1, le=1440)
    cleanup_proxies_days: int = Field(30, ge=0, le=3650)
    cleanup_projects_days: int = Field(7, ge=0, le=3650)
    cleanup_videos_days: int = Field(30, ge=0, le=3650)
    """0 = không dọn loại đó. Video cũ bị xoá file, giữ dòng DB để đối soát."""
    disk_warn_free_gb: float = Field(5.0, ge=0.0, le=1000.0)
    disk_watch_paths: str = ""
    """Thêm đường dẫn cần theo dõi, cách nhau bằng os.pathsep; trống = ổ của DATA_DIR + C:."""

    # --- Đăng bài Facebook ---
    fb_page_id: str = ""
    fb_graph_version: str = "v23.0"
    publish_check_sec: int = Field(30, ge=5, le=3600)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "autoclip.db"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path.as_posix()}"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def out_dir(self) -> Path:
        return self.data_dir / "out"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def concat_tools_path(self) -> Path:
        return self.data_dir / "concat_tools.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
