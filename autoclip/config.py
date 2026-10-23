"""Cấu hình toàn cục, đọc từ biến môi trường và file `.env` (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Concat / ReiCut ---
    concat_bin: Path = Path(r"D:\work\reicut\src\target\app\concat.exe")
    # Thư mục ngoài home được phép ghi; chuỗi ngăn cách bằng os.pathsep, truyền nguyên cho Concat.
    concat_mcp_roots: str = ""
    concat_mcp_mode: Literal["headless", "window", "auto"] = "headless"
    concat_timeout_sec: float = 600.0

    # --- Dữ liệu ---
    data_dir: Path = _REPO_ROOT / "data"

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
