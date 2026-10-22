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
    llm_provider: Literal["none", "openai", "anthropic"] = "none"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # --- Ingest ---
    min_scene_sec: float = 0.8
    whisper_enabled: bool = True
    whisper_model: str = "small"
    whisper_device: str = "auto"
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
