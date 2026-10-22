"""Chạy migration Alembic từ code (CLI, test, worker)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine

_MIGRATIONS = Path(__file__).resolve().parent / "migrations"


def _config() -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS))
    cfg.attributes["configure_logger"] = False
    return cfg


def upgrade(engine: Engine, revision: str = "head") -> None:
    """Nâng schema của `engine` lên `revision`."""
    cfg = _config()
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, revision)
