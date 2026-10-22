"""Môi trường Alembic: URL lấy từ config (hoặc `-x url=...`), metadata từ models."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context

from autoclip.db.engine import make_engine
from autoclip.db.models import Base

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str | None:
    return context.get_x_argument(as_dictionary=True).get("url") or config.attributes.get("url")


def run_migrations_offline() -> None:
    context.configure(
        url=_url() or config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    engine = make_engine(_url())
    with engine.connect() as conn:
        _run(conn)
    engine.dispose()


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
