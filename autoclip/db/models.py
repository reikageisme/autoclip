"""SQLAlchemy models theo schema trong CLAUDE.md."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[str]: JSON}


class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String)
    sha1: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    duration: Mapped[float] = mapped_column(Float)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    fps: Mapped[float] = mapped_column(Float)
    has_audio: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    segments: Mapped[list[Segment]] = relationship(
        back_populates="clip", cascade="all, delete-orphan"
    )


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(primary_key=True)
    clip_id: Mapped[int] = mapped_column(ForeignKey("clips.id", ondelete="CASCADE"), index=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    transcript: Mapped[str | None] = mapped_column(String)
    tags: Mapped[list[str]] = mapped_column(default=list)
    hook_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    phash: Mapped[str | None] = mapped_column(String(64))
    thumb_path: Mapped[str | None] = mapped_column(String)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)

    clip: Mapped[Clip] = relationship(back_populates="segments")


class Recipe(Base):
    __tablename__ = "recipes"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    source_project: Mapped[str | None] = mapped_column(String)
    aspect: Mapped[str] = mapped_column(String(16))
    duration_target: Mapped[list[float]] = mapped_column(JSON)
    """[min, max] giây (lưu JSON)."""
    spec: Mapped[dict[str, Any]] = mapped_column(JSON)


class RecipeVariant(Base):
    __tablename__ = "recipe_variants"

    id: Mapped[int] = mapped_column(primary_key=True)
    recipe_id: Mapped[int] = mapped_column(ForeignKey("recipes.id", ondelete="CASCADE"), index=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    recipe_id: Mapped[int] = mapped_column(ForeignKey("recipes.id"), index=True)
    variant_id: Mapped[int | None] = mapped_column(ForeignKey("recipe_variants.id"))
    seed: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    """queued | running | qa | done | failed"""
    error: Mapped[str | None] = mapped_column(String)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Render(Base):
    __tablename__ = "renders"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    output_path: Mapped[str] = mapped_column(String)
    qa_score: Mapped[float | None] = mapped_column(Float)
    qa_notes: Mapped[str | None] = mapped_column(String)
    duration: Mapped[float | None] = mapped_column(Float)
    size_bytes: Mapped[int | None] = mapped_column(Integer)


class UsageLog(Base):
    __tablename__ = "usage_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    segment_id: Mapped[int] = mapped_column(ForeignKey("segments.id"), index=True)
    render_id: Mapped[int] = mapped_column(ForeignKey("renders.id"), index=True)
    used_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    render_id: Mapped[int] = mapped_column(ForeignKey("renders.id"), index=True)
    platform: Mapped[str] = mapped_column(String(32))
    views: Mapped[int | None] = mapped_column(Integer)
    avg_watch: Mapped[float | None] = mapped_column(Float)
    likes: Mapped[int | None] = mapped_column(Integer)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
