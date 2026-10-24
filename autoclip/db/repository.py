"""SqlClipRepository — hiện thực `ClipRepository` trên SQLAlchemy."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from sqlalchemy import Engine, exists, func, select
from sqlalchemy.orm import Session, sessionmaker

from autoclip.contracts import ClipInfo, SegmentFilter, SegmentInfo
from autoclip.db.engine import make_session_factory
from autoclip.db.models import Clip, Segment

log = logging.getLogger(__name__)


def _clip_info(row: Clip) -> ClipInfo:
    return ClipInfo(
        id=row.id,
        path=row.path,
        sha1=row.sha1,
        duration=row.duration,
        width=row.width,
        height=row.height,
        fps=row.fps,
        has_audio=row.has_audio,
        created_at=_as_utc(row.created_at),
        source_path=row.source_path,
    )


def _as_utc(dt: datetime | None) -> datetime | None:
    """SQLite không lưu múi giờ; mọi thời điểm trong DB đều là UTC."""
    return dt.replace(tzinfo=UTC) if dt is not None and dt.tzinfo is None else dt


def _segment_info(row: Segment) -> SegmentInfo:
    return SegmentInfo(
        id=row.id,
        clip_id=row.clip_id,
        start=row.start,
        end=row.end,
        transcript=row.transcript,
        tags=list(row.tags or []),
        hook_score=row.hook_score,
        phash=row.phash,
        thumb_path=row.thumb_path,
        embedding=row.embedding,
    )


class SqlClipRepository:
    def __init__(self, engine: Engine) -> None:
        self._sessions: sessionmaker[Session] = make_session_factory(engine)

    def get_clip_by_sha1(self, sha1: str) -> ClipInfo | None:
        with self._sessions() as s:
            row = s.scalar(select(Clip).where(Clip.sha1 == sha1))
            return _clip_info(row) if row else None

    def get_clip(self, clip_id: int) -> ClipInfo | None:
        with self._sessions() as s:
            row = s.get(Clip, clip_id)
            return _clip_info(row) if row else None

    def set_source_path(self, clip_id: int, source_path: str) -> None:
        with self._sessions.begin() as s:
            row = s.get(Clip, clip_id)
            if row is not None:
                row.source_path = source_path

    def add_clip(self, clip: ClipInfo) -> ClipInfo:
        with self._sessions.begin() as s:
            existing = s.scalar(select(Clip).where(Clip.sha1 == clip.sha1))
            if existing:
                log.debug("add_clip: sha1 %s đã có (id=%s)", clip.sha1, existing.id)
                return _clip_info(existing)
            row = Clip(**clip.model_dump(exclude={"id", "created_at"}, exclude_none=True))
            if clip.created_at:
                row.created_at = clip.created_at
            s.add(row)
            s.flush()
            return _clip_info(row)

    def add_segments(self, clip_id: int, segments: list[SegmentInfo]) -> list[SegmentInfo]:
        with self._sessions.begin() as s:
            rows = [
                Segment(clip_id=clip_id, **seg.model_dump(exclude={"id", "clip_id"}))
                for seg in segments
            ]
            s.add_all(rows)
            s.flush()
            return [_segment_info(r) for r in rows]

    def list_segments(self, filter: SegmentFilter | None = None) -> list[SegmentInfo]:
        f = filter or SegmentFilter()
        q = select(Segment)
        if f.clip_id is not None:
            q = q.where(Segment.clip_id == f.clip_id)
        if f.tags_any:
            tag = func.json_each(Segment.tags).table_valued("value")
            q = q.where(exists(select(1).select_from(tag).where(tag.c.value.in_(f.tags_any))))
        if f.min_hook_score is not None:
            q = q.where(Segment.hook_score >= f.min_hook_score)
        if f.min_duration is not None:
            q = q.where(Segment.end - Segment.start >= f.min_duration)
        if f.max_duration is not None:
            q = q.where(Segment.end - Segment.start <= f.max_duration)
        if f.exclude_ids:
            q = q.where(Segment.id.not_in(f.exclude_ids))
        q = q.order_by(Segment.id)
        if f.limit is not None:
            q = q.limit(f.limit)
        with self._sessions() as s:
            return [_segment_info(r) for r in s.scalars(q)]
