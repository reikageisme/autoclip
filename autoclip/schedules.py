"""Lịch chạy batch: CRUD nhẹ trên bảng `batch_schedules` + tính lần kế tiếp."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from autoclip.db.models import BatchSchedule


def utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def create_schedule(
    engine,
    *,
    recipe: str,
    count: int,
    run_at: datetime,
    repeat_minutes: int = 0,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> BatchSchedule:
    with Session(engine, expire_on_commit=False) as db:
        row = BatchSchedule(
            recipe=recipe,
            count=count,
            seed=seed,
            options=options or {},
            run_at=utc(run_at),
            repeat_minutes=max(0, repeat_minutes),
            enabled=True,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row


def list_schedules(engine) -> list[BatchSchedule]:
    with Session(engine, expire_on_commit=False) as db:
        return list(db.scalars(select(BatchSchedule).order_by(BatchSchedule.run_at)))


def due_schedules(engine, now: datetime | None = None) -> list[BatchSchedule]:
    now = utc(now or datetime.now(UTC))
    with Session(engine, expire_on_commit=False) as db:
        rows = db.scalars(
            select(BatchSchedule)
            .where(BatchSchedule.enabled.is_(True), BatchSchedule.run_at <= now)
            .order_by(BatchSchedule.run_at)
        ).all()
        return list(rows)


def mark_run(engine, schedule_id: int, now: datetime | None = None) -> None:
    now = utc(now or datetime.now(UTC))
    with Session(engine) as db:
        row = db.get(BatchSchedule, schedule_id)
        if row is None:
            return
        row.last_run_at = now
        if row.repeat_minutes > 0:
            row.run_at = now + timedelta(minutes=row.repeat_minutes)
        else:
            row.enabled = False
        db.commit()


def defer(engine, schedule_id: int, minutes: int = 5) -> None:
    with Session(engine) as db:
        row = db.get(BatchSchedule, schedule_id)
        if row is None:
            return
        row.run_at = utc(datetime.now(UTC)) + timedelta(minutes=max(1, minutes))
        db.commit()


def set_enabled(engine, schedule_id: int, enabled: bool) -> bool:
    with Session(engine) as db:
        row = db.get(BatchSchedule, schedule_id)
        if row is None:
            return False
        row.enabled = enabled
        if enabled and row.run_at <= utc(datetime.now(UTC)):
            row.run_at = utc(datetime.now(UTC)) + timedelta(seconds=5)
        db.commit()
        return True


def delete_schedule(engine, schedule_id: int) -> bool:
    with Session(engine) as db:
        row = db.get(BatchSchedule, schedule_id)
        if row is None:
            return False
        db.delete(row)
        db.commit()
        return True


def serialize(row: BatchSchedule) -> dict[str, Any]:
    return {
        "id": row.id,
        "recipe": row.recipe,
        "count": row.count,
        "seed": row.seed,
        "options": row.options or {},
        "run_at": utc(row.run_at).isoformat(),
        "repeat_minutes": row.repeat_minutes,
        "enabled": row.enabled,
        "last_run_at": utc(row.last_run_at).isoformat() if row.last_run_at else None,
        "created_at": utc(row.created_at).isoformat() if row.created_at else None,
    }
