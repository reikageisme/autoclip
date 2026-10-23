"""Nhập CSV lượt xem và tính hiệu quả theo kiểu video.

CSV chấp nhận các cột linh hoạt (tiếng Việt hoặc tiếng Anh):
`render_id`/`id`/`video`/`file`, `views`/`luot_xem`, `avg_watch`, `likes`,
`platform`, `captured_at`. Số kiểu `1.234`, `12,345`, `1.2K`, `3M` đều đọc được.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from autoclip.db.models import Job, Metric, Recipe, Render


class MetricsError(ValueError):
    pass


@dataclass
class MetricImportReport:
    imported: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    recipes: dict[str, dict[str, float | int]] = field(default_factory=dict)


def _fold(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _pick(row: dict[str, str], *names: str) -> str:
    folded = {_fold(k): (v or "").strip() for k, v in row.items()}
    for name in names:
        value = folded.get(_fold(name))
        if value:
            return value
    return ""


_NUMBER = re.compile(r"^([0-9][0-9., ]*)\s*([kmb])?$", re.I)


def _number(text: str) -> float | None:
    raw = (text or "").strip()
    if not raw:
        return None
    match = _NUMBER.match(raw)
    if not match:
        raise ValueError(f"số không hợp lệ: {raw!r}")
    digits, suffix = match.groups()
    digits = digits.replace(" ", "")
    if "," in digits and "." in digits:
        if digits.rfind(",") > digits.rfind("."):
            digits = digits.replace(".", "").replace(",", ".")
        else:
            digits = digits.replace(",", "")
    elif "," in digits:
        digits = digits.replace(",", "." if len(digits.rsplit(",", 1)[1]) <= 2 else "")
    elif digits.count(".") > 1:
        digits = digits.replace(".", "")
    value = float(digits)
    return value * {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(
        (suffix or "").lower(), 1
    )


def _int(text: str) -> int | None:
    raw = (text or "").strip()
    if re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]{3})+", raw):
        raw = raw.replace(".", "")
    value = _number(raw)
    return None if value is None else int(round(value))


def _date(text: str) -> datetime:
    raw = (text or "").strip()
    if not raw:
        return datetime.now(UTC)
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"ngày không hợp lệ: {raw!r}") from exc


def _find_render(db: Session, raw: str) -> Render | None:
    raw = raw.strip()
    if not raw:
        return None
    if raw.isdigit():
        row = db.get(Render, int(raw))
        if row is not None:
            return row
    target = Path(raw).name.lower()
    stem = Path(raw).stem.lower()
    rows = db.scalars(select(Render)).all()
    for row in rows:
        name = Path(row.output_path).name.lower()
        if name == target or Path(row.output_path).stem.lower() == stem:
            return row
    return None


def _latest_per_render(db: Session) -> dict[int, Metric]:
    latest: dict[int, Metric] = {}
    for metric in db.scalars(select(Metric).order_by(Metric.id)):
        current = latest.get(metric.render_id)
        key = ((metric.captured_at.timestamp() if metric.captured_at else 0.0), metric.id)
        if current is None or key >= (
            (current.captured_at.timestamp() if current.captured_at else 0.0),
            current.id,
        ):
            latest[metric.render_id] = metric
    return latest


def recipe_scores(engine) -> dict[str, dict[str, float | int]]:
    """Trung bình views của lần đo mới nhất cho mỗi video, gom theo tên kiểu video."""
    with Session(engine) as db:
        latest = _latest_per_render(db)
        if not latest:
            return {}
        rows = db.execute(
            select(Render.id, Recipe.name).join(Job, Job.id == Render.job_id).join(
                Recipe, Recipe.id == Job.recipe_id
            )
        ).all()
    by_render = {rid: name for rid, name in rows}
    grouped: dict[str, list[float]] = {}
    for render_id, metric in latest.items():
        name = by_render.get(render_id)
        if name is None or metric.views is None:
            continue
        grouped.setdefault(name, []).append(float(metric.views))
    out: dict[str, dict[str, float | int]] = {}
    for name, views in grouped.items():
        out[name] = {
            "avg_views": round(sum(views) / len(views), 1),
            "total_views": int(sum(views)),
            "samples": len(views),
        }
    return out


def best_recipe(engine, names: list[str]) -> str | None:
    """Tên kiểu có views trung bình cao nhất trong số các kiểu đang có."""
    scores = recipe_scores(engine)
    ranked = [
        (float(scores[n]["avg_views"]), int(scores[n]["samples"]), n)
        for n in names
        if n in scores
    ]
    return max(ranked)[2] if ranked else None


def import_metrics(
    engine,
    text: str,
    *,
    platform: str = "facebook",
    default_render_id: int | None = None,
) -> MetricImportReport:
    report = MetricImportReport()
    clean = text.lstrip("\ufeff")
    if not clean.strip():
        raise MetricsError("CSV đang trống")
    try:
        dialect = csv.Sniffer().sniff(clean[:4096], delimiters=",;\t")
        reader = csv.DictReader(io.StringIO(clean), dialect=dialect)
    except csv.Error:
        reader = csv.DictReader(io.StringIO(clean))
    if not reader.fieldnames:
        raise MetricsError("CSV không có dòng tiêu đề")
    with Session(engine, expire_on_commit=False) as db:
        for index, row in enumerate(reader, start=2):
            if not any((v or "").strip() for v in row.values()):
                continue
            try:
                ref = _pick(row, "render_id", "id", "video", "file", "filename", "ten_file")
                render = _find_render(db, ref) if ref else None
                if render is None and default_render_id is not None:
                    render = db.get(Render, default_render_id)
                if render is None:
                    raise ValueError(f"không tìm thấy video {ref!r}")
                views = _int(_pick(row, "views", "view", "luot_xem", "luot xem"))
                avg_watch = _number(
                    _pick(row, "avg_watch", "avg_watch_sec", "watch", "thoi_luong_xem")
                )
                likes = _int(_pick(row, "likes", "like", "cam_xuc"))
                if views is None and avg_watch is None and likes is None:
                    raise ValueError("dòng không có views/avg_watch/likes")
                db.add(
                    Metric(
                        render_id=render.id,
                        platform=_pick(row, "platform", "kenh") or platform,
                        views=views,
                        avg_watch=avg_watch,
                        likes=likes,
                        captured_at=_date(_pick(row, "captured_at", "date", "ngay", "thoi_gian")),
                    )
                )
                report.imported += 1
            except (ValueError, TypeError) as exc:
                report.skipped += 1
                if len(report.errors) < 50:
                    report.errors.append(f"dòng {index}: {exc}")
        db.commit()
    if report.imported == 0 and report.skipped:
        raise MetricsError("; ".join(report.errors) or "không nhập được dòng nào")
    report.recipes = recipe_scores(engine)
    return report


def list_metrics(engine, limit: int = 100) -> list[dict[str, Any]]:
    with Session(engine) as db:
        rows = db.execute(
            select(Metric, Render, Recipe.name)
            .join(Render, Render.id == Metric.render_id)
            .join(Job, Job.id == Render.job_id)
            .join(Recipe, Recipe.id == Job.recipe_id)
            .order_by(Metric.id.desc())
            .limit(max(1, min(limit, 1000)))
        ).all()
    return [
        {
            "id": m.id,
            "render_id": r.id,
            "video": Path(r.output_path).name,
            "recipe": name,
            "platform": m.platform,
            "views": m.views,
            "avg_watch": m.avg_watch,
            "likes": m.likes,
            "captured_at": m.captured_at.isoformat() if m.captured_at else None,
        }
        for m, r, name in rows
    ]
