"""Đăng video lên Facebook Page qua Graph API, có hẹn giờ cục bộ.

Chỉ những dòng `Publication` do người dùng tạo bằng tay (nút Đăng/Hẹn giờ) mới được
gửi đi; service không tự đăng video đã duyệt. Lịch chạy trên máy này, vì vậy service
phải đang chạy khi tới giờ. Token Page lưu bằng DPAPI (xem `autoclip.secrets`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from autoclip.config import Settings
from autoclip.db.models import Publication, Render
from autoclip.secrets import SecretStore

log = logging.getLogger(__name__)

TOKEN_NAME = "facebook_page_token"
GRAPH_ROOT = "https://graph.facebook.com"
MIN_LEAD = timedelta(seconds=5)


class PublishError(RuntimeError):
    pass


@dataclass
class PublishResult:
    publication_id: int
    status: str
    post_id: str | None = None
    error: str | None = None


def facebook_token(settings: Settings) -> str | None:
    return SecretStore(settings.data_dir).get(TOKEN_NAME)


def save_facebook_token(settings: Settings, token: str) -> None:
    text = token.strip()
    if not text:
        raise PublishError("token Page đang trống")
    SecretStore(settings.data_dir).set(TOKEN_NAME, text)


def clear_facebook_token(settings: Settings) -> None:
    SecretStore(settings.data_dir).delete(TOKEN_NAME)


def _caption(render: Render) -> str:
    return (render.title or render.headline or "").strip()


def check_page(settings: Settings, token: str | None = None) -> str:
    """Gọi Graph API để kiểm tra Page id + token; trả tên Page."""
    token = token or facebook_token(settings)
    if not settings.fb_page_id.strip():
        raise PublishError("chưa nhập Page ID trong Cài đặt")
    if not token:
        raise PublishError("chưa nhập token Page trong Cài đặt")
    url = (
        f"{GRAPH_ROOT}/{settings.fb_graph_version}/"
        f"{settings.fb_page_id.strip()}?fields=name&access_token={token}"
    )
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url)
    except httpx.HTTPError as exc:
        raise PublishError(f"không gọi được Graph API: {exc}") from exc
    if resp.status_code >= 400:
        raise PublishError(_graph_error(resp))
    return str(resp.json().get("name") or settings.fb_page_id)


def _graph_error(resp: httpx.Response) -> str:
    try:
        error = resp.json().get("error") or {}
        message = error.get("message") or resp.text
        code = error.get("code")
        code_text = f" (code {code})" if code else ""
        return f"Graph API {resp.status_code}{code_text}: {message}"
    except ValueError:
        return f"Graph API {resp.status_code}: {resp.text[:300]}"


def publish_video(
    render: Render,
    settings: Settings,
    *,
    token: str | None = None,
    caption: str | None = None,
) -> str:
    """Upload MP4 lên Page, trả id bài viết. Hàm đồng bộ (gọi trong scheduler/worker)."""
    token = token or facebook_token(settings)
    page_id = settings.fb_page_id.strip()
    if not page_id:
        raise PublishError("chưa nhập Page ID trong Cài đặt")
    if not token:
        raise PublishError("chưa nhập token Page trong Cài đặt")
    path = Path(render.output_path)
    if not path.is_file():
        raise PublishError(f"không còn file video {path}")
    data = {
        "access_token": token,
        "description": (caption if caption is not None else _caption(render))[:5000],
        "published": "true",
    }
    url = f"{GRAPH_ROOT}/{settings.fb_graph_version}/{page_id}/videos"
    try:
        with path.open("rb") as fh, httpx.Client(
            timeout=httpx.Timeout(1200.0, connect=30.0)
        ) as client:
            resp = client.post(
                url,
                data=data,
                files={"source": (path.name, fh, "video/mp4")},
            )
    except httpx.HTTPError as exc:
        raise PublishError(f"upload lên Facebook lỗi: {exc}") from exc
    except OSError as exc:
        raise PublishError(f"không đọc được video: {exc}") from exc
    if resp.status_code >= 400:
        raise PublishError(_graph_error(resp))
    post_id = str(resp.json().get("id") or "")
    if not post_id:
        raise PublishError(f"Graph API không trả id bài viết: {resp.text[:300]}")
    log.info("đã đăng render %s lên Page %s: %s", render.id, page_id, post_id)
    return post_id


def schedule_publication(
    engine,
    render_id: int,
    *,
    settings: Settings,
    at: datetime | None = None,
    caption: str | None = None,
) -> Publication:
    """Tạo lịch đăng; `at` None = ngay (scheduler sẽ gửi ở lần kiểm tra kế tiếp)."""
    if not settings.fb_page_id.strip():
        raise PublishError("chưa nhập Page ID trong Cài đặt")
    if not facebook_token(settings):
        raise PublishError("chưa nhập token Page trong Cài đặt")
    when = at or datetime.now(UTC)
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    when = max(when, datetime.now(UTC) + MIN_LEAD)
    with Session(engine, expire_on_commit=False) as db:
        render = db.get(Render, render_id)
        if render is None:
            raise PublishError(f"không có video #{render_id}")
        if not Path(render.output_path).is_file():
            raise PublishError("file video không còn trên đĩa")
        row = Publication(
            render_id=render.id,
            platform="facebook",
            page_id=settings.fb_page_id.strip(),
            caption=(caption if caption is not None else _caption(render))[:5000] or None,
            scheduled_at=when,
            status="scheduled",
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row


def cancel_publication(engine, publication_id: int) -> bool:
    with Session(engine) as db:
        row = db.get(Publication, publication_id)
        if row is None or row.status != "scheduled":
            return False
        row.status = "cancelled"
        db.commit()
        return True


def retry_publication(engine, publication_id: int) -> bool:
    with Session(engine) as db:
        row = db.get(Publication, publication_id)
        if row is None or row.status not in {"failed", "cancelled"}:
            return False
        row.status = "scheduled"
        row.scheduled_at = datetime.now(UTC) + MIN_LEAD
        row.error = None
        db.commit()
        return True


def list_publications(engine, limit: int = 100) -> list[dict]:
    with Session(engine) as db:
        rows = db.scalars(
            select(Publication).order_by(Publication.id.desc()).limit(max(1, min(limit, 500)))
        ).all()
        return [
            {
                "id": r.id,
                "render_id": r.render_id,
                "page_id": r.page_id,
                "caption": r.caption,
                "scheduled_at": r.scheduled_at.isoformat() if r.scheduled_at else None,
                "status": r.status,
                "fb_post_id": r.fb_post_id,
                "error": r.error,
                "attempts": r.attempts,
                "published_at": r.published_at.isoformat() if r.published_at else None,
            }
            for r in rows
        ]


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.max.replace(tzinfo=UTC)
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def publish_due(
    engine,
    settings: Settings,
    *,
    now: datetime | None = None,
    limit: int = 5,
    publisher: Callable[..., str] = publish_video,
) -> list[PublishResult]:
    """Gửi các lịch đã tới giờ. Mỗi dòng được claim (scheduled → publishing) trước khi gọi."""
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    claimed: list[int] = []
    with Session(engine, expire_on_commit=False) as db:
        pending = db.scalars(
            select(Publication)
            .where(Publication.status == "scheduled")
            .order_by(Publication.scheduled_at)
        ).all()
        rows = [row for row in pending if _utc(row.scheduled_at) <= now][:limit]
        for row in rows:
            row.status = "publishing"
            row.attempts += 1
            claimed.append(row.id)
        db.commit()

    out: list[PublishResult] = []
    for pid in claimed:
        with Session(engine, expire_on_commit=False) as db:
            row = db.get(Publication, pid)
            if row is None:
                continue
            render = db.get(Render, row.render_id)
            try:
                if render is None:
                    raise PublishError("không còn dòng render")
                token = facebook_token(settings)
                post_id = publisher(render, settings, token=token, caption=row.caption)
                row.status = "published"
                row.fb_post_id = post_id
                row.published_at = datetime.now(UTC)
                row.error = None
            except Exception as exc:
                log.warning("đăng publication %s lỗi: %s", pid, exc)
                row.status = "failed"
                row.error = f"{type(exc).__name__}: {exc}"
            db.commit()
            out.append(
                PublishResult(
                    publication_id=pid,
                    status=row.status,
                    post_id=row.fb_post_id,
                    error=row.error,
                )
            )
    return out
