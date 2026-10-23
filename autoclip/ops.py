"""Vận hành: theo dõi dung lượng ổ đĩa và dọn dữ liệu tạm/cũ.

Mọi đường dẫn bị xoá đều nằm dưới `DATA_DIR` (proxies, projects, out, thumbs)
và phải cũ hơn số ngày cấu hình. `0` = tắt dọn loại đó.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from autoclip.config import Settings

log = logging.getLogger(__name__)


@dataclass
class DiskInfo:
    path: str
    total_gb: float
    free_gb: float
    low: bool
    error: str | None = None


@dataclass
class CleanupReport:
    deleted: int = 0
    bytes_freed: int = 0
    errors: list[str] = field(default_factory=list)
    details: dict[str, int] = field(default_factory=dict)

    @property
    def mb_freed(self) -> float:
        return round(self.bytes_freed / (1024 * 1024), 1)


def _drive_anchor(path: Path) -> str:
    anchor = path.resolve().anchor
    return anchor or str(path)


def watch_paths(settings: Settings) -> list[str]:
    raw = [p.strip() for p in settings.disk_watch_paths.split(os.pathsep) if p.strip()]
    if raw:
        return list(dict.fromkeys(raw))
    paths = [_drive_anchor(Path(settings.data_dir))]
    if os.name == "nt":
        paths.append("C:\\")
    return list(dict.fromkeys(paths))


def disk_status(settings: Settings) -> list[DiskInfo]:
    out: list[DiskInfo] = []
    for raw in watch_paths(settings):
        try:
            usage = shutil.disk_usage(raw)
            free = usage.free / (1024**3)
            out.append(
                DiskInfo(
                    path=raw,
                    total_gb=round(usage.total / (1024**3), 1),
                    free_gb=round(free, 1),
                    low=free < settings.disk_warn_free_gb,
                )
            )
        except OSError as exc:
            out.append(DiskInfo(path=raw, total_gb=0, free_gb=0, low=False, error=str(exc)))
    return out


def _old_enough(path: Path, cutoff: float) -> bool:
    try:
        return path.stat().st_mtime < cutoff
    except OSError:
        return False


def _remove(path: Path) -> tuple[int, int, str | None]:
    try:
        if path.is_dir():
            size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            shutil.rmtree(path)
        else:
            size = path.stat().st_size
            path.unlink()
        return 1, size, None
    except OSError as exc:
        return 0, 0, f"{path}: {exc}"


def _key(path: Path | str) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def protected_paths(engine) -> set[str]:
    """File còn dùng: proxy của clip trong kho, video đã duyệt hoặc đang chờ đăng."""
    from sqlalchemy import or_, select
    from sqlalchemy.orm import Session

    from autoclip.db.models import Clip, Publication, Render

    keep: set[str] = set()
    with Session(engine) as db:
        keep.update(_key(p) for p in db.scalars(select(Clip.path)))
        waiting = select(Publication.render_id).where(
            Publication.status.in_(("scheduled", "publishing"))
        )
        rows = db.execute(
            select(Render.output_path, Render.project_path).where(
                or_(Render.review == "approved", Render.id.in_(waiting))
            )
        )
        for out, project in rows:
            keep.add(_key(out))
            if project:
                keep.add(_key(project))
    return keep


def _clean_folder(
    root: Path,
    days: int,
    *,
    now: datetime,
    dirs_only: bool = False,
    suffix: tuple[str, ...] | None = None,
    keep: set[str] | None = None,
    dry_run: bool = False,
) -> tuple[int, int, list[str]]:
    if days <= 0 or not root.is_dir():
        return 0, 0, []
    cutoff = (now - timedelta(days=days)).timestamp()
    deleted = size = 0
    errors: list[str] = []
    for path in root.iterdir():
        if dirs_only and not path.is_dir():
            continue
        if suffix and path.suffix.lower() not in suffix:
            continue
        if not _old_enough(path, cutoff):
            continue
        if keep and _key(path) in keep:
            continue
        if dry_run:
            deleted += 1
            continue
        n, b, error = _remove(path)
        deleted += n
        size += b
        if error:
            errors.append(error)
    return deleted, size, errors


def cleanup(
    settings: Settings,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
    proxies_days: int | None = None,
    projects_days: int | None = None,
    videos_days: int | None = None,
    keep: set[str] | None = None,
) -> CleanupReport:
    """Xoá proxy/project/video cũ dưới DATA_DIR. `dry_run` chỉ đếm.

    `keep` (xem `protected_paths`) là các đường dẫn không bao giờ xoá.
    """
    now = now or datetime.now(UTC)
    report = CleanupReport()
    data = Path(settings.data_dir).resolve()
    jobs = [
        (
            "proxies",
            data / "proxies",
            settings.cleanup_proxies_days if proxies_days is None else proxies_days,
            False,
            None,
        ),
        (
            "projects",
            data / "projects",
            settings.cleanup_projects_days if projects_days is None else projects_days,
            True,
            None,
        ),
        (
            "videos",
            data / "out",
            settings.cleanup_videos_days if videos_days is None else videos_days,
            False,
            (".mp4", ".mov", ".mkv", ".webm"),
        ),
    ]
    for name, root, days, dirs_only, suffix in jobs:
        if days <= 0:
            report.details[name] = 0
            continue
        root = root.resolve()
        if data not in root.parents and root != data:
            report.errors.append(f"bỏ qua đường dẫn ngoài DATA_DIR: {root}")
            continue
        deleted, size, errors = _clean_folder(
            root, days, now=now, dirs_only=dirs_only, suffix=suffix, keep=keep, dry_run=dry_run
        )
        report.details[name] = deleted
        report.deleted += deleted
        if dry_run:
            continue
        report.bytes_freed += size
        report.errors.extend(errors)
        log.info("dọn %s: %d mục, %.1f MB", name, deleted, size / (1024 * 1024))
    return report
