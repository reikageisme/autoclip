"""Tải video nguồn của khách (page Facebook của khách, hoặc link khác yt-dlp hỗ trợ).

Mỗi video lưu thành `<dest>/<id>.mp4` kèm `<id>.info.json` (link gốc, tiêu đề, ngày đăng)
để truy nguồn. `<dest>/.downloaded.txt` là archive của yt-dlp: chạy lại không tải trùng.
Chỉ dùng cho nội dung khách có quyền (xem memory/CLAUDE.md); không né chặn hay vượt đăng nhập.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ARCHIVE_NAME = ".downloaded.txt"


@dataclass
class FetchResult:
    dest: Path
    downloaded: list[Path] = field(default_factory=list)
    skipped: int = 0
    """Số video đã có trong archive từ các lần trước (yt-dlp không tải lại)."""
    errors: list[str] = field(default_factory=list)


def expand_urls(items: list[str]) -> list[str]:
    """Link giữ nguyên; đường dẫn file .txt → các dòng link trong đó (bỏ dòng trống, #)."""
    urls: list[str] = []
    for item in items:
        path = Path(item)
        if not item.startswith(("http://", "https://")) and path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
        else:
            urls.append(item)
    return list(dict.fromkeys(urls))


def source_name(url: str) -> str:
    """Tên thư mục con cho một nguồn: tên page Facebook, hoặc domain."""
    m = re.match(r"https?://(?:[\w-]+\.)?(?:facebook|fb)\.com/([^/?#]+)", url)
    name = m.group(1) if m and m.group(1) not in {"watch", "reel", "share", "videos"} else ""
    if not name:
        host = re.match(r"https?://([^/?#]+)", url)
        name = host.group(1) if host else "misc"
    return re.sub(r"[^\w.-]+", "_", name).strip("._") or "misc"


def build_options(
    dest: Path,
    *,
    limit: int | None = None,
    cookies_from_browser: str | None = None,
    cookies_file: Path | None = None,
    sleep_sec: float = 3.0,
) -> dict[str, Any]:
    """Tuỳ chọn yt-dlp: mp4 tốt nhất, info.json, archive chống tải trùng, nghỉ giữa các video."""
    opts: dict[str, Any] = {
        "outtmpl": str(dest / "%(id)s.%(ext)s"),
        "format": "bv*[ext=mp4]+ba[ext=m4a]/bv*+ba/b",
        "merge_output_format": "mp4",
        "writeinfojson": True,
        "download_archive": str(dest / ARCHIVE_NAME),
        "ignoreerrors": True,
        "noprogress": True,
        "quiet": True,
        "no_warnings": False,
        "sleep_interval": sleep_sec,
        "max_sleep_interval": sleep_sec * 2,
        "restrictfilenames": True,
    }
    if limit:
        opts["playlistend"] = limit
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if cookies_file:
        opts["cookiefile"] = str(cookies_file)
    return opts


def fetch(
    urls: list[str],
    dest_root: Path,
    *,
    limit: int | None = None,
    cookies_from_browser: str | None = None,
    cookies_file: Path | None = None,
    sleep_sec: float = 3.0,
    name: str | None = None,
    ydl_factory: Any = None,
) -> FetchResult:
    """Tải các link (video lẻ hoặc trang danh sách video) về `dest_root/<nguồn>/`."""
    if ydl_factory is None:
        from yt_dlp import YoutubeDL

        ydl_factory = YoutubeDL

    folder = name or (source_name(urls[0]) if urls else "")
    dest = dest_root / folder if folder else dest_root
    dest.mkdir(parents=True, exist_ok=True)
    result = FetchResult(dest=dest)
    archive = dest / ARCHIVE_NAME
    before = set(archive.read_text(encoding="utf-8").splitlines()) if archive.exists() else set()

    opts = build_options(
        dest,
        limit=limit,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
        sleep_sec=sleep_sec,
    )

    def on_progress(d: dict[str, Any]) -> None:
        if d.get("status") == "finished" and d.get("filename"):
            log.info("tải xong %s", d["filename"])

    opts["progress_hooks"] = [on_progress]

    with ydl_factory(opts) as ydl:
        for url in urls:
            log.info("fetch %s → %s", url, dest)
            try:
                code = ydl.download([url])
            except Exception as exc:  # yt-dlp ném nhiều loại lỗi khác nhau
                result.errors.append(f"{url}: {exc}")
                continue
            if code:
                result.errors.append(f"{url}: yt-dlp trả mã {code} (xem log)")

    after = set(archive.read_text(encoding="utf-8").splitlines()) if archive.exists() else set()
    for line in sorted(after - before):
        video_id = line.split()[-1]
        files = sorted(dest.glob(f"{video_id}.mp4"))
        result.downloaded.extend(files)
    result.skipped = len(before)
    return result
