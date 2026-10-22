"""Chạy ingest thật 2 lần trên một thư mục clip; lần 2 phải không thêm bản ghi mới.

    uv run python scripts/smoke_ingest.py <folder> [--whisper]
"""

from __future__ import annotations

import argparse
import logging
import sys

from autoclip.config import get_settings
from autoclip.db import SqlClipRepository, make_engine
from autoclip.db.migrate import upgrade
from autoclip.ingest import IngestOptions, ingest_folder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder")
    parser.add_argument("--whisper", action="store_true", help="bật Whisper (tải model lần đầu)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    engine = make_engine()
    upgrade(engine)
    repo = SqlClipRepository(engine)
    options = IngestOptions.from_settings(settings)
    options.whisper_enabled = args.whisper

    reports = []
    for run in (1, 2):
        report = ingest_folder(args.folder, repo, options)
        reports.append(report)
        print(
            f"lần {run}: thấy {report.clips_seen} clip, thêm {report.clips_added}, "
            f"bỏ qua {report.clips_skipped}, {report.segments_added} segment "
            f"(loại trùng {report.segments_deduped}), {len(report.errors)} lỗi, "
            f"{report.elapsed_sec:.1f}s"
        )
        for err in report.errors:
            print(f"  lỗi: {err.path}: {err.message}")

    print(f"DB: {settings.db_path} — tổng {len(repo.list_segments())} segment")
    second = reports[1]
    if second.clips_added or second.segments_added:
        print("THẤT BẠI: lần 2 vẫn thêm bản ghi mới (không idempotent)")
        return 1
    if reports[0].errors:
        return 1
    print("OK: lần 2 không tạo bản ghi trùng")
    return 0


if __name__ == "__main__":
    sys.exit(main())
