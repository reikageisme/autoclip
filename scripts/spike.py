"""Spike P0: nối các clip thành video 1080x1920 có title + captions, export MP4, in thời gian.

uv run python scripts/spike.py clip1.mp4 clip2.mp4 clip3.mp4
uv run python scripts/spike.py            # dùng data/testclips/*.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from autoclip.config import get_settings
from autoclip.spike import run_spike


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clips", nargs="*", type=Path)
    parser.add_argument("--title", default="AutoClip spike")
    parser.add_argument("-v", "--verbose", action="store_true", help="log DEBUG mọi lệnh Concat")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    clips = args.clips or sorted((get_settings().data_dir / "testclips").glob("*.mp4"))[:3]
    if not clips:
        print("Không có clip: truyền đường dẫn clip hoặc chạy run-autoclip.ps1 để sinh clip test")
        return 2
    r = run_spike(clips, title=args.title)
    print(f"MP4:     {r.output}")
    print(f"Video:   {r.video_sec:.2f}s từ {len(clips)} clip")
    print(f"Dựng:    {r.build_sec:.2f}s")
    print(f"Render:  {r.render_sec:.2f}s  ({r.render_ratio:.2f} giây render / giây video)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
