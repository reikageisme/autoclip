"""`python -m autoclip.concat_client`: chạy concat headless, lưu tools/list ra concat_tools.json."""

from __future__ import annotations

import logging

from autoclip.concat_client import SyncConcatClient
from autoclip.config import get_settings


def main() -> None:
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    with SyncConcatClient() as cc:
        version = cc.call_tool("concat_version") if "concat_version" in cc.tools else None
        print(f"{len(cc.tools)} tool → {get_settings().concat_tools_path}")
        if version is not None:
            print(f"concat_version: {version}")


if __name__ == "__main__":
    main()
