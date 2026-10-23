"""Spike P0: dựng video 1080x1920 từ vài clip bằng Concat headless và đo thời gian render.

project_create → media_import → clips_add_sequence → title_add → captions_add
→ project_save → export_start/export_wait → project_close.
Tên và tham số tool theo data/concat_tools.json (tools/list thật).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from autoclip.concat_client import ConcatError, SyncConcatClient
from autoclip.config import Settings, get_settings
from autoclip.ingest import probe

log = logging.getLogger(__name__)

WIDTH, HEIGHT, FPS = 1080, 1920, 30
EXPORT_POLL_SEC = 600.0


@dataclass
class SpikeResult:
    output: Path
    project: Path
    video_sec: float
    """Thời lượng file MP4 xuất ra (ffprobe)."""
    build_sec: float
    """Thời gian dựng timeline (import → save)."""
    render_sec: float
    """Thời gian export_start → xong."""

    @property
    def render_ratio(self) -> float:
        """Giây render cho mỗi giây video."""
        return self.render_sec / self.video_sec if self.video_sec else float("nan")


def _require(result: Any, key: str, tool: str) -> Any:
    if isinstance(result, dict) and key in result:
        return result[key]
    raise ConcatError(tool, f"kết quả thiếu '{key}': {result!r}")


def _export_done(status: Any) -> tuple[bool, str | None]:
    """(xong chưa, lỗi nếu có) từ kết quả export_wait/export_status."""
    if not isinstance(status, dict):
        return False, None
    state = str(status.get("state") or status.get("status") or "").lower()
    if state in {"done", "finished", "complete", "completed", "succeeded"}:
        return True, None
    if state in {"failed", "error", "cancelled", "canceled"}:
        return True, str(status.get("error") or status.get("message") or state)
    return False, None


def run_spike(
    clips: list[Path],
    *,
    title: str = "AutoClip spike",
    settings: Settings | None = None,
    name: str | None = None,
) -> SpikeResult:
    s = settings or get_settings()
    clips = [Path(c).resolve() for c in clips]
    missing = [c for c in clips if not c.is_file()]
    if missing:
        raise FileNotFoundError(f"không thấy clip: {', '.join(map(str, missing))}")
    durations = [probe(c).duration for c in clips]
    total = sum(durations)

    name = name or f"spike-{datetime.now():%Y%m%d-%H%M%S}"
    s.projects_dir.mkdir(parents=True, exist_ok=True)
    s.out_dir.mkdir(parents=True, exist_ok=True)
    project = s.projects_dir / name
    output = s.out_dir / f"{name}.mp4"

    with SyncConcatClient(s) as cc:
        t0 = time.perf_counter()
        cc.call_tool(
            "project_create",
            {
                "location": str(s.projects_dir),
                "name": name,
                "video": {"width": WIDTH, "height": HEIGHT, "rateNum": FPS, "rateDen": 1},
            },
        )
        path = str(project)
        try:
            media_ids = [
                str(
                    _require(
                        cc.call_tool("media_import", {"path": path, "file": str(c)}),
                        "createdId",
                        "media_import",
                    )
                )
                for c in clips
            ]
            cc.call_tool("clips_add_sequence", {"path": path, "mediaIds": media_ids, "start": 0})
            cc.call_tool(
                "title_add",
                {
                    "path": path,
                    "start": 0,
                    "duration": round(min(3.0, total), 3),
                    "style": {
                        "content": title,
                        "fontWeight": 800,
                        "fontSize": 0.06,
                        "color": "#FFD400",
                        "strokeWidth": 0.008,
                        "strokeColor": "#000000",
                    },
                    "offsetY": -0.3,
                    "above": True,
                },
            )
            captions, cursor = [], 0.0
            for i, (clip, dur) in enumerate(zip(clips, durations, strict=True), start=1):
                captions.append(
                    {
                        "start": round(cursor, 3),
                        "end": round(cursor + dur, 3),
                        "text": f"Cảnh {i}: {clip.stem}",
                    }
                )
                cursor += dur
            cc.call_tool("captions_add", {"path": path, "captions": captions, "offsetY": 0.35})
            cc.call_tool("project_save", {"path": path})
            build_sec = time.perf_counter() - t0

            t1 = time.perf_counter()
            job = str(
                _require(
                    cc.call_tool("export_start", {"path": path, "output": str(output)}),
                    "job",
                    "export_start",
                )
            )
            while True:
                status = cc.call_tool(
                    "export_wait",
                    {"job": job, "timeoutSeconds": EXPORT_POLL_SEC},
                    timeout=EXPORT_POLL_SEC + 60,
                )
                done, error = _export_done(status)
                if error:
                    raise ConcatError("export_wait", error)
                if done:
                    break
                log.info("export đang chạy: %s", status)
            render_sec = time.perf_counter() - t1
        finally:
            try:
                cc.call_tool("project_close", {"path": path, "save": False})
            except ConcatError as exc:
                log.warning("project_close: %s", exc)

    if not output.is_file():
        raise ConcatError("export_wait", f"export báo xong nhưng không thấy {output}")
    return SpikeResult(
        output=output,
        project=project,
        video_sec=probe(output).duration,
        build_sec=build_sec,
        render_sec=render_sec,
    )
