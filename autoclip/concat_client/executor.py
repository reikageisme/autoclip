"""Chạy danh sách `ConcatCommand` của compiler, thay tham chiếu bằng id thật của Concat.

Quy ước (xem docstring của autoclip.recipes.compiler):
- Giá trị ``"@media:<file>"`` → ``createdId`` của lệnh ``media_import`` file đó.
- ``"@track:<tên>"`` / ``"@clip:<tên>"`` → ``createdId`` của lệnh có ``"@as"`` tương ứng.
- ``"@timeline"`` → id timeline đang mở (đọc bằng ``timeline_read`` khi cần).
- Giá trị ``"@clip:<i>"`` → ``createdId`` của lệnh có ``"@as": "clip:<i>"``.
- Khoá bắt đầu bằng ``@`` là chỉ thị, không gửi tới Concat.
- Tool có tham số ``path`` mà lệnh chưa có → chèn thư mục project.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from autoclip.concat_client.client import ConcatError
from autoclip.contracts import ConcatCommand

log = logging.getLogger(__name__)


class _Client(Protocol):
    tools: dict[str, Any]

    def call_tool(self, name: str, args: dict[str, Any] | None = None) -> Any: ...


def _resolve(value: Any, refs: dict[str, str], tool: str) -> Any:
    if isinstance(value, str) and (
        value.startswith(("@media:", "@clip:", "@track:")) or value == "@timeline"
    ):
        key = value[1:]
        if key not in refs:
            raise ConcatError(tool, f"tham chiếu chưa có id: {value}")
        return refs[key]
    if isinstance(value, list):
        return [_resolve(v, refs, tool) for v in value]
    if isinstance(value, dict):
        return {k: _resolve(v, refs, tool) for k, v in value.items()}
    return value


def _takes_path(cc: _Client, tool: str) -> bool:
    schema = getattr(cc.tools.get(tool), "input_schema", None) or {}
    return "path" in schema.get("properties", {})


def execute(cc: _Client, commands: list[ConcatCommand], project_path: str) -> dict[str, str]:
    """Chạy tuần tự; trả về bảng tham chiếu → id thật (media:<file>, clip:<i>)."""
    refs: dict[str, str] = {}
    for n, command in enumerate(commands):
        alias = command.args.get("@as")
        args = {k: v for k, v in command.args.items() if not k.startswith("@")}
        if "timeline" not in refs and '"@timeline"' in json.dumps(args):
            read = cc.call_tool("timeline_read", {"path": project_path})
            refs["timeline"] = str(read["timeline"]["id"])
        args = _resolve(args, refs, command.tool)
        if "path" not in args and _takes_path(cc, command.tool):
            args["path"] = project_path
        result = cc.call_tool(command.tool, args)
        created = result.get("createdId") if isinstance(result, dict) else None
        if command.tool == "media_import":
            media_key = f"media:{command.args['file']}"
            if created is None and media_key not in refs:
                raise ConcatError("media_import", f"không có createdId cho {command.args['file']}")
            if created is not None:
                refs[media_key] = str(created)
        if alias:
            if created is None:
                raise ConcatError(command.tool, f"lệnh #{n} cần createdId cho {alias}: {result!r}")
            refs[str(alias)] = str(created)
    return refs


EXPORT_POLL_SEC = 600.0


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


def export_mp4(cc: Any, project_path: str, output: str) -> None:
    """export_start rồi export_wait tới khi xong; lỗi export → ConcatError."""
    started = cc.call_tool("export_start", {"path": project_path, "output": output})
    job = started.get("job") if isinstance(started, dict) else None
    if not job:
        raise ConcatError("export_start", f"kết quả thiếu 'job': {started!r}")
    while True:
        status = cc.call_tool(
            "export_wait",
            {"job": str(job), "timeoutSeconds": EXPORT_POLL_SEC},
            timeout=EXPORT_POLL_SEC + 60,
        )
        done, error = _export_done(status)
        if error:
            raise ConcatError("export_wait", error)
        if done:
            return
        log.info("export đang chạy: %s", status)
