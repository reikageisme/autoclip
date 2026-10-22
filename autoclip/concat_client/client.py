"""Client MCP (stdio) tới `concat --mcp headless`.

Chỉ giao tiếp qua MCP — không link tới crate nào của Concat
(xem RANH GIỚI GIẤY PHÉP trong CLAUDE.md).

Dùng async:

    async with ConcatClient() as cc:
        await cc.call_tool("project_create", {...})

hoặc sync (script, CLI, worker):

    with SyncConcatClient() as cc:
        cc.call_tool("project_create", {...})
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from anyio.from_thread import BlockingPortal, start_blocking_portal
from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client

from autoclip.config import Settings, get_settings
from autoclip.contracts import ConcatCommand

log = logging.getLogger(__name__)


class ConcatError(RuntimeError):
    """Concat trả về lỗi cho một lần gọi tool (`isError`)."""

    def __init__(self, tool: str, message: str) -> None:
        super().__init__(f"{tool}: {message}")
        self.tool = tool
        self.message = message


def _text_of(result: types.CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, types.TextContent))


def _decode(result: types.CallToolResult) -> Any:
    """Lấy giá trị trả về của tool: structured content nếu có, không thì JSON/text trong content."""
    if result.structured_content is not None:
        return result.structured_content
    text = _text_of(result)
    try:
        return json.loads(text)
    except ValueError:
        return text


class ConcatClient:
    """Một tiến trình concat headless. Mỗi job nên dùng một client + một thư mục project riêng
    (Concat khoá thư mục project đang mở bằng `concat.lock`)."""

    def __init__(self, settings: Settings | None = None, *, dump_tools: bool = True) -> None:
        self._settings = settings or get_settings()
        self._dump_tools = dump_tools
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self.tools: dict[str, types.Tool] = {}

    # ------------------------------------------------------------------ lifecycle

    def _server_params(self) -> StdioServerParameters:
        s = self._settings
        env = dict(os.environ)
        roots = [r for r in s.concat_mcp_roots.split(os.pathsep) if r]
        if s.data_dir.as_posix() not in {Path(r).as_posix() for r in roots}:
            roots.append(str(s.data_dir))
        env["CONCAT_MCP_ROOTS"] = os.pathsep.join(roots)
        return StdioServerParameters(
            command=str(s.concat_bin), args=["--mcp", s.concat_mcp_mode], env=env
        )

    async def __aenter__(self) -> Self:
        params = self._server_params()
        if not Path(params.command).is_file():
            raise FileNotFoundError(f"Không thấy CONCAT_BIN: {params.command}")
        log.debug("spawn %s %s", params.command, " ".join(params.args))
        self._stack = AsyncExitStack()
        try:
            read, write = await self._stack.enter_async_context(stdio_client(params))
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
            await self.refresh_tools()
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = self._session = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("ConcatClient chưa được mở (dùng `async with`)")
        return self._session

    # ------------------------------------------------------------------ tools

    async def refresh_tools(self) -> list[types.Tool]:
        tools: list[types.Tool] = []
        cursor: str | None = None
        while True:
            params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
            page = await self.session.list_tools(params=params)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if not cursor:
                break
        self.tools = {t.name: t for t in tools}
        log.debug("tools/list: %d tool", len(tools))
        if self._dump_tools:
            self.save_tools(self._settings.concat_tools_path)
        return tools

    async def list_tools(self) -> list[dict[str, Any]]:
        """Schema các tool (dạng JSON của MCP, camelCase)."""
        return [
            t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in self.tools.values()
        ]

    def save_tools(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            t.model_dump(mode="json", by_alias=True, exclude_none=True)
            for t in sorted(self.tools.values(), key=lambda t: t.name)
        ]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("đã lưu %d tool vào %s", len(data), path)

    async def call_tool(
        self, name: str, args: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> Any:
        """Gọi một tool, trả về kết quả đã decode. Lỗi tool → `ConcatError`."""
        if self.tools and name not in self.tools:
            raise ConcatError(name, "tool không có trong tools/list")
        args = args or {}
        log.debug("→ %s %s", name, json.dumps(args, ensure_ascii=False, default=str))
        t0 = time.perf_counter()
        result = await self.session.call_tool(
            name, args, read_timeout_seconds=timeout or self._settings.concat_timeout_sec
        )
        dt = time.perf_counter() - t0
        if not isinstance(result, types.CallToolResult):
            raise ConcatError(name, f"kết quả không hỗ trợ: {type(result).__name__}")
        if result.is_error:
            msg = _text_of(result) or "lỗi không rõ"
            log.debug("✗ %s (%.2fs) %s", name, dt, msg)
            raise ConcatError(name, msg)
        log.debug("← %s (%.2fs)", name, dt)
        return result if name == "preview_frame" else _decode(result)

    async def run(self, commands: list[ConcatCommand]) -> list[Any]:
        """Chạy tuần tự danh sách lệnh (ví dụ output của compiler)."""
        return [await self.call_tool(c.tool, c.args) for c in commands]

    async def preview_frame(self, t: float, **extra: Any) -> bytes:
        """Ảnh frame đã composite tại giây `t` của project đang mở (bytes ảnh).

        Tên tham số thời gian đọc từ schema `preview_frame` trong tools/list, không đoán.
        `extra` truyền thêm các tham số khác của tool (xem data/concat_tools.json).
        """
        args = {self._time_arg("preview_frame"): t, **extra}
        result: types.CallToolResult = await self.call_tool("preview_frame", args)
        for c in result.content:
            if isinstance(c, types.ImageContent):
                return base64.b64decode(c.data)
        raise ConcatError("preview_frame", "kết quả không có ảnh")

    def _time_arg(self, tool: str) -> str:
        schema = self.tools[tool].input_schema if tool in self.tools else {}
        props: dict[str, Any] = schema.get("properties", {})
        numeric = [k for k, v in props.items() if v.get("type") in ("number", "integer")]
        required = [k for k in numeric if k in schema.get("required", [])]
        candidates = required or numeric
        if len(candidates) != 1:
            raise ConcatError(
                tool, f"không xác định được tham số thời gian từ schema (ứng viên: {candidates})"
            )
        return candidates[0]


class SyncConcatClient:
    """Bọc `ConcatClient` để dùng từ code đồng bộ (chạy event loop trong thread riêng)."""

    def __init__(self, settings: Settings | None = None, *, dump_tools: bool = True) -> None:
        self._inner = ConcatClient(settings, dump_tools=dump_tools)
        self._portal_cm = None
        self._portal: BlockingPortal | None = None

    def __enter__(self) -> Self:
        self._portal_cm = start_blocking_portal()
        self._portal = self._portal_cm.__enter__()
        try:
            self._portal.call(self._inner.__aenter__)
        except BaseException:
            self._portal_cm.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._portal is not None:
                self._portal.call(self._inner.__aexit__, None, None, None)
        finally:
            if self._portal_cm is not None:
                self._portal_cm.__exit__(*exc)
            self._portal = self._portal_cm = None

    def _call(self, fn, *args: Any, **kwargs: Any) -> Any:
        if self._portal is None:
            raise RuntimeError("SyncConcatClient chưa được mở (dùng `with`)")
        return self._portal.call(lambda: fn(*args, **kwargs))

    @property
    def tools(self) -> dict[str, types.Tool]:
        return self._inner.tools

    def list_tools(self) -> list[dict[str, Any]]:
        return self._call(self._inner.list_tools)

    def call_tool(
        self, name: str, args: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> Any:
        return self._call(self._inner.call_tool, name, args, timeout=timeout)

    def run(self, commands: list[ConcatCommand]) -> list[Any]:
        return self._call(self._inner.run, commands)

    def preview_frame(self, t: float, **extra: Any) -> bytes:
        return self._call(self._inner.preview_frame, t, **extra)
