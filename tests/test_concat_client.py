from __future__ import annotations

import pytest
from mcp import types

from autoclip.concat_client import ConcatClient, ConcatError
from autoclip.concat_client.client import _decode
from autoclip.config import Settings


def _client_with(schema: dict) -> ConcatClient:
    cc = ConcatClient(Settings(), dump_tools=False)
    cc.tools = {"preview_frame": types.Tool(name="preview_frame", input_schema=schema)}
    return cc


def test_time_arg_from_required_numeric():
    cc = _client_with(
        {
            "type": "object",
            "properties": {
                "at": {"type": "number"},
                "width": {"type": "integer"},
                "project": {"type": "string"},
            },
            "required": ["at", "project"],
        }
    )
    assert cc._time_arg("preview_frame") == "at"


def test_time_arg_ambiguous_raises():
    cc = _client_with(
        {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}
    )
    with pytest.raises(ConcatError):
        cc._time_arg("preview_frame")


def test_decode_prefers_structured_then_json_then_text():
    structured = types.CallToolResult(content=[], structured_content={"ok": 1})
    assert _decode(structured) == {"ok": 1}
    js = types.CallToolResult(content=[types.TextContent(type="text", text='{"id": "p1"}')])
    assert _decode(js) == {"id": "p1"}
    plain = types.CallToolResult(content=[types.TextContent(type="text", text="saved")])
    assert _decode(plain) == "saved"


def test_server_params_adds_data_dir_to_roots(tmp_path):
    cc = ConcatClient(Settings(data_dir=tmp_path, concat_mcp_roots=r"E:\media"), dump_tools=False)
    p = cc._server_params()
    assert p.args == ["--mcp", "headless"]
    assert p.env["CONCAT_MCP_ROOTS"].split(";") == [r"E:\media", str(tmp_path)]
