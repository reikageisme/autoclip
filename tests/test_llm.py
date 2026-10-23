from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from autoclip.config import Settings
from autoclip.llm import LLMClient, LLMError


@pytest.fixture
def fake_openai():
    """Server OpenAI-compatible giả: ghi lại request, trả 1 câu trả lời cố định."""
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "auth": self.headers["Authorization"], "body": body})
            out = json.dumps({"choices": [{"message": {"content": '{"score": 0.8}'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1", seen
    server.shutdown()


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_openai_compat_sends_model_key_and_image(fake_openai):
    url, seen = fake_openai
    s = _settings(
        llm_provider="openai_compat",
        llm_base_url=url,
        llm_api_key="sk-x",
        llm_model="text-m",
        vision_model="vis-m",
    )
    vision = LLMClient.from_settings(s, vision=True)
    assert vision.available
    assert vision.chat("chấm điểm", image_jpeg_b64="QUJD") == '{"score": 0.8}'
    req = seen[-1]
    assert req["path"] == "/v1/chat/completions"
    assert req["auth"] == "Bearer sk-x"
    assert req["body"]["model"] == "vis-m"
    parts = req["body"]["messages"][0]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,QUJD")

    text = LLMClient.from_settings(s)
    text.chat("hi")
    assert seen[-1]["body"]["model"] == "text-m"
    assert seen[-1]["body"]["messages"][0]["content"] == "hi"


def test_vision_off_without_vision_model():
    s = _settings(llm_provider="openai_compat", llm_base_url="http://x/v1", llm_api_key="k")
    assert not LLMClient.from_settings(s, vision=True).available


def test_unconfigured_raises():
    with pytest.raises(LLMError):
        LLMClient("none", None).chat("x")


def test_empty_content_is_an_error(monkeypatch):
    """Model suy luận hết max_tokens trả content rỗng → phải báo lỗi, không coi là trả lời."""
    client = LLMClient("openai_compat", "m", base_url="http://x/v1", api_key="k")
    monkeypatch.setattr(
        client,
        "_post",
        lambda *a, **k: {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]},
    )
    with pytest.raises(LLMError, match="finish_reason=length"):
        client.chat("x")
