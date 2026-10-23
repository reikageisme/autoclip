"""Client LLM tối giản dùng chung (text + 1 ảnh), cấu hình hoàn toàn qua Settings.

Provider:
- ``openai_compat``: endpoint kiểu OpenAI tại ``LLM_BASE_URL`` (ví dụ 9router), key ``LLM_API_KEY``.
- ``openai``: api.openai.com, key ``OPENAI_API_KEY``.
- ``anthropic``: api.anthropic.com, key ``ANTHROPIC_API_KEY``.
- ``none``: tắt — ``available`` là False, mọi nơi dùng phải có fallback không-LLM.

Không hardcode model: text dùng ``LLM_MODEL``, ảnh dùng ``VISION_MODEL``.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from autoclip.config import Settings, get_settings

log = logging.getLogger(__name__)

_OPENAI_URL = "https://api.openai.com/v1"
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        provider: str,
        model: str | None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_settings(cls, settings: Settings | None = None, *, vision: bool = False) -> LLMClient:
        s = settings or get_settings()
        model = s.vision_model if vision else s.llm_model
        provider = s.llm_provider
        if provider == "openai_compat":
            key = s.llm_api_key
            base = s.llm_base_url
        elif provider == "openai":
            key = s.openai_api_key
            base = s.llm_base_url or _OPENAI_URL
        else:
            key = s.anthropic_api_key if provider == "anthropic" else None
            base = None
        return cls(
            provider,
            model,
            base_url=base,
            api_key=key.get_secret_value() if key else None,
            timeout=s.llm_timeout_sec,
        )

    @property
    def available(self) -> bool:
        if self.provider == "none" or not self.model or not self.api_key:
            return False
        return self.provider == "anthropic" or bool(self.base_url)

    def chat(self, prompt: str, *, image_jpeg_b64: str | None = None, max_tokens: int = 400) -> str:
        """Gửi 1 lượt user (text + tuỳ chọn 1 ảnh JPEG base64), trả về text của model."""
        if not self.available:
            raise LLMError(f"LLM chưa cấu hình (provider={self.provider}, model={self.model})")
        if self.provider == "anthropic":
            return self._anthropic(prompt, image_jpeg_b64, max_tokens)
        return self._openai(prompt, image_jpeg_b64, max_tokens)

    def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        log.debug("LLM → %s model=%s", url, payload.get("model"))
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

    def _openai(self, prompt: str, image: str | None, max_tokens: int) -> str:
        content: Any = prompt
        if image:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
            ]
        data = self._post(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            {"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"phản hồi lạ: {str(data)[:300]}") from exc
        if not content.strip():
            # model suy luận (reasoning) có thể tiêu hết max_tokens trước khi trả lời
            raise LLMError(f"nội dung rỗng (finish_reason={choice.get('finish_reason')})")
        return content

    def _anthropic(self, prompt: str, image: str | None, max_tokens: int) -> str:
        content: list[dict[str, Any]] = []
        if image:
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": image},
                }
            )
        content.append({"type": "text", "text": prompt})
        data = self._post(
            _ANTHROPIC_URL,
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": content}],
            },
            {"x-api-key": str(self.api_key), "anthropic-version": "2023-06-01"},
        )
        try:
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise LLMError(f"phản hồi lạ: {str(data)[:300]}") from exc
