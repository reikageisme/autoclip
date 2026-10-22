"""Quality scoring for rendered videos."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from collections.abc import Callable, Sequence
from typing import Any

from autoclip.contracts import QAResult
from autoclip.ingest._compat import construct

from .checks import (
    CheckResult,
    check_aspect,
    check_audio,
    check_black_or_red_frames,
    check_duration,
    duration_target,
)
from .frames import MediaInfo, grab_frames, grab_frames_from_callable, probe_media

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"frames": 0.35, "duration": 0.20, "audio": 0.20, "aspect": 0.25}
PASS_THRESHOLD = 0.60


def _spec_aspect(spec: Any) -> Any:
    if spec is None:
        return None
    if isinstance(spec, dict):
        return spec.get("aspect")
    return getattr(spec, "aspect", None)


def _score_from_checks(
    checks: Sequence[CheckResult],
    weights: dict[str, float] | None = None,
) -> float:
    merged = dict(DEFAULT_WEIGHTS)
    if weights:
        merged.update(weights)
    total = sum(merged.get(check.name, 0.0) for check in checks)
    if total <= 0:
        return 0.0 if any(not check.passed for check in checks) else 1.0
    return sum(merged.get(check.name, 0.0) * check.score for check in checks) / total


def _merge_vision(
    score: float,
    passed: bool,
    notes: list[str],
    vision_result: Any,
) -> tuple[float, bool, list[str]]:
    if vision_result is None:
        return score, passed, notes
    vision_score: float | None = None
    vision_passed: bool | None = None
    vision_notes: list[str] = []
    if isinstance(vision_result, (tuple, list)) and vision_result:
        try:
            vision_score = float(vision_result[0])
        except (TypeError, ValueError):
            vision_score = None
        if len(vision_result) > 1 and isinstance(vision_result[1], (list, tuple)):
            vision_notes = [str(item) for item in vision_result[1]]
    elif isinstance(vision_result, dict):
        try:
            vision_score = float(vision_result.get("score"))
        except (TypeError, ValueError):
            vision_score = None
        vision_passed = vision_result.get("passed")
        vision_notes = [str(item) for item in vision_result.get("notes", [])]
    elif hasattr(vision_result, "score"):
        vision_score = float(vision_result.score)
        vision_passed = getattr(vision_result, "passed", None)
        vision_notes = [str(item) for item in getattr(vision_result, "notes", []) or []]

    if vision_score is None:
        return score, passed, notes
    final = 0.8 * score + 0.2 * max(0.0, min(1.0, vision_score))
    if vision_passed is not None:
        passed = passed and bool(vision_passed)
    passed = passed and final >= PASS_THRESHOLD
    notes.extend(f"vision: {note}" for note in vision_notes)
    return final, passed, notes


def _call_vision(vision: Any, frames: Sequence[Any]) -> Any:
    try:
        if hasattr(vision, "score"):
            return vision.score(frames)
        return vision(frames)
    except Exception:
        logger.exception("vision QA scorer failed")
        return None


def evaluate(
    path: str | Any,
    spec: Any = None,
    *,
    frames_count: int = 5,
    grab_frame_fn: Callable[[float], Any] | None = None,
    vision_scorer: Any = None,
    weights: dict[str, float] | None = None,
) -> QAResult:
    """Score a rendered video and return the shared :class:`QAResult`.

    Black/red frames, duration, audio and aspect are always deterministic.
    A vision scorer is optional and only runs when explicitly supplied.
    """
    info: MediaInfo = probe_media(path)
    if grab_frame_fn is not None:
        frames = grab_frames_from_callable(grab_frame_fn, frames_count, info.duration)
    else:
        frames = grab_frames(path, frames_count, info.duration, info=info)

    checks: list[CheckResult] = [check_black_or_red_frames(frames)]
    target = duration_target(spec)
    if target is not None:
        checks.append(check_duration(info.duration, target))
    checks.append(check_audio(path))
    aspect = _spec_aspect(spec)
    if aspect is not None:
        checks.append(check_aspect(info.width, info.height, aspect))

    score = _score_from_checks(checks, weights)
    passed = all(check.passed for check in checks) and score >= PASS_THRESHOLD
    notes = [f"{check.name}: {check.detail}" for check in checks if not check.passed]

    if vision_scorer is not None:
        score, passed, notes = _merge_vision(
            score, passed, notes, _call_vision(vision_scorer, frames)
        )

    return construct(
        QAResult,
        score=round(max(0.0, min(1.0, score)), 4),
        passed=bool(passed),
        notes=notes,
    )


def _frames_to_jpeg_b64(frames: Sequence[Any]) -> str | None:
    if not frames:
        return None
    try:
        import numpy as np
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    array = np.asarray(frames[len(frames) // 2]).astype("uint8")
    image = Image.fromarray(array).convert("RGB")
    image.thumbnail((640, 640))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse_vision_payload(text: str) -> tuple[float, list[str]]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return 0.0, ["vision response was not valid JSON"]
    try:
        score = float(payload.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    notes = [str(item) for item in payload.get("notes", [])]
    return max(0.0, min(1.0, score)), notes


class VisionLLMScorer:
    """Optional vision model scorer (OpenAI or Anthropic), skipped without a key."""

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        client: Callable[[Sequence[Any]], Any] | None = None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model or (
            "claude-3-5-haiku-latest" if self.provider == "anthropic" else "gpt-4o-mini"
        )
        self.api_key = api_key or (
            os.environ.get("ANTHROPIC_API_KEY")
            if self.provider == "anthropic"
            else os.environ.get("OPENAI_API_KEY")
        )
        self.timeout = timeout
        self._client = client

    @property
    def available(self) -> bool:
        return self._client is not None or bool(self.api_key)

    def score(self, frames: Sequence[Any]) -> tuple[float, list[str]]:
        if self._client is not None:
            result = self._client(frames)
            if isinstance(result, tuple):
                return float(result[0]), [str(item) for item in result[1]]
            if isinstance(result, (int, float)):
                return float(result), []
            return 0.0, ["invalid vision client result"]
        if not self.api_key:
            return 0.0, ["vision scorer unavailable (no API key)"]
        encoded = _frames_to_jpeg_b64(frames)
        if encoded is None:
            return 0.0, ["vision scorer unavailable (no frames)"]
        prompt = (
            "Score this rendered short-form video keyframe from 0 to 1 for visual "
            "quality: composition, readability, and whether it looks broken. "
            'Return ONLY JSON: {"score": number, "notes": ["..."]}.'
        )
        try:
            if self.provider == "anthropic":
                text = self._call_anthropic(encoded, prompt)
            else:
                text = self._call_openai(encoded, prompt)
        except Exception as exc:
            logger.exception("vision QA request failed")
            return 0.0, [f"vision request failed: {exc}"]
        return _parse_vision_payload(text)

    def _request(self, request: Any) -> dict[str, Any]:
        import urllib.request

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _call_openai(self, encoded: str, prompt: str) -> str:
        import urllib.request

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                        },
                    ],
                }
            ],
            "temperature": 0,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        data = self._request(request)
        return str(data["choices"][0]["message"]["content"])

    def _call_anthropic(self, encoded: str, prompt: str) -> str:
        import urllib.request

        payload = {
            "model": self.model,
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encoded,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": str(self.api_key),
                "anthropic-version": "2023-06-01",
            },
        )
        data = self._request(request)
        return str(data["content"][0]["text"])
