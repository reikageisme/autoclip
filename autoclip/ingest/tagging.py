"""Segment tagging and hook scoring.

``HeuristicTagger`` is the mandatory offline implementation.  It measures
brightness, inter-frame motion, presence of a face (OpenCV Haar cascade) and
whether the scene has speech.  ``VisionLLMTagger`` is optional and silently
unavailable when no API key is configured.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .frames import to_gray
from .transcribe import normalize_text

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"motion": 0.45, "face": 0.30, "speech": 0.25}


class TaggerUnavailable(RuntimeError):
    """Raised when an optional tagger cannot run."""


@dataclass(slots=True)
class TagResult:
    tags: list[str] = field(default_factory=list)
    hook_score: float = 0.0


@runtime_checkable
class Tagger(Protocol):
    """Tag a set of frames plus the scene transcript."""

    def tag(self, frames: Sequence[Any], transcript: str) -> TagResult: ...


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def combine_hook_score(
    motion: float,
    has_face: bool,
    has_speech: bool,
    weights: dict[str, float] | None = None,
    motion_full: float = 0.15,
) -> float:
    """Weighted hook score in 0-1 from normalized motion, face and speech."""
    merged = dict(DEFAULT_WEIGHTS)
    if weights:
        merged.update(weights)
    total = sum(max(0.0, float(value)) for value in merged.values()) or 1.0
    motion_score = _clamp(motion / motion_full if motion_full > 0 else 0.0)
    score = (
        merged.get("motion", 0.0) * motion_score
        + merged.get("face", 0.0) * (1.0 if has_face else 0.0)
        + merged.get("speech", 0.0) * (1.0 if has_speech else 0.0)
    ) / total
    return round(_clamp(score), 4)


def frame_statistics(frames: Sequence[Any]) -> tuple[float, float]:
    """Return ``(brightness, motion)`` normalized to 0-1."""
    import numpy as np

    grays = [to_gray(frame) for frame in frames if frame is not None]
    if not grays:
        return 0.0, 0.0
    brightness = float(np.mean([gray.mean() for gray in grays])) / 255.0
    if len(grays) < 2:
        return _clamp(brightness), 0.0
    diffs = [
        float(np.abs(grays[index + 1] - grays[index]).mean()) / 255.0
        for index in range(len(grays) - 1)
    ]
    return _clamp(brightness), _clamp(float(np.mean(diffs)))


class HeuristicTagger:
    """Offline tagger using brightness, frame difference, Haar faces and speech."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        motion_full: float = 0.15,
        motion_threshold: float = 0.02,
        brightness_high: float = 0.60,
        brightness_low: float = 0.18,
        face_detector: Callable[[Sequence[Any]], bool] | None = None,
    ) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.motion_full = motion_full
        self.motion_threshold = motion_threshold
        self.brightness_high = brightness_high
        self.brightness_low = brightness_low
        self._face_detector = face_detector
        self._cascade: Any | None = None

    def _haar_face(self, frames: Sequence[Any]) -> bool:
        if self._face_detector is not None:
            return bool(self._face_detector(frames))
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.debug("opencv not installed; face tag disabled")
            return False
        if self._cascade is None:
            cascade_path = os.path.join(
                getattr(cv2, "data", None).haarcascades
                if getattr(cv2, "data", None)
                else "",
                "haarcascade_frontalface_default.xml",
            )
            if not cascade_path or not os.path.exists(cascade_path):
                logger.debug("Haar cascade not found; face tag disabled")
                self._cascade = False
                return False
            self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade is False or self._cascade.empty():
            return False
        for frame in frames:
            if frame is None:
                continue
            gray = to_gray(frame)
            if gray.size == 0:
                continue
            image = np.clip(gray, 0, 255).astype("uint8")
            image = cv2.equalizeHist(image)
            faces = self._cascade.detectMultiScale(
                image, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24)
            )
            if len(faces) > 0:
                return True
        return False

    def tag(self, frames: Sequence[Any], transcript: str) -> TagResult:
        brightness, motion = frame_statistics(frames)
        has_face = self._haar_face(frames)
        has_speech = bool(normalize_text(transcript))

        tags: list[str] = []
        if brightness >= self.brightness_high:
            tags.append("bright")
        elif brightness <= self.brightness_low:
            tags.append("dark")
        tags.append("action" if motion >= self.motion_threshold else "static")
        if has_face:
            tags.append("face")
        if has_speech:
            tags.append("speech")

        score = combine_hook_score(motion, has_face, has_speech, self.weights, self.motion_full)
        return TagResult(tags=tags, hook_score=score)


def _frames_to_jpeg_b64(frames: Sequence[Any]) -> str | None:
    if not frames:
        return None
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    middle = frames[len(frames) // 2]
    image = Image.fromarray(middle).convert("RGB")
    image.thumbnail((640, 640))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse_tag_payload(text: str) -> TagResult:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return TagResult(tags=[], hook_score=0.0)
    tags = [str(tag) for tag in payload.get("tags", []) if isinstance(tag, str)]
    try:
        score = _clamp(float(payload.get("hook_score", 0.0)))
    except (TypeError, ValueError):
        score = 0.0
    return TagResult(tags=tags, hook_score=score)


class VisionLLMTagger:
    """Optional vision model tagger (OpenAI or Anthropic), skipped without a key."""

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        client: Callable[[Sequence[Any], str], TagResult] | None = None,
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

    def tag(self, frames: Sequence[Any], transcript: str) -> TagResult:
        if self._client is not None:
            return self._client(frames, transcript)
        if not self.api_key:
            raise TaggerUnavailable("no vision API key configured")
        encoded = _frames_to_jpeg_b64(frames)
        if encoded is None:
            return TagResult()
        prompt = (
            "You are tagging short-form video clips. Given the keyframe and "
            "transcript, return ONLY JSON: "
            '{"tags": ["action"|"face"|"speech"|"static"|"bright"|"dark", ...], '
            '"hook_score": number between 0 and 1}. Transcript: '
            + normalize_text(transcript)
        )
        if self.provider == "anthropic":
            return self._call_anthropic(encoded, prompt)
        return self._call_openai(encoded, prompt)

    def _request(self, request: Any) -> dict[str, Any]:
        import urllib.request

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _call_openai(self, encoded: str, prompt: str) -> TagResult:
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
        text = data["choices"][0]["message"]["content"]
        return _parse_tag_payload(text)

    def _call_anthropic(self, encoded: str, prompt: str) -> TagResult:
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
        text = data["content"][0]["text"]
        return _parse_tag_payload(text)
