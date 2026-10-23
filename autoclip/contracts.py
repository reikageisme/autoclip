"""Hợp đồng chung giữa các phần của AutoClip.

CHỈ chứa kiểu dữ liệu và Protocol — không có logic. Chỉ lead được sửa file này;
agent khác cần đổi thì ghi vào mục "Yêu cầu gửi lead" trong báo cáo.

Quy ước đơn vị: thời gian tính bằng giây (float); vị trí/kích thước là tỉ lệ 0–1 của khung hình.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

Range = tuple[float, float]
"""Khoảng [min, max], ghi ra JSON dạng mảng 2 phần tử."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- Media / ingest


class ClipInfo(_Model):
    """Một file clip source. `id` là None cho tới khi được lưu vào DB."""

    id: int | None = None
    path: str
    sha1: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    created_at: datetime | None = None


class SegmentInfo(_Model):
    """Một đoạn (cảnh) trong clip. `id`/`clip_id` là None cho tới khi được lưu vào DB."""

    id: int | None = None
    clip_id: int | None = None
    start: float
    end: float
    transcript: str | None = None
    tags: list[str] = Field(default_factory=list)
    hook_score: float = 0.0
    """0–1, càng cao càng hợp làm hook."""
    phash: str | None = None
    """Perceptual hash dạng hex của keyframe giữa segment."""
    thumb_path: str | None = None
    embedding: bytes | None = None


class IngestError(_Model):
    path: str
    message: str


class IngestReport(_Model):
    folder: str
    clips_seen: int = 0
    clips_added: int = 0
    clips_skipped: int = 0
    """Đã có trong DB (trùng sha1)."""
    segments_added: int = 0
    segments_deduped: int = 0
    errors: list[IngestError] = Field(default_factory=list)
    elapsed_sec: float = 0.0


class SegmentFilter(_Model):
    """Điều kiện lọc cho `ClipRepository.list_segments`. Mọi trường đều tuỳ chọn."""

    clip_id: int | None = None
    tags_any: list[str] = Field(default_factory=list)
    min_hook_score: float | None = None
    min_duration: float | None = None
    max_duration: float | None = None
    exclude_ids: list[int] = Field(default_factory=list)
    limit: int | None = None


@runtime_checkable
class ClipRepository(Protocol):
    """Kho clip/segment. Ingest chỉ được ghi DB qua interface này."""

    def get_clip_by_sha1(self, sha1: str) -> ClipInfo | None: ...

    def get_clip(self, clip_id: int) -> ClipInfo | None: ...

    def add_clip(self, clip: ClipInfo) -> ClipInfo:
        """Lưu clip, trả về bản có `id`. Trùng sha1 → trả về bản đã có, không tạo mới."""
        ...

    def add_segments(self, clip_id: int, segments: list[SegmentInfo]) -> list[SegmentInfo]:
        """Lưu các segment của clip, trả về bản có `id`."""
        ...

    def list_segments(self, filter: SegmentFilter | None = None) -> list[SegmentInfo]: ...


# ---------------------------------------------------------------- CapCut / recipe


class DraftStatus(StrEnum):
    OK = "ok"
    ENCRYPTED = "encrypted"
    CORRUPT = "corrupt"
    MISSING = "missing"


class MusicSpec(_Model):
    mode: str = "none"
    """none | library | source (giữ audio gốc của clip)."""
    beat_sync: bool = False


class SlotSpec(_Model):
    role: str
    """hook | body | outro | ..."""
    dur: Range | None = None
    repeat: tuple[int, int] | None = None
    """Số lần lặp slot [min, max]; None = 1 lần."""
    want_tags: list[str] = Field(default_factory=list)
    speed: Range | None = None
    transition: list[str] = Field(default_factory=list)
    """Tên transition (theo từ vựng của effect_map) để random chọn khi vào slot này."""


class TextSpec(_Model):
    kind: str
    """headline | captions | title | cta | ..."""
    slot: str | None = None
    """Role của slot mà text gắn vào; None = toàn video."""
    style: str | None = None
    pos: Range | None = None
    """Tâm (x, y) tỉ lệ khung hình."""
    source: str | None = None
    """Nguồn nội dung, ví dụ "whisper" cho captions, "llm" cho headline."""


class VariationSpec(_Model):
    palette: list[str] = Field(default_factory=list)
    fonts: list[str] = Field(default_factory=list)
    shuffle_body: bool = False


class UnmappedItem(_Model):
    """Thành phần CapCut không có tương đương trong Concat."""

    kind: str
    """effect | transition | sticker | font | filter | animation | ..."""
    name: str
    note: str | None = None


class RecipeSpec(_Model):
    """Nội dung recipe.json."""

    name: str
    """Mã kiểu (chữ thường, không dấu) — dùng làm tên file và trong DB."""
    label: str | None = None
    """Tên hiển thị cho người dùng, ví dụ "Hook nhanh · có phụ đề"."""
    description: str | None = None
    aspect: str = "9:16"
    duration_target: Range
    music: MusicSpec = Field(default_factory=MusicSpec)
    slots: list[SlotSpec]
    texts: list[TextSpec] = Field(default_factory=list)
    variation: VariationSpec = Field(default_factory=VariationSpec)
    source_project: str | None = None
    unmapped: list[UnmappedItem] = Field(default_factory=list)


# ---------------------------------------------------------------- Render


class RenderItem(_Model):
    """Một clip trên timeline, theo thứ tự xuất hiện."""

    slot_role: str
    segment_id: int | None
    clip_path: str
    src_start: float
    src_end: float
    media_duration: float | None = None
    """Độ dài file nguồn (ClipInfo.duration) — compiler cần để cắt đuôi theo `src_end`."""
    media_width: int | None = None
    media_height: int | None = None
    """Kích thước nguồn — để phóng cho kín khung khi tỉ lệ khác khung output."""
    focus_x: float | None = None
    """Vị trí ngang của chủ thể trong nguồn (0–1); nguồn ngang cắt dọc sẽ bám theo đây."""
    speed: float = 1.0
    transition: str | None = None
    """Transition vào item này; None = cut."""


class RenderText(_Model):
    """Text đã điền nội dung."""

    kind: str
    text: str
    slot: str | None = None
    style: str | None = None
    pos: Range | None = None
    start: float | None = None
    """Giây trên timeline output; None = theo slot / toàn video."""
    end: float | None = None


class RenderPlan(_Model):
    recipe: str
    seed: int
    items: list[RenderItem]
    texts: list[RenderText] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    """Tham số biến thể đã chọn (màu, font, ...)."""


class ConcatCommand(_Model):
    """Một lần gọi tool MCP của Concat."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- QA


class QAResult(_Model):
    score: float
    """0–1."""
    passed: bool
    notes: list[str] = Field(default_factory=list)
