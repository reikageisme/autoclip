"""Đọc project CapCut (draft JSON) an toàn, không bao giờ crash.

CapCut lưu thời gian bằng micro giây; module này đổi mọi mốc thời gian sang
giây ngay khi đọc. Các trường lạ được giữ lại (``extra="allow"``) nhưng không
được diễn giải. Text style của CapCut là một chuỗi JSON lồng trong material,
được parse phòng thủ: lỗi ở một text không làm hỏng cả draft.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from autoclip.contracts import DraftStatus

logger = logging.getLogger(__name__)

DRAFT_FILENAMES = ("draft_content.json", "draft_info.json")
_MICROSECONDS = 1_000_000.0
_ASPECTS = (
    (9 / 16, "9:16"),
    (16 / 9, "16:9"),
    (1.0, "1:1"),
    (4 / 5, "4:5"),
    (3 / 4, "3:4"),
    (21 / 9, "21:9"),
)


class DraftError(RuntimeError):
    """Draft không đọc được, kèm trạng thái phát hiện được."""

    def __init__(self, status: DraftStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class _CapCutModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class TextInfo(_CapCutModel):
    """Nội dung + style của một material text, đã bóc khỏi chuỗi JSON."""

    text: str = ""
    font_family: str | None = None
    font_path: str | None = None
    font_size: float | None = None
    bold: bool = False
    italic: bool = False
    color: str | None = None
    has_stroke: bool = False
    alignment: str | None = None
    styles: list[dict[str, Any]] = Field(default_factory=list)
    parse_error: bool = False


class VideoMaterial(_CapCutModel):
    id: str = ""
    path: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    kind: str = "video"
    name: str = ""


class AudioMaterial(_CapCutModel):
    id: str = ""
    path: str = ""
    duration: float = 0.0
    kind: str = ""
    name: str = ""


class TextMaterial(_CapCutModel):
    id: str = ""
    text: TextInfo = Field(default_factory=TextInfo)
    raw_content: str = ""


class EffectMaterial(_CapCutModel):
    id: str = ""
    name: str = ""
    effect_id: str = ""
    resource_id: str = ""
    category: str = ""
    value: float = 0.0
    kind: str = "effect"


class TransitionMaterial(_CapCutModel):
    id: str = ""
    name: str = ""
    duration: float = 0.0
    is_overlap: bool = False
    effect_id: str = ""
    resource_id: str = ""


class StickerMaterial(_CapCutModel):
    id: str = ""
    name: str = ""
    resource_id: str = ""
    category: str = ""


class FilterMaterial(_CapCutModel):
    id: str = ""
    name: str = ""
    effect_id: str = ""
    resource_id: str = ""
    value: float = 0.0


class CapCutMaterials(_CapCutModel):
    videos: list[VideoMaterial] = Field(default_factory=list)
    audios: list[AudioMaterial] = Field(default_factory=list)
    texts: list[TextMaterial] = Field(default_factory=list)
    effects: list[EffectMaterial] = Field(default_factory=list)
    transitions: list[TransitionMaterial] = Field(default_factory=list)
    stickers: list[StickerMaterial] = Field(default_factory=list)
    filters: list[FilterMaterial] = Field(default_factory=list)
    images: list[VideoMaterial] = Field(default_factory=list)


class CapCutSegment(_CapCutModel):
    id: str = ""
    material_id: str = ""
    track_type: str = ""
    start: float = 0.0
    duration: float = 0.0
    end: float = 0.0
    source_start: float = 0.0
    source_duration: float = 0.0
    speed: float = 1.0
    volume: float = 1.0
    visible: bool = True
    render_index: int = 0
    transform: dict[str, Any] = Field(default_factory=dict)
    extra_material_refs: list[str] = Field(default_factory=list)
    text: TextInfo | None = None
    transition: TransitionMaterial | None = None
    effects: list[EffectMaterial] = Field(default_factory=list)
    filters: list[FilterMaterial] = Field(default_factory=list)
    sticker: StickerMaterial | None = None
    path: str = ""


class CapCutTrack(_CapCutModel):
    id: str = ""
    type: str = ""
    attribute: int = 0
    segments: list[CapCutSegment] = Field(default_factory=list)


class CapCutDraft(_CapCutModel):
    name: str = "draft"
    draft_dir: str = ""
    draft_file: str = ""
    aspect: str = "9:16"
    canvas_width: int = 1080
    canvas_height: int = 1920
    fps: float = 30.0
    duration: float = 0.0
    tracks: list[CapCutTrack] = Field(default_factory=list)
    materials: CapCutMaterials = Field(default_factory=CapCutMaterials)
    warnings: list[str] = Field(default_factory=list)

    def segments_of(self, *kinds: str) -> list[CapCutSegment]:
        wanted = {kind.lower() for kind in kinds}
        found: list[CapCutSegment] = []
        for track in self.tracks:
            if track.type.lower() in wanted:
                found.extend(track.segments)
        return sorted(found, key=lambda segment: (segment.start, segment.end))

    @property
    def visual_segments(self) -> list[CapCutSegment]:
        return self.segments_of("video", "main")

    @property
    def audio_segments(self) -> list[CapCutSegment]:
        return self.segments_of("audio")

    @property
    def text_segments(self) -> list[CapCutSegment]:
        return self.segments_of("text")

    @property
    def effect_segments(self) -> list[CapCutSegment]:
        return self.segments_of("effect")

    @property
    def sticker_segments(self) -> list[CapCutSegment]:
        return self.segments_of("sticker")


def _as_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _as_int(value: Any, default: int = 0) -> int:
    return int(_as_number(value, float(default)))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _seconds(value: Any) -> float:
    return _as_number(value) / _MICROSECONDS


def _time_range(value: Any) -> tuple[float, float, float]:
    payload = _as_dict(value)
    start = _seconds(payload.get("start"))
    duration = _seconds(payload.get("duration"))
    if duration <= 0:
        end = _seconds(payload.get("end"))
        duration = max(0.0, end - start)
    return start, duration, start + duration


def _hex_color(value: Any) -> str | None:
    if isinstance(value, list) and len(value) >= 3:
        try:
            channels = [max(0, min(255, round(float(channel) * 255))) for channel in value[:3]]
        except (TypeError, ValueError):
            return None
        return "#{:02x}{:02x}{:02x}".format(*channels)
    if isinstance(value, str):
        match = re.fullmatch(r"#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?", value.strip())
        if match:
            return value.strip().lower()
    return None


def _text_info(raw_content: Any) -> TextInfo:
    payload: dict[str, Any] = {}
    parse_error = False
    if isinstance(raw_content, dict):
        payload = raw_content
    elif isinstance(raw_content, str) and raw_content.strip():
        try:
            parsed = json.loads(raw_content)
        except (TypeError, ValueError):
            parse_error = True
        else:
            if isinstance(parsed, dict):
                payload = parsed
            else:
                parse_error = True

    text = ""
    for key in ("text", "content", "text_content"):
        candidate = payload.get(key)
        if isinstance(candidate, str) and candidate:
            text = candidate
            break

    styles = [style for style in _as_list(payload.get("styles")) if isinstance(style, dict)]
    first = styles[0] if styles else {}
    font = _as_dict(first.get("font"))
    font_path = _as_text(font.get("path")) or None
    family = _as_text(font.get("family") or font.get("name") or first.get("fontFamily")) or None
    if family is None and font_path:
        family = Path(font_path.replace("\\", "/")).stem or None

    fill = _as_dict(first.get("fill"))
    fill_content = _as_dict(fill.get("content"))
    solid = _as_dict(fill_content.get("solid"))
    color = _hex_color(solid.get("color")) or _hex_color(first.get("color"))
    stroke = _as_dict(first.get("stroke"))
    weight = _as_number(first.get("fontWeight") or first.get("weight"), 300.0)
    alignment = first.get("alignment") or first.get("align")

    return TextInfo(
        text=text,
        font_family=family,
        font_path=font_path,
        font_size=_as_number(first.get("size") or first.get("fontSize"), 0.0) or None,
        bold=bool(first.get("bold")) or weight >= 600,
        italic=bool(first.get("italic")),
        color=color,
        has_stroke=_as_number(stroke.get("width"), 0.0) > 0,
        alignment=_as_text(alignment) or None,
        styles=styles,
        parse_error=parse_error,
    )


def _material_id(material: dict[str, Any]) -> str:
    return _as_text(material.get("id") or material.get("material_id"))


def _parse_video_material(raw: dict[str, Any], kind: str) -> VideoMaterial:
    return VideoMaterial(
        id=_material_id(raw),
        path=_as_text(raw.get("path")),
        duration=_seconds(raw.get("duration")),
        width=_as_int(raw.get("width")),
        height=_as_int(raw.get("height")),
        kind=_as_text(raw.get("type")) or kind,
        name=_as_text(raw.get("material_name") or raw.get("name")),
    )


def _parse_audio_material(raw: dict[str, Any]) -> AudioMaterial:
    return AudioMaterial(
        id=_material_id(raw),
        path=_as_text(raw.get("path")),
        duration=_seconds(raw.get("duration")),
        kind=_as_text(raw.get("type")),
        name=_as_text(raw.get("material_name") or raw.get("name")),
    )


def _parse_effect_material(raw: dict[str, Any]) -> EffectMaterial:
    return EffectMaterial(
        id=_material_id(raw),
        name=_as_text(raw.get("name") or raw.get("effect_name")),
        effect_id=_as_text(raw.get("effect_id")),
        resource_id=_as_text(raw.get("resource_id")),
        category=_as_text(raw.get("category_name") or raw.get("category")),
        value=_as_number(raw.get("value")),
        kind=_as_text(raw.get("type")) or "effect",
    )


def _parse_transition_material(raw: dict[str, Any]) -> TransitionMaterial:
    return TransitionMaterial(
        id=_material_id(raw),
        name=_as_text(raw.get("name")),
        duration=_seconds(raw.get("duration")),
        is_overlap=bool(raw.get("is_overlap")),
        effect_id=_as_text(raw.get("effect_id")),
        resource_id=_as_text(raw.get("resource_id")),
    )


def _parse_sticker_material(raw: dict[str, Any]) -> StickerMaterial:
    return StickerMaterial(
        id=_material_id(raw),
        name=_as_text(raw.get("name") or raw.get("sticker_name")),
        resource_id=_as_text(raw.get("resource_id")),
        category=_as_text(raw.get("category_name") or raw.get("category")),
    )


def _parse_filter_material(raw: dict[str, Any]) -> FilterMaterial:
    return FilterMaterial(
        id=_material_id(raw),
        name=_as_text(raw.get("name")),
        effect_id=_as_text(raw.get("effect_id")),
        resource_id=_as_text(raw.get("resource_id")),
        value=_as_number(raw.get("value")),
    )


def _parse_materials(raw_materials: Any) -> CapCutMaterials:
    raw = _as_dict(raw_materials)
    videos = [
        _parse_video_material(item, "video")
        for item in _as_list(raw.get("videos"))
        if isinstance(item, dict)
    ]
    images = [
        _parse_video_material(item, "image")
        for item in _as_list(raw.get("images"))
        if isinstance(item, dict)
    ]
    texts = []
    for item in _as_list(raw.get("texts")):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        texts.append(
            TextMaterial(
                id=_material_id(item),
                text=_text_info(content),
                raw_content=content if isinstance(content, str) else json.dumps(content),
            )
        )
    return CapCutMaterials(
        videos=videos,
        audios=[
            _parse_audio_material(item)
            for item in _as_list(raw.get("audios"))
            if isinstance(item, dict)
        ],
        texts=texts,
        effects=[
            _parse_effect_material(item)
            for item in _as_list(raw.get("effects"))
            if isinstance(item, dict)
        ],
        transitions=[
            _parse_transition_material(item)
            for item in _as_list(raw.get("transitions"))
            if isinstance(item, dict)
        ],
        stickers=[
            _parse_sticker_material(item)
            for item in _as_list(raw.get("stickers"))
            if isinstance(item, dict)
        ],
        filters=[
            _parse_filter_material(item)
            for item in _as_list(raw.get("filters"))
            if isinstance(item, dict)
        ],
        images=images,
    )


def _infer_track_type(raw: dict[str, Any]) -> str:
    explicit = _as_text(raw.get("type")).lower()
    if explicit:
        return explicit
    for segment in _as_list(raw.get("segments")):
        if not isinstance(segment, dict):
            continue
        if isinstance(segment.get("clip"), dict) or segment.get("material_id"):
            return "video"
    return "video"


_MaterialT = TypeVar("_MaterialT")


def _index_by_id(items: list[_MaterialT]) -> dict[str, _MaterialT]:
    indexed: dict[str, _MaterialT] = {}
    for item in items:
        identifier = getattr(item, "id", "")
        if identifier:
            indexed[identifier] = item
    return indexed


def _parse_segment(
    raw: dict[str, Any],
    track_type: str,
    indexes: dict[str, dict[str, Any]],
    warnings: list[str],
) -> CapCutSegment:
    start, duration, end = _time_range(raw.get("target_timerange") or raw.get("target_time_range"))
    source_start, source_duration, _ = _time_range(
        raw.get("source_timerange") or raw.get("source_time_range")
    )
    refs = [_as_text(reference) for reference in _as_list(raw.get("extra_material_refs"))]
    clip = _as_dict(raw.get("clip"))
    material_id = _as_text(raw.get("material_id"))

    text = None
    text_material = indexes["texts"].get(material_id)
    if isinstance(text_material, TextMaterial):
        text = text_material.text
    if text is None and track_type == "text":
        for reference in refs:
            candidate = indexes["texts"].get(reference)
            if isinstance(candidate, TextMaterial):
                text = candidate.text
                break
        if text is None:
            warnings.append(f"text segment {material_id or '<no id>'} has no text material")

    transition = None
    for reference in refs:
        candidate = indexes["transitions"].get(reference)
        if isinstance(candidate, TransitionMaterial):
            transition = candidate
            break

    effects = [
        candidate
        for reference in refs
        if isinstance((candidate := indexes["effects"].get(reference)), EffectMaterial)
    ]
    filters = [
        candidate
        for reference in refs
        if isinstance((candidate := indexes["filters"].get(reference)), FilterMaterial)
    ]
    sticker = None
    for reference in refs:
        candidate = indexes["stickers"].get(reference)
        if isinstance(candidate, StickerMaterial):
            sticker = candidate
            break

    path = ""
    for kind in ("videos", "images", "audios"):
        material = indexes[kind].get(material_id)
        if material is not None and getattr(material, "path", ""):
            path = material.path
            break

    if track_type == "text" and text is not None and text.parse_error:
        warnings.append(f"text segment {material_id or '<no id>'} has unparseable style JSON")

    return CapCutSegment(
        id=_as_text(raw.get("id")),
        material_id=material_id,
        track_type=track_type,
        start=start,
        duration=duration,
        end=end,
        source_start=source_start,
        source_duration=source_duration,
        speed=_as_number(raw.get("speed"), 1.0) or 1.0,
        volume=_as_number(raw.get("volume") or clip.get("volume"), 1.0),
        visible=bool(raw.get("visible", True)),
        render_index=_as_int(raw.get("render_index")),
        transform=_as_dict(clip.get("transform")),
        extra_material_refs=refs,
        text=text,
        transition=transition,
        effects=effects,
        filters=filters,
        sticker=sticker,
        path=path,
    )


def _parse_track(
    raw: dict[str, Any],
    indexes: dict[str, dict[str, Any]],
    warnings: list[str],
) -> CapCutTrack:
    track_type = _infer_track_type(raw)
    segments = [
        _parse_segment(segment, track_type, indexes, warnings)
        for segment in _as_list(raw.get("segments"))
        if isinstance(segment, dict)
    ]
    return CapCutTrack(
        id=_as_text(raw.get("id")),
        type=track_type,
        attribute=_as_int(raw.get("attribute")),
        segments=sorted(segments, key=lambda segment: (segment.start, segment.end)),
    )


def _aspect_from_size(width: float, height: float, ratio: str = "") -> str:
    text = ratio.strip().replace(" ", "")
    if re.fullmatch(r"\d+:\d+", text):
        return text
    if width <= 0 or height <= 0:
        return "9:16"
    value = width / height
    return min(_ASPECTS, key=lambda pair: abs(pair[0] - value))[1]


def _find_draft_file(draft_dir: Path) -> Path | None:
    if draft_dir.is_file():
        return draft_dir if draft_dir.name in DRAFT_FILENAMES else None
    for filename in DRAFT_FILENAMES:
        candidate = draft_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _read_draft_file(path: Path) -> tuple[DraftStatus, dict[str, Any] | None, str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        return DraftStatus.CORRUPT, None, f"cannot read {path.name}: {error}"
    if not raw.strip():
        return DraftStatus.CORRUPT, None, f"{path.name} is empty"
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    stripped = raw.lstrip()
    if not stripped.startswith(b"{"):
        return DraftStatus.ENCRYPTED, None, f"{path.name} is not plain JSON (encrypted?)"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        return DraftStatus.CORRUPT, None, f"{path.name} is not valid JSON: {error}"
    if not isinstance(payload, dict):
        return DraftStatus.CORRUPT, None, f"{path.name} is not a JSON object"
    return DraftStatus.OK, payload, ""


def detect(draft_dir: str | Path) -> DraftStatus:
    """Phát hiện trạng thái draft: ok / encrypted / corrupt / missing. Không raise."""
    try:
        path = Path(draft_dir)
        if not path.exists():
            return DraftStatus.MISSING
        draft_file = _find_draft_file(path)
        if draft_file is None:
            return DraftStatus.MISSING
        status, _, _ = _read_draft_file(draft_file)
        return status
    except OSError:
        return DraftStatus.MISSING
    except Exception:
        logger.exception("unexpected error while detecting draft %s", draft_dir)
        return DraftStatus.CORRUPT


def _split_canvas(raw: dict[str, Any]) -> tuple[int, int, str]:
    canvas = _as_dict(raw.get("canvas_config") or raw.get("canvas"))
    width = _as_int(canvas.get("width") or raw.get("canvas_width") or raw.get("width"), 1080)
    height = _as_int(canvas.get("height") or raw.get("canvas_height") or raw.get("height"), 1920)
    ratio = _as_text(canvas.get("ratio") or raw.get("ratio"))
    if width <= 0:
        width = 1080
    if height <= 0:
        height = 1920
    return width, height, _aspect_from_size(width, height, ratio)


def load(draft_dir: str | Path) -> CapCutDraft:
    """Đọc draft thành model tối thiểu. Raise ``DraftError`` nếu không đọc được."""
    path = Path(draft_dir)
    draft_file = _find_draft_file(path)
    if draft_file is None:
        raise DraftError(DraftStatus.MISSING, f"không tìm thấy draft trong {path}")
    status, payload, message = _read_draft_file(draft_file)
    if status is not DraftStatus.OK or payload is None:
        raise DraftError(status, message)

    warnings: list[str] = []
    materials = _parse_materials(payload.get("materials"))
    indexes = {
        "videos": _index_by_id(materials.videos),
        "images": _index_by_id(materials.images),
        "audios": _index_by_id(materials.audios),
        "texts": _index_by_id(materials.texts),
        "effects": _index_by_id(materials.effects),
        "transitions": _index_by_id(materials.transitions),
        "stickers": _index_by_id(materials.stickers),
        "filters": _index_by_id(materials.filters),
    }
    tracks = [
        _parse_track(track, indexes, warnings)
        for track in _as_list(payload.get("tracks"))
        if isinstance(track, dict)
    ]
    width, height, aspect = _split_canvas(payload)
    duration = _seconds(payload.get("duration"))
    if duration <= 0:
        duration = max((segment.end for track in tracks for segment in track.segments), default=0.0)
    fps = _as_number(payload.get("fps") or _as_dict(payload.get("canvas_config")).get("fps"), 30.0)
    if fps <= 0:
        fps = 30.0
    name = _as_text(payload.get("name") or payload.get("draft_name"))
    if not name.strip():
        name = path.name if path.is_dir() else path.parent.name
    name = name.strip() or "draft"

    return CapCutDraft(
        name=name,
        draft_dir=str(path),
        draft_file=str(draft_file),
        aspect=aspect,
        canvas_width=width,
        canvas_height=height,
        fps=fps,
        duration=duration,
        tracks=tracks,
        materials=materials,
        warnings=warnings,
    )


def draft_file_of(draft_dir: str | Path) -> Path | None:
    """Trả về file draft được chọn, nếu có."""
    return _find_draft_file(Path(draft_dir))


def iter_draft_dirs(samples_dir: str | Path) -> list[Path]:
    """Liệt kê các thư mục con (và chính nó) có file draft."""
    root = Path(samples_dir)
    if not root.is_dir():
        return []
    candidates: list[Path] = []
    if _find_draft_file(root) is not None:
        candidates.append(root)
    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if child.is_dir() and _find_draft_file(child) is not None:
            candidates.append(child)
    return candidates
