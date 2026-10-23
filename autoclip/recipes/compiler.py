"""Biên dịch ``RecipeSpec`` + ``RenderPlan`` thành danh sách lệnh Concat.

Hàm ``compile_plan`` là hàm thuần: không I/O, không gọi Concat, không đọc file.
Mọi thời gian là giây; vị trí/kích thước text là tỉ lệ 0–1 của khung hình.

Quy ước thực thi (``RenderPlan`` không biết id mà Concat sẽ sinh ra):

- Tham chiếu là chuỗi giữ chỗ, executor (``autoclip.concat_client.executor``) thay
  bằng id thật: ``@media:<clip_path>`` → ``createdId`` của ``media_import`` file đó;
  ``@clip:<i>`` → clip thứ ``i`` của plan.
- Khoá bắt đầu bằng ``@`` trong ``args`` là chỉ thị cho executor, không gửi tới
  Concat: ``"@as": "clip:<i>"`` ghi ``createdId`` của lệnh vào tham chiếu đó.
- ``params["project_path"]`` (tuỳ chọn) được chèn vào mọi lệnh dưới khoá ``path``;
  không có thì executor tự chèn.

Mỗi item được đặt riêng tại con trỏ timeline (``clips_add_sequence`` một media),
cắt đầu theo ``src_start``, cắt đuôi theo ``src_end`` (cần ``media_duration``),
rồi mới đổi tốc độ — nên clip sau luôn bắt đầu đúng chỗ clip trước kết thúc.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from autoclip.contracts import ConcatCommand, RecipeSpec, RenderItem, RenderPlan

from . import effect_map

_ASPECT_SIZES: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "3:4": (1080, 1440),
    "21:9": (1920, 823),
}

_STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "plain": {"fontWeight": 400},
    "bold": {"fontWeight": 800},
    "stroke": {"fontWeight": 500, "strokeWidth": 0.008, "strokeColor": "#000000"},
    "bold-stroke": {
        "fontWeight": 800,
        "fontSize": 0.055,
        "maxWidth": 0.86,
        "strokeWidth": 0.008,
        "strokeColor": "#000000",
    },
    "caption": {"fontSize": 0.04, "fontWeight": 600},
    "word-pop": {
        "fontSize": 0.05,
        "maxWidth": 0.86,
        "fontWeight": 800,
        "strokeWidth": 0.006,
        "strokeColor": "#000000",
    },
}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _params(plan: RenderPlan) -> dict[str, Any]:
    params = _get(plan, "params", {})
    return params if isinstance(params, dict) else {}


def _video_size(aspect: str, params: dict[str, Any]) -> tuple[int, int]:
    width = params.get("width")
    height = params.get("height")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height
    key = str(aspect or "9:16").replace(" ", "")
    if key in _ASPECT_SIZES:
        return _ASPECT_SIZES[key]
    if ":" in key:
        try:
            left, right = (int(part) for part in key.split(":", 1))
        except (TypeError, ValueError):
            left = right = 0
        if left > 0 and right > 0:
            if right >= left:
                return 1080, round(1080 * right / left)
            return 1920, round(1920 * right / left)
    return _ASPECT_SIZES["9:16"]


def _fps_pair(value: Any) -> tuple[int, int]:
    try:
        rate = Fraction(float(value)).limit_denominator(100_000)
    except (TypeError, ValueError, ZeroDivisionError):
        return 30, 1
    if rate <= 0:
        return 30, 1
    return rate.numerator, rate.denominator


def _command(tool: str, args: dict[str, Any], params: dict[str, Any]) -> ConcatCommand:
    payload = dict(args)
    project_path = params.get("project_path") or params.get("path")
    if project_path:
        payload["path"] = str(project_path)
    return ConcatCommand(tool=tool, args=payload)


def _fill_scale(item: RenderItem, width: int, height: int) -> float:
    """Hệ số phóng để nguồn khác tỉ lệ phủ kín khung (scale 1 = vừa khung, có viền)."""
    w, h = _get(item, "media_width"), _get(item, "media_height")
    if not w or not h:
        return 1.0
    src, frame = w / h, width / height
    return max(src / frame, frame / src)


def _timeline_duration(item: RenderItem) -> float:
    speed = float(_get(item, "speed", 1.0) or 1.0)
    if speed <= 0:
        speed = 1.0
    source = float(_get(item, "src_end", 0.0)) - float(_get(item, "src_start", 0.0))
    if source <= 0:
        raise ValueError(f"RenderItem {_get(item, 'segment_id')!r} has src_end <= src_start")
    return source / speed


def planned_duration(plan: RenderPlan) -> float:
    """Tổng thời lượng timeline mà plan mong đợi, tính từ src range + speed."""
    return round(sum(_timeline_duration(item) for item in plan.items), 6)


def _timeline_starts(items: list[RenderItem], start: float = 0.0) -> list[float]:
    starts: list[float] = []
    cursor = start
    for item in items:
        starts.append(cursor)
        cursor += _timeline_duration(item)
    return starts


def _transition_for(
    spec: RecipeSpec,
    item: RenderItem,
    role_indexes: dict[str, int],
) -> str | None:
    explicit = _get(item, "transition")
    if explicit:
        return effect_map.transition_id(str(explicit)) or None
    role = str(_get(item, "slot_role", ""))
    options: list[str] = []
    for slot in spec.slots:
        if slot.role == role:
            options = list(slot.transition)
            break
    if not options:
        return None
    index = role_indexes.get(role, 0)
    role_indexes[role] = index + 1
    return effect_map.transition_id(options[index % len(options)]) or None


def _text_style(
    text: Any,
    spec: RecipeSpec,
    params: dict[str, Any],
) -> dict[str, Any]:
    style_name = str(_get(text, "style", "") or "plain")
    style = dict(_STYLE_PRESETS.get(style_name, _STYLE_PRESETS["plain"]))
    fonts = params.get("fonts") or spec.variation.fonts
    if isinstance(fonts, str):
        fonts = [fonts]
    font = fonts[0] if isinstance(fonts, list) and fonts else None
    family = effect_map.font_family(str(font)) if font else None
    if family:
        style["fontFamily"] = family
    palette = params.get("palette") or spec.variation.palette
    if isinstance(palette, str):
        palette = [palette]
    if isinstance(palette, list) and palette:
        style["color"] = str(palette[0])
    return style


def _offset_y(text: Any, default: float) -> float:
    pos = _get(text, "pos")
    if isinstance(pos, (tuple, list)) and len(pos) >= 2:
        try:
            return round(max(-3.0, min(3.0, float(pos[1]) - 0.5)), 4)
        except (TypeError, ValueError):
            return default
    return default


def _text_window(
    text: Any,
    index: int,
    count: int,
    text_start: float,
    text_end: float,
    default_duration: float,
) -> tuple[float, float]:
    start = _get(text, "start")
    end = _get(text, "end")
    if start is None:
        start = text_start + (text_end - text_start) * index / max(1, count)
    start = max(0.0, float(start))
    if end is None:
        if count > 1:
            end = text_start + (text_end - text_start) * (index + 1) / count
        else:
            end = start + default_duration
    end = float(end)
    if end <= start:
        end = start + 0.5
    return round(start, 3), round(end, 3)


def compile_plan(spec: RecipeSpec, plan: RenderPlan) -> list[ConcatCommand]:
    """Sinh danh sách ``ConcatCommand`` theo thứ tự thực thi."""
    items = list(plan.items)
    if not items:
        raise ValueError("RenderPlan has no items")
    params = _params(plan)
    width, height = _video_size(spec.aspect, params)
    rate_num, rate_den = _fps_pair(params.get("fps", 30))
    commands: list[ConcatCommand] = [
        _command(
            "project_set_video",
            {
                "video": {
                    "width": width,
                    "height": height,
                    "rateNum": rate_num,
                    "rateDen": rate_den,
                }
            },
            params,
        )
    ]

    seen_paths: set[str] = set()
    for item in items:
        path = str(_get(item, "clip_path", ""))
        if path and path not in seen_paths:
            seen_paths.add(path)
            commands.append(_command("media_import", {"file": path}, params))

    role_indexes: dict[str, int] = {}
    cursor = float(params.get("start", 0.0) or 0.0)
    for index, item in enumerate(items):
        clip_ref = f"@clip:{index}"
        src_start = float(_get(item, "src_start", 0.0) or 0.0)
        src_end = float(_get(item, "src_end", 0.0) or 0.0)
        media_duration = _get(item, "media_duration")
        if media_duration is None:
            raise ValueError(f"RenderItem {index} thiếu media_duration (cần để cắt đuôi)")
        media_duration = float(media_duration)
        if src_end > media_duration + 0.01:
            raise ValueError(f"RenderItem {index}: src_end {src_end} > media {media_duration}")

        commands.append(
            _command(
                "clips_add_sequence",
                {
                    "mediaIds": [f"@media:{_get(item, 'clip_path', '')}"],
                    "start": round(cursor, 3),
                    "@as": f"clip:{index}",
                },
                params,
            )
        )
        if src_start > 0.001:
            commands.append(
                _command(
                    "clip_trim",
                    {
                        "clipId": clip_ref,
                        "edge": "start",
                        "delta": round(src_start, 3),
                        "ripple": True,
                    },
                    params,
                )
            )
        if media_duration - src_end > 0.001:
            commands.append(
                _command(
                    "clip_trim",
                    {
                        "clipId": clip_ref,
                        "edge": "end",
                        "delta": round(src_end - media_duration, 3),
                    },
                    params,
                )
            )
        fill = _fill_scale(item, width, height) if params.get("fit", "fill") == "fill" else 1.0
        if fill > 1.001:
            commands.append(
                _command("clip_transform", {"clipId": clip_ref, "scale": round(fill, 4)}, params)
            )
        speed = float(_get(item, "speed", 1.0) or 1.0)
        if abs(speed - 1.0) > 0.001 and speed > 0:
            commands.append(
                _command("clip_speed", {"clipId": clip_ref, "speed": round(speed, 4)}, params)
            )
        transition = _transition_for(spec, item, role_indexes) if index > 0 else None
        if transition:
            duration = min(0.4, max(0.1, _timeline_duration(item) / 4.0))
            commands.append(
                _command(
                    "clip_update",
                    {
                        "clipId": clip_ref,
                        "patch": {
                            "transitionIn": {"id": transition, "duration": round(duration, 3)}
                        },
                    },
                    params,
                )
            )
            for frame in effect_map.keyframes_for(transition):
                commands.append(
                    _command(
                        "keyframe_set",
                        {
                            "clipId": clip_ref,
                            "property": frame["property"],
                            "at": frame["at"],
                            "value": frame["value"],
                        },
                        params,
                    )
                )
        cursor += _timeline_duration(item)

    effects = params.get("effects") or []
    if isinstance(effects, list):
        total = planned_duration(plan)
        for effect in effects:
            if not isinstance(effect, dict):
                continue
            effect_id = effect.get("effect_id") or effect.get("id")
            if not effect_id:
                continue
            start = max(0.0, float(effect.get("start", 0.0) or 0.0))
            duration = float(effect.get("duration", total) or total)
            commands.append(
                _command(
                    "effect_layer_add",
                    {
                        "effectId": str(effect_id),
                        "name": str(effect.get("name") or effect_id),
                        "start": round(start, 3),
                        "duration": round(max(0.1, duration), 3),
                    },
                    params,
                )
            )

    total = planned_duration(plan)
    captions = [text for text in plan.texts if _is_caption(text)]
    headlines = [text for text in plan.texts if not _is_caption(text)]

    for index, text in enumerate(headlines):
        content = str(_get(text, "text", "") or "")
        if not content.strip():
            continue
        start, end = _text_window(text, index, len(headlines), 0.0, total, 2.5)
        style = _text_style(text, spec, params)
        style["content"] = content
        commands.append(
            _command(
                "title_add",
                {
                    "start": start,
                    "duration": round(end - start, 3),
                    "style": style,
                    "offsetY": _offset_y(text, -0.3),
                    "above": True,
                },
                params,
            )
        )

    if captions:
        style = _text_style(captions[0], spec, params)
        caption_payload: list[dict[str, Any]] = []
        for index, text in enumerate(captions):
            content = str(_get(text, "text", "") or "")
            if not content.strip():
                continue
            start, end = _text_window(text, index, len(captions), 0.0, total, 1.5)
            caption_payload.append({"start": start, "end": end, "text": content})
        if caption_payload:
            commands.append(
                _command(
                    "captions_add",
                    {
                        "captions": caption_payload,
                        "style": style,
                        "offsetY": _offset_y(captions[0], 0.35),
                    },
                    params,
                )
            )

    commands.extend(_brand_commands(params, total, width, height))
    return commands


def _is_caption(text: Any) -> bool:
    kind = str(_get(text, "kind", "") or "").lower()
    style = str(_get(text, "style", "") or "").lower()
    return kind == "captions" or style in {"caption", "word-pop"}


_CORNERS = {
    "top-left": (-1, -1),
    "top-right": (1, -1),
    "bottom-left": (-1, 1),
    "bottom-right": (1, 1),
}


def _brand_commands(
    params: dict[str, Any], total: float, width: int, height: int
) -> list[ConcatCommand]:
    """Logo, cảnh kết CTA, nhạc nền từ ``params["brand"]`` (xem autoclip.brand)."""
    brand = params.get("brand") or {}
    out: list[ConcatCommand] = []
    outro = brand.get("outro")
    end = total + (float(outro["seconds"]) if outro else 0.0)

    if outro:
        # Sau cảnh cuối không còn hình: nền timeline hiện màu thương hiệu.
        out.append(
            _command(
                "timeline_background",
                {
                    "timelineId": "@timeline",
                    "background": {"kind": "color", "color": outro["background"]},
                },
                params,
            )
        )
        out.append(
            _command(
                "title_add",
                {
                    "start": round(total, 3),
                    "duration": round(float(outro["seconds"]), 3),
                    "style": {
                        "content": outro["text"],
                        "fontWeight": 800,
                        "fontSize": 0.04,
                        "maxWidth": 0.92,
                        "color": outro["color"],
                    },
                    "offsetY": 0.0,
                    "above": True,
                },
                params,
            )
        )

    logo = brand.get("logo")
    if logo:
        size = float(logo["size"])
        sx, sy = _CORNERS.get(logo["corner"], (1, -1))
        margin = 0.04
        # scale 1 = vừa khung (logo vuông vừa bề ngang): cao logo theo tỉ lệ khung
        half_w = size / 2
        half_h = size * (width / height) / 2
        out += [
            _command("media_import", {"file": logo["path"]}, params),
            _command("track_add", {"@as": "track:logo"}, params),
            _command(
                "clip_add",
                {
                    "mediaId": f"@media:{logo['path']}",
                    "trackId": "@track:logo",
                    "start": 0,
                    "@as": "clip:logo",
                },
                params,
            ),
            _command(
                "clip_trim",
                {"clipId": "@clip:logo", "edge": "end", "delta": round(end - 5.0, 3)},
                params,
            ),
            _command(
                "clip_transform",
                {
                    "clipId": "@clip:logo",
                    "scale": round(size, 4),
                    "offsetX": round(sx * (0.5 - half_w - margin), 4),
                    "offsetY": round(sy * (0.5 - half_h - margin), 4),
                },
                params,
            ),
            _command(
                "clip_update",
                {"clipId": "@clip:logo", "patch": {"opacity": float(logo["opacity"])}},
                params,
            ),
        ]

    music = brand.get("music")
    if music:
        length = float(music["duration"])
        out += [
            _command("media_import", {"file": music["path"]}, params),
            _command("track_add", {"bottom": True, "@as": "track:music"}, params),
            _command(
                "clip_add",
                {
                    "mediaId": f"@media:{music['path']}",
                    "trackId": "@track:music",
                    "start": 0,
                    "@as": "clip:music",
                },
                params,
            ),
        ]
        if length > end + 0.05:
            out.append(
                _command(
                    "clip_trim",
                    {"clipId": "@clip:music", "edge": "end", "delta": round(end - length, 3)},
                    params,
                )
            )
        out.append(
            _command(
                "clip_update",
                {
                    "clipId": "@clip:music",
                    "patch": {"volume": float(music["volume"]), "fadeIn": 0.5, "fadeOut": 1.5},
                },
                params,
            )
        )
    return out
