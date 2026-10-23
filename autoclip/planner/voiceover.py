"""Planner chế độ lồng tiếng: mỗi câu kịch bản = một cảnh dài đúng bằng câu đọc.

Tiếng gốc của clip tắt; giọng đọc (WAV đã tổng hợp) nằm trên lane riêng; phụ đề chạy
theo đúng câu đang đọc. Ưu tiên đoạn không có lời nói (tiếng gốc đằng nào cũng tắt).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from autoclip.contracts import RecipeSpec, RenderItem, RenderPlan, RenderText, SegmentInfo
from autoclip.diversity import make_variant
from autoclip.planner.base import (
    PlanError,
    Pool,
    _caption_chunks,
    caption_y,
    clean_headline,
    focus_of,
)
from autoclip.vision_tag import line_scenes, scene_tags

MIN_SPEED = 0.7


@dataclass
class Line:
    text: str
    audio: Path
    seconds: float


def _choose(
    rng: random.Random, free: list[SegmentInfo], need: float, wanted: set[str] | None = None
) -> SegmentInfo:
    """Đoạn đủ dài cho câu đọc; câu nhắc tới cảnh nào (nhãn AI) thì ưu tiên cảnh đó."""
    if not free:
        raise PlanError("hết segment chưa dùng — ingest thêm clip nguồn")
    fits = [s for s in free if s.end - s.start >= need]
    close = [s for s in free if s.end - s.start >= need * MIN_SPEED]
    pool = fits or close or sorted(free, key=lambda s: s.start - s.end)[:5]
    if wanted:
        matched = [s for s in pool if scene_tags(s.tags) & wanted]
        pool = matched or pool
    quiet = [s for s in pool if not (s.transcript or "").strip()]
    return rng.choice(quiet or pool)


def plan_voiceover(
    spec: RecipeSpec,
    pool: Pool,
    seed: int,
    exclude: set[int],
    lines: list[Line],
    *,
    headline: str = "",
    title: str = "",
) -> RenderPlan:
    if not lines:
        raise ValueError("kịch bản trống")
    rng = random.Random(f"voice:{seed}")
    variant = make_variant(spec, seed)
    body = next((s for s in spec.slots if s.role == "body"), None)
    transitions = list(body.transition) if body and body.transition else []
    free = pool.usable(exclude)

    items: list[RenderItem] = []
    texts: list[RenderText] = []
    voice: list[dict] = []
    cursor = 0.0
    for i, line in enumerate(lines):
        seg = _choose(rng, free, line.seconds, line_scenes(line.text))
        free = [s for s in free if s.id != seg.id]
        seg_len = seg.end - seg.start
        if seg_len >= line.seconds:
            start = seg.start + rng.uniform(0, seg_len - line.seconds)
            src = (round(start, 3), round(start + line.seconds, 3))
            speed = 1.0
        else:  # đoạn hơi ngắn: chậm lại cho vừa câu đọc
            src = (round(seg.start, 3), round(seg.end, 3))
            speed = round(max(MIN_SPEED, seg_len / line.seconds), 3)
        clip = pool.clips[seg.clip_id]
        role = "hook" if i == 0 else "outro" if i == len(lines) - 1 else "body"
        transition = rng.choice(transitions) if transitions and i > 0 else None
        items.append(
            RenderItem(
                slot_role=role,
                segment_id=seg.id,
                clip_path=clip.path,
                src_start=src[0],
                src_end=src[1],
                media_duration=clip.duration,
                media_width=clip.width,
                media_height=clip.height,
                focus_x=focus_of(seg),
                speed=speed,
                transition=None if transition == "cut" else transition,
            )
        )
        length = (src[1] - src[0]) / speed
        voice.append({"path": str(line.audio), "start": round(cursor, 3)})
        for t in _caption_chunks(line.text, cursor, cursor + length, per=10):
            t.pos = (0.5, caption_y(seg, variant.caption_y))
            texts.append(t)
        cursor += length

    head = clean_headline(headline)
    heads = [t for t in spec.texts if t.kind != "captions"]
    if head and heads:
        texts.append(
            RenderText(
                kind=heads[0].kind,
                text=head,
                slot=heads[0].slot,
                style=heads[0].style,
                pos=(0.5, variant.headline_y),
                start=0.0,
                end=round(min(2.8, cursor), 3),
            )
        )
    return RenderPlan(
        recipe=spec.name,
        seed=seed,
        items=items,
        texts=texts,
        params={
            "palette": [variant.color] if variant.color else [],
            "planner": "voiceover",
            "angle": variant.angle,
            "title": title,
            "mute_source": True,
            "voiceover": voice,
            "script": [line.text for line in lines],
        },
    )
