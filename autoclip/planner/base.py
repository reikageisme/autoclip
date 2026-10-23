"""Phần dùng chung của các planner: kho segment, cắt đoạn, dựng RenderPlan."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from autoclip.contracts import (
    ClipInfo,
    ClipRepository,
    RecipeSpec,
    RenderItem,
    RenderPlan,
    RenderText,
    SegmentInfo,
    SlotSpec,
)
from autoclip.diversity import Variant

MIN_ITEM_SEC = 0.8
CAPTION_WORDS = 8
"""Chữ mỗi phụ đề. Ít clip chữ hơn: Concat render nhiều clip chữ ngắn (≥~14 trong vài giây)
thì có lúc mất hình bên dưới (khung đen) — gộp phụ đề dài hơn tránh được."""
HEADLINE_MAX_CHARS = 40


class PlanError(ValueError):
    """Không lập được plan (thường là hết segment chưa dùng)."""


@dataclass
class Pool:
    segments: list[SegmentInfo]
    clips: dict[int, ClipInfo]

    def usable(self, exclude: set[int]) -> list[SegmentInfo]:
        return [s for s in self.segments if s.id not in exclude and s.end - s.start >= MIN_ITEM_SEC]


def load_pool(repo: ClipRepository) -> Pool:
    """Mọi segment có file nguồn còn tồn tại."""
    segments = repo.list_segments()
    clips: dict[int, ClipInfo] = {}
    for seg in segments:
        if seg.clip_id not in clips:
            clips[seg.clip_id] = repo.get_clip(seg.clip_id)
    ok = {cid for cid, c in clips.items() if c and Path(c.path).is_file()}
    return Pool([s for s in segments if s.clip_id in ok], {c: clips[c] for c in ok})


def want_duration(slot: SlotSpec, rng: random.Random) -> float:
    return rng.uniform(*slot.dur) if slot.dur else 2.5


def cut(seg: SegmentInfo, want: float, rng: random.Random) -> tuple[float, float]:
    """Đoạn nguồn [start, end] dài tối đa `want` trong segment.

    Có lời thoại thì lấy từ đầu segment để caption khớp tiếng (DB chưa lưu mốc từng từ).
    """
    seg_len = seg.end - seg.start
    length = min(want, seg_len)
    offset = 0.0 if seg.transcript else rng.uniform(0, seg_len - length)
    start = round(seg.start + offset, 3)
    return start, round(start + length, 3)


def _caption_chunks(
    text: str, start: float, end: float, per: int = CAPTION_WORDS
) -> list[RenderText]:
    words = text.split()
    if not words:
        return []
    chunks = [words[i : i + per] for i in range(0, len(words), per)]
    step = (end - start) / len(chunks)
    return [
        RenderText(
            kind="captions",
            text=" ".join(c),
            style="word-pop",
            start=round(start + i * step, 3),
            end=round(start + (i + 1) * step - 0.02, 3),
        )
        for i, c in enumerate(chunks)
    ]


def caption_y(seg: SegmentInfo, default: float) -> float:
    """Phụ đề né chữ in sẵn: chữ ở dưới → lên cao; chữ cả trên lẫn dưới → giữa khung."""
    tags = set(seg.tags)
    if "text_bottom" in tags:
        return 0.5 if "text_top" in tags else 0.3
    return default


def focus_of(seg: SegmentInfo) -> float | None:
    for tag in seg.tags:
        if tag.startswith("fx:"):
            try:
                return float(tag[3:])
            except ValueError:
                return None
    return None


def clean_headline(text: str | None) -> str:
    """Headline 1 dòng, viết hoa, cắt theo từ cho vừa HEADLINE_MAX_CHARS."""
    words = " ".join((text or "").replace("\n", " ").split()).strip(" \"'“”").split()
    out: list[str] = []
    for w in words:
        if len(" ".join([*out, w])) > HEADLINE_MAX_CHARS:
            break
        out.append(w)
    return " ".join(out).upper()


def assemble(
    spec: RecipeSpec,
    picks: list[tuple[SlotSpec, SegmentInfo, float]],
    pool: Pool,
    variant: Variant,
    rng: random.Random,
    *,
    headline: str | None,
    title: str | None,
    planner: str,
) -> RenderPlan:
    """(slot, segment, độ dài mong muốn) theo thứ tự → RenderPlan hoàn chỉnh."""
    items: list[RenderItem] = []
    texts: list[RenderText] = []
    want_captions = any(t.kind == "captions" for t in spec.texts)
    cursor = 0.0
    for slot, seg, want in picks:
        src_start, src_end = cut(seg, want, rng)
        speed = rng.uniform(*slot.speed) if slot.speed else 1.0
        if slot.role == "body":
            speed *= variant.speed_jitter
        speed = round(speed, 3)
        transition = rng.choice(slot.transition) if slot.transition else None
        clip = pool.clips[seg.clip_id]
        items.append(
            RenderItem(
                slot_role=slot.role,
                segment_id=seg.id,
                clip_path=clip.path,
                src_start=src_start,
                src_end=src_end,
                media_duration=clip.duration,
                media_width=clip.width,
                media_height=clip.height,
                focus_x=focus_of(seg),
                speed=speed,
                transition=None if transition == "cut" else transition,
            )
        )
        out_len = (src_end - src_start) / speed
        if want_captions and seg.transcript:
            words = seg.transcript.split()
            share = (src_end - src_start) / (seg.end - seg.start)
            spoken = " ".join(words[: max(1, round(len(words) * share))])
            for t in _caption_chunks(spoken, cursor, cursor + out_len):
                t.pos = (0.5, caption_y(seg, variant.caption_y))
                texts.append(t)
        cursor += out_len

    headline = clean_headline(headline)
    for t in spec.texts:
        if t.kind != "captions" and headline:
            texts.append(
                RenderText(
                    kind=t.kind,
                    text=headline,
                    slot=t.slot,
                    style=t.style,
                    pos=(0.5, variant.headline_y),
                    start=0.0,
                    end=round(min(2.8, cursor), 3),
                )
            )

    return RenderPlan(
        recipe=spec.name,
        seed=variant.seed,
        items=items,
        texts=texts,
        params={
            "palette": [variant.color] if variant.color else [],
            "planner": planner,
            "angle": variant.angle,
            "title": (title or "").strip(),
        },
    )
