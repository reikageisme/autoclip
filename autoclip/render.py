"""End-to-end thô (P2): recipe + segment trong DB → RenderPlan → compiler → Concat → MP4.

`build_plan` là planner tối giản, không LLM: chọn segment ngẫu nhiên theo seed (tái lập
được), ưu tiên tag slot muốn, không dùng lại segment trong một video. Planner thật
(LLM viết hook/caption, chống lặp qua nhiều render) là việc của P3.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from autoclip.concat_client import SyncConcatClient
from autoclip.concat_client.executor import execute, export_mp4
from autoclip.config import Settings, get_settings
from autoclip.contracts import (
    ClipRepository,
    RecipeSpec,
    RenderItem,
    RenderPlan,
    RenderText,
    SegmentInfo,
)
from autoclip.ingest import probe
from autoclip.recipes.compiler import compile_plan

log = logging.getLogger(__name__)

MIN_ITEM_SEC = 0.8
CAPTION_WORDS = 4


class PlanError(ValueError):
    pass


def load_recipe(path: Path) -> RecipeSpec:
    return RecipeSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _pick(
    rng: random.Random,
    pool: list[SegmentInfo],
    used: set[int],
    want: float,
    tags: list[str],
) -> SegmentInfo:
    free = [s for s in pool if s.id not in used and s.end - s.start >= MIN_ITEM_SEC]
    if not free:
        raise PlanError("hết segment chưa dùng — ingest thêm clip")
    long_enough = [s for s in free if s.end - s.start >= want] or free
    tagged = [s for s in long_enough if set(tags) & set(s.tags)] if tags else []
    candidates = tagged or long_enough
    if tags:
        candidates.sort(key=lambda s: -s.hook_score)
        candidates = candidates[: max(3, len(candidates) // 3)]
    return rng.choice(candidates)


def _caption_chunks(text: str, start: float, end: float) -> list[RenderText]:
    words = text.split()
    if not words:
        return []
    chunks = [words[i : i + CAPTION_WORDS] for i in range(0, len(words), CAPTION_WORDS)]
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


def build_plan(spec: RecipeSpec, repo: ClipRepository, seed: int) -> RenderPlan:
    rng = random.Random(seed)
    pool = repo.list_segments()
    clips = {}
    for seg in pool:
        if seg.clip_id not in clips:
            clips[seg.clip_id] = repo.get_clip(seg.clip_id)
    pool = [s for s in pool if clips.get(s.clip_id) and Path(clips[s.clip_id].path).is_file()]
    if not pool:
        raise PlanError("DB chưa có segment nào của file còn tồn tại — chạy `autoclip ingest`")

    slots = []
    for slot in spec.slots:
        count = rng.randint(*slot.repeat) if slot.repeat else 1
        slots.extend([slot] * count)
    used: set[int] = set()
    items: list[RenderItem] = []
    texts: list[RenderText] = []
    cursor = 0.0
    for slot in slots:
        want = rng.uniform(*slot.dur) if slot.dur else 2.5
        seg = _pick(rng, pool, used, want, slot.want_tags)
        used.add(seg.id)
        seg_len = seg.end - seg.start
        length = min(want, seg_len)
        # Có lời: lấy từ đầu segment để caption khớp tiếng (chưa lưu mốc thời gian từng từ).
        offset = 0.0 if seg.transcript else rng.uniform(0, seg_len - length)
        src_start = round(seg.start + offset, 3)
        src_end = round(src_start + length, 3)
        speed = round(rng.uniform(*slot.speed), 3) if slot.speed else 1.0
        transition = rng.choice(slot.transition) if slot.transition else None
        clip = clips[seg.clip_id]
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
                speed=speed,
                transition=None if transition == "cut" else transition,
            )
        )
        out_len = length / speed
        if seg.transcript and any(t.kind == "captions" for t in spec.texts):
            words = seg.transcript.split()
            spoken = " ".join(words[: max(1, round(len(words) * length / seg_len))])
            texts.extend(_caption_chunks(spoken, cursor, cursor + out_len))
        cursor += out_len

    for t in spec.texts:
        if t.kind == "captions":
            continue
        hook = next((s for s in pool if s.id in used and s.transcript), None)
        words = (hook.transcript or "").split()[:4] if hook else []
        if words:
            texts.append(
                RenderText(
                    kind=t.kind,
                    text=" ".join(words).upper(),
                    slot=t.slot,
                    style=t.style,
                    pos=t.pos,
                    start=0.0,
                    end=round(min(2.5, cursor), 3),
                )
            )

    palette = spec.variation.palette
    return RenderPlan(
        recipe=spec.name,
        seed=seed,
        items=items,
        texts=texts,
        params={"palette": [rng.choice(palette)] if palette else []},
    )


@dataclass
class RenderResult:
    output: Path
    plan: RenderPlan
    video_sec: float
    render_sec: float
    commands: int


def render(
    spec: RecipeSpec,
    plan: RenderPlan,
    *,
    settings: Settings | None = None,
    name: str | None = None,
) -> RenderResult:
    s = settings or get_settings()
    name = name or f"{spec.name}-s{plan.seed}-{datetime.now():%Y%m%d-%H%M%S}"
    s.projects_dir.mkdir(parents=True, exist_ok=True)
    s.out_dir.mkdir(parents=True, exist_ok=True)
    project = str(s.projects_dir / name)
    output = s.out_dir / f"{name}.mp4"
    commands = compile_plan(spec, plan)

    with SyncConcatClient(s) as cc:
        cc.call_tool("project_create", {"location": str(s.projects_dir), "name": name})
        try:
            execute(cc, commands, project)
            cc.call_tool("project_save", {"path": project})
            t0 = time.perf_counter()
            export_mp4(cc, project, str(output))
            render_sec = time.perf_counter() - t0
        finally:
            try:
                cc.call_tool("project_close", {"path": project, "save": False})
            except Exception as exc:  # đóng project lỗi không được che lỗi thật
                log.warning("project_close: %s", exc)

    return RenderResult(
        output=output,
        plan=plan,
        video_sec=probe(output).duration,
        render_sec=render_sec,
        commands=len(commands),
    )
