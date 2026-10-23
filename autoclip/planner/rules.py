"""Planner không LLM: chọn theo tag + hook_score, chữ lấy từ lời thoại."""

from __future__ import annotations

import random

from autoclip.contracts import RecipeSpec, RenderPlan, SegmentInfo, SlotSpec
from autoclip.diversity import expand_slots, make_variant
from autoclip.planner.base import PlanError, Pool, assemble, want_duration


def pick_segment(
    rng: random.Random, free: list[SegmentInfo], want: float, tags: list[str]
) -> SegmentInfo:
    """Ưu tiên đủ dài; slot có tag → trong nhóm khớp tag, lấy ngẫu nhiên trong top hook."""
    if not free:
        raise PlanError("hết segment chưa dùng — ingest thêm clip nguồn")
    long_enough = [s for s in free if s.end - s.start >= want] or free
    tagged = [s for s in long_enough if set(tags) & set(s.tags)] if tags else []
    candidates = tagged or long_enough
    if tags:
        candidates = sorted(candidates, key=lambda s: -s.hook_score)
        candidates = candidates[: max(3, len(candidates) // 3)]
    return rng.choice(candidates)


def _length(picks: list[tuple[SlotSpec, SegmentInfo, float]]) -> float:
    """Ước lượng thời lượng output (tốc độ ~1)."""
    return sum(min(want, seg.end - seg.start) for _, seg, want in picks)


def first_words(text: str | None, n: int) -> str:
    return " ".join((text or "").split()[:n])


class RulePlanner:
    name = "rules"
    available = True

    def pick(
        self,
        spec: RecipeSpec,
        pool: Pool,
        seed: int,
        exclude: set[int],
    ) -> list[tuple[SlotSpec, SegmentInfo, float]]:
        rng = random.Random(f"picks:{seed}")
        free = pool.usable(exclude)
        picks = []
        for slot in expand_slots(spec, rng):
            want = want_duration(slot, rng)
            seg = pick_segment(rng, free, want, slot.want_tags)
            free = [s for s in free if s.id != seg.id]
            picks.append((slot, seg, want))
        # Segment ngắn hơn mong muốn → video hụt; chèn thêm body trước outro cho đủ.
        body = next((s for s in spec.slots if s.role == "body"), None)
        low = spec.duration_target[0]
        while body and _length(picks) < low + 1.0:  # dư 1s vì tốc độ có thể > 1
            want = want_duration(body, rng)
            seg = pick_segment(rng, free, want, body.want_tags)
            free = [s for s in free if s.id != seg.id]
            at = len(picks) - 1 if picks and picks[-1][0].role == "outro" else len(picks)
            picks.insert(at, (body, seg, want))
        return picks

    def plan(self, spec: RecipeSpec, pool: Pool, seed: int, exclude: set[int]) -> RenderPlan:
        picks = self.pick(spec, pool, seed, exclude)
        hook_text = next((seg.transcript for _, seg, _ in picks if seg.transcript), None)
        return assemble(
            spec,
            picks,
            pool,
            make_variant(spec, seed),
            random.Random(f"assemble:{seed}"),
            headline=first_words(hook_text, 5),
            title=first_words(" ".join(s.transcript or "" for _, s, _ in picks), 25),
            planner=self.name,
        )
