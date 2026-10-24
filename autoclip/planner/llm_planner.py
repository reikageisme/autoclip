"""Planner dùng LLM: chọn segment cho slot + viết headline và tiêu đề bài đăng.

Số slot và độ dài vẫn do seed quyết (như RulePlanner) để video luôn hợp recipe; LLM
chỉ chọn segment nào vào slot nào trong danh sách ứng viên và viết chữ. Trả lời được
cache theo hash prompt (data/llm_cache) → cùng seed + cùng kho → cùng plan.
Mọi lỗi (mạng, JSON sai, id lạ, trùng) → dùng lựa chọn của RulePlanner, ghi lý do.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

from autoclip.config import Settings, get_settings
from autoclip.contracts import RecipeSpec, RenderPlan, SegmentInfo, SlotSpec
from autoclip.diversity import Variant, make_variant
from autoclip.llm import LLMClient, LLMError
from autoclip.planner.base import Pool, assemble
from autoclip.planner.rules import RulePlanner, rule_headline
from autoclip.planner.vi_text import vi_ratio

log = logging.getLogger(__name__)

MAX_CANDIDATES = 40
MAX_TOKENS = 4000
TEXT_CHARS = 90

PROMPT = """Bạn là biên tập viên video ngắn (TikTok/Reels) cho một doanh nghiệp Việt Nam.
Dựng 1 video theo các slot dưới đây, theo đúng thứ tự. Với mỗi slot chọn đúng 1 đoạn
trong danh sách ỨNG VIÊN (theo id), không dùng 1 id 2 lần.
- Slot "hook" mở đầu phải gây chú ý ngay (ưu tiên đoạn có lời nói hấp dẫn, hook cao).
- Các slot "body" nên kể thành mạch, tránh 2 đoạn liền nhau quá giống nhau.
- Slot "outro" nên là lời kêu gọi / thông tin liên hệ nếu có.
Viết thêm:
- "headline": tối đa 6 từ tiếng Việt, đặt ở đầu video, góc nội dung: {angle}.
  Chỉ dùng thông tin có trong lời thoại, không bịa con số hay cam kết.
- "title": mô tả bài đăng tối đa 120 ký tự, kèm 2-3 hashtag.

SLOT: {slots}

ỨNG VIÊN (id, giây, tags, hook 0-1, lời thoại):
{candidates}

Chỉ trả về JSON: {{"picks": [id, ...], "headline": "...", "title": "..."}}"""


def _candidate_line(seg: SegmentInfo) -> str:
    text = seg.transcript if vi_ratio(seg.transcript) >= 0.6 else ""  # bỏ chữ Whisper đoán bừa
    text = " ".join((text or "").split())[:TEXT_CHARS] or "-"
    tags = ",".join(seg.tags) or "-"
    return f"{seg.id} | {seg.end - seg.start:.1f}s | {tags} | {seg.hook_score:.2f} | {text}"


def _parse(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"không có JSON: {text[:200]!r}")
    return json.loads(m.group(0))


class LLMPlanner:
    name = "llm"

    def __init__(self, llm: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient.from_settings(self.settings)
        self.rules = RulePlanner()
        self.cache_dir = self.settings.data_dir / "llm_cache"

    @property
    def available(self) -> bool:
        return self.llm.available

    def _cache_path(self, prompt: str) -> Path:
        key = hashlib.sha1(f"{self.llm.model}\n{prompt}".encode()).hexdigest()
        return self.cache_dir / f"{key}.txt"

    def _ask(self, prompt: str) -> tuple[str, bool]:
        """(câu trả lời, lấy từ cache?)."""
        cached = self._cache_path(prompt)
        if cached.is_file() and cached.stat().st_size:
            return cached.read_text(encoding="utf-8"), True
        # model suy luận cần chỗ cho phần reasoning trước câu trả lời
        return self.llm.chat(prompt, max_tokens=MAX_TOKENS), False

    def _remember(self, prompt: str, answer: str) -> None:
        """Chỉ cache câu trả lời đã qua kiểm tra — câu lỗi phải được hỏi lại lần sau."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(prompt).write_text(answer, encoding="utf-8")

    def _candidates(
        self,
        base: list[tuple[SlotSpec, SegmentInfo, float]],
        pool: Pool,
        exclude: set[int],
        seed: int,
    ) -> list[SegmentInfo]:
        """Ứng viên: lựa chọn của RulePlanner + phần còn lại của kho, xáo theo seed."""
        rng = random.Random(f"cands:{seed}")
        rest = [s for s in pool.usable(exclude) if s.id not in {seg.id for _, seg, _ in base}]
        rng.shuffle(rest)
        out = [seg for _, seg, _ in base] + rest
        out = out[:MAX_CANDIDATES]
        rng.shuffle(out)
        return out

    def plan(self, spec: RecipeSpec, pool: Pool, seed: int, exclude: set[int]) -> RenderPlan:
        base = self.rules.pick(spec, pool, seed, exclude)  # số slot + độ dài theo seed
        variant: Variant = make_variant(spec, seed)
        cands = self._candidates(base, pool, exclude, seed)
        by_id = {s.id: s for s in cands}
        slots_desc = ", ".join(
            f"{i + 1}.{slot.role}(~{want:.1f}s)" for i, (slot, _, want) in enumerate(base)
        )
        prompt = PROMPT.format(
            angle=variant.angle,
            slots=slots_desc,
            candidates="\n".join(_candidate_line(s) for s in cands),
        )

        picks = base
        headline = rule_headline([s for _, s, _ in base], seed)
        title = ""
        used = "rules(fallback)"
        try:
            answer, from_cache = self._ask(prompt)
            data = _parse(answer)
            ids = [int(i) for i in data.get("picks", [])]
            if len(ids) != len(base):
                raise ValueError(f"cần {len(base)} id, nhận {len(ids)}")
            if len(set(ids)) != len(ids) or any(i not in by_id for i in ids):
                raise ValueError(f"id trùng hoặc ngoài danh sách: {ids}")
            picks = [(slot, by_id[i], want) for (slot, _, want), i in zip(base, ids, strict=True)]
            headline = str(data.get("headline") or headline)
            title = str(data.get("title") or "")
            used = self.name
            if not from_cache:
                self._remember(prompt, answer)
        except (LLMError, ValueError, TypeError, json.JSONDecodeError) as exc:
            log.warning("LLM planner lỗi, dùng rules (seed %s): %s", seed, exc)

        return assemble(
            spec,
            picks,
            pool,
            variant,
            random.Random(f"assemble:{seed}"),
            headline=headline,
            title=title,
            planner=used,
        )
