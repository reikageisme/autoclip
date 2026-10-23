"""Đa dạng hoá theo seed + chống lặp segment qua các render gần đây.

- `make_variant`: mọi lựa chọn "trang trí" của một video (màu, vị trí chữ, nhịp, góc
  nội dung) suy ra từ seed → cùng seed cùng kết quả, seed khác ra video khác rõ rệt.
- `recent_segment_ids`: segment đã dùng trong N render gần nhất (bảng usage_log), để
  planner loại ra.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from sqlalchemy import Engine, select

from autoclip.contracts import RecipeSpec, SlotSpec
from autoclip.db.models import Render, UsageLog

# Góc nội dung gợi ý cho LLM viết headline — mỗi seed một góc để headline không na ná.
ANGLES = [
    "tốc độ giao hàng",
    "chi phí tiết kiệm",
    "uy tín và an toàn hàng hoá",
    "quy trình chuyên nghiệp",
    "dịch vụ trọn gói",
    "thời điểm nhập hàng",
    "câu hỏi gây tò mò",
    "con số cụ thể",
]
CAPTION_Y = [0.72, 0.76, 0.8]
HEADLINE_Y = [0.16, 0.2, 0.24]


@dataclass(frozen=True)
class Variant:
    seed: int
    color: str | None
    caption_y: float
    headline_y: float
    angle: str
    speed_jitter: float
    """Nhân thêm vào tốc độ body (0.95–1.1) để nhịp mỗi video khác nhau."""


def make_variant(spec: RecipeSpec, seed: int) -> Variant:
    rng = random.Random(f"variant:{seed}")
    palette = spec.variation.palette
    return Variant(
        seed=seed,
        color=rng.choice(palette) if palette else None,
        caption_y=rng.choice(CAPTION_Y),
        headline_y=rng.choice(HEADLINE_Y),
        angle=rng.choice(ANGLES),
        speed_jitter=round(rng.uniform(0.95, 1.1), 3),
    )


def expand_slots(spec: RecipeSpec, rng: random.Random) -> list[SlotSpec]:
    """Danh sách slot theo thứ tự timeline, đã nhân `repeat` theo seed."""
    out: list[SlotSpec] = []
    for slot in spec.slots:
        out.extend([slot] * (rng.randint(*slot.repeat) if slot.repeat else 1))
    return out


def recent_segment_ids(engine: Engine, last_n_renders: int) -> set[int]:
    """Segment đã xuất hiện trong `last_n_renders` render gần nhất."""
    if last_n_renders <= 0:
        return set()
    recent = select(Render.id).order_by(Render.id.desc()).limit(last_n_renders).subquery()
    q = select(UsageLog.segment_id).where(UsageLog.render_id.in_(select(recent.c.id)))
    with engine.connect() as conn:
        return {row[0] for row in conn.execute(q)}
