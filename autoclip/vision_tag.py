"""Gắn nhãn cảnh bằng AI nhìn ảnh (tuỳ chọn, chạy nền): kho, container, xe tải…

Đọc thumbnail của từng đoạn, hỏi model nhận ảnh (`VISION_MODEL`), thêm nhãn `scene:<tên>`
vào đoạn. Lồng tiếng dùng nhãn này để chọn cảnh khớp câu đọc ("container" → cảnh container).
Mỗi ảnh ~10–20 giây với model miễn phí, nên chỉ chạy cho đoạn chưa có nhãn cảnh.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from autoclip.db.models import Segment
from autoclip.llm import LLMClient

log = logging.getLogger(__name__)

SCENES = {
    "kho": ["kho", "nhà kho", "lưu kho"],
    "container": ["container", "cont", "thùng hàng"],
    "xe_tai": ["xe", "xe tải", "vận chuyển", "giao hàng", "chở"],
    "hang_hoa": ["hàng", "hàng hoá", "hàng hóa", "sản phẩm", "nguồn hàng"],
    "dong_goi": ["đóng gói", "đóng hàng", "kiểm đếm", "kiểm hàng", "niêm phong"],
    "nguoi": ["đội ngũ", "nhân viên", "chúng tôi", "khách"],
    "van_phong": ["tư vấn", "văn phòng", "liên hệ", "hỗ trợ"],
    "su_kien": ["sự kiện", "hội thảo", "triển lãm"],
    "cua_hang": ["cửa hàng", "showroom", "kinh doanh", "bán"],
    "ngoai_troi": ["cửa khẩu", "bãi", "đường"],
    "man_hinh": ["màn hình", "đơn hàng", "hệ thống"],
    "logo": ["thương hiệu", "giang minh"],
}

PROMPT = (
    "Ảnh là một cảnh trong kho video của công ty vận chuyển/nhập hàng. Chọn tối đa 3 nhãn "
    "đúng NHẤT với nội dung chính của ảnh từ danh sách: " + ", ".join(SCENES) + ". "
    'Chỉ trả về JSON: {"tags": ["..."]}'
)


def scene_tags(seg_tags: list[str]) -> set[str]:
    return {t.removeprefix("scene:") for t in seg_tags if t.startswith("scene:")}


def line_scenes(text: str) -> set[str]:
    """Nhãn cảnh mà một câu kịch bản nhắc tới."""
    low = text.lower()
    return {scene for scene, words in SCENES.items() if any(w in low for w in words)}


def ask(llm: LLMClient, image: Path) -> list[str]:
    data = base64.b64encode(image.read_bytes()).decode()
    answer = llm.chat(PROMPT, image_jpeg_b64=data, max_tokens=3000)
    m = re.search(r"\{.*\}", answer, re.S)
    tags = json.loads(m.group(0)).get("tags", []) if m else []
    return [t for t in (str(x).strip().lower() for x in tags) if t in SCENES][:3]


def tag_segments(
    engine: Engine,
    llm: LLMClient,
    *,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """(số đoạn đã gắn, số lỗi) cho các đoạn có thumbnail mà chưa có nhãn cảnh."""
    with Session(engine) as db:
        todo = [
            (s.id, s.thumb_path)
            for s in db.scalars(select(Segment).order_by(Segment.id))
            if s.thumb_path and not scene_tags(s.tags) and Path(s.thumb_path).is_file()
        ]
    if limit:
        todo = todo[:limit]
    done = failed = 0
    for n, (sid, thumb) in enumerate(todo, start=1):
        try:
            found = ask(llm, Path(thumb)) or ["khac"]
        except Exception as exc:  # một ảnh lỗi không dừng cả kho
            log.warning("gắn nhãn cảnh đoạn %s lỗi: %s", sid, exc)
            failed += 1
            continue
        with Session(engine) as db:
            seg = db.get(Segment, sid)
            seg.tags = [*seg.tags, *(f"scene:{t}" for t in found)]
            db.commit()
        done += 1
        if progress:
            progress(n, len(todo))
    return done, failed
