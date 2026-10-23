"""Kịch bản lồng tiếng: AI viết theo chủ đề, hoặc người dùng tự gõ.

Mỗi dòng kịch bản là một câu đọc + một cảnh. AI viết bản khác nhau cho mỗi video (góc nội
dung theo seed); không có AI thì dùng kịch bản người dùng gõ.
"""

from __future__ import annotations

import json
import logging
import re

from autoclip.llm import LLMClient, LLMError

log = logging.getLogger(__name__)

MAX_WORDS = 18

PROMPT = """Bạn viết lời thoại cho video ngắn (TikTok/Reels) tiếng Việt.
Chủ đề: {topic}
Góc nội dung cho video này: {angle}
{brand}
Viết {lines} câu ngắn, mỗi câu tối đa {words} từ, đọc trôi chảy, giọng tự nhiên:
- Câu 1 là câu mở đầu gây tò mò hoặc nêu vấn đề.
- Các câu giữa nêu lợi ích cụ thể, không bịa số liệu hay cam kết.
- Câu cuối kêu gọi liên hệ / nhắn tin.
Không dùng emoji, không đánh số, không ký tự đặc biệt.
Thêm "headline" tối đa 6 từ và "title" (mô tả bài đăng ≤120 ký tự + 2-3 hashtag).
Chỉ trả về JSON: {{"lines": ["...", "..."], "headline": "...", "title": "..."}}"""


def clean_lines(text: str) -> list[str]:
    """Kịch bản người dùng gõ: mỗi dòng một câu, bỏ dòng trống/đánh số, cắt câu quá dài."""
    out = []
    for raw in text.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", raw).strip()
        if not line:
            continue
        words = line.split()
        while len(words) > MAX_WORDS:
            out.append(" ".join(words[:MAX_WORDS]))
            words = words[MAX_WORDS:]
        out.append(" ".join(words))
    return out


def write_script(
    llm: LLMClient,
    topic: str,
    *,
    lines: int,
    angle: str,
    brand_name: str = "",
) -> tuple[list[str], str, str]:
    """(các câu, headline, title) do AI viết; lỗi → LLMError/ValueError."""
    prompt = PROMPT.format(
        topic=topic.strip(),
        angle=angle,
        brand=f"Thương hiệu: {brand_name}" if brand_name else "",
        lines=lines,
        words=MAX_WORDS,
    )
    answer = llm.chat(prompt, max_tokens=4000)
    m = re.search(r"\{.*\}", answer, re.S)
    if not m:
        raise ValueError(f"AI không trả JSON: {answer[:200]!r}")
    data = json.loads(m.group(0))
    out = clean_lines("\n".join(str(x) for x in data.get("lines", [])))
    if len(out) < 2:
        raise ValueError("AI trả kịch bản quá ngắn")
    return out, str(data.get("headline") or ""), str(data.get("title") or "")


__all__ = ["LLMError", "clean_lines", "write_script"]
