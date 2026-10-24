"""Nhận ra chữ chép lời "rác" để không đưa lên video.

Whisper bị ép nghe tiếng Việt (WHISPER_LANGUAGE=vi) nên với đoạn tiếng Trung, tiếng Anh
hay tiếng ồn xưởng nó vẫn đoán ra chữ: "Chrogelo", "And I have to"... Âm tiết tiếng Việt
có cấu trúc rất chặt (phụ âm đầu + vần + phụ âm cuối giới hạn), nên kiểm tra từng từ
theo cấu trúc đó loại được phần lớn chữ đoán bừa mà không cần từ điển.
"""

from __future__ import annotations

import re
import unicodedata

# Dấu thanh (huyền, sắc, ngã, hỏi, nặng) bỏ đi; giữ dấu mũ/trăng/móc vì là một phần chữ cái.
_TONES = {"̀", "́", "̃", "̉", "̣"}
_SYLLABLE = re.compile(
    r"^(ngh|ng|nh|ch|gh|gi|kh|ph|th|tr|qu|[bcdđghklmnprstvx])?"
    r"[aăâeêioôơuưy]{1,3}"
    r"(ch|nh|ng|[cmnpt])?$"
)
_STRIP = ".,!?;:…\"'“”‘’()[]-–—"


def _base(word: str) -> str:
    nfd = unicodedata.normalize("NFD", word.lower())
    return unicodedata.normalize("NFC", "".join(c for c in nfd if c not in _TONES))


def is_vietnamese_word(word: str) -> bool:
    """Một âm tiết tiếng Việt hợp lệ (hoặc con số như 20, 1.5, 50k)."""
    w = word.strip(_STRIP)
    if not w:
        return True
    if any(ch.isdigit() for ch in w):
        return True
    return bool(_SYLLABLE.match(_base(w)))


def vi_ratio(text: str | None) -> float:
    """Tỉ lệ từ là âm tiết tiếng Việt (chuỗi rỗng = 0)."""
    words = [w for w in (text or "").split() if w.strip(_STRIP)]
    if not words:
        return 0.0
    return sum(is_vietnamese_word(w) for w in words) / len(words)


def clauses(text: str | None) -> list[list[str]]:
    """Tách lời thoại thành các vế theo dấu câu Whisper đặt."""
    out: list[list[str]] = []
    cur: list[str] = []
    for w in (text or "").split():
        cur.append(w)
        if w[-1] in ".,!?;:…":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def headline_candidates(texts: list[str | None], lo: int = 3, hi: int = 8) -> list[str]:
    """Vế câu trọn vẹn làm được tiêu đề: bắt đầu bằng chữ hoa (đầu câu), 3–8 từ,
    toàn âm tiết tiếng Việt, không lặp từ quá nửa. Cụm cắt giữa câu → bỏ."""
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        for words in clauses(text):
            if not lo <= len(words) <= hi or not words[0][:1].isupper():
                continue
            phrase = " ".join(words).strip(_STRIP)
            key = phrase.lower()
            if key in seen or vi_ratio(phrase) < 1.0:
                continue
            if len({_base(w.strip(_STRIP)) for w in words}) <= len(words) / 2:
                continue
            seen.add(key)
            out.append(phrase)
    return out
