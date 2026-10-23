"""Bố cục khung hình: chữ in sẵn (phụ đề gốc, bảng tên) và điểm nhìn chính.

- `text_bands`: có dòng chữ ở dải trên / dưới khung không → nhãn `text_top` / `text_bottom`,
  để phụ đề của mình không đè lên chữ có sẵn trong video gốc.
- `focus_x`: vị trí ngang của người/vật chính (0 trái → 1 phải) → nhãn `fx:0.62`, để khi
  cắt footage ngang thành khung dọc thì bám theo chủ thể thay vì luôn lấy chính giữa.
Chạy trên ảnh nhỏ (thumbnail ~360px) là đủ, nên gắn nhãn lại kho cũ không cần giải mã video.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

TOP_BAND = 0.30
BOTTOM_BAND = 0.62
_cascade: Any = None


def _gray(image: Any) -> Any:
    import cv2
    import numpy as np

    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr.astype("uint8"), cv2.COLOR_RGB2GRAY)
    return arr.astype("uint8")


def text_lines(image: Any) -> list[tuple[float, float, float, float]]:
    """Các hộp trông như dòng chữ: (x, y, w, h) theo tỉ lệ khung."""
    import cv2

    gray = _gray(image)
    h, w = gray.shape[:2]
    if w > 480:
        scale = 480 / w
        gray = cv2.resize(gray, (480, int(h * scale)))
        h, w = gray.shape[:2]
    grad = cv2.morphologyEx(
        gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    joined = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 30), 3))
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < w * 0.25 or ch < h * 0.012 or ch > h * 0.09 or cw / max(ch, 1) < 4:
            continue
        fill = cv2.countNonZero(bw[y : y + ch, x : x + cw]) / float(cw * ch)
        if not 0.18 <= fill <= 0.75:  # chữ: nhiều nét nhưng không đặc như mảng màu
            continue
        center = (x + cw / 2) / w
        if not 0.3 <= center <= 0.7:  # phụ đề/tiêu đề in sẵn thường căn giữa
            continue
        if _glyphs(gray[y : y + ch, x : x + cw]) < 5:
            continue  # đèn trần, biển số, lan can: ít "ký tự" rời
        lines.append((x / w, y / h, cw / w, ch / h))
    return lines


def _glyphs(box: Any) -> int:
    """Số mảng rời cao gần bằng dòng — xấp xỉ số ký tự trong một dòng chữ."""
    import cv2

    if box.size == 0:
        return 0
    _, bw = cv2.threshold(box, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if cv2.countNonZero(bw) > bw.size / 2:
        bw = cv2.bitwise_not(bw)  # chữ sáng trên nền tối hay ngược lại
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw)
    height = box.shape[0]
    return sum(
        1
        for i in range(1, n)
        if 0.35 * height <= stats[i, cv2.CC_STAT_HEIGHT] <= height
        and stats[i, cv2.CC_STAT_WIDTH] <= 1.6 * stats[i, cv2.CC_STAT_HEIGHT]
    )


def text_bands(images: list[Any]) -> set[str]:
    """`text_bottom` / `text_top` khi đa số ảnh có dòng chữ ở dải đó."""
    if not images:
        return set()
    top = bottom = 0
    for image in images:
        lines = text_lines(image)
        top += any(y + h / 2 < TOP_BAND for _, y, _, h in lines)
        bottom += any(y + h / 2 > BOTTOM_BAND for _, y, _, h in lines)
    need = len(images) / 2
    return {name for name, n in (("text_top", top), ("text_bottom", bottom)) if n > need}


def _faces(gray: Any) -> list[tuple[int, int, int, int]]:
    global _cascade
    import cv2

    if _cascade is None:
        path = os.path.join(
            getattr(getattr(cv2, "data", None), "haarcascades", ""),
            "haarcascade_frontalface_default.xml",
        )
        _cascade = cv2.CascadeClassifier(path) if os.path.exists(path) else False
    if _cascade is False or _cascade.empty():
        return []
    return list(_cascade.detectMultiScale(cv2.equalizeHist(gray), 1.15, 5, minSize=(24, 24)))


def focus_x(image: Any) -> float:
    """Vị trí ngang của chủ thể: mặt người lớn nhất, không có thì trọng tâm vùng nhiều chi tiết."""
    import cv2
    import numpy as np

    gray = _gray(image)
    h, w = gray.shape[:2]
    faces = _faces(gray)
    if faces:
        x, _, fw, _ = max(faces, key=lambda f: f[2] * f[3])
        return round((x + fw / 2) / w, 3)
    edges = np.abs(cv2.Sobel(cv2.GaussianBlur(gray, (5, 5), 0), cv2.CV_32F, 1, 0))
    column = edges.sum(axis=0)
    if column.sum() <= 0:
        return 0.5
    centroid = float((column * np.arange(w)).sum() / column.sum()) / w
    return round(0.5 + (centroid - 0.5) * 0.8, 3)  # kéo nhẹ về giữa cho đỡ lệch


def layout_tags(images: list[Any], landscape: bool) -> list[str]:
    tags = sorted(text_bands(images))
    if landscape and images:
        tags.append(f"fx:{focus_x(images[len(images) // 2]):.2f}")
    return tags
