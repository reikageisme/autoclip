from __future__ import annotations

import cv2
import numpy as np

from autoclip.contracts import RenderItem
from autoclip.ingest.layout import focus_x, layout_tags, text_bands
from autoclip.recipes.compiler import _reframe_offset


def _frame(text_at: float | None = None) -> np.ndarray:
    img = np.full((640, 360, 3), 90, np.uint8)
    cv2.rectangle(img, (40, 150), (320, 420), (60, 120, 200), -1)  # một mảng hình, không phải chữ
    if text_at is not None:
        y = int(640 * text_at)
        cv2.putText(
            img, "Van chuyen hang", (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )
    return img


def test_text_bands_bottom_only():
    assert text_bands([_frame(0.85)]) == {"text_bottom"}
    assert text_bands([_frame(0.12)]) == {"text_top"}
    assert text_bands([_frame()]) == set()


def test_focus_follows_subject_on_the_right():
    img = np.full((360, 640, 3), 40, np.uint8)
    for x in range(430, 600, 12):
        cv2.line(img, (x, 60), (x, 300), (255, 255, 255), 3)
    assert focus_x(img) > 0.6
    assert any(t.startswith("fx:") for t in layout_tags([img], landscape=True))
    assert not any(t.startswith("fx:") for t in layout_tags([img], landscape=False))


def test_reframe_offset_moves_toward_subject_and_stays_covered():
    item = RenderItem(
        slot_role="body",
        segment_id=1,
        clip_path="x",
        src_start=0,
        src_end=1,
        media_duration=10,
        media_width=1920,
        media_height=1080,
        focus_x=0.8,
    )
    fill = (1920 / 1080) / (1080 / 1920)
    off = _reframe_offset(item, 1080, 1920, fill)
    assert off < 0 and abs(off) <= (fill - 1) / 2 + 1e-6
    item.focus_x = 0.5
    assert _reframe_offset(item, 1080, 1920, fill) == 0
    vertical = item.model_copy(update={"media_width": 1080, "media_height": 1920, "focus_x": 0.9})
    assert _reframe_offset(vertical, 1080, 1920, 1.0) == 0
