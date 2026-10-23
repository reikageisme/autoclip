"""Kiểu video (recipe) cho người dùng: có sẵn + tự tạo, và form đơn giản để sửa.

- Kiểu có sẵn: `recipes/*.json` trong bộ cài (chỉ đọc).
- Kiểu tự tạo: `<DATA_DIR>/recipes/*.json` — nằm trong dữ liệu người dùng nên còn
  nguyên khi cập nhật app. Mã luôn bắt đầu bằng `my-` để không trùng kiểu có sẵn.
- `StyleForm`: vài thông số người dùng hiểu được (thời lượng, số cảnh, chuyển cảnh,
  chữ…), đổi qua lại với `RecipeSpec` mà planner/compiler dùng.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from autoclip.contracts import (
    MusicSpec,
    RecipeSpec,
    SlotSpec,
    TextSpec,
    VariationSpec,
)

BUILTIN_DIR = Path(__file__).resolve().parents[1] / "recipes"
TRANSITIONS = ["cut", "zoom", "whip", "dissolve", "slide", "push", "glitch", "fade"]
"""Chuyển cảnh panel cho chọn (tên theo từ vựng effect_map)."""


class StyleForm(BaseModel):
    label: str = Field(..., min_length=1, max_length=60)
    description: str = ""
    duration_min: float = Field(15, ge=5, le=180)
    duration_max: float = Field(30, ge=5, le=180)
    body_min: int = Field(4, ge=1, le=20)
    body_max: int = Field(6, ge=1, le=20)
    scene_min: float = Field(2.0, ge=0.5, le=10)
    scene_max: float = Field(3.5, ge=0.5, le=10)
    hook_speed: float = Field(1.1, ge=1.0, le=2.0)
    transitions: list[str] = Field(default_factory=lambda: ["cut", "zoom"])
    headline: bool = True
    captions: bool = True
    colors: list[str] = Field(default_factory=lambda: ["#FFD400", "#FFFFFF"])

    @model_validator(mode="after")
    def _ranges(self) -> StyleForm:
        if self.duration_max < self.duration_min:
            self.duration_min, self.duration_max = self.duration_max, self.duration_min
        if self.body_max < self.body_min:
            self.body_min, self.body_max = self.body_max, self.body_min
        if self.scene_max < self.scene_min:
            self.scene_min, self.scene_max = self.scene_max, self.scene_min
        self.transitions = [t for t in self.transitions if t in TRANSITIONS] or ["cut"]
        self.colors = [c for c in self.colors if re.fullmatch(r"#[0-9A-Fa-f]{6}", c)] or ["#FFFFFF"]
        return self


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text.replace("đ", "d").replace("Đ", "D"))
    ascii_text = ascii_text.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-") or "kieu"


def form_to_spec(form: StyleForm, name: str) -> RecipeSpec:
    scene = (form.scene_min, form.scene_max)
    texts: list[TextSpec] = []
    if form.headline:
        texts.append(
            TextSpec(
                slot="hook",
                kind="headline",
                style="bold-stroke",
                pos=(0.5, 0.2),
                source="transcript",
            )
        )
    if form.captions:
        texts.append(TextSpec(kind="captions", source="whisper", style="word-pop", pos=(0.5, 0.78)))
    return RecipeSpec(
        name=name,
        label=form.label.strip(),
        description=form.description.strip() or None,
        aspect="9:16",
        duration_target=(form.duration_min, form.duration_max),
        music=MusicSpec(mode="source"),
        slots=[
            SlotSpec(
                role="hook", dur=scene, want_tags=["speech", "action"], speed=(1.0, form.hook_speed)
            ),
            SlotSpec(
                role="body",
                repeat=(form.body_min, form.body_max),
                dur=scene,
                transition=form.transitions,
            ),
            SlotSpec(role="outro", dur=scene, want_tags=["speech"]),
        ],
        texts=texts,
        variation=VariationSpec(palette=form.colors, shuffle_body=True),
    )


def spec_to_form(spec: RecipeSpec) -> StyleForm:
    """Đọc ngược một recipe bất kỳ thành form (lấy gần đúng nếu recipe phức tạp hơn form)."""
    hook = next((s for s in spec.slots if s.role == "hook"), None)
    body = next((s for s in spec.slots if s.role == "body"), None)
    any_slot = body or hook or (spec.slots[0] if spec.slots else None)
    scene = (any_slot.dur if any_slot and any_slot.dur else None) or (2.0, 3.5)
    repeat = body.repeat if body and body.repeat else (1, 1)
    return StyleForm(
        label=spec.label or spec.name,
        description=spec.description or "",
        duration_min=spec.duration_target[0],
        duration_max=spec.duration_target[1],
        body_min=repeat[0],
        body_max=repeat[1],
        scene_min=scene[0],
        scene_max=scene[1],
        hook_speed=(hook.speed[1] if hook and hook.speed else 1.0),
        transitions=list(body.transition) if body and body.transition else ["cut"],
        headline=any(t.kind != "captions" for t in spec.texts),
        captions=any(t.kind == "captions" for t in spec.texts),
        colors=list(spec.variation.palette) or ["#FFFFFF"],
    )


def segments_per_video(spec: RecipeSpec, avg_segment_sec: float | None = None) -> float:
    """Số đoạn trung bình một video dùng: số slot, hoặc nhiều hơn nếu đoạn trong kho ngắn."""
    slots = 0.0
    want = 0.0
    for slot in spec.slots:
        n = (slot.repeat[0] + slot.repeat[1]) / 2 if slot.repeat else 1
        slots += n
        want += n * ((slot.dur[0] + slot.dur[1]) / 2 if slot.dur else 2.5)
    if avg_segment_sec and avg_segment_sec > 0 and want > 0:
        # đoạn ngắn hơn cảnh mong muốn → planner phải lấy thêm cảnh để đủ thời lượng
        need = spec.duration_target[0] + 1.0
        per_scene = min(want / slots, avg_segment_sec)
        slots = max(slots, need / per_scene)
    return max(1.0, slots)


def estimate_videos(spec: RecipeSpec, available: int, avg_segment_sec: float | None) -> int:
    return math.floor(available / segments_per_video(spec, avg_segment_sec))


class Styles:
    def __init__(self, custom_dir: Path, builtin_dir: Path = BUILTIN_DIR) -> None:
        self.custom_dir = custom_dir
        self.builtin_dir = builtin_dir

    def _files(self) -> list[tuple[Path, bool]]:
        out = [(f, True) for f in sorted(self.builtin_dir.glob("*.json"))]
        if self.custom_dir.is_dir():
            out += [(f, False) for f in sorted(self.custom_dir.glob("*.json"))]
        return out

    def list(self) -> list[tuple[str, RecipeSpec | None, bool, str | None]]:
        """(file, spec hoặc None nếu hỏng, có sẵn?, lỗi)."""
        out = []
        for f, builtin in self._files():
            try:
                out.append((f.name, self._read(f), builtin, None))
            except Exception as exc:  # file hỏng vẫn liệt kê để người dùng thấy
                out.append((f.name, None, builtin, str(exc)))
        return out

    @staticmethod
    def _read(path: Path) -> RecipeSpec:
        return RecipeSpec.model_validate_json(path.read_text(encoding="utf-8"))

    def path(self, file: str) -> Path | None:
        """File của một kiểu theo tên — chỉ trong 2 thư mục kiểu, không đi ra ngoài."""
        if Path(file).name != file or not file.endswith(".json"):
            return None
        for folder in (self.custom_dir, self.builtin_dir):
            candidate = folder / file
            if candidate.is_file():
                return candidate
        return None

    def load(self, file: str) -> RecipeSpec | None:
        path = self.path(file)
        return self._read(path) if path else None

    def is_builtin(self, file: str) -> bool:
        path = self.path(file)
        return bool(path and path.parent == self.builtin_dir)

    def save(self, form: StyleForm, file: str | None = None) -> str:
        """Lưu kiểu tự tạo; `file` của kiểu tự tạo cũ thì ghi đè, còn lại tạo mới."""
        self.custom_dir.mkdir(parents=True, exist_ok=True)
        if file and not self.is_builtin(file) and self.path(file):
            target = self.custom_dir / file
            name = target.stem
        else:
            base = "my-" + slugify(form.label)
            name, n = base, 2
            while (self.custom_dir / f"{name}.json").exists() or (
                self.builtin_dir / f"{name}.json"
            ).exists():
                name, n = f"{base}-{n}", n + 1
            target = self.custom_dir / f"{name}.json"
        spec = form_to_spec(form, name)
        target.write_text(
            json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target.name

    def delete(self, file: str) -> bool:
        path = self.path(file)
        if not path or path.parent != self.custom_dir:
            return False
        path.unlink()
        return True
