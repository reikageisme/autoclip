"""Trích recipe từ draft CapCut và báo cáo nhanh một thư mục samples.

``extract_recipe`` chỉ đọc model ``CapCutDraft`` đã parse, không chạm file.
``report`` duyệt từng thư mục con, chịu được draft mã hoá/hỏng và luôn trả về
markdown. CLI:

    python -m autoclip.recipes.extract <draft_dir> -o recipes/<name>.json
    python -m autoclip.recipes.extract --report samples/
    python -m autoclip.recipes.extract --all samples/ --output-dir recipes/
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from unicodedata import normalize as unicode_normalize

from autoclip.contracts import (
    DraftStatus,
    MusicSpec,
    RecipeSpec,
    SlotSpec,
    TextSpec,
    UnmappedItem,
    VariationSpec,
)

from . import effect_map
from .capcut_reader import (
    CapCutDraft,
    CapCutSegment,
    DraftError,
    detect,
    iter_draft_dirs,
    load,
)

_POSITION_LIMIT = 1.5


def _slug(value: str) -> str:
    text = unicode_normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "draft"


def _round(value: float) -> float:
    return round(float(value), 2)


def _as_range(values: list[float], pad: float = 0.15) -> tuple[float, float] | None:
    if not values:
        return None
    low = min(values)
    high = max(values)
    if high - low < 0.05:
        low, high = low * (1 - pad), high * (1 + pad)
    return (_round(max(0.1, low)), _round(max(low + 0.1, high)))


def _style_name(info: Any) -> str:
    bold = bool(getattr(info, "bold", False))
    stroke = bool(getattr(info, "has_stroke", False))
    if bold and stroke:
        return "bold-stroke"
    if bold:
        return "bold"
    if stroke:
        return "stroke"
    return "plain"


def _position(segment: CapCutSegment) -> tuple[float, float]:
    transform = segment.transform or {}
    x = float(transform.get("x", 0.0) or 0.0)
    y = float(transform.get("y", 0.0) or 0.0)
    if abs(x) <= _POSITION_LIMIT and abs(y) <= _POSITION_LIMIT:
        x = (x + 1.0) / 2.0
        y = (y + 1.0) / 2.0
    return (_round(min(1.0, max(0.0, x))), _round(min(1.0, max(0.0, y))))


def _role_for_start(start: float, visual: list[CapCutSegment]) -> str | None:
    if not visual:
        return None
    if start < visual[0].end:
        return "hook"
    if start >= visual[-1].start:
        return "outro"
    return "body"


def _transitions_in(segments: list[CapCutSegment]) -> list[str]:
    found: list[str] = []
    for segment in segments:
        transition = segment.transition
        if transition is None:
            continue
        mapped = effect_map.transition_id(transition.name)
        if mapped and mapped not in found:
            found.append(mapped)
    return found


def _slot_speed(segments: list[CapCutSegment]) -> tuple[float, float] | None:
    speeds = [segment.speed for segment in segments if segment.speed > 0]
    if not speeds or all(abs(speed - 1.0) < 0.01 for speed in speeds):
        return None
    return (_round(min(speeds)), _round(max(speeds)))


def _slot(role: str, segments: list[CapCutSegment]) -> SlotSpec:
    durations = [segment.duration for segment in segments if segment.duration > 0]
    repeat = None
    if role == "body" and segments:
        buckets = Counter(round(duration * 2) / 2 for duration in durations)
        common = max(buckets.values()) if buckets else 0
        repeat = (max(1, common), max(1, len(segments)))
        if common == len(segments):
            repeat = (len(segments), len(segments))
    return SlotSpec(
        role=role,
        dur=_as_range(durations),
        repeat=repeat,
        want_tags=[],
        speed=_slot_speed(segments),
        transition=_transitions_in(segments),
    )


def _build_slots(visual: list[CapCutSegment]) -> list[SlotSpec]:
    if not visual:
        return []
    if len(visual) == 1:
        return [_slot("hook", visual)]
    if len(visual) == 2:
        return [_slot("hook", visual[:1]), _slot("outro", visual[1:])]
    return [
        _slot("hook", visual[:1]),
        _slot("body", visual[1:-1]),
        _slot("outro", visual[-1:]),
    ]


def _collect_effects(draft: CapCutDraft, visual: list[CapCutSegment]) -> list[UnmappedItem]:
    unmapped: list[UnmappedItem] = []
    seen: set[tuple[str, str]] = set()
    sources: list[tuple[str, Any]] = []
    for segment in visual:
        sources.extend(("effect", effect) for effect in segment.effects)
        sources.extend(("filter", effect) for effect in segment.filters)
    for track in draft.tracks:
        if track.type.lower() not in {"effect", "filter", "adjust"}:
            continue
        for segment in track.segments:
            sources.extend(("effect", effect) for effect in segment.effects)
            sources.extend(("filter", effect) for effect in segment.filters)
    for kind, material in sources:
        name = getattr(material, "name", "") or getattr(material, "effect_id", "")
        if not name:
            continue
        mapped, known = effect_map.map_effect(name)
        if mapped or known:
            continue
        key = (kind, name)
        if key in seen:
            continue
        seen.add(key)
        unmapped.append(
            UnmappedItem(kind=kind, name=name, note="không có effect Concat tương đương")
        )
    return unmapped


def _collect_stickers(draft: CapCutDraft, visual: list[CapCutSegment]) -> list[UnmappedItem]:
    names: set[str] = set()
    for segment in visual:
        if segment.sticker is not None:
            names.add(segment.sticker.name or segment.sticker.id)
    for track in draft.tracks:
        if track.type.lower() != "sticker":
            continue
        for segment in track.segments:
            if segment.sticker is not None:
                names.add(segment.sticker.name or segment.sticker.id)
    return [
        UnmappedItem(kind="sticker", name=name, note="ReiCut không có sticker tương đương")
        for name in sorted(name for name in names if name)
    ]


def _collect_texts(
    draft: CapCutDraft,
    visual: list[CapCutSegment],
) -> tuple[list[TextSpec], bool, list[str], list[str]]:
    segments = [segment for segment in draft.text_segments if segment.text is not None]
    if not segments:
        return [], False, [], []
    durations = [segment.duration for segment in segments]
    median = statistics.median(durations) if durations else 0.0
    captions = len(segments) >= 3 and median <= 7.0
    texts: list[TextSpec] = []
    fonts: list[str] = []
    palette: list[str] = []
    unmapped: list[str] = []
    for segment in segments:
        info = segment.text
        if info is None:
            continue
        family = info.font_family
        if family:
            mapped, known = effect_map.map_font(family)
            if mapped and mapped not in fonts:
                fonts.append(mapped)
            if not known:
                unmapped.append(family)
        if info.color and info.color not in palette:
            palette.append(info.color)
        texts.append(
            TextSpec(
                kind="captions" if captions else "headline",
                slot=_role_for_start(segment.start, visual),
                style=_style_name(info),
                pos=_position(segment),
                source="draft" if captions else "capcut",
            )
        )
    return texts, captions, fonts, unmapped


def _collect_unmapped_transitions(visual: list[CapCutSegment]) -> list[UnmappedItem]:
    unmapped: list[UnmappedItem] = []
    seen: set[str] = set()
    for segment in visual:
        transition = segment.transition
        if transition is None or not transition.name or effect_map.is_cut(transition.name):
            continue
        if effect_map.transition_id(transition.name) is not None or transition.name in seen:
            continue
        seen.add(transition.name)
        unmapped.append(
            UnmappedItem(
                kind="transition",
                name=transition.name,
                note="không map được, dùng cut",
            )
        )
    return unmapped


def _music(draft: CapCutDraft) -> MusicSpec:
    has_audio = any(material.path for material in draft.materials.audios)
    has_audio = has_audio or bool(draft.audio_segments)
    return MusicSpec(mode="source" if has_audio else "none", beat_sync=False)


def _text_colors(draft: CapCutDraft) -> list[str]:
    colors: list[str] = []
    for segment in draft.text_segments:
        if segment.text is not None and segment.text.color:
            colors.append(segment.text.color)
    return colors


def extract_recipe(draft: CapCutDraft) -> RecipeSpec:
    """Suy ra ``RecipeSpec`` từ draft đã parse."""
    visual = [segment for segment in draft.visual_segments if segment.duration > 0]
    total = max(draft.duration, sum(segment.duration for segment in visual))
    texts, _captions, fonts, font_unmapped = _collect_texts(draft, visual)
    palette = sorted(set(_text_colors(draft)))
    if not fonts:
        fonts = ["Hanken Grotesk"]
    if not palette:
        palette = ["#ffffff"]
    unmapped = [
        *_collect_unmapped_transitions(visual),
        *_collect_effects(draft, visual),
        *_collect_stickers(draft, visual),
        *(
            UnmappedItem(kind="font", name=name, note="font không có trong bảng map")
            for name in font_unmapped
        ),
    ]
    unique: dict[tuple[str, str], UnmappedItem] = {}
    for item in unmapped:
        unique.setdefault((item.kind, item.name), item)
    ordered_unmapped = sorted(unique.values(), key=lambda item: (item.kind, item.name.lower()))
    low = _round(max(0.0, total * 0.9))
    high = _round(max(low + 0.1, total * 1.1))
    return RecipeSpec(
        name=f"{_slug(draft.name)}-recipe",
        aspect=draft.aspect,
        duration_target=(low, high),
        music=_music(draft),
        slots=_build_slots(visual),
        texts=texts,
        variation=VariationSpec(palette=palette, fonts=fonts, shuffle_body=True),
        source_project=draft.name,
        unmapped=ordered_unmapped,
    )


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _draft_row(root: Path, draft_dir: Path) -> tuple[str, list[str]]:
    relative = draft_dir.relative_to(root).as_posix() if draft_dir != root else root.name
    status = detect(draft_dir)
    row = [
        _markdown_cell(relative),
        status.value,
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0.0s",
        "",
    ]
    if status is not DraftStatus.OK:
        notes = {
            DraftStatus.ENCRYPTED: "file không bắt đầu bằng { (mã hoá?)",
            DraftStatus.CORRUPT: "JSON hỏng hoặc rỗng",
            DraftStatus.MISSING: "không có draft_content.json/draft_info.json",
        }
        row[-1] = notes.get(status, "")
        return status.value, row
    try:
        draft = load(draft_dir)
    except DraftError as error:
        row[1] = error.status.value
        row[-1] = _markdown_cell(str(error))
        return error.status.value, row
    effects = {
        effect.name or effect.id
        for segment in draft.visual_segments
        for effect in [*segment.effects, *segment.filters]
    }
    stickers = {
        segment.sticker.name or segment.sticker.id
        for segment in draft.sticker_segments
        if segment.sticker is not None
    }
    transitions = {
        segment.transition.name
        for segment in draft.visual_segments
        if segment.transition is not None and segment.transition.name
    }
    row[2] = str(len(draft.tracks))
    row[3] = str(len(draft.visual_segments))
    row[4] = str(len(draft.audio_segments))
    row[5] = str(len(draft.text_segments))
    row[6] = str(len(effects))
    row[7] = str(len(stickers))
    row[8] = str(len(transitions))
    row[9] = f"{draft.duration:.1f}s"
    notes: list[str] = []
    if draft.warnings:
        notes.append(f"{len(draft.warnings)} cảnh báo")
    if draft.materials.audios:
        notes.append(f"{len(draft.materials.audios)} audio material")
    row[-1] = _markdown_cell("; ".join(notes))
    return status.value, row


def report(samples_dir: str | Path) -> str:
    """Tạo bảng markdown tổng hợp mọi draft trong ``samples_dir``."""
    root = Path(samples_dir)
    draft_dirs = iter_draft_dirs(root)
    rows: list[list[str]] = []
    counts: Counter[str] = Counter()
    total_duration = 0.0
    for draft_dir in draft_dirs:
        status, row = _draft_row(root, draft_dir)
        counts[status] += 1
        rows.append(row)
        if status == DraftStatus.OK.value:
            with contextlib.suppress(DraftError):
                total_duration += load(draft_dir).duration
    header = [
        "Draft",
        "Trạng thái",
        "Tracks",
        "Video",
        "Audio",
        "Text",
        "Effect",
        "Sticker",
        "Transition",
        "Thời lượng",
        "Ghi chú",
    ]
    lines = [
        "# Báo cáo draft CapCut",
        "",
        f"- Thư mục: `{root}`",
        f"- Tìm thấy: **{len(draft_dirs)}** draft",
        f"- ok: **{counts[DraftStatus.OK.value]}**, encrypted: "
        f"**{counts[DraftStatus.ENCRYPTED.value]}**, corrupt: "
        f"**{counts[DraftStatus.CORRUPT.value]}**, "
        f"missing: **{counts[DraftStatus.MISSING.value]}**",
        f"- Tổng thời lượng draft đọc được: **{total_duration:.1f}s**",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    if rows:
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
    else:
        lines.append(
            "| _(không có draft)_ |  |  |  |  |  |  |  |  |  | "
            "Thêm project CapCut vào samples/ rồi chạy lại. |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(samples_dir: str | Path, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report(samples_dir), encoding="utf-8")
    return path


def _dump_recipe(spec: RecipeSpec) -> str:
    payload = spec.model_dump(mode="json", exclude_none=True)
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trích recipe CapCut hoặc báo cáo samples")
    parser.add_argument("draft_dir", nargs="?", help="thư mục project CapCut")
    parser.add_argument("-o", "--output", help="file recipe.json đầu ra")
    parser.add_argument(
        "--report",
        nargs="?",
        const="samples",
        metavar="SAMPLES_DIR",
        help="chỉ chạy báo cáo (mặc định samples/)",
    )
    parser.add_argument(
        "--report-output",
        default=str(Path("data") / "capcut_report.md"),
        help="file markdown báo cáo",
    )
    parser.add_argument(
        "--all",
        metavar="SAMPLES_DIR",
        help="trích mọi draft đọc được trong SAMPLES_DIR",
    )
    parser.add_argument(
        "--output-dir",
        default="recipes",
        help="thư mục recipe khi dùng --all",
    )
    args = parser.parse_args(argv)

    if args.report is not None:
        path = write_report(args.report, args.report_output)
        print(f"Đã ghi báo cáo: {path}")
        return 0

    if args.all is not None:
        root = Path(args.all)
        output_dir = Path(args.output_dir)
        written = 0
        skipped = 0
        for draft_dir in iter_draft_dirs(root):
            if detect(draft_dir) is not DraftStatus.OK:
                skipped += 1
                continue
            spec = extract_recipe(load(draft_dir))
            output = output_dir / f"{spec.name}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(_dump_recipe(spec), encoding="utf-8")
            written += 1
            print(f"recipe: {output}")
        print(f"Đã trích {written} recipe, bỏ qua {skipped} draft không đọc được")
        return 0

    if not args.draft_dir:
        parser.error("cần <draft_dir>, --report hoặc --all")
    draft_dir = Path(args.draft_dir)
    status = detect(draft_dir)
    if status is not DraftStatus.OK:
        print(f"Không đọc được draft ({status.value}): {draft_dir}", file=sys.stderr)
        return 2
    spec = extract_recipe(load(draft_dir))
    output = Path(args.output) if args.output else Path("recipes") / f"{spec.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_dump_recipe(spec), encoding="utf-8")
    print(f"Đã ghi recipe: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
