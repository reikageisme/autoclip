"""Bảng map tên transition/effect/font của CapCut sang gần nhất trong ReiCut.

Khoá là tên đã chuẩn hoá (chữ thường, bỏ khoảng trắng thừa). Giá trị là id
effect package thật của ReiCut (ví dụ ``concat.zoom-transition``). Tên không
có trong bảng được coi là không map được: compiler dùng ``cut`` và extract ghi
vào ``RecipeSpec.unmapped``.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

CUT_NAMES = frozenset({"", "cut", "none", "no transition", "không", "无", "없음", "なし"})

TRANSITIONS: dict[str, str] = {
    "zoom": "concat.zoom-transition",
    "zoom in": "concat.zoom-transition",
    "zoomin": "concat.zoom-transition",
    "变焦": "concat.zoom-transition",
    "放大": "concat.zoom-transition",
    "zoom blur": "concat.zoom-blur-transition",
    "zoombur": "concat.zoom-blur-transition",
    "模糊变焦": "concat.zoom-blur-transition",
    "dissolve": "concat.dissolve",
    "cross dissolve": "concat.dissolve",
    "crossdissolve": "concat.dissolve",
    "叠化": "concat.dissolve",
    "fade": "concat.fade-black",
    "fade black": "concat.fade-black",
    "fade to black": "concat.fade-black",
    "black fade": "concat.fade-black",
    "闪黑": "concat.fade-black",
    "fade white": "concat.fade-white",
    "fade to white": "concat.fade-white",
    "white fade": "concat.fade-white",
    "闪白": "concat.fade-white",
    "glitch": "concat.glitch-transition",
    "故障": "concat.glitch-transition",
    "whip": "concat.whip-pan",
    "whip pan": "concat.whip-pan",
    "whippan": "concat.whip-pan",
    "甩入": "concat.whip-pan",
    "slide": "concat.slide",
    "滑动": "concat.slide",
    "push": "concat.push",
    "推": "concat.push",
    "spin": "concat.spin",
    "旋转": "concat.spin",
    "swirl": "concat.swirl",
    "漩涡": "concat.swirl",
    "blur": "concat.blur-dissolve",
    "模糊": "concat.blur-dissolve",
    "cube": "concat.cube",
    "立方体": "concat.cube",
    "iris": "concat.iris",
    "光圈": "concat.iris",
    "clock wipe": "concat.clock-wipe",
    "clockwipe": "concat.clock-wipe",
    "时钟": "concat.clock-wipe",
    "shape wipe": "concat.shape-wipe",
    "shapewipe": "concat.shape-wipe",
    "形状": "concat.shape-wipe",
    "linear wipe": "concat.linear-wipe",
    "linearwipe": "concat.linear-wipe",
    "线性": "concat.linear-wipe",
    "film burn": "concat.film-burn",
    "filmburn": "concat.film-burn",
    "胶片烧灼": "concat.film-burn",
    "light leak": "concat.light-leak-transition",
    "lightleak": "concat.light-leak-transition",
    "漏光": "concat.light-leak-transition",
    "rgb split": "concat.rgb-split-wipe",
    "rgbsplit": "concat.rgb-split-wipe",
    "色差": "concat.rgb-split-wipe",
}

EFFECTS: dict[str, str] = {
    "vignette": "concat.vignette",
    "暗角": "concat.vignette",
    "film grain": "concat.film-grain",
    "filmgrain": "concat.film-grain",
    "噪点": "concat.film-grain",
    "颗粒": "concat.film-grain",
    "vhs": "concat.vhs",
    "retro": "concat.vhs",
    "复古": "concat.vhs",
    "pixelate": "concat.pixelate",
    "pixel": "concat.pixelate",
    "像素": "concat.pixelate",
    "gaussian blur": "concat.gaussian-blur",
    "gaussianblur": "concat.gaussian-blur",
    "高斯模糊": "concat.gaussian-blur",
    "zoom blur": "concat.zoom-blur",
    "zoomblur": "concat.zoom-blur",
    "glitch": "concat.glitch-warp",
    "故障": "concat.glitch-warp",
    "mono": "concat.mono",
    "monochrome": "concat.mono",
    "black and white": "concat.mono",
    "黑白": "concat.mono",
    "noir": "concat.noir",
    "vivid": "concat.vivid",
    "鲜艳": "concat.vivid",
    "warm": "concat.warm",
    "暖色": "concat.warm",
    "cool": "concat.cool",
    "冷色": "concat.cool",
    "sharpen": "concat.sharpen",
    "锐化": "concat.sharpen",
    "motion blur": "concat.motion-blur",
    "motionblur": "concat.motion-blur",
    "动态模糊": "concat.motion-blur",
    "soft light": "concat.soft-light",
    "柔光": "concat.soft-light",
    "high contrast": "concat.high-contrast-mono",
    "高对比": "concat.high-contrast-mono",
    "vintage": "concat.vintage",
    "retro film": "concat.vintage",
    "bloom": "concat.glow",
    "glow": "concat.glow",
    "发光": "concat.glow",
}

FONTS: dict[str, str] = {
    "arial": "Arial",
    "bebas neue": "Bebas Neue",
    "bebasneue": "Bebas Neue",
    "calibri": "Calibri",
    "cambria": "Cambria",
    "candara": "Candara",
    "cascadia": "Cascadia",
    "consolas": "Consolas",
    "courier new": "Courier New",
    "dengxian": "DengXian",
    "georgia": "Georgia",
    "gill sans": "Gill Sans",
    "hanken grotesk": "Hanken Grotesk",
    "impact": "Impact",
    "inter": "Inter",
    "lato": "Lato",
    "lucida sans": "Lucida Sans",
    "microsoft yahei": "Microsoft YaHei",
    "montserrat": "Montserrat",
    "noto sans": "Noto Sans",
    "noto sans sc": "Noto Sans SC",
    "open sans": "Open Sans",
    "oswald": "Oswald",
    "pingfang sc": "PingFang SC",
    "poppins": "Poppins",
    "roboto": "Roboto",
    "segoe ui": "Segoe UI",
    "simhei": "SimHei",
    "simsun": "SimSun",
    "source han sans": "Source Han Sans",
    "source han sans sc": "Source Han Sans SC",
    "tahoma": "Tahoma",
    "times new roman": "Times New Roman",
    "trebuchet ms": "Trebuchet MS",
    "verdana": "Verdana",
}

KEYFRAMES: dict[str, list[dict[str, float | str]]] = {
    "concat.zoom-transition": [
        {"property": "scale", "at": 0.0, "value": 1.0},
        {"property": "scale", "at": 1.0, "value": 1.08},
    ],
    "concat.zoom-blur-transition": [
        {"property": "scale", "at": 0.0, "value": 1.0},
        {"property": "scale", "at": 1.0, "value": 1.12},
    ],
    "concat.whip-pan": [
        {"property": "offsetX", "at": 0.0, "value": 0.0},
        {"property": "offsetX", "at": 1.0, "value": 0.06},
    ],
    "concat.spin": [
        {"property": "rotation", "at": 0.0, "value": 0.0},
        {"property": "rotation", "at": 1.0, "value": 8.0},
    ],
}


def normalize(name: str | None) -> str:
    """Chuẩn hoá tên để tra bảng: thường hoá, NFC, gộp khoảng trắng."""
    if not name:
        return ""
    text = unicodedata.normalize("NFC", str(name)).strip().lower()
    text = re.sub(r"[\s_\-]+", " ", text)
    return text.strip()


_TRANSITION_IDS = frozenset(TRANSITIONS.values())
_EFFECT_IDS = frozenset(EFFECTS.values())


def is_cut(name: str | None) -> bool:
    return normalize(name) in CUT_NAMES


def transition_id(name: str | None) -> str | None:
    """Trả về id transition Concat, hoặc None nếu là cut/không map được."""
    raw = str(name or "").strip().lower()
    if raw in _TRANSITION_IDS:
        return raw
    key = normalize(name)
    if not key or key in CUT_NAMES:
        return None
    return TRANSITIONS.get(key) or TRANSITIONS.get(key.replace(" ", ""))


def effect_id(name: str | None) -> str | None:
    raw = str(name or "").strip().lower()
    if raw in _EFFECT_IDS:
        return raw
    key = normalize(name)
    if not key:
        return None
    return EFFECTS.get(key) or EFFECTS.get(key.replace(" ", ""))


def font_family(name: str | None) -> str | None:
    key = normalize(name)
    if not key:
        return None
    direct = FONTS.get(key) or FONTS.get(key.replace(" ", ""))
    if direct:
        return direct
    compact = key.replace(" ", "")
    for alias, family in FONTS.items():
        if alias.replace(" ", "") == compact:
            return family
    return None


def map_transition(name: str | None) -> tuple[str, bool]:
    """Trả về ``(effect_id, mapped)``; fallback ``("cut", False)``."""
    raw = str(name or "").strip().lower()
    if raw in _TRANSITION_IDS:
        return raw, True
    key = normalize(name)
    if not key or key in CUT_NAMES:
        return "cut", True
    effect = TRANSITIONS.get(key) or TRANSITIONS.get(key.replace(" ", ""))
    if effect is None:
        return "cut", False
    return effect, True


def map_effect(name: str | None) -> tuple[str | None, bool]:
    raw = str(name or "").strip().lower()
    if raw in _EFFECT_IDS:
        return raw, True
    key = normalize(name)
    if not key:
        return None, True
    effect = EFFECTS.get(key) or EFFECTS.get(key.replace(" ", ""))
    return (effect, True) if effect else (None, False)


def map_font(name: str | None) -> tuple[str | None, bool]:
    family = font_family(name)
    if family:
        return family, True
    return (normalize(name), False) if normalize(name) else (None, True)


def keyframes_for(effect: str | None) -> list[dict[str, Any]]:
    if not effect:
        return []
    return [dict(frame) for frame in KEYFRAMES.get(effect, [])]
