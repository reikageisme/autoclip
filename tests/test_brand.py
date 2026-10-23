from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from autoclip.brand import BrandKit, apply_brand, load_brand, save_brand
from autoclip.concat_client.executor import execute
from autoclip.contracts import ConcatCommand, RenderItem, RenderPlan, RenderText
from autoclip.recipes.compiler import compile_plan
from autoclip.render import load_recipe

ROOT = Path(__file__).resolve().parents[1]
TOOLS = {
    t["name"]: t["inputSchema"]
    for t in json.loads(
        (ROOT / "tests" / "fixtures" / "concat_tools.json").read_text(encoding="utf-8")
    )
}


def _plan(tmp_path: Path) -> RenderPlan:
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    return RenderPlan(
        recipe="hook-3cut-caption",
        seed=3,
        items=[
            RenderItem(
                slot_role="hook",
                segment_id=1,
                clip_path=str(clip),
                src_start=0,
                src_end=6,
                media_duration=20,
                speed=1.0,
            ),
            RenderItem(
                slot_role="outro",
                segment_id=2,
                clip_path=str(clip),
                src_start=6,
                src_end=10,
                media_duration=20,
                speed=1.0,
            ),
        ],
        texts=[RenderText(kind="captions", text="xin chào", start=0, end=2)],
    )


@pytest.fixture
def music(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg required")
    folder = tmp_path / "music"
    folder.mkdir()
    for name, sec in (("a.mp3", 30), ("b.mp3", 4)):
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=duration={sec}",
                str(folder / name),
            ],
            check=True,
        )
    return folder


def test_brand_roundtrip(tmp_path):
    assert load_brand(tmp_path) == BrandKit()
    kit = BrandKit(outro_enabled=True, outro_text="GIANG MINH\nHotline 0909", music_volume=0.2)
    save_brand(tmp_path, kit)
    assert load_brand(tmp_path) == kit


def test_brand_commands_are_valid_and_complete(tmp_path, music):
    logo = tmp_path / "logo.png"
    logo.write_bytes(b"png")
    kit = BrandKit(
        logo_path=str(logo),
        logo_corner="bottom-left",
        outro_enabled=True,
        outro_text="GIANG MINH\nHotline 0909",
        outro_seconds=3,
        music_folder=str(music),
    )
    plan = apply_brand(_plan(tmp_path), kit)
    brand = plan.params["brand"]
    assert set(brand) == {"logo", "outro", "music"}
    assert brand["music"]["volume"] == kit.music_volume_speech  # có phụ đề = có lời
    assert apply_brand(_plan(tmp_path), kit).params == plan.params  # cùng seed, cùng bài

    cmds = compile_plan(load_recipe(ROOT / "recipes" / "hook-3cut-caption.json"), plan)
    for c in cmds:
        sent = {k for k in c.args if not k.startswith("@")}
        assert not sent - set(TOOLS[c.tool].get("properties", {})), (c.tool, sent)
        assert not set(TOOLS[c.tool].get("required", [])) - sent - {"path"}, (c.tool, sent)
    tools = [c.tool for c in cmds]
    assert "timeline_background" in tools and tools.count("track_add") == 3  # video, logo, nhạc
    logo_tf = next(
        c for c in cmds if c.tool == "clip_transform" and c.args["clipId"] == "@clip:logo"
    )
    assert logo_tf.args["offsetX"] < 0 and logo_tf.args["offsetY"] > 0  # góc dưới-trái
    outro = [c for c in cmds if c.tool == "title_add"][-1]
    assert outro.args["start"] == 10.0 and outro.args["duration"] == 3.0


class FakeCC:
    def __init__(self):
        from mcp import types

        schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        self.tools = {
            n: types.Tool(name=n, input_schema=schema) for n in ("timeline_background", "track_add")
        }
        self.calls = []

    def call_tool(self, name, args=None):
        self.calls.append((name, args))
        if name == "timeline_read":
            return {"timeline": {"id": "t9"}}
        return {"createdId": "x1"}


def test_executor_resolves_timeline_and_tracks():
    cc = FakeCC()
    refs = execute(
        cc,
        [
            ConcatCommand(tool="track_add", args={"@as": "track:logo"}),
            ConcatCommand(
                tool="timeline_background",
                args={"timelineId": "@timeline", "background": {"kind": "none"}},
            ),
        ],
        "P",
    )
    assert refs["track:logo"] == "x1" and refs["timeline"] == "t9"
    assert cc.calls[-1][1]["timelineId"] == "t9"
