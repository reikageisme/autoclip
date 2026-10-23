from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoclip.contracts import RenderItem, RenderPlan, RenderText
from autoclip.recipes.capcut_reader import load
from autoclip.recipes.compiler import compile_plan, planned_duration
from autoclip.recipes.effect_map import map_transition, transition_id
from autoclip.recipes.extract import extract_recipe

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "capcut"
SNAPSHOTS = Path(__file__).resolve().parent / "snapshots"


def _plan() -> RenderPlan:
    return RenderPlan(
        recipe="basic-hook-recipe",
        seed=7,
        items=[
            RenderItem(
                slot_role="hook",
                segment_id=1,
                clip_path="C:/media/hook.mp4",
                src_start=0.0,
                src_end=2.0,
                media_duration=2.0,
                speed=1.0,
            ),
            RenderItem(
                slot_role="body",
                segment_id=2,
                clip_path="C:/media/body.mp4",
                src_start=1.5,
                src_end=5.5,
                media_duration=12.0,
                speed=1.3,
                transition="Zoom",
            ),
            RenderItem(
                slot_role="outro",
                segment_id=3,
                clip_path="C:/media/outro.mp4",
                src_start=0.0,
                src_end=2.0,
                media_duration=6.0,
                speed=1.0,
                transition="Glitch Pro",
            ),
        ],
        texts=[
            RenderText(
                kind="headline",
                text="WOW!",
                slot="hook",
                style="bold-stroke",
                pos=(0.5, 0.2),
                start=0.1,
                end=2.2,
            ),
            RenderText(
                kind="captions",
                text="hello",
                style="word-pop",
                pos=(0.5, 0.85),
                start=2.3,
                end=3.0,
            ),
            RenderText(
                kind="captions",
                text="world",
                style="word-pop",
                pos=(0.5, 0.85),
                start=3.1,
                end=3.8,
            ),
        ],
        params={
            "project_path": "D:/work/autoclip/data/tmp/basic",
            "palette": ["#ffffff"],
            "fonts": ["Inter"],
            "effects": [
                {"effect_id": "concat.vignette", "start": 0.0, "duration": 7.0},
            ],
        },
    )


def test_compile_basic_matches_snapshot() -> None:
    spec = extract_recipe(load(FIXTURES / "basic"))
    commands = compile_plan(spec, _plan())
    actual = [command.model_dump(mode="json") for command in commands]
    snapshot = json.loads((SNAPSHOTS / "basic_commands.json").read_text(encoding="utf-8"))
    assert actual == snapshot


def test_plan_duration_fits_recipe_target() -> None:
    spec = extract_recipe(load(FIXTURES / "basic"))
    duration = planned_duration(_plan())
    low, high = spec.duration_target
    assert low <= duration <= high


def test_compile_command_sequence() -> None:
    spec = extract_recipe(load(FIXTURES / "basic"))
    commands = compile_plan(spec, _plan())
    tools = [command.tool for command in commands]
    assert tools[0] == "project_set_video"
    assert tools.count("media_import") == 3
    assert tools.count("clips_add_sequence") == 3
    assert tools.count("clip_speed") == 1
    assert "clip_update" in tools
    assert tools.count("keyframe_set") == 2
    assert tools.count("title_add") == 1
    assert tools.count("captions_add") == 1
    assert tools.count("effect_layer_add") == 1
    assert all(command.args["path"] == "D:/work/autoclip/data/tmp/basic" for command in commands)
    unknow = [command for command in commands if command.tool == "clip_update"]
    assert len(unknow) == 1
    assert unknow[0].args["patch"]["transitionIn"]["id"] == "concat.zoom-transition"


def test_compile_unknown_transition_falls_back_to_cut() -> None:
    spec = extract_recipe(load(FIXTURES / "basic"))
    plan = _plan()
    plan.items[1].transition = "Definitely Not A Transition"
    commands = compile_plan(spec, plan)
    assert not any(command.tool == "clip_update" for command in commands)


def test_compile_empty_plan_raises() -> None:
    spec = extract_recipe(load(FIXTURES / "basic"))
    plan = RenderPlan(recipe=spec.name, seed=1, items=[])
    with pytest.raises(ValueError):
        compile_plan(spec, plan)


def test_transition_ids_accept_canonical_names() -> None:
    assert transition_id("concat.zoom-transition") == "concat.zoom-transition"
    assert map_transition("Zoom") == ("concat.zoom-transition", True)
    assert map_transition("Not a transition") == ("cut", False)


TOOLS = {
    t["name"]: t["inputSchema"]
    for t in json.loads(
        (Path(__file__).resolve().parents[1] / "fixtures" / "concat_tools.json").read_text(
            encoding="utf-8"
        )
    )
}


def test_every_command_matches_concat_schema() -> None:
    """Đối chiếu với tools/list thật: đúng tên tool, đủ khoá bắt buộc, không khoá lạ."""
    spec = extract_recipe(load(FIXTURES / "basic"))
    for command in compile_plan(spec, _plan()):
        assert command.tool in TOOLS, command.tool
        schema = TOOLS[command.tool]
        sent = {k for k in command.args if not k.startswith("@")}
        assert not sent - set(schema.get("properties", {})), (command.tool, sent)
        assert not set(schema.get("required", [])) - sent, (command.tool, sent)


def test_items_are_trimmed_to_source_range_and_chained() -> None:
    spec = extract_recipe(load(FIXTURES / "basic"))
    commands = compile_plan(spec, _plan())
    adds = [c for c in commands if c.tool == "clips_add_sequence"]
    # hook 2s, body 4s @1.3 = 3.077s, outro 2s
    assert [a.args["start"] for a in adds] == [0.0, 2.0, 5.077]
    assert [a.args["@as"] for a in adds] == ["clip:0", "clip:1", "clip:2"]
    trims = [
        (c.args["clipId"], c.args["edge"], c.args["delta"])
        for c in commands
        if c.tool == "clip_trim"
    ]
    assert trims == [("@clip:1", "start", 1.5), ("@clip:1", "end", -6.5), ("@clip:2", "end", -4.0)]
    order = [c.tool for c in commands if c.args.get("clipId") == "@clip:1"]
    assert order.index("clip_speed") > order.index("clip_trim")


def test_missing_media_duration_raises() -> None:
    spec = extract_recipe(load(FIXTURES / "basic"))
    plan = _plan()
    plan.items[0].media_duration = None
    with pytest.raises(ValueError):
        compile_plan(spec, plan)
