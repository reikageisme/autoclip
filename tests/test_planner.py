from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoclip.config import Settings
from autoclip.contracts import ClipInfo, SegmentInfo
from autoclip.diversity import make_variant
from autoclip.planner import LLMPlanner, PlanError, Pool, RulePlanner
from autoclip.planner.base import HEADLINE_MAX_CHARS, clean_headline
from autoclip.recipes.compiler import compile_plan
from autoclip.render import load_recipe

RECIPE = Path(__file__).resolve().parents[1] / "recipes" / "hook-3cut-caption.json"


def make_pool(tmp_path: Path, n_clips: int = 4, per_clip: int = 6) -> Pool:
    clips, segs, sid = {}, [], 1
    for cid in range(1, n_clips + 1):
        f = tmp_path / f"c{cid}.mp4"
        f.write_bytes(b"x")
        clips[cid] = ClipInfo(
            id=cid,
            path=str(f),
            sha1=f"{cid:040d}",
            duration=30.0,
            width=884,
            height=1918,
            fps=30.0,
            has_audio=True,
        )
        for k in range(per_clip):
            segs.append(
                SegmentInfo(
                    id=sid,
                    clip_id=cid,
                    start=k * 4.0,
                    end=k * 4.0 + 4.0,
                    tags=["speech", "action"] if k % 2 == 0 else ["action"],
                    hook_score=0.4 + 0.1 * (k % 3),
                    transcript=f"đoạn {sid} hàng về đúng hẹn mỗi ngày" if k % 2 == 0 else "",
                )
            )
            sid += 1
    return Pool(segs, clips)


def test_rule_planner_reproducible_distinct_and_compilable(tmp_path):
    spec, pool = load_recipe(RECIPE), make_pool(tmp_path)
    a = RulePlanner().plan(spec, pool, 7, set())
    assert a == RulePlanner().plan(spec, pool, 7, set())
    b = RulePlanner().plan(spec, pool, 8, set())
    assert [i.segment_id for i in a.items] != [i.segment_id for i in b.items]
    ids = [i.segment_id for i in a.items]
    assert len(ids) == len(set(ids))
    assert a.params["planner"] == "rules"
    compile_plan(spec, a)


def test_exclude_is_respected_and_exhaustion_raises(tmp_path):
    spec, pool = load_recipe(RECIPE), make_pool(tmp_path)
    exclude = {s.id for s in pool.segments[:10]}
    plan = RulePlanner().plan(spec, pool, 3, exclude)
    assert not {i.segment_id for i in plan.items} & exclude
    with pytest.raises(PlanError):
        RulePlanner().plan(spec, pool, 3, {s.id for s in pool.segments})


def test_variant_differs_by_seed():
    spec = load_recipe(RECIPE)
    assert make_variant(spec, 1) == make_variant(spec, 1)
    variants = {(v.color, v.caption_y, v.angle) for v in (make_variant(spec, s) for s in range(20))}
    assert len(variants) > 10


def test_clean_headline():
    assert clean_headline(' "hàng về đúng hẹn" ') == "HÀNG VỀ ĐÚNG HẸN"
    assert len(clean_headline("rất " * 30)) <= HEADLINE_MAX_CHARS


class FakeLLM:
    model = "fake"
    available = True

    def __init__(self, answer):
        self.answer = answer
        self.prompts: list[str] = []

    def chat(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.answer(prompt) if callable(self.answer) else self.answer


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "data")


def _pick_from_prompt(prompt: str, n_slots: int, reverse: bool = False) -> list[int]:
    lines = prompt.split("ỨNG VIÊN (")[-1].splitlines()[1:]
    ids = [int(line.split("|")[0]) for line in lines if "|" in line]
    return (ids[::-1] if reverse else ids)[:n_slots]


def test_llm_planner_uses_answer_and_caches(tmp_path):
    spec, pool = load_recipe(RECIPE), make_pool(tmp_path)
    n_slots = len(RulePlanner().pick(spec, pool, 5, set()))

    def answer(prompt):
        ids = _pick_from_prompt(prompt, n_slots, reverse=True)
        return json.dumps(
            {"picks": ids, "headline": "Hàng về đúng hẹn", "title": "Nhập hàng #logistics"}
        )

    llm = FakeLLM(answer)
    planner = LLMPlanner(llm=llm, settings=_settings(tmp_path))
    plan = planner.plan(spec, pool, 5, set())
    assert plan.params["planner"] == "llm"
    assert plan.params["title"] == "Nhập hàng #logistics"
    assert any(t.text == "HÀNG VỀ ĐÚNG HẸN" for t in plan.texts)
    assert [i.segment_id for i in plan.items] == _pick_from_prompt(llm.prompts[0], n_slots, True)
    # lần 2: lấy từ cache, không gọi LLM nữa, cùng kết quả
    again = planner.plan(spec, pool, 5, set())
    assert again == plan and len(llm.prompts) == 1


@pytest.mark.parametrize(
    "answer",
    ["không phải json", json.dumps({"picks": [1, 1, 1]}), json.dumps({"picks": [99999] * 3})],
)
def test_llm_planner_falls_back_to_rules(tmp_path, answer):
    spec, pool = load_recipe(RECIPE), make_pool(tmp_path)
    plan = LLMPlanner(llm=FakeLLM(answer), settings=_settings(tmp_path)).plan(spec, pool, 5, set())
    rules = RulePlanner().plan(spec, pool, 5, set())
    assert plan.params["planner"] == "rules(fallback)"
    assert [i.segment_id for i in plan.items] == [i.segment_id for i in rules.items]


def test_voiceover_prefers_scene_matching_line(tmp_path):
    from autoclip.planner.voiceover import Line, plan_voiceover
    from autoclip.vision_tag import line_scenes

    spec, pool = load_recipe(RECIPE), make_pool(tmp_path)
    target = pool.segments[17]
    target.tags = [*target.tags, "scene:container"]
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    assert line_scenes("Container chạy đều mỗi tuần") == {"container"}
    plan = plan_voiceover(spec, pool, 5, set(), [Line("Container chạy đều mỗi tuần", wav, 3.0)])
    assert plan.items[0].segment_id == target.id
    assert plan.params["mute_source"] and plan.params["voiceover"][0]["start"] == 0.0
