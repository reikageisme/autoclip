from __future__ import annotations

from pathlib import Path

import pytest

from autoclip.concat_client import ConcatError
from autoclip.concat_client.executor import execute
from autoclip.contracts import ClipInfo, ConcatCommand, SegmentInfo
from autoclip.recipes.compiler import compile_plan
from autoclip.render import PlanError, build_plan, load_recipe

RECIPE = Path(__file__).resolve().parents[1] / "recipes" / "hook-3cut-caption.json"


class Repo:
    def __init__(self, tmp_path: Path, n_clips: int = 3, segs_per_clip: int = 6) -> None:
        self.clips: dict[int, ClipInfo] = {}
        self.segs: list[SegmentInfo] = []
        sid = 1
        for cid in range(1, n_clips + 1):
            f = tmp_path / f"c{cid}.mp4"
            f.write_bytes(b"x")
            self.clips[cid] = ClipInfo(
                id=cid,
                path=str(f),
                sha1=f"{cid:040d}",
                duration=30.0,
                width=1080,
                height=1920,
                fps=30.0,
                has_audio=True,
            )
            for k in range(segs_per_clip):
                self.segs.append(
                    SegmentInfo(
                        id=sid,
                        clip_id=cid,
                        start=k * 4.0,
                        end=k * 4.0 + 4.0,
                        tags=["speech", "action"] if k == 0 else ["action"],
                        hook_score=0.9 if k == 0 else 0.3,
                        transcript="hàng về đúng hẹn mỗi ngày nhé" if k == 0 else "",
                    )
                )
                sid += 1

    def get_clip(self, clip_id: int) -> ClipInfo | None:
        return self.clips.get(clip_id)

    def list_segments(self, filter=None) -> list[SegmentInfo]:
        return list(self.segs)


def test_plan_is_reproducible_and_valid(tmp_path):
    spec = load_recipe(RECIPE)
    repo = Repo(tmp_path)
    a, b = build_plan(spec, repo, 42), build_plan(spec, repo, 42)
    assert a == b
    assert a != build_plan(spec, repo, 43)
    ids = [i.segment_id for i in a.items]
    assert len(ids) == len(set(ids)), "không dùng lại segment trong một video"
    assert a.items[0].slot_role == "hook" and a.items[-1].slot_role == "outro"
    for item in a.items:
        seg = next(s for s in repo.segs if s.id == item.segment_id)
        assert seg.start <= item.src_start < item.src_end <= seg.end + 1e-6
        assert item.media_duration == 30.0
    compile_plan(spec, a)  # compile được, không lỗi


def test_plan_runs_out_of_segments(tmp_path):
    with pytest.raises(PlanError):
        build_plan(load_recipe(RECIPE), Repo(tmp_path, n_clips=1, segs_per_clip=2), 1)


class FakeCC:
    def __init__(self) -> None:
        from mcp import types

        props = {"type": "object", "properties": {"path": {"type": "string"}}}
        self.tools = {
            n: types.Tool(name=n, input_schema=props)
            for n in ("media_import", "clips_add_sequence", "clip_trim")
        }
        self.calls: list[tuple[str, dict]] = []
        self._n = 0

    def call_tool(self, name, args=None):
        self.calls.append((name, args))
        self._n += 1
        return {"createdId": f"id{self._n}"}


def test_executor_resolves_refs_and_strips_directives():
    cc = FakeCC()
    cmds = [
        ConcatCommand(tool="media_import", args={"file": "C:/a.mp4"}),
        ConcatCommand(
            tool="clips_add_sequence",
            args={"mediaIds": ["@media:C:/a.mp4"], "start": 0, "@as": "clip:0"},
        ),
        ConcatCommand(tool="clip_trim", args={"clipId": "@clip:0", "edge": "end", "delta": -1.0}),
    ]
    refs = execute(cc, cmds, "P")
    assert refs == {"media:C:/a.mp4": "id1", "clip:0": "id2"}
    assert cc.calls[1] == ("clips_add_sequence", {"mediaIds": ["id1"], "start": 0, "path": "P"})
    assert cc.calls[2][1]["clipId"] == "id2"


def test_executor_unknown_ref_raises():
    with pytest.raises(ConcatError):
        execute(FakeCC(), [ConcatCommand(tool="clip_trim", args={"clipId": "@clip:9"})], "P")
