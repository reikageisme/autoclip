"""End-to-end: recipe + segment trong DB → RenderPlan → compiler → Concat → MP4.

`build_plan` gọi planner (autoclip.planner): RulePlanner hoặc LLMPlanner, theo seed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from autoclip.concat_client import SyncConcatClient
from autoclip.concat_client.executor import execute, export_mp4
from autoclip.config import Settings, get_settings
from autoclip.contracts import ClipRepository, RecipeSpec, RenderPlan
from autoclip.ingest import probe
from autoclip.planner import PlanError, load_pool, make_planner
from autoclip.recipes.compiler import compile_plan

__all__ = ["PlanError", "RenderResult", "build_plan", "load_recipe", "render"]

log = logging.getLogger(__name__)


def load_recipe(path: Path) -> RecipeSpec:
    return RecipeSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))


def build_plan(
    spec: RecipeSpec,
    repo: ClipRepository,
    seed: int,
    *,
    exclude: set[int] | None = None,
    use_llm: bool = False,
) -> RenderPlan:
    """RenderPlan cho `seed` từ kho segment (LLMPlanner nếu use_llm và LLM đã cấu hình)."""
    return make_planner(use_llm).plan(spec, load_pool(repo), seed, exclude or set())


@dataclass
class RenderResult:
    output: Path
    plan: RenderPlan
    video_sec: float
    render_sec: float
    commands: int


def render(
    spec: RecipeSpec,
    plan: RenderPlan,
    *,
    settings: Settings | None = None,
    name: str | None = None,
) -> RenderResult:
    s = settings or get_settings()
    name = name or f"{spec.name}-s{plan.seed}-{datetime.now():%Y%m%d-%H%M%S}"
    s.projects_dir.mkdir(parents=True, exist_ok=True)
    s.out_dir.mkdir(parents=True, exist_ok=True)
    project = str(s.projects_dir / name)
    output = s.out_dir / f"{name}.mp4"
    commands = compile_plan(spec, plan)

    with SyncConcatClient(s) as cc:
        cc.call_tool("project_create", {"location": str(s.projects_dir), "name": name})
        try:
            execute(cc, commands, project)
            cc.call_tool("project_save", {"path": project})
            t0 = time.perf_counter()
            export_mp4(cc, project, str(output))
            render_sec = time.perf_counter() - t0
        finally:
            try:
                cc.call_tool("project_close", {"path": project, "save": False})
            except Exception as exc:  # đóng project lỗi không được che lỗi thật
                log.warning("project_close: %s", exc)

    return RenderResult(
        output=output,
        plan=plan,
        video_sec=probe(output).duration,
        render_sec=render_sec,
        commands=len(commands),
    )
