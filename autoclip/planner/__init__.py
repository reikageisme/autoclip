"""Planner: chọn segment cho từng slot của recipe và điền chữ → `RenderPlan`.

- `RulePlanner`: không LLM — theo tag + hook_score, chữ lấy từ lời thoại. Luôn chạy được.
- `LLMPlanner`: LLM (qua `autoclip.llm`, ví dụ 9router) chọn segment và viết headline/
  tiêu đề; trả lời sai hoặc lỗi mạng → tự rơi về `RulePlanner`.
"""

from __future__ import annotations

from autoclip.planner.base import PlanError, Pool, load_pool
from autoclip.planner.llm_planner import LLMPlanner
from autoclip.planner.rules import RulePlanner

__all__ = ["LLMPlanner", "PlanError", "Pool", "RulePlanner", "load_pool", "make_planner"]


def make_planner(use_llm: bool = True):
    """LLMPlanner nếu LLM đã cấu hình và được phép, ngược lại RulePlanner."""
    if use_llm:
        planner = LLMPlanner()
        if planner.available:
            return planner
    return RulePlanner()
