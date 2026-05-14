"""统一提示词管理 — 所有提示词按模块组织，按需导入。"""

from prompts.main_prompts import rewrite_queries_prompt, system_prompt
from prompts.financial_report_analysis import (
    analyze_prompt,
    analyze_system_prompt,
    reduce_prompt,
    reduce_system_prompt,
)

__all__ = [
    "system_prompt",
    "rewrite_queries_prompt",
    "analyze_prompt",
    "analyze_system_prompt",
    "reduce_prompt",
    "reduce_system_prompt",
]
