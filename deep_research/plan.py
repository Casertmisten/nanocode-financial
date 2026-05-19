"""Plan Agent — 将研究问题拆解为子任务列表。"""

import json
import logging

import llm
from deep_research.prompts import plan_system_prompt
from deep_research.schemas import SubTask

log = logging.getLogger(__name__)


def plan(query: str, usage_out: dict | None = None) -> tuple[str, list[SubTask]]:
    """执行 Plan Agent，返回研究框架和子任务列表。"""
    log.info("Plan Agent 开始: query=%s", query[:80])

    user_prompt = f"请将以下研究问题拆解为子任务：\n\n{query}"
    response = llm.call_llm(plan_system_prompt, user_prompt, usage_out=usage_out)

    # 解析 JSON（兼容 LLM 输出前后可能有的 markdown 代码块标记）
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        log.error("Plan JSON 解析失败: %s, 原始响应: %s", e, text[:300])
        raise ValueError(f"Plan Agent 返回格式错误: {e}") from e

    outline = data.get("research_outline", "")
    raw_tasks = data.get("sub_tasks", [])

    sub_tasks = []
    for i, t in enumerate(raw_tasks):
        sub_tasks.append(SubTask(
            id=t.get("id", i + 1),
            title=t.get("title", f"子任务 {i + 1}"),
            description=t.get("description", ""),
            tools=t.get("tools", []),
            search_queries=t.get("search_queries", []),
        ))

    log.info("Plan 完成: %d 个子任务, 框架: %s", len(sub_tasks), outline[:50])
    return outline, sub_tasks
