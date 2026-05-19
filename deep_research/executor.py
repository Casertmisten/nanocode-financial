"""Executor Agent — 单子任务多轮工具调用循环。"""

import asyncio
import json
import logging

import config
import tools
from deep_research.prompts import executor_system_prompt
from deep_research.schemas import ExecutorResult, SubTask

log = logging.getLogger(__name__)

# 不允许 Executor 使用的工具（文件操作类）
_BLOCKED_TOOLS = {"read", "write", "edit", "glob", "grep", "bash", "rag_ingest"}


def _filter_schema(allowed_tools: list[str]) -> list[dict]:
    """从完整工具 schema 中筛选允许的工具。"""
    full_schema = tools.make_schema()
    allowed = set(allowed_tools) - _BLOCKED_TOOLS
    return [s for s in full_schema if s["function"]["name"] in allowed]


async def execute_task(sub_task: SubTask) -> ExecutorResult:
    """执行单个子任务，返回搜集结果。"""
    log.info("Executor 启动: task_id=%d, title=%s", sub_task.id, sub_task.title)

    import llm
    task_schema = _filter_schema(sub_task.tools)
    if not task_schema:
        log.warning("子任务 %d 无可用工具: %s", sub_task.id, sub_task.tools)

    user_prompt = f"""研究子任务：{sub_task.title}

详细描述：{sub_task.description}

建议搜索关键词：{', '.join(sub_task.search_queries)}

请使用提供的工具搜集信息，完成这个子任务。"""

    messages = [{"role": "user", "content": user_prompt}]
    raw_data: list[str] = []
    max_turns = config.RESEARCH_EXECUTOR_MAX_TURNS
    content_parts: list[str] = []
    last_turn_had_tools = False
    accumulated_usage: dict = {}  # 累积 token 用量

    for turn in range(max_turns):
        content_parts = []
        tool_calls_map: dict[int, dict] = {}
        usage_turn: dict = {}

        async for chunk in llm.async_stream_chat(
            messages, executor_system_prompt, tools=task_schema if task_schema else None,
            usage_out=usage_turn,
        ):
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})

            text = delta.get("content")
            if text:
                content_parts.append(text)

            tc_deltas = delta.get("tool_calls")
            if tc_deltas:
                for tc in tc_deltas:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    entry = tool_calls_map[idx]
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        entry["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        entry["function"]["arguments"] += fn["arguments"]

        # 无工具调用，LLM 输出最终摘要
        if not tool_calls_map:
            last_turn_had_tools = False
            break

        # 累积 token 用量
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            accumulated_usage[key] = accumulated_usage.get(key, 0) + usage_turn.get(key, 0)

        last_turn_had_tools = True

        # 记录 assistant 消息
        assistant_msg: dict = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
            "tool_calls": [tool_calls_map[i] for i in sorted(tool_calls_map)],
        }
        messages.append(assistant_msg)

        # 执行工具调用
        for idx in sorted(tool_calls_map):
            tc = tool_calls_map[idx]
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            log.info("Executor[%d] 工具调用: %s(%s)", sub_task.id, tool_name,
                     json.dumps(tool_args, ensure_ascii=False)[:100])

            tool_result = await asyncio.to_thread(tools.run_tool, tool_name, tool_args)
            raw_data.append(f"[{tool_name}] {tool_result[:500]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(tool_result),
            })

        if turn == max_turns - 1:
            log.warning("Executor[%d] 达到最大轮次 %d", sub_task.id, max_turns)

    summary = "".join(content_parts) if content_parts else "未能获取有效信息。"

    # 从摘要中提取来源
    sources = []
    in_source_section = False
    for line in summary.split("\n"):
        line = line.strip()
        if "信息来源" in line or "来源" in line and line.startswith("##"):
            in_source_section = True
            continue
        if in_source_section and line.startswith("- "):
            sources.append(line[2:])
        elif in_source_section and line.startswith("##"):
            in_source_section = False

    log.info("Executor[%d] 完成: 摘要长度=%d, 来源数=%d", sub_task.id, len(summary), len(sources))
    return ExecutorResult(
        task_id=sub_task.id,
        title=sub_task.title,
        summary=summary,
        sources=sources,
        raw_data=raw_data,
        complete=not last_turn_had_tools,
        usage=accumulated_usage,
    )
