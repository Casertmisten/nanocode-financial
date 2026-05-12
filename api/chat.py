"""SSE 流式对话 API — 异步 agentic loop + 工具调用。"""

import json
import logging
import os
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
import db
import llm
import tools

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

# 默认系统提示词
_DEFAULT_SYSTEM_PROMPT = "你是一个金融分析助手"


def _load_system_prompt() -> str:
    """加载系统提示词，优先使用 prompts/rag_system.txt，支持 {cwd} 占位符。"""
    prompt_path = os.path.join(config.BASE_DIR, "prompts", "rag_system.txt")
    if os.path.isfile(prompt_path):
        text = open(prompt_path, encoding="utf-8").read()
        return text.replace("{cwd}", config.BASE_DIR)
    return _DEFAULT_SYSTEM_PROMPT


SYSTEM_PROMPT = _load_system_prompt()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str | None = None


def _sse(event: str, data: dict) -> str:
    """构造一条 SSE 事件字符串。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _messages_to_llm_format(messages: list[dict]) -> list[dict]:
    """将数据库消息转换为 LLM API 格式，去掉 _session_id 等内部字段。"""
    result = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # 跳过 tool 类型消息（工具结果已嵌入 assistant 消息的 tool_calls 中）
        if role == "tool":
            continue
        entry: dict = {"role": role, "content": content}
        # 如果有 tool_calls，还原为 LLM 格式
        if msg.get("tool_calls"):
            try:
                tc_list = json.loads(msg["tool_calls"]) if isinstance(msg["tool_calls"], str) else msg["tool_calls"]
                entry["tool_calls"] = tc_list
                # assistant 带 tool_calls 时 content 可能为空
                if not content:
                    entry["content"] = None
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(entry)
    return result


async def _agentic_loop(
    messages: list[dict],
    model: str | None,
) -> AsyncIterator[str]:
    """异步 agentic loop，yield SSE 事件字符串。"""
    tools_schema = tools.make_schema()
    # 深拷贝避免修改原始列表
    llm_messages = list(messages)
    # 循环次数上限，防止无限循环
    max_turns = 20

    for _ in range(max_turns):
        # 累积当前轮的 content 和 tool_calls
        content_parts: list[str] = []
        tool_calls_map: dict[int, dict] = {}  # index -> {id, name, arguments}

        # 流式调用 LLM
        async for chunk in llm.async_stream_chat(
            llm_messages, SYSTEM_PROMPT, tools=tools_schema, model=model,
        ):
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})

            # 处理文本 token
            text = delta.get("content")
            if text:
                content_parts.append(text)
                yield _sse("token", {"content": text})

            # 处理工具调用增量
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

        # 如果没有工具调用，本轮结束
        if not tool_calls_map:
            break

        # 按序组装 assistant 消息（含 tool_calls）
        assistant_msg: dict = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
            "tool_calls": [tool_calls_map[i] for i in sorted(tool_calls_map)],
        }
        llm_messages.append(assistant_msg)

        # 执行每个工具调用
        for idx in sorted(tool_calls_map):
            tc = tool_calls_map[idx]
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            # yield 工具开始事件
            yield _sse("tool_start", {"tool": tool_name, "args": tool_args})

            # 执行工具
            tool_result = tools.run_tool(tool_name, tool_args)

            # yield 工具结束事件
            yield _sse("tool_end", {"tool": tool_name, "result": tool_result})

            # 将工具结果添加到消息列表
            llm_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(tool_result),
            })


@router.post("/chat")
async def chat(req: ChatRequest):
    """SSE 流式对话接口。"""
    # 检查会话是否存在
    session = await db.get_session(req.session_id)
    if not session:
        raise HTTPException(404, "会话不存在")

    # 保存用户消息
    await db.add_message(req.session_id, "user", req.message)

    # 首条消息时自动更新会话标题
    existing = await db.get_messages(req.session_id)
    if len(existing) == 1:
        # 取消息前 30 字符作为标题
        title = req.message[:30].replace("\n", " ")
        if len(req.message) > 30:
            title += "..."
        await db.update_session(req.session_id, title=title)

    # 加载历史消息并转换为 LLM 格式
    messages_raw = await db.get_messages(req.session_id)
    llm_messages = _messages_to_llm_format(messages_raw)

    async def stream():
        """SSE 流式生成器。"""
        full_content: list[str] = []

        async for sse_str in _agentic_loop(llm_messages, req.model):
            # 解析事件类型
            lines = sse_str.strip().split("\n")
            event_type = lines[0].replace("event: ", "")
            data_str = lines[1].replace("data: ", "", 1)
            data = json.loads(data_str)

            if event_type == "token":
                full_content.append(data["content"])
                yield sse_str
            elif event_type == "tool_start":
                yield sse_str
            elif event_type == "tool_end":
                yield sse_str
            elif event_type == "done":
                yield sse_str

        # agentic loop 结束，保存 assistant 消息到数据库
        content = "".join(full_content)
        # 从 llm_messages 的最后状态获取 tool_calls 信息
        # 查找最后一条 assistant 消息中的 tool_calls
        saved_tool_calls = None
        for msg in reversed(llm_messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                saved_tool_calls = json.dumps(msg["tool_calls"], ensure_ascii=False)
                break

        message_id = await db.add_message(
            req.session_id, "assistant", content,
            tool_calls=saved_tool_calls,
        )

        # 发送完成事件
        yield _sse("done", {"message_id": message_id})

    return StreamingResponse(stream(), media_type="text/event-stream")
