"""SSE 流式对话 API — 异步 agentic loop + 工具调用。"""

import json
import os
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
import db
import llm
import tools
from utils import BaseLogger

log = BaseLogger.getLogger("chat")

router = APIRouter(prefix="/api", tags=["chat"])


_DEFAULT_SYSTEM_PROMPT = "你是一个金融分析助手"


def _load_system_prompt() -> str:
    """加载系统提示词，优先使用 prompts/rag_system.txt，支持 {cwd} 占位符。"""
    prompt_path = os.path.join(config.BASE_DIR, "prompts", "rag_system.txt")
    if os.path.isfile(prompt_path):
        with open(prompt_path, encoding="utf-8") as f:
            text = f.read()
        return text.replace("{cwd}", config.BASE_DIR)
    return _DEFAULT_SYSTEM_PROMPT


SYSTEM_PROMPT = _load_system_prompt()
log.info("系统提示词加载完成，长度: %d", len(SYSTEM_PROMPT))


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
        if role == "tool":
            continue
        entry: dict = {"role": role, "content": content}
        if msg.get("tool_calls"):
            try:
                tc_list = json.loads(msg["tool_calls"]) if isinstance(msg["tool_calls"], str) else msg["tool_calls"]
                entry["tool_calls"] = tc_list
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
    llm_messages = list(messages)
    max_turns = 20

    for turn in range(max_turns):
        content_parts: list[str] = []
        tool_calls_map: dict[int, dict] = {}  # index -> {id, name, arguments}

        async for chunk in llm.async_stream_chat(
            llm_messages, SYSTEM_PROMPT, tools=tools_schema, model=model,
        ):
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})

            text = delta.get("content")
            if text:
                content_parts.append(text)
                yield _sse("token", {"content": text})

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

        if not tool_calls_map:
            break

        assistant_msg: dict = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
            "tool_calls": [tool_calls_map[i] for i in sorted(tool_calls_map)],
        }
        llm_messages.append(assistant_msg)

        for idx in sorted(tool_calls_map):
            tc = tool_calls_map[idx]
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            log.info("工具调用 [%d/%d]: %s(%s)", idx + 1, len(tool_calls_map), tool_name, json.dumps(tool_args, ensure_ascii=False)[:200])

            yield _sse("tool_start", {"tool": tool_name, "args": tool_args})

            tool_result = tools.run_tool(tool_name, tool_args)
            log.info("工具结果 [%s]: %s", tool_name, str(tool_result)[:200])

            yield _sse("tool_end", {"tool": tool_name, "result": tool_result})

            llm_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(tool_result),
            })

        if turn == max_turns - 1:
            log.warning("agentic loop 达到最大轮次 %d", max_turns)


@router.post("/chat")
async def chat(req: ChatRequest):
    """SSE 流式对话接口。"""
    log.info("收到对话请求: session=%s, model=%s, 消息长度=%d", req.session_id, req.model, len(req.message))

    session = await db.get_session(req.session_id)
    if not session:
        raise HTTPException(404, "会话不存在")

    await db.add_message(req.session_id, "user", req.message)

    messages_raw = await db.get_messages(req.session_id)
    if len(messages_raw) == 1:
        title = req.message[:30].replace("\n", " ")
        if len(req.message) > 30:
            title += "..."
        await db.update_session(req.session_id, title=title)
        log.info("新会话自动标题: %s", title)

    llm_messages = _messages_to_llm_format(messages_raw)
    log.info("加载历史消息 %d 条，转换为 %d 条 LLM 格式", len(messages_raw), len(llm_messages))

    async def stream():
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

        log.info("对话完成: session=%s, message_id=%d, 回复长度=%d", req.session_id, message_id, len(content))

        # 发送完成事件
        yield _sse("done", {"message_id": message_id})

    return StreamingResponse(stream(), media_type="text/event-stream")
