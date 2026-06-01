"""SSE 流式对话 API — 异步 agentic loop + 工具调用。"""

import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
import db
import llm
import tools
from prompts.main_prompts import system_prompt as _SYSTEM_PROMPT_TEMPLATE
from utils import BaseLogger
import memory as memory_module
import asyncio
from intent import classify_intent
from intent.schemas import IntentResult

log = BaseLogger.getLogger("chat")

router = APIRouter(prefix="/api", tags=["chat"])


SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.replace("{cwd}", config.BASE_DIR)
log.info("系统提示词加载完成，长度: %d", len(SYSTEM_PROMPT))


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str | None = None
    doc_ids: list[int] | None = None
    pro_mode: bool = False


# FRA 守门：检查选中的文档中是否包含财报或研报
_FRA_REQUIRED_TAGS = {"financial", "report"}


async def _has_financial_docs(doc_ids: list[int] | None) -> bool:
    """检查 doc_ids 中是否包含财报或研报类文档。"""
    if not doc_ids:
        return False
    for did in doc_ids:
        doc = await db.get_document(did)
        if not doc:
            continue
        tags = doc.get("tags", "[]")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []
        if any(t in _FRA_REQUIRED_TAGS for t in tags):
            return True
    return False


async def _run_workflow_stream(
    workflow_name: str,
    workflow_fn,
    query: str,
    entities: dict,
    session_id: str,
) -> AsyncIterator[str]:
    """在工作流线程中执行同步工作流，通过 SSE 流式推送进度和结果。"""
    import stock_investment.prompts as si_prompts
    import sector_rotation.prompts as sr_prompts

    # 选择正确的步骤名称映射
    if workflow_name == "stock_investment":
        step_names = si_prompts.STEP_NAMES
    elif workflow_name == "sector_rotation":
        step_names = sr_prompts.STEP_NAMES
    else:
        step_names = {}

    total = len(step_names)

    def progress_cb(step_name: str, idx: int, total_steps: int):
        """进度回调——将进度信息通过队列传递。"""
        pass  # 进度通过 yield 推送，这里不做操作

    # 在线程中执行工作流
    def _exec():
        return workflow_fn(query, entities=entities)

    result = await asyncio.to_thread(_exec)
    return result


def _sse(event: str, data: dict) -> str:
    """构造一条 SSE 事件字符串。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _messages_to_llm_format(messages: list[dict]) -> list[dict]:
    """将数据库消息转换为 LLM API 格式，保留完整的 tool_calls / tool 消息链。"""
    result = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "tool":
            call_id = msg.get("tool_call_id")
            if call_id:
                result.append({"role": "tool", "tool_call_id": call_id, "content": content})
            continue

        entry: dict = {"role": role, "content": content}
        if role == "assistant" and msg.get("tool_calls"):
            try:
                tc_list = json.loads(msg["tool_calls"]) if isinstance(msg["tool_calls"], str) else msg["tool_calls"]
                entry["tool_calls"] = tc_list
                if not content:
                    entry["content"] = None
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(entry)
    return result


async def _record_token_usage(usage: dict, model: str | None, session_id: str | None):
    """记录一轮 token 用量。"""
    if not usage.get("total_tokens"):
        return
    try:
        await db.add_token_usage(
            model=model or config.MODEL,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            source="chat",
            session_id=session_id,
        )
    except Exception as e:
        log.warning("记录 token 用量失败: %s", e)


async def _build_system_prompt(memory_injection: str | None = None) -> str:
    """构建完整系统提示词：替换文档列表 + 追加记忆。"""
    prompt = SYSTEM_PROMPT
    try:
        docs = await db.list_documents(status="ready")
        doc_lines = "\n".join(f"- {d['title']} ({d['file_type']})" for d in docs) if docs else "（知识库为空，暂无文档）"
    except Exception:
        doc_lines = "（知识库为空，暂无文档）"
    prompt = prompt.replace("{documents}", doc_lines)
    if memory_injection:
        prompt += "\n\n" + memory_injection
    return prompt


async def _agentic_loop(
    messages: list[dict],
    model: str | None,
    doc_filepaths: list[str] | None = None,
    session_id: str | None = None,
    system_prompt: str | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """异步 agentic loop，yield (event_type, data) 元组。"""
    tools_schema = tools.make_schema()
    llm_messages = list(messages)
    max_turns = config.AGENT_MAX_TURNS
    _sys = system_prompt or SYSTEM_PROMPT

    for turn in range(max_turns):
        content_parts: list[str] = []
        tool_calls_map: dict[int, dict] = {}
        usage_turn: dict = {}

        async for chunk in llm.async_stream_chat(
            llm_messages, _sys, tools=tools_schema, model=model,
            usage_out=usage_turn,
        ):
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})

            text = delta.get("content")
            if text:
                content_parts.append(text)
                yield ("token", {"content": text})

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

        # 流式结束后若 usage 为空，用字符数估算兜底
        if not usage_turn.get("total_tokens"):
            from memory.context import estimate_tokens
            estimated = estimate_tokens(llm_messages)
            output_chars = sum(len(c) for c in content_parts)
            for tc in tool_calls_map.values():
                output_chars += len(tc.get("function", {}).get("arguments", ""))
            est_output = int(output_chars / 2.0) or 1
            usage_turn.update({
                "prompt_tokens": estimated,
                "completion_tokens": est_output,
                "total_tokens": estimated + est_output,
            })

        if not tool_calls_map:
            await _record_token_usage(usage_turn, model, session_id)
            # 保存最终 assistant 消息（纯文本）
            final_msg_id = None
            if session_id:
                try:
                    final_msg_id = await db.add_message(
                        session_id, "assistant", "".join(content_parts))
                except Exception as e:
                    log.warning("保存最终 assistant 消息失败: %s", e)
            yield ("final", {"message_id": final_msg_id})
            break

        assistant_msg: dict = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
            "tool_calls": [tool_calls_map[i] for i in sorted(tool_calls_map)],
        }
        llm_messages.append(assistant_msg)

        # 模型未输出文本就直接调用工具，通知客户端
        if not content_parts:
            yield ("status", {"message": "正在调用工具..."})

        # 增量保存：assistant 消息（含 tool_calls）
        if session_id:
            try:
                tc_json = json.dumps(assistant_msg["tool_calls"], ensure_ascii=False)
                await db.add_message(session_id, "assistant", assistant_msg["content"] or "",
                                     tool_calls=tc_json)
            except Exception as e:
                log.warning("保存中间 assistant 消息失败: %s", e)

        for idx in sorted(tool_calls_map):
            tc = tool_calls_map[idx]
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            log.info("工具调用 [%d/%d]: %s(%s)", idx + 1, len(tool_calls_map), tool_name, json.dumps(tool_args, ensure_ascii=False)[:200])

            # 为 rag_query 注入文件路径过滤
            if tool_name == "rag_query" and doc_filepaths:
                tool_args["file_paths"] = doc_filepaths

            yield ("tool_start", {"tool": tool_name, "args": tool_args})

            try:
                tool_result = await asyncio.to_thread(tools.run_tool, tool_name, tool_args)
            except Exception as e:
                log.warning("工具执行失败 [%s]: %s", tool_name, e)
                tool_result = f"工具执行失败: {e}"

            log.info("工具结果 [%s]: %s", tool_name, str(tool_result)[:200])

            yield ("tool_end", {"tool": tool_name, "result": tool_result})

            llm_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(tool_result),
            })

            # 增量保存：tool 结果
            if session_id:
                try:
                    await db.add_message(session_id, "tool", str(tool_result),
                                         tool_call_id=tc["id"])
                except Exception as e:
                    log.warning("保存 tool 消息失败: %s", e)

        await _record_token_usage(usage_turn, model, session_id)

        if turn == max_turns - 1:
            log.warning("agentic loop 达到最大轮次 %d", max_turns)
            yield ("final", {"message_id": None})


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

    # 三层记忆注入
    compressed_msgs, memory_injection = await memory_module.inject_memory(
        req.session_id, messages_raw, req.message,
    )
    llm_messages = _messages_to_llm_format(compressed_msgs)
    log.info("加载历史消息 %d 条，转换为 %d 条 LLM 格式", len(messages_raw), len(llm_messages))

    # 解析选中的文档为文件路径
    doc_filepaths: list[str] | None = None
    if req.doc_ids:
        doc_filepaths = []
        for did in req.doc_ids:
            doc = await db.get_document(did)
            if doc and doc.get("filepath"):
                doc_filepaths.append(doc["filepath"])
        log.info("文档过滤: %d 个文档, %d 个文件路径", len(req.doc_ids), len(doc_filepaths))
        if not doc_filepaths:
            doc_filepaths = None

    async def stream():
        message_id = None
        turn_system_prompt = await _build_system_prompt(memory_injection)

        # ── 专业模式：意图识别 + 工作流路由 ──
        routed = False

        if req.pro_mode:
            intent_result = classify_intent(req.message)
            log.info("意图识别: intent=%s, entities=%s", intent_result.intent, intent_result.entities)

            if intent_result.intent == "stock_investment":
                # 个股投资决策工作流
                import stock_investment
                yield _sse("workflow_start", {"workflow": "stock_investment", "label": "个股投资决策"})

                def _progress(step_name, idx, total):
                    pass  # 进度暂不通过 SSE 推送，工作流完成后一次性返回

                report = await asyncio.to_thread(
                    stock_investment.run, req.message,
                    intent_result.entities, _progress,
                )
                yield _sse("token", {"content": report})
                # 保存工作流结果为 assistant 消息
                if req.session_id:
                    try:
                        _, msg_id = await db.add_message(req.session_id, "assistant", report), None
                    except Exception:
                        pass
                routed = True

            elif intent_result.intent == "sector_rotation":
                # 行业轮动工作流
                import sector_rotation
                yield _sse("workflow_start", {"workflow": "sector_rotation", "label": "行业轮动与机会发现"})

                def _progress(step_name, idx, total):
                    pass

                report = await asyncio.to_thread(
                    sector_rotation.run, req.message,
                    intent_result.entities, _progress,
                )
                yield _sse("token", {"content": report})
                if req.session_id:
                    try:
                        await db.add_message(req.session_id, "assistant", report)
                    except Exception:
                        pass
                routed = True

            elif intent_result.intent == "fra":
                # FRA 工作流：需要检查是否勾选了财报/研报文档
                has_docs = await _has_financial_docs(req.doc_ids)
                if has_docs:
                    import financial_report_analysis
                    yield _sse("workflow_start", {"workflow": "fra", "label": "财报深度分析"})

                    report = await asyncio.to_thread(
                        financial_report_analysis.run, req.message,
                    )
                    yield _sse("token", {"content": report})
                    if req.session_id:
                        try:
                            await db.add_message(req.session_id, "assistant", report)
                        except Exception:
                            pass
                    routed = True
                else:
                    # 未勾选财报/研报文档，回退到通用对话
                    log.info("FRA 意图但未勾选财报文档，回退到通用对话")
                    routed = False

        if not routed:
            # 通用对话：走原有 agentic loop
            async for event_type, data in _agentic_loop(
                llm_messages, req.model, doc_filepaths, req.session_id,
                system_prompt=turn_system_prompt,
            ):
                if event_type == "final":
                    message_id = data.get("message_id")
                    continue
                yield _sse(event_type, data)

        log.info("对话完成: session=%s, message_id=%s", req.session_id, message_id)

        yield _sse("done", {"message_id": message_id})

    return StreamingResponse(stream(), media_type="text/event-stream")
