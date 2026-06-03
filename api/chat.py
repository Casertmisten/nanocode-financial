"""SSE 流式对话 API — 异步 agentic loop + 工具调用。"""

import json
import httpx
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
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
    retry: bool = False


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
    """异步 agentic loop，yield (event_type, data) 元组。含错误恢复机制。"""
    tools_schema = tools.make_schema()
    llm_messages = list(messages)
    max_turns = config.AGENT_MAX_TURNS
    _sys = system_prompt or SYSTEM_PROMPT
    _max_tokens = config.LLM_MAX_TOKENS

    for turn in range(max_turns):
        content_parts: list[str] = []
        tool_calls_map: dict[int, dict] = {}
        usage_turn: dict = {}
        finish_reason: str | None = None

        # ── 错误恢复：LLM 调用重试循环 ──
        _rate_retries = 0
        _ctx_retries = 0

        while True:
            try:
                async for chunk in llm.async_stream_chat(
                    llm_messages, _sys, tools=tools_schema, model=model,
                    usage_out=usage_turn, max_tokens=_max_tokens,
                ):
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})

                    # 捕获 finish_reason
                    fr = choice.get("finish_reason")
                    if fr:
                        finish_reason = fr

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

                break  # 调用成功，退出重试循环

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                # 429 限流：间隔 2 秒重试，最多 5 次
                if status_code == 429:
                    _rate_retries += 1
                    if _rate_retries >= 5:
                        yield ("error", {"code": 429, "message": "模型限流，请稍后重试"})
                        return
                    yield ("status", {"message": f"模型限流，2秒后重试 ({_rate_retries}/5)..."})
                    await asyncio.sleep(2)
                    content_parts.clear()
                    tool_calls_map.clear()
                    usage_turn.clear()
                    finish_reason = None
                    continue

                # 401：API Key 错误，直接终止
                if status_code == 401:
                    yield ("error", {"code": 401, "message": "API Key 填写错误，请检查配置"})
                    return

                # 403：模型欠费或无法使用，直接终止
                if status_code == 403:
                    yield ("error", {"code": 403, "message": "模型欠费或无法使用"})
                    return

                # 400：检查是否为上下文超限
                if status_code == 400:
                    err_text = ""
                    try:
                        err_text = e.response.text
                    except Exception:
                        pass
                    if any(kw in err_text.lower() for kw in
                           ("context", "token", "length", "too many", "maximum")):
                        _ctx_retries += 1
                        if _ctx_retries > 2:
                            yield ("error", {"code": "context_overflow",
                                             "message": "上下文长度超限，压缩后仍无法继续"})
                            return
                        yield ("status", {"message": "上下文超限，正在压缩..."})
                        # 使用已有的压缩逻辑
                        if session_id:
                            try:
                                messages_raw = await db.get_messages(session_id)
                                compressed, _ = await memory_module.inject_memory(
                                    session_id, messages_raw, "")
                                llm_messages = _messages_to_llm_format(compressed)
                            except Exception:
                                llm_messages = llm_messages[-10:]
                        else:
                            llm_messages = llm_messages[-10:]
                        content_parts.clear()
                        tool_calls_map.clear()
                        usage_turn.clear()
                        finish_reason = None
                        continue
                    yield ("error", {"code": 400, "message": f"请求错误: {err_text[:200]}"})
                    return

                yield ("error", {"code": status_code, "message": str(e)})
                return

            except Exception as e:
                log.warning("LLM 调用异常: %s", e)
                yield ("error", {"code": "unknown", "message": f"未知错误: {e}"})
                return

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

        # ── 错误恢复：max_tokens 用完，模型话说一半 ──
        if finish_reason == "length":
            if not tool_calls_map:
                # 纯文本截断：追加已输出内容，让模型继续生成
                _max_tokens = min(_max_tokens * 2, config.LLM_MAX_TOKENS * 4)
                partial = "".join(content_parts)
                if partial:
                    llm_messages.append({"role": "assistant", "content": partial})
                    llm_messages.append({"role": "user", "content": "请继续"})
                yield ("status", {"message": "输出被截断，正在继续生成..."})
                content_parts.clear()
                continue
            else:
                # 工具调用截断：扩充 max_tokens 重试整个 turn
                _max_tokens = min(_max_tokens * 2, config.LLM_MAX_TOKENS * 4)
                yield ("status", {"message": "工具调用被截断，正在重试..."})
                content_parts.clear()
                tool_calls_map.clear()
                usage_turn.clear()
                finish_reason = None
                continue

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
async def chat(req: ChatRequest, request: Request):
    """SSE 流式对话接口。"""
    log.info("收到对话请求: session=%s, model=%s, retry=%s, 消息长度=%d", req.session_id, req.model, req.retry, len(req.message))

    session = await db.get_session(req.session_id)
    if not session:
        raise HTTPException(404, "会话不存在")

    if req.retry:
        # 重试：删除最后一条用户消息之后的所有消息（assistant/tool），不重复添加用户消息
        messages_raw = await db.get_messages(req.session_id)
        last_user = None
        for m in reversed(messages_raw):
            if m["role"] == "user":
                last_user = m
                break
        if last_user:
            deleted = await db.delete_messages_after(req.session_id, last_user["id"])
            log.info("重试模式: 删除 %d 条旧消息", deleted)
        messages_raw = await db.get_messages(req.session_id)
    else:
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

            def _make_wf_callback(queue: asyncio.Queue):
                """构造工作流进度回调（纯异步，无需 call_soon_threadsafe）。"""
                def _cb(*args):
                    if len(args) == 1 and isinstance(args[0], dict):
                        ev = args[0]
                        queue.put_nowait((ev.get("type", "workflow_event"), ev))
                    elif len(args) == 3:
                        step_name, idx, total = args
                        queue.put_nowait(("workflow_step", {"step": step_name, "idx": idx, "total": total}))
                return _cb

            async def _run_workflow_and_drain(run_fn, wf_name, label, entities=None):
                """通用工作流执行 + SSE 事件推送。"""
                yield _sse("workflow_start", {"workflow": wf_name, "label": label})
                wf_queue: asyncio.Queue = asyncio.Queue()
                _wf_cb = _make_wf_callback(wf_queue)

                async def _drain():
                    # entities=None 的工作流（如 FRA）只传 query + progress_cb
                    if entities is None:
                        report = await run_fn(req.message, _wf_cb)
                    else:
                        report = await run_fn(req.message, entities, _wf_cb)
                    await wf_queue.put(("workflow_done", {"report": report}))
                    if req.session_id:
                        try:
                            await db.add_message(req.session_id, "assistant", report)
                        except Exception:
                            pass

                drain_task = asyncio.create_task(_drain())
                try:
                    while True:
                        try:
                            ev_type, ev_data = await asyncio.wait_for(wf_queue.get(), timeout=0.5)
                        except asyncio.TimeoutError:
                            if drain_task.done():
                                break
                            continue
                        if ev_type == "workflow_done":
                            break  # 报告已通过 token 事件流式推送
                        yield _sse(ev_type, ev_data)
                finally:
                    if not drain_task.done():
                        drain_task.cancel()

            intent_result = classify_intent(req.message)
            log.info("意图识别: intent=%s, entities=%s", intent_result.intent, intent_result.entities)

            if intent_result.intent == "stock_investment":
                import stock_investment
                async for sse_str in _run_workflow_and_drain(
                    stock_investment.run, "stock_investment", "个股投资决策",
                    entities=intent_result.entities,
                ):
                    yield sse_str
                routed = True

            elif intent_result.intent == "sector_rotation":
                import sector_rotation
                async for sse_str in _run_workflow_and_drain(
                    sector_rotation.run, "sector_rotation", "行业轮动与机会发现",
                    entities=intent_result.entities,
                ):
                    yield sse_str
                routed = True

            elif intent_result.intent == "fra":
                has_docs = await _has_financial_docs(req.doc_ids)
                if has_docs:
                    import financial_report_analysis
                    async for sse_str in _run_workflow_and_drain(
                        financial_report_analysis.run, "fra", "财报深度分析",
                    ):
                        yield sse_str
                    routed = True
                else:
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
                # 客户端断开则立即停止
                if await request.is_disconnected():
                    log.info("客户端断开，停止生成: session=%s", req.session_id)
                    break
            else:
                log.info("对话完成: session=%s, message_id=%s", req.session_id, message_id)
                yield _sse("done", {"message_id": message_id})
            return

        log.info("对话完成(工作流): session=%s", req.session_id)

    return StreamingResponse(stream(), media_type="text/event-stream")
