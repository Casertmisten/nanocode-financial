"""L3: 当前会话记忆 — token 估算、压缩触发、摘要管理。"""

import json

import db
import config
import llm
from memory.prompts import COMPRESS_PROMPT
from utils import BaseLogger

log = BaseLogger.getLogger("memory.context")

# 中文约 1.5 token/字符，英文约 0.25 token/字符，取折中估算
_CHARS_PER_TOKEN = 2.0


def estimate_tokens(messages: list[dict]) -> int:
    """估算消息列表的总 token 数。"""
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        total += len(content) / _CHARS_PER_TOKEN
        # tool_calls 的 JSON 也算
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total += len(json.dumps(tool_calls, ensure_ascii=False)) / _CHARS_PER_TOKEN
    return int(total)


def _count_rounds(messages: list[dict]) -> int:
    """统计 user/assistant 对话轮数。"""
    count = 0
    for msg in messages:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            count += 1
    # user + assistant 为 1 轮
    return count // 2


def _serialize_messages_for_summary(messages: list[dict]) -> str:
    """将消息序列化为摘要 prompt 所需的文本格式。

    保留工具调用轨迹（工具名+参数），仅用占位符替代工具返回的大段结果。
    """
    import json as _json
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "tool":
            # 工具返回结果直接用占位符替代
            parts.append("[tool结果] （如需完整数据可重新调用工具获取）")
            continue

        if role == "assistant":
            tool_calls_raw = msg.get("tool_calls")
            call_lines = []
            if tool_calls_raw:
                try:
                    tc_list = _json.loads(tool_calls_raw) if isinstance(tool_calls_raw, str) else tool_calls_raw
                    for tc in tc_list:
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        args_str = fn.get("arguments", "{}")
                        if name:
                            try:
                                args = _json.loads(args_str) if isinstance(args_str, str) else args_str
                                # 只取每个参数的前 50 字
                                short_args = {k: str(v)[:50] for k, v in args.items()}
                                call_lines.append(f"  → {name}({', '.join(f'{k}={v}' for k, v in short_args.items())})")
                            except (_json.JSONDecodeError, TypeError):
                                call_lines.append(f"  → {name}(...)")
                except (_json.JSONDecodeError, TypeError):
                    pass
            text = content[:500] if content else ""
            if call_lines:
                text += "\n[工具调用]\n" + "\n".join(call_lines)
            if text:
                parts.append(f"[{role}] {text}")
            continue

        if not content:
            continue
        parts.append(f"[{role}] {content[:500]}")
    return "\n".join(parts)


async def compress_if_needed(session_id: str, messages: list[dict]) -> list[dict]:
    """检查 token 预算，超限时压缩最早的消息。

    Returns:
        压缩后的消息列表（可能包含摘要替代原始消息）
    """
    max_tokens = config.MEMORY_SESSION_MAX_TOKENS
    compress_rounds = config.MEMORY_COMPRESS_ROUNDS
    min_keep_rounds = config.MEMORY_MIN_KEEP_ROUNDS

    result = list(messages)

    # 加载已有摘要
    existing_summaries = await db.get_session_summaries(session_id)
    summarized_ranges = set()
    for s in existing_summaries:
        summarized_ranges.add((s["start_msg_id"], s["end_msg_id"]))

    while estimate_tokens(result) > max_tokens:
        if _count_rounds(result) <= min_keep_rounds:
            log.warning("会话消息已达最小保留轮数，停止压缩: session=%s", session_id)
            break

        # 找出最早的 N 轮对话（跳过已被摘要的和 system 消息）
        to_compress = []
        remaining = []
        round_count = 0
        in_round = False

        for msg in result:
            msg_id = msg.get("id")

            # 如果这条消息已被摘要覆盖，跳过
            if msg_id and any(start <= msg_id <= end for start, end in summarized_ranges):
                continue

            # 不压缩 system 角色消息
            if msg.get("role") == "system":
                remaining.append(msg)
                continue

            if round_count < compress_rounds:
                to_compress.append(msg)
                # user 消息开始一轮
                if msg.get("role") == "user":
                    in_round = True
                # assistant 消息结束一轮
                if in_round and msg.get("role") == "assistant":
                    round_count += 1
                    in_round = False
            else:
                remaining.append(msg)

        if not to_compress or round_count == 0:
            log.info("无可压缩的消息: session=%s", session_id)
            break

        # 生成摘要
        text = _serialize_messages_for_summary(to_compress)
        try:
            summary = llm.call_llm("你是一个对话压缩助手。", COMPRESS_PROMPT.format(messages=text))
        except Exception:
            log.warning("压缩摘要生成失败，停止压缩", exc_info=True)
            break

        # 保存摘要到数据库
        start_id = min((m.get("id", 0) for m in to_compress if m.get("id")), default=0)
        end_id = max((m.get("id", 0) for m in to_compress if m.get("id")), default=0)
        await db.add_session_summary(session_id, start_id, end_id, summary)
        summarized_ranges.add((start_id, end_id))

        # 用摘要消息替代被压缩的消息
        summary_msg = {
            "role": "system",
            "content": f"[对话摘要] {summary}",
        }
        result = [summary_msg] + remaining
        log.info("会话压缩完成: session=%s, 压缩 %d 轮, 摘要长度=%d", session_id, round_count, len(summary))

    return result
