"""三层记忆系统：用户画像 / 跨会话记忆 / 当前会话压缩。"""

from memory.profile import load_profile, render_profile_markdown, add_candidate
from memory.session_memory import (
    generate_summary,
    save_session_memory,
    retrieve_memories,
    render_memories_markdown,
)
from memory.context import compress_if_needed


async def inject_memory(session_id: str, messages: list[dict],
                         user_message: str) -> tuple[list[dict], str]:
    """三层记忆注入入口。

    Args:
        session_id: 当前会话 ID
        messages: 从数据库加载的原始消息列表
        user_message: 当前用户输入的消息文本

    Returns:
        (压缩后的消息列表, 追加到 system prompt 的记忆片段)
    """
    # L3: 压缩超长对话
    compressed = await compress_if_needed(session_id, messages)

    # L1: 用户画像
    profile = load_profile()
    profile_md = render_profile_markdown(profile)

    # L2: 跨会话记忆检索
    memories = retrieve_memories(user_message)
    memories_md = render_memories_markdown(memories)

    # 合并注入片段
    parts = []
    if profile_md:
        parts.append(profile_md)
    if memories_md:
        parts.append(memories_md)
    memory_injection = "\n\n".join(parts)

    return compressed, memory_injection


async def save_after_turn(session_id: str, messages: list[dict],
                           session_type: str = "chat"):
    """对话结束后保存记忆（L2 写入 + L1 候选提取）。

    Args:
        session_id: 会话 ID
        messages: 本次会话的全部消息
        session_type: "chat" 或 "deep_research"
    """
    if not messages:
        return

    # 序列化消息文本
    from memory.context import _serialize_messages_for_summary
    text = _serialize_messages_for_summary(messages)

    # L2: 生成摘要并保存
    try:
        summary, topics, profile_candidates = generate_summary(text)
        save_session_memory(session_id, summary, topics, session_type)
    except Exception:
        import logging
        logging.getLogger("memory").warning("保存跨会话记忆失败", exc_info=True)
        return

    # L1: 提取画像候选
    if profile_candidates:
        try:
            add_candidate(session_id, profile_candidates)
        except Exception:
            import logging
            logging.getLogger("memory").warning("保存画像候选失败", exc_info=True)
