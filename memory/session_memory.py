"""L2: 跨会话记忆 — 摘要生成、ChromaDB 存取、时间衰减检索。"""

import json
import math
import re
from datetime import datetime, timezone

import chromadb

import config
import llm
from memory.prompts import SUMMARY_PROMPT, CROSS_SESSION_TEMPLATE
from utils import BaseLogger

log = BaseLogger.getLogger("memory.session_memory")


def _get_collection() -> chromadb.Collection:
    """获取 session_memory collection（不存在则创建）。"""
    client = config.get_chroma_client()
    return client.get_or_create_collection(
        name=config.MEMORY_SESSION_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _get_embedding(text: str) -> list[float]:
    """调用 embedding API 获取向量。复用 LlamaIndex 的 embedding 模型。"""
    from rag.indexer import get_index

    _, embed_model = get_index(
        config.CHROMA_PERSIST_DIR,
        config.EMBEDDING_API_URL,
        config.EMBEDDING_API_KEY,
        config.EMBEDDING_MODEL,
    )
    return embed_model.get_text_embedding(text)


def generate_summary(messages_text: str) -> tuple[str, list[str], dict]:
    """调用 LLM 生成会话摘要。

    Returns:
        (summary, topics, profile_candidates)
    """
    prompt = SUMMARY_PROMPT.format(messages=messages_text)
    result = llm.call_llm("你是一个对话摘要助手。", prompt)

    # 解析 XML 标签
    summary = _extract_tag(result, "summary") or result[:200]
    topics_raw = _extract_tag(result, "topics") or "[]"
    candidates_raw = _extract_tag(result, "profile_candidates") or "{}"

    try:
        topics = json.loads(topics_raw)
    except json.JSONDecodeError:
        topics = []

    try:
        profile_candidates = json.loads(candidates_raw)
    except json.JSONDecodeError:
        profile_candidates = {}

    return summary.strip(), topics, profile_candidates


def _extract_tag(text: str, tag: str) -> str | None:
    """提取 XML 标签内容。"""
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def save_session_memory(session_id: str, summary: str, topics: list[str],
                         session_type: str = "chat"):
    """将会话摘要向量化并保存到 ChromaDB。"""
    if not summary.strip():
        log.warning("摘要为空，跳过保存: session=%s", session_id)
        return

    collection = _get_collection()
    embedding = _get_embedding(summary)
    now = datetime.now(timezone.utc).isoformat()

    doc_id = f"sm_{session_id}_{int(datetime.now().timestamp())}"
    collection.add(
        ids=[doc_id],
        documents=[summary],
        embeddings=[embedding],
        metadatas=[{
            "session_id": session_id,
            "created_at": now,
            "session_type": session_type,
            "topics": json.dumps(topics, ensure_ascii=False),
        }],
    )
    log.info("跨会话记忆已保存: session=%s, type=%s, topics=%s", session_id, session_type, topics)


def retrieve_memories(query: str, top_k: int | None = None,
                       inject_k: int | None = None) -> list[dict]:
    """检索相关的跨会话记忆，应用时间衰减。

    Returns:
        [{"summary": str, "session_id": str, "created_at": str, "score": float}]
    """
    top_k = top_k or config.MEMORY_CROSS_SESSION_TOP_K
    inject_k = inject_k or config.MEMORY_CROSS_SESSION_INJECT_K

    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    query_embedding = _get_embedding(query)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        log.warning("跨会话记忆检索失败", exc_info=True)
        return []

    if not results["ids"][0]:
        return []

    now = datetime.now(timezone.utc)
    memories = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # cosine distance → similarity score
        raw_score = 1.0 - distance

        # 时间衰减
        created_at = meta.get("created_at", now.isoformat())
        try:
            created = datetime.fromisoformat(created_at)
            days_since = (now - created).days
        except (ValueError, TypeError):
            days_since = 0

        decay = math.exp(-config.MEMORY_TIME_DECAY_LAMBDA * days_since)
        adjusted_score = raw_score * decay

        memories.append({
            "summary": doc,
            "session_id": meta.get("session_id", ""),
            "created_at": created_at[:10],
            "score": round(adjusted_score, 4),
        })

    # 按衰减后分数排序，取 top inject_k
    memories.sort(key=lambda m: m["score"], reverse=True)
    return memories[:inject_k]


def render_memories_markdown(memories: list[dict]) -> str:
    """将检索到的记忆渲染为 Markdown 注入片段。"""
    if not memories:
        return ""

    items = []
    for m in memories:
        items.append(f"- [{m['created_at']}] {m['summary']}")

    return CROSS_SESSION_TEMPLATE.format(memory_items="\n".join(items))
