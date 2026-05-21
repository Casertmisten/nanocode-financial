# 三层记忆系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为金融智能分析助手实现三层记忆系统（用户画像 / 跨会话记忆 / 当前会话压缩）。

**Architecture:** 在现有 FastAPI + SQLite + ChromaDB 架构上以最小侵入方式添加 `memory/` 模块，通过在 `api/chat.py` 的消息准备阶段注入三层记忆，不改 RAG 和 Deep Research 核心逻辑。

**Tech Stack:** Python 3.12+, FastAPI, aiosqlite, ChromaDB, httpx, LlamaIndex embeddings

---

## 文件结构

| 操作 | 文件 | 职责 |
|---|---|---|
| 新建 | `memory/__init__.py` | 对外统一接口 |
| 新建 | `memory/profile.py` | L1: 用户画像读写、模板渲染、候选池管理、LLM 合并更新 |
| 新建 | `memory/session_memory.py` | L2: 跨会话摘要生成、ChromaDB 存取、时间衰减检索 |
| 新建 | `memory/context.py` | L3: token 估算、压缩触发、摘要 CRUD |
| 新建 | `memory/prompts.py` | 三层记忆相关 prompt 模板 |
| 修改 | `config.py` | 新增 Memory 配置常量 |
| 修改 | `db.py` | 新增 `session_summaries` 表 + CRUD 函数 |
| 修改 | `api/chat.py` | 在消息准备阶段注入 L1/L2/L3；对话结束后触发 L2 写入 + L1 候选 |
| 修改 | `deep_research/__init__.py` | 研究完成后写入 L2 |

---

## Task 1: 配置常量 + 数据库表

**Files:**
- Modify: `config.py:57-66` (在 Deep Research 配置之后、Web 服务配置之前)
- Modify: `db.py:36-133` (在 `init_db` 的 `executescript` 中追加建表语句)

- [ ] **Step 1: 在 `config.py` 添加 Memory 配置**

在 `# --- Web 服务 ---` 注释之前插入：

```python
# --- Memory ---
MEMORY_PROFILE_PATH = os.path.join(BASE_DIR, "data", "profile.json")
MEMORY_CANDIDATES_PATH = os.path.join(BASE_DIR, "data", "profile_candidates.json")
MEMORY_SESSION_COLLECTION = "session_memory"
MEMORY_SESSION_MAX_TOKENS = int(os.environ.get("MEMORY_SESSION_MAX_TOKENS", "6000"))
MEMORY_COMPRESS_ROUNDS = 3
MEMORY_TIME_DECAY_LAMBDA = float(os.environ.get("MEMORY_TIME_DECAY_LAMBDA", "0.05"))
MEMORY_CROSS_SESSION_TOP_K = int(os.environ.get("MEMORY_CROSS_SESSION_TOP_K", "5"))
MEMORY_CROSS_SESSION_INJECT_K = int(os.environ.get("MEMORY_CROSS_SESSION_INJECT_K", "3"))
MEMORY_PROFILE_UPDATE_INTERVAL = int(os.environ.get("MEMORY_PROFILE_UPDATE_INTERVAL", "3"))
MEMORY_MIN_KEEP_ROUNDS = 5
```

- [ ] **Step 2: 在 `db.py` 的 `init_db` 中追加 `session_summaries` 建表语句**

在 `executescript` 的 SQL 字符串末尾（`web_search_cache` 表之后、闭合引号之前）追加：

```sql
CREATE TABLE IF NOT EXISTS session_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    start_msg_id INTEGER,
    end_msg_id INTEGER,
    summary TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_summaries_session
    ON session_summaries(session_id);
```

- [ ] **Step 3: 在 `db.py` 末尾追加 session_summaries CRUD 函数**

```python
# ---------------------------------------------------------------------------
# Session Summaries CRUD（L3 当前会话压缩）
# ---------------------------------------------------------------------------

async def add_session_summary(session_id: str, start_msg_id: int, end_msg_id: int,
                               summary: str, token_count: int = 0) -> int:
    """添加会话摘要，返回摘要 ID"""
    now = _now()
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO session_summaries (session_id, start_msg_id, end_msg_id, "
            "summary, token_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, start_msg_id, end_msg_id, summary, token_count, now),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def get_session_summaries(session_id: str) -> list[dict]:
    """获取会话的所有摘要，按起始消息 ID 排序"""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM session_summaries WHERE session_id = ? ORDER BY start_msg_id ASC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        await db.close()
```

- [ ] **Step 4: 提交**

```bash
git add config.py db.py
git commit -m "feat: 添加 Memory 配置常量和 session_summaries 数据库表"
```

---

## Task 2: memory/prompts.py — Prompt 模板

**Files:**
- Create: `memory/prompts.py`

- [ ] **Step 1: 创建 `memory/prompts.py`**

```python
"""三层记忆系统相关 prompt 模板。"""

# --- L2: 会话摘要生成（含 L1 画像候选提取）---
SUMMARY_PROMPT = """你是一个对话摘要助手。请对以下对话进行摘要，要求：
1. 提炼用户的核心问题和助手的回答要点
2. 提取 3-5 个主题关键词，以 JSON 数组格式输出
3. 摘要长度控制在 200 字以内
4. 如果对话中暴露了用户的投资偏好、关注行业/个股、报告风格偏好，请额外列出

输出格式（严格遵循）：
<summary>
摘要内容...
</summary>
<topics>
["主题1", "主题2", "主题3"]
</topics>
<profile_candidates>
{"focus_sectors": [], "watched_stocks": [], "preferred_markets": [], "risk_tolerance": "", "report_style": ""}
</profile_candidates>

对话内容：
{messages}"""

# --- L1: 画像候选合并 ---
MERGE_PROFILE_PROMPT = """你是一个用户画像管理助手。

以下是当前的用户画像：
{current_profile}

以下是近期对话中提取的用户偏好候选：
{candidates}

请合并以上信息，生成更新后的用户画像。规则：
1. 新信息覆盖旧信息中的冲突字段
2. 列表类字段（如关注行业、关注个股）取并集
3. 如果候选中没有某字段的新信息，保留原值
4. 严格保持 JSON 格式输出，只输出 JSON，不要其他内容

输出格式：
{{"preferred_markets": [], "focus_sectors": [], "watched_stocks": [], "risk_tolerance": "", "report_style": "", "language": "中文"}}"""

# --- L1: 用户画像 Markdown 模板 ---
PROFILE_TEMPLATE = """## 用户画像
- 关注市场：{preferred_markets}
- 关注行业：{focus_sectors}
- 关注个股：{watched_stocks}
- 风险偏好：{risk_tolerance}
- 报告风格：{report_style}"""

# --- L3: 对话压缩摘要 ---
COMPRESS_PROMPT = """请将以下对话内容压缩为一段简洁的摘要，保留关键信息（用户意图、结论、重要数据点）。
摘要长度控制在 150 字以内。只输出摘要文本，不要其他内容。

对话内容：
{messages}"""

# --- L2: 跨会话记忆注入模板 ---
CROSS_SESSION_TEMPLATE = """## 历史对话记忆
以下是与你之前对话相关的记忆片段：
{memory_items}"""
```

- [ ] **Step 2: 创建 `memory/__init__.py`（空壳，后续 task 填充）**

```python
"""三层记忆系统：用户画像 / 跨会话记忆 / 当前会话压缩。"""
```

- [ ] **Step 3: 提交**

```bash
git add memory/
git commit -m "feat: 创建 memory 模块骨架和 prompt 模板"
```

---

## Task 3: memory/profile.py — L1 用户画像

**Files:**
- Create: `memory/profile.py`
- Modify: `memory/__init__.py`

- [ ] **Step 1: 创建 `memory/profile.py`**

```python
"""L1: 用户画像管理 — JSON 读写、Markdown 渲染、候选池管理、LLM 合并更新。"""

import json
import os
from datetime import datetime, timezone

import config
import llm
from memory.prompts import MERGE_PROFILE_PROMPT, PROFILE_TEMPLATE
from utils import BaseLogger

log = BaseLogger.getLogger("memory.profile")

_DEFAULT_PROFILE = {
    "preferred_markets": [],
    "focus_sectors": [],
    "watched_stocks": [],
    "risk_tolerance": "",
    "report_style": "",
    "language": "中文",
}


def load_profile() -> dict | None:
    """加载用户画像 JSON，文件不存在返回 None。"""
    path = config.MEMORY_PROFILE_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("profile", _DEFAULT_PROFILE)
    except Exception:
        log.warning("加载用户画像失败", exc_info=True)
        return None


def render_profile_markdown(profile: dict | None) -> str:
    """将用户画像渲染为 Markdown 片段。画像为空时返回空字符串。"""
    if not profile:
        return ""
    # 过滤掉空值字段
    values = {}
    for key, label in [
        ("preferred_markets", "关注市场"),
        ("focus_sectors", "关注行业"),
        ("watched_stocks", "关注个股"),
        ("risk_tolerance", "风险偏好"),
        ("report_style", "报告风格"),
    ]:
        val = profile.get(key)
        if val and val != [] and val != "":
            if isinstance(val, list):
                values[label] = "、".join(str(v) for v in val)
            else:
                values[label] = str(val)
    if not values:
        return ""
    return PROFILE_TEMPLATE.format(**values)


def add_candidate(session_id: str, fields: dict):
    """将一轮会话提取的画像候选写入候选池。"""
    path = config.MEMORY_CANDIDATES_PATH
    candidates_data = {"candidates": [], "session_count": 0}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                candidates_data = json.load(f)
        except Exception:
            log.warning("读取候选池失败，重新初始化", exc_info=True)

    candidates_data["candidates"].append({
        "session_id": session_id,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "fields": {k: v for k, v in fields.items() if v and v != [] and v != ""},
    })
    candidates_data["session_count"] += 1

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(candidates_data, f, ensure_ascii=False, indent=2)

    log.info("画像候选已添加: session=%s, 候选池=%d 轮", session_id, candidates_data["session_count"])

    # 检查是否触发画像更新
    if candidates_data["session_count"] >= config.MEMORY_PROFILE_UPDATE_INTERVAL:
        _update_profile(candidates_data)


def _update_profile(candidates_data: dict):
    """用 LLM 合并候选并更新 profile.json。"""
    current_profile = load_profile() or _DEFAULT_PROFILE
    candidates_text = json.dumps(candidates_data["candidates"], ensure_ascii=False, indent=2)

    prompt = MERGE_PROFILE_PROMPT.format(
        current_profile=json.dumps(current_profile, ensure_ascii=False, indent=2),
        candidates=candidates_text,
    )

    try:
        result = llm.call_llm("你是一个用户画像管理助手。", prompt)
        # 提取 JSON（可能包裹在 ```json ... ``` 中）
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        new_profile = json.loads(result)
    except Exception:
        log.warning("LLM 合并画像失败，跳过本次更新", exc_info=True)
        # 清空候选池避免无限累积
        _clear_candidates()
        return

    # 写入 profile.json
    profile_data = {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "profile": new_profile,
    }
    path = config.MEMORY_PROFILE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, ensure_ascii=False, indent=2)

    log.info("用户画像已更新: %s", json.dumps(new_profile, ensure_ascii=False)[:200])
    _clear_candidates()


def _clear_candidates():
    """清空候选池。"""
    path = config.MEMORY_CANDIDATES_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"candidates": [], "session_count": 0}, f)
```

- [ ] **Step 2: 提交**

```bash
git add memory/profile.py
git commit -m "feat: L1 用户画像模块（加载、渲染、候选池、LLM 合并更新）"
```

---

## Task 4: memory/session_memory.py — L2 跨会话记忆

**Files:**
- Create: `memory/session_memory.py`

本 task 依赖 LlamaIndex 的 embedding 接口和 ChromaDB，与现有 `rag/indexer.py` 中的 embedding 注册模式一致。

- [ ] **Step 1: 创建 `memory/session_memory.py`**

```python
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
    """调用 embedding API 获取向量。复用 LlamaIndex 的 OpenAIEmbedding。"""
    from rag.indexer import get_index
    import os

    # 复用现有 LlamaIndex embedding 模型
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
    query_embedding = _get_embedding(query)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()) if collection.count() > 0 else 0,
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
```

- [ ] **Step 2: 提交**

```bash
git add memory/session_memory.py
git commit -m "feat: L2 跨会话记忆模块（摘要生成、ChromaDB 存取、时间衰减检索）"
```

---

## Task 5: memory/context.py — L3 当前会话压缩

**Files:**
- Create: `memory/context.py`

- [ ] **Step 1: 创建 `memory/context.py`**

```python
"""L3: 当前会话记忆 — token 估算、压缩触发、摘要管理。"""

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
            import json
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
    """将消息序列化为摘要 prompt 所需的文本格式。"""
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "tool":
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

        # 找出最早的 N 轮对话（跳过已被摘要的）
        to_compress = []
        remaining = []
        round_count = 0
        in_round = False
        compressed_ids = set()

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
                if msg_id:
                    compressed_ids.add(msg_id)
                # user 消息开始一轮
                if msg.get("role") == "user":
                    in_round = True
                # assistant 消息结束一轮（如果有内容或 tool_calls）
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
        start_id = min(m.get("id", 0) for m in to_compress if m.get("id"))
        end_id = max(m.get("id", 0) for m in to_compress if m.get("id"))
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
```

- [ ] **Step 2: 提交**

```bash
git add memory/context.py
git commit -m "feat: L3 当前会话压缩模块（token 估算、自动压缩、摘要存储）"
```

---

## Task 6: memory/__init__.py — 统一接口

**Files:**
- Modify: `memory/__init__.py`

- [ ] **Step 1: 完善 `memory/__init__.py`**

```python
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
```

- [ ] **Step 2: 提交**

```bash
git add memory/__init__.py
git commit -m "feat: memory 模块统一接口（inject_memory + save_after_turn）"
```

---

## Task 7: 集成到 api/chat.py

**Files:**
- Modify: `api/chat.py`

这是核心集成点。改动集中在 `chat` 函数和 `stream` 闭包中。

- [ ] **Step 1: 在 `api/chat.py` 顶部添加 import**

在现有 import 块中追加：

```python
import memory as memory_module
```

- [ ] **Step 2: 修改 `chat` 函数，在消息准备阶段注入三层记忆**

在 `api/chat.py` 的 `chat` 函数中，`llm_messages = _messages_to_llm_format(messages_raw)` 之后、`async def stream()` 之前，插入记忆注入逻辑。

替换原始的 `llm_messages = _messages_to_llm_format(messages_raw)` 为：

```python
    # 三层记忆注入
    compressed_msgs, memory_injection = await memory_module.inject_memory(
        req.session_id, messages_raw, req.message,
    )
    llm_messages = _messages_to_llm_format(compressed_msgs)
```

- [ ] **Step 3: 修改 `stream` 闭包，将记忆注入追加到 system prompt**

在 `stream()` 函数内，`turn_system_prompt` 构建完成后（`doc_lines` 替换之后），追加记忆注入：

在 `turn_system_prompt = SYSTEM_PROMPT.replace("{documents}", doc_lines)` 行之后，`except` 块之后，添加：

```python
        # 追加三层记忆到 system prompt
        if memory_injection:
            turn_system_prompt = turn_system_prompt + "\n\n" + memory_injection
```

- [ ] **Step 4: 在对话结束后保存记忆**

在 `stream()` 函数末尾，`yield _sse("done", ...)` 之前，添加：

```python
            # 保存三层记忆（异步后台执行，不阻塞响应）
            import asyncio
            try:
                all_session_msgs = await db.get_messages(req.session_id)
                asyncio.create_task(
                    memory_module.save_after_turn(req.session_id, all_session_msgs)
                )
            except Exception:
                log.warning("触发记忆保存失败", exc_info=True)
```

- [ ] **Step 5: 提交**

```bash
git add api/chat.py
git commit -m "feat: 在对话流程中集成三层记忆注入和保存"
```

---

## Task 8: 集成到 Deep Research

**Files:**
- Modify: `deep_research/__init__.py`

在 Deep Research 的 `execute` 函数完成报告生成后，将报告内容写入 L2 跨会话记忆。

- [ ] **Step 1: 在 `deep_research/__init__.py` 的 `execute` 函数末尾，报告保存之后，添加 L2 记忆写入**

在 `return results_list, report, report_id, filepath` 之前插入：

```python
    # 保存到 L2 跨会话记忆
    try:
        from memory.session_memory import save_session_memory
        topics = [query[:20]]
        save_session_memory(
            session_id or f"research_{report_id}",
            report[:1000],  # 截取前 1000 字作为摘要
            topics,
            session_type="deep_research",
        )
    except Exception:
        log.warning("Deep Research 报告保存到跨会话记忆失败", exc_info=True)
```

- [ ] **Step 2: 提交**

```bash
git add deep_research/__init__.py
git commit -m "feat: Deep Research 报告自动保存到跨会话记忆"
```

---

## Task 9: 端到端验证

- [ ] **Step 1: 确认数据库初始化正常**

```bash
cd /home/caser/文档/code/nanocode-financial
uv run python -c "import db; import asyncio; asyncio.run(db.init_db()); print('DB init OK')"
```

Expected: `DB init OK`

- [ ] **Step 2: 验证 memory 模块可导入**

```bash
uv run python -c "import memory; print('Memory module OK')"
```

Expected: `Memory module OK`

- [ ] **Step 3: 启动 Web 服务进行交互验证**

```bash
uv run python web.py
```

验证：
1. 发送一条金融问题（如"贵州茅台最近怎么样"），确认正常响应
2. 检查 `data/profile_candidates.json` 是否生成（首次会话后应有候选）
3. 再发送几条消息后，检查 ChromaDB `session_memory` collection 是否有数据

- [ ] **Step 4: 提交最终验证状态**

```bash
git add -A
git commit -m "feat: 三层记忆系统实现完成"
```

---

## 自检清单

| 检查项 | 状态 |
|---|---|
| Spec 中 L1 画像存储和更新 | Task 3 覆盖 |
| Spec 中 L1 Markdown 模板注入 | Task 3 + Task 6 + Task 7 覆盖 |
| Spec 中 L1 候选池 + 累积触发 | Task 3 覆盖 |
| Spec 中 L2 ChromaDB session_memory collection | Task 4 覆盖 |
| Spec 中 L2 摘要生成 + 主题提取 | Task 4 覆盖 |
| Spec 中 L2 时间衰减检索 | Task 4 覆盖 |
| Spec 中 L2 每轮 RAG 注入 | Task 6 + Task 7 覆盖 |
| Spec 中 L2 Deep Research 报告写入 | Task 8 覆盖 |
| Spec 中 L3 session_summaries 表 | Task 1 覆盖 |
| Spec 中 L3 token 估算 + 压缩触发 | Task 5 覆盖 |
| Spec 中 L3 保护规则（最近 5 轮不压缩） | Task 5 覆盖 |
| Spec 中 L3 已有摘要复用 | Task 5 覆盖 |
| Spec 中 config.py 新增配置 | Task 1 覆盖 |
| Spec 中 memory/ 模块结构 | Task 2-6 覆盖 |
| 无 placeholder | 已检查，无 TBD/TODO |
| 类型一致性 | 已检查，函数签名和调用一致 |
