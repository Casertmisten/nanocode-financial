# 后端-前端连接 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建 FastAPI Web 服务层，连接前端 3 个 HTML 页面与后端核心模块，同时保持 CLI 入口独立可用。

**Architecture:** 新增 `web.py`（FastAPI 入口）+ `api/`（路由层）+ `db.py`（SQLite CRUD）+ `llm.py`（流式 LLM 调用层，从 nanocode.py 抽取）。CLI 和 Web 共享 datasource/、rag/、financial_report_analysis/ 等核心模块。

**Tech Stack:** FastAPI, uvicorn, httpx, aiosqlite, SSE (Server-Sent Events)

**设计文档:** `docs/superpowers/specs/2026-05-11-backend-frontend-integration-design.md`

---

## 阶段 1：MVP 骨架（对话 + 会话持久化）

### Task 1: 添加依赖 + 更新 config.py

**Files:**
- Modify: `pyproject.toml`
- Modify: `config.py`

- [ ] **Step 1: 添加 FastAPI 相关依赖到 pyproject.toml**

在 `dependencies` 数组中追加：

```toml
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "httpx>=0.27.0",
    "aiosqlite>=0.20.0",
    "python-multipart>=0.0.9",
```

- [ ] **Step 2: 更新 config.py 添加数据库路径**

在 `config.py` 的 `AKSHARE_CACHE_TTL` 行之后追加：

```python
# --- Web 服务 ---
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "data", "app.db"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "data", "uploads"))
```

- [ ] **Step 3: 安装依赖**

Run: `uv sync`

Expected: 依赖安装成功，无报错

- [ ] **Step 4: 创建 data/uploads 目录 + 更新 .gitignore**

Run: `mkdir -p data/uploads`

在 `.gitignore` 末尾追加：

```
data/app.db
data/uploads/
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml config.py .gitignore uv.lock
git commit -m "feat: 添加 FastAPI/Web 依赖和配置项"
```

---

### Task 2: 创建 db.py（SQLite 初始化 + CRUD）

**Files:**
- Create: `db.py`

- [ ] **Step 1: 编写 db.py**

```python
"""SQLite 数据库初始化与会话/消息/文档 CRUD。"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '新对话',
    model TEXT NOT NULL DEFAULT '',
    knowledge_base_ids TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user','assistant','system','tool')),
    content TEXT NOT NULL DEFAULT '',
    tool_calls TEXT,
    tool_result TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL UNIQUE,
    file_size INTEGER DEFAULT 0,
    file_type TEXT NOT NULL,
    source TEXT DEFAULT 'upload',
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending','processing','ready','error')),
    chunks INTEGER DEFAULT 0,
    tags TEXT DEFAULT '[]',
    description TEXT DEFAULT '',
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fra_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    query TEXT NOT NULL,
    content TEXT NOT NULL,
    filepath TEXT,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get_db() -> aiosqlite.Connection:
    """获取数据库连接（带 WAL 模式和外键支持）。"""
    db = await aiosqlite.connect(config.DB_PATH)
    db.row_factory = sqlite3.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    """初始化数据库表。"""
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    db = await get_db()
    await db.executescript(_SCHEMA)
    await db.commit()
    await db.close()


# ---- Sessions ----

async def create_session(session_id: str, title: str = "新对话", model: str = "") -> dict:
    db = await get_db()
    now = _now()
    await db.execute(
        "INSERT INTO sessions (id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, title, model, now, now),
    )
    await db.commit()
    await db.close()
    return {"id": session_id, "title": title, "model": model, "created_at": now, "updated_at": now}


async def list_sessions(limit: int = 50) -> list[dict]:
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, title, model, knowledge_base_ids, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    )
    rows = await cursor.fetchall()
    await db.close()
    return [dict(r) for r in rows]


async def get_session(session_id: str) -> dict | None:
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, title, model, knowledge_base_ids, created_at, updated_at FROM sessions WHERE id=?",
        (session_id,),
    )
    row = await cursor.fetchone()
    await db.close()
    return dict(row) if row else None


async def update_session(session_id: str, **fields) -> bool:
    if not fields:
        return False
    fields["updated_at"] = _now()
    set_clause = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [session_id]
    db = await get_db()
    cursor = await db.execute(f"UPDATE sessions SET {set_clause} WHERE id=?", values)
    await db.commit()
    ok = cursor.rowcount > 0
    await db.close()
    return ok


async def delete_session(session_id: str) -> bool:
    db = await get_db()
    cursor = await db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    await db.commit()
    ok = cursor.rowcount > 0
    await db.close()
    return ok


# ---- Messages ----

async def add_message(session_id: str, role: str, content: str, tool_calls: str | None = None, tool_result: str | None = None) -> int:
    db = await get_db()
    cursor = await db.execute(
        "INSERT INTO messages (session_id, role, content, tool_calls, tool_result, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, role, content, tool_calls, tool_result, _now()),
    )
    await db.commit()
    msg_id = cursor.lastrowid
    await db.close()
    return msg_id


async def get_messages(session_id: str, limit: int = 100) -> list[dict]:
    db = await get_db()
    cursor = await db.execute(
        "SELECT id, role, content, tool_calls, tool_result, created_at FROM messages WHERE session_id=? ORDER BY created_at ASC LIMIT ?",
        (session_id, limit),
    )
    rows = await cursor.fetchall()
    await db.close()
    return [dict(r) for r in rows]


# ---- Documents ----

async def add_document(title: str, filename: str, filepath: str, file_size: int, file_type: str, source: str = "upload", description: str = "", tags: list[str] | None = None) -> int:
    db = await get_db()
    now = _now()
    cursor = await db.execute(
        "INSERT INTO documents (title, filename, filepath, file_size, file_type, source, description, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (title, filename, filepath, file_size, file_type, source, description, json.dumps(tags or [], ensure_ascii=False), now, now),
    )
    await db.commit()
    doc_id = cursor.lastrowid
    await db.close()
    return doc_id


async def list_documents(status: str | None = None, file_type: str | None = None, limit: int = 100) -> list[dict]:
    db = await get_db()
    clauses = []
    params = []
    if status:
        clauses.append("status=?")
        params.append(status)
    if file_type:
        clauses.append("file_type=?")
        params.append(file_type)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    cursor = await db.execute(f"SELECT * FROM documents{where} ORDER BY created_at DESC LIMIT ?", params)
    rows = await cursor.fetchall()
    await db.close()
    return [dict(r) for r in rows]


async def get_document(doc_id: int) -> dict | None:
    db = await get_db()
    cursor = await db.execute("SELECT * FROM documents WHERE id=?", (doc_id,))
    row = await cursor.fetchone()
    await db.close()
    return dict(row) if row else None


async def update_document(doc_id: int, **fields) -> bool:
    if not fields:
        return False
    fields["updated_at"] = _now()
    set_clause = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [doc_id]
    db = await get_db()
    cursor = await db.execute(f"UPDATE documents SET {set_clause} WHERE id=?", values)
    await db.commit()
    ok = cursor.rowcount > 0
    await db.close()
    return ok


async def delete_document(doc_id: int) -> bool:
    db = await get_db()
    cursor = await db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    await db.commit()
    ok = cursor.rowcount > 0
    await db.close()
    return ok


async def get_document_stats() -> dict:
    db = await get_db()
    cursor = await db.execute("SELECT status, COUNT(*) as cnt FROM documents GROUP BY status")
    rows = await cursor.fetchall()
    await db.close()
    stats = {"total": 0, "ready": 0, "pending": 0, "processing": 0, "error": 0}
    for r in rows:
        stats[r["status"]] = r["cnt"]
        stats["total"] += r["cnt"]
    return stats


# ---- Settings ----

async def get_settings() -> dict:
    db = await get_db()
    cursor = await db.execute("SELECT key, value FROM settings")
    rows = await cursor.fetchall()
    await db.close()
    return {r["key"]: json.loads(r["value"]) for r in rows}


async def save_settings(data: dict):
    db = await get_db()
    now = _now()
    for key, value in data.items():
        await db.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, ensure_ascii=False), now),
        )
    await db.commit()
    await db.close()
```

- [ ] **Step 2: 验证 db.py 可导入**

Run: `uv run python -c "import db; print('db.py OK')"`

Expected: `db.py OK`

- [ ] **Step 3: Commit**

```bash
git add db.py
git commit -m "feat: 添加 SQLite 数据库层 db.py"
```

---

### Task 3: 创建 llm.py（流式 LLM 调用层）

**Files:**
- Create: `llm.py`

从 `nanocode.py` 的 `call_api()` 抽取，改用 `httpx` 支持流式。

- [ ] **Step 1: 编写 llm.py**

```python
"""LLM 调用层 — 同步/异步流式迭代器，CLI 和 Web 共用。"""

import json
import logging
from typing import AsyncIterator, Iterator

import httpx

import config

log = logging.getLogger(__name__)


def _build_body(messages: list[dict], system_prompt: str, tools: list[dict] | None = None, model: str | None = None, stream: bool = False) -> dict:
    all_messages = [{"role": "system", "content": system_prompt}] + messages
    body: dict = {
        "model": model or config.MODEL,
        "max_tokens": 8192,
        "messages": all_messages,
        "stream": stream,
    }
    if tools:
        body["tools"] = tools
    return body


def stream_chat(messages: list[dict], system_prompt: str, tools: list[dict] | None = None, model: str | None = None) -> Iterator[dict]:
    """同步流式迭代器（CLI 用）。Yield SSE 事件字典。"""
    body = _build_body(messages, system_prompt, tools, model, stream=True)
    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", config.API_URL, json=body, headers={"Authorization": f"Bearer {config.API_KEY}", "Content-Type": "application/json"}) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue


async def async_stream_chat(messages: list[dict], system_prompt: str, tools: list[dict] | None = None, model: str | None = None) -> AsyncIterator[dict]:
    """异步流式迭代器（Web SSE 用）。Yield SSE 事件字典。"""
    body = _build_body(messages, system_prompt, tools, model, stream=True)
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", config.API_URL, json=body, headers={"Authorization": f"Bearer {config.API_KEY}", "Content-Type": "application/json"}) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue


def call_llm(system_prompt: str, user_content: str, model: str | None = None) -> str:
    """同步非流式调用（FRA pipeline 用）。"""
    body = _build_body([{"role": "user", "content": user_content}], system_prompt, model=model)
    resp = httpx.post(config.API_URL, json=body, headers={"Authorization": f"Bearer {config.API_KEY}", "Content-Type": "application/json"}, timeout=120.0)
    resp.raise_for_status()
    result = resp.json()
    return result["choices"][0]["message"]["content"]
```

- [ ] **Step 2: 验证 llm.py 可导入**

Run: `uv run python -c "import llm; print('llm.py OK')"`

Expected: `llm.py OK`

- [ ] **Step 3: Commit**

```bash
git add llm.py
git commit -m "feat: 添加 LLM 流式调用层 llm.py"
```

---

### Task 4: 创建 api/ 路由包

**Files:**
- Create: `api/__init__.py`
- Create: `api/sessions.py`

- [ ] **Step 1: 创建 api/__init__.py**

```python
"""API 路由包。"""
```

- [ ] **Step 2: 创建 api/sessions.py**

```python
"""会话管理路由。"""

from uuid import uuid4

from fastapi import APIRouter, HTTPException

import db

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
async def list_sessions():
    sessions = await db.list_sessions()
    for s in sessions:
        if isinstance(s.get("knowledge_base_ids"), str):
            import json
            s["knowledge_base_ids"] = json.loads(s["knowledge_base_ids"])
    return sessions


@router.post("")
async def create_session():
    session_id = uuid4().hex[:16]
    session = await db.create_session(session_id)
    return session


@router.get("/{session_id}")
async def get_session(session_id: str):
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    messages = await db.get_messages(session_id)
    session["messages"] = messages
    if isinstance(session.get("knowledge_base_ids"), str):
        import json
        session["knowledge_base_ids"] = json.loads(session["knowledge_base_ids"])
    return session


@router.patch("/{session_id}")
async def update_session(session_id: str, body: dict):
    allowed = {"title", "model", "knowledge_base_ids"}
    fields = {k: v for k, v in body.items() if k in allowed}
    if "knowledge_base_ids" in fields and not isinstance(fields["knowledge_base_ids"], str):
        import json
        fields["knowledge_base_ids"] = json.dumps(fields["knowledge_base_ids"], ensure_ascii=False)
    ok = await db.update_session(session_id, **fields)
    if not ok:
        raise HTTPException(404, "会话不存在")
    return {"ok": True}


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    ok = await db.delete_session(session_id)
    if not ok:
        raise HTTPException(404, "会话不存在")
    return {"ok": True}
```

- [ ] **Step 3: Commit**

```bash
git add api/
git commit -m "feat: 添加会话管理 API 路由"
```

---

### Task 5: 创建 api/chat.py（SSE 对话）

**Files:**
- Create: `api/chat.py`

- [ ] **Step 1: 创建 api/chat.py**

```python
"""对话 API — SSE 流式输出 + 工具调用。"""

import json
import logging
import os

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

import config
import db
import llm
from tools import make_schema, run_tool

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _load_system_prompt() -> str:
    prompt_path = os.path.join(config.BASE_DIR, "prompts", "rag_system.txt")
    if os.path.exists(prompt_path):
        return open(prompt_path).read().replace("{cwd}", os.getcwd())
    return f"你是一个金融分析助手。当前工作目录: {os.getcwd()}"


async def _run_agentic_stream(messages: list[dict], system_prompt: str, tools_schema: list[dict]):
    """异步 agentic loop，逐事件 yield SSE 格式字符串。"""
    while True:
        full_content = ""
        tool_calls_map: dict[int, dict] = {}

        async for event in llm.async_stream_chat(messages, system_prompt, tools=tools_schema):
            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})

            # 文本内容
            content = delta.get("content", "")
            if content:
                full_content += content
                yield f"event: token\ndata: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

            # 工具调用
            tc_list = delta.get("tool_calls") or []
            for tc in tc_list:
                idx = tc.get("index", 0)
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {"id": tc.get("id", ""), "function": {"name": "", "arguments": ""}}
                if tc.get("id"):
                    tool_calls_map[idx]["id"] = tc["id"]
                if tc.get("function"):
                    fn = tc["function"]
                    if fn.get("name"):
                        tool_calls_map[idx]["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        tool_calls_map[idx]["function"]["arguments"] += fn["arguments"]

            finish = choices[0].get("finish_reason")
            if finish and finish != "tool_calls":
                # 流结束
                pass

        # 保存 assistant 消息
        if full_content or tool_calls_map:
            tc_json = json.dumps(list(tool_calls_map.values()), ensure_ascii=False) if tool_calls_map else None
            msg_id = await db.add_message(messages[-1].get("_session_id", ""), "assistant", full_content, tool_calls=tc_json)

        if not tool_calls_map:
            yield f"event: done\ndata: {json.dumps({'message_id': msg_id if (full_content or tool_calls_map) else 0}, ensure_ascii=False)}\n\n"
            break

        # 执行工具调用
        for idx, tc in sorted(tool_calls_map.items()):
            tool_name = tc["function"]["name"]
            tool_args = json.loads(tc["function"]["arguments"])
            yield f"event: tool_start\ndata: {json.dumps({'tool': tool_name, 'args': tool_args}, ensure_ascii=False)}\n\n"

            result = run_tool(tool_name, tool_args)

            yield f"event: tool_end\ndata: {json.dumps({'tool': tool_name, 'result': result[:500]}, ensure_ascii=False)}\n\n"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        # 添加 assistant 消息到 messages（含 tool_calls）用于下一轮
        assistant_msg = {"role": "assistant", "content": full_content}
        if tool_calls_map:
            assistant_msg["tool_calls"] = list(tool_calls_map.values())
        messages.append(assistant_msg)


@router.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "")
    message = body.get("message", "")
    model = body.get("model")

    if not session_id or not message:
        from fastapi import HTTPException
        raise HTTPException(400, "缺少 session_id 或 message")

    # 保存用户消息
    await db.add_message(session_id, "user", message)

    # 更新会话标题（首条消息时）
    session = await db.get_session(session_id)
    if session and session["title"] == "新对话":
        title = message[:30] + ("..." if len(message) > 30 else "")
        await db.update_session(session_id, title=title)

    # 加载历史
    history = await db.get_messages(session_id)
    messages = []
    for m in history[:-1]:  # 不包含刚保存的用户消息
        msg = {"role": m["role"], "content": m["content"]}
        if m.get("tool_calls"):
            msg["tool_calls"] = json.loads(m["tool_calls"])
        if m["role"] == "tool":
            msg["tool_call_id"] = str(m.get("id", ""))
        messages.append(msg)
    messages.append({"role": "user", "content": message, "_session_id": session_id})

    system_prompt = _load_system_prompt()
    tools_schema = make_schema()

    return StreamingResponse(
        _run_agentic_stream(messages, system_prompt, tools_schema),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 2: Commit**

```bash
git add api/chat.py
git commit -m "feat: 添加 SSE 流式对话 API"
```

---

### Task 6: 创建 web.py（FastAPI 入口）

**Files:**
- Create: `web.py`

- [ ] **Step 1: 编写 web.py**

```python
"""Web 入口 — FastAPI 应用，托管前端 + API 路由。"""

import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import db
from api.chat import router as chat_router
from api.sessions import router as sessions_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="FinAssist", version="0.1.0")


@app.on_event("startup")
async def startup():
    await db.init_db()


# API 路由
app.include_router(chat_router)
app.include_router(sessions_router)


# 健康检查
@app.get("/api/health")
async def health():
    return {"status": "ok"}


# 模型列表
@app.get("/api/models")
async def list_models():
    return [
        {"id": "qwen-max", "name": "Qwen-Max", "provider": "阿里"},
        {"id": "qwen-plus", "name": "Qwen-Plus", "provider": "阿里"},
        {"id": "qwen-turbo", "name": "Qwen-Turbo", "provider": "阿里"},
        {"id": "glm-4-plus", "name": "GLM-4-Plus", "provider": "智谱"},
        {"id": "glm-4-flash", "name": "GLM-4-Flash", "provider": "智谱"},
        {"id": "deepseek-v3", "name": "DeepSeek-V3", "provider": "DeepSeek"},
    ]


# 知识库列表（从文档表聚合）
@app.get("/api/knowledge-bases")
async def list_knowledge_bases():
    docs = await db.list_documents(status="ready")
    return [
        {"id": "default", "name": "默认知识库", "docs": len(docs), "updated": docs[0]["updated_at"][:10] if docs else ""},
    ]


# 前端静态文件（放在最后，避免覆盖 API 路由）
from pathlib import Path
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
```

- [ ] **Step 2: 验证 Web 服务可启动**

Run: `uv run python -c "from web import app; print('FastAPI app OK')"`

Expected: `FastAPI app OK`

- [ ] **Step 3: Commit**

```bash
git add web.py
git commit -m "feat: 添加 FastAPI Web 入口"
```

---

### Task 7: 创建前端 API 工具层

**Files:**
- Create: `frontend/js/api.js`

- [ ] **Step 1: 创建 frontend/js/ 目录并编写 api.js**

```javascript
// 统一 API 工具层
const API_BASE = '/api';

async function api(method, path, body = null) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(`${API_BASE}${path}`, opts);
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`API ${res.status}: ${text}`);
    }
    return res.json();
}

/**
 * SSE POST 请求 — 用于对话流式输出
 * @param {string} path - API 路径
 * @param {object} body - 请求体
 * @param {function} onToken - 收到 token 事件回调 (content: string)
 * @param {function} onToolStart - 工具开始回调 (tool: string, args: object)
 * @param {function} onToolEnd - 工具结束回调 (tool: string, result: string)
 * @param {function} onDone - 完成回调 (messageId: number)
 * @param {function} onError - 错误回调 (error: Error)
 */
async function ssePost(path, body, { onToken, onToolStart, onToolEnd, onDone, onError }) {
    const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    if (!res.ok) {
        const text = await res.text();
        if (onError) onError(new Error(`API ${res.status}: ${text}`));
        return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = '';
        for (const line of lines) {
            if (line.startsWith('event: ')) {
                currentEvent = line.slice(7);
            } else if (line.startsWith('data: ')) {
                const dataStr = line.slice(6);
                try {
                    const data = JSON.parse(dataStr);
                    switch (currentEvent) {
                        case 'token':
                            if (onToken) onToken(data.content);
                            break;
                        case 'tool_start':
                            if (onToolStart) onToolStart(data.tool, data.args);
                            break;
                        case 'tool_end':
                            if (onToolEnd) onToolEnd(data.tool, data.result);
                            break;
                        case 'done':
                            if (onDone) onDone(data.message_id);
                            break;
                    }
                } catch (e) {
                    // 忽略解析错误
                }
                currentEvent = '';
            }
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/js/
git commit -m "feat: 添加前端 API 工具层"
```

---

### Task 8: 改造前端 chat.html 对接真实 API

**Files:**
- Modify: `frontend/chat.html`

- [ ] **Step 1: 在 chat.html 的 `</body>` 前引入 api.js**

在 `<script>` 标签之前（即第 757 行 `<script>` 之前）添加：

```html
  <script src="js/api.js"></script>
```

- [ ] **Step 2: 替换 mock 数据加载为真实 API**

在 `<script>` 块内，替换以下内容：

**替换 1：** 删除 mock `knowledgeBases` 和 `models` 数组（第 766-784 行），改为从 API 加载：

```javascript
    // --- 从 API 加载数据 ---
    let knowledgeBases = [];
    let models = [];

    async function loadInitialData() {
      try {
        const [kbData, modelData] = await Promise.all([
          api('GET', '/knowledge-bases'),
          api('GET', '/models'),
        ]);
        knowledgeBases = kbData;
        models = modelData;
        renderKbList();
        renderModelList();
      } catch (e) {
        console.error('加载初始数据失败:', e);
      }
    }
```

**替换 2：** 修改 `newChat()` 函数，改为调用 API 创建会话：

```javascript
    async function newChat() {
      try {
        const session = await api('POST', '/sessions');
        currentChatId = session.id;
        messages = [];
        messagesContainer.innerHTML = '';
        welcomeScreen.style.display = 'flex';
        messagesContainer.style.display = 'none';
        chatList.querySelectorAll('.sidebar-item').forEach(el => el.classList.remove('active'));
        document.querySelector('.top-bar-title').textContent = '新对话';
        await loadSessionList();
      } catch (e) {
        console.error('创建会话失败:', e);
      }
    }
```

**替换 3：** 添加 `loadSessionList()` 函数，加载侧边栏会话列表：

```javascript
    async function loadSessionList() {
      try {
        const sessions = await api('GET', '/sessions');
        chatList.innerHTML = sessions.map(s => `
          <li class="sidebar-item ${s.id === currentChatId ? 'active' : ''}" data-id="${s.id}" onclick="loadSession('${s.id}')">
            <svg class="item-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            <span class="item-text">${escapeHtml(s.title)}</span>
            <button class="item-delete" title="删除" onclick="event.stopPropagation(); deleteChat('${s.id}')">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          </li>`).join('');
      } catch (e) {
        console.error('加载会话列表失败:', e);
      }
    }
```

**替换 4：** 修改 `deleteChat()` 改为调用 API：

```javascript
    async function deleteChat(id) {
      try {
        await api('DELETE', `/sessions/${id}`);
        await loadSessionList();
      } catch (e) {
        console.error('删除会话失败:', e);
      }
    }
```

**替换 5：** 添加 `loadSession()` 函数，加载历史会话：

```javascript
    async function loadSession(sessionId) {
      try {
        const data = await api('GET', `/sessions/${sessionId}`);
        currentChatId = sessionId;
        chatList.querySelectorAll('.sidebar-item').forEach(el => el.classList.remove('active'));
        const item = chatList.querySelector(`[data-id="${sessionId}"]`);
        if (item) item.classList.add('active');
        document.querySelector('.top-bar-title').textContent = data.title;

        messages = [];
        welcomeScreen.style.display = 'none';
        messagesContainer.style.display = 'block';
        messagesContainer.innerHTML = '';

        for (const m of (data.messages || [])) {
          messages.push({ role: m.role, content: m.content, time: new Date(m.created_at) });
        }
        renderMessages();
        scrollToBottom();

        if (window.innerWidth <= 768) toggleSidebar();
      } catch (e) {
        console.error('加载会话失败:', e);
      }
    }
```

**替换 6：** 替换 `simulateReply()` 为真实 SSE 调用：

```javascript
    function simulateReply(userText) {
      isLoading = true;

      // 显示打字指示器
      const typingEl = document.createElement('div');
      typingEl.className = 'typing-indicator';
      typingEl.id = 'typingIndicator';
      typingEl.innerHTML = `
        <div class="message-avatar" style="background: linear-gradient(135deg, var(--color-primary), var(--color-primary-container)); color: white;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/></svg>
        </div>
        <div class="typing-dots"><span></span><span></span><span></span></div>`;
      messagesContainer.appendChild(typingEl);
      scrollToBottom();

      let fullReply = '';

      ssePost('/chat', {
        session_id: currentChatId,
        message: userText,
        model: currentModel,
      }, {
        onToken: (content) => {
          fullReply += content;
          typingEl.remove();
          if (messages.length > 0 && messages[messages.length - 1].role === 'assistant' && !messages[messages.length - 1]._finished) {
            messages[messages.length - 1].content = fullReply;
            renderMessages();
          } else {
            messages.push({ role: 'assistant', content: fullReply, time: new Date(), _finished: false });
            renderMessages();
          }
          scrollToBottom();
        },
        onDone: () => {
          if (messages.length > 0 && messages[messages.length - 1].role === 'assistant') {
            messages[messages.length - 1]._finished = true;
          }
          typingEl.remove();
          isLoading = false;
        },
        onError: (err) => {
          typingEl.remove();
          addMessage('assistant', '抱歉，回复时出现错误：' + err.message);
          isLoading = false;
        },
      });
    }
```

**替换 7：** 替换 `renderMessages()` 中 assistant 消息的渲染，支持流式更新（将 `msg.content` 用 `innerHTML` 而非 `textContent` 渲染，因为可能包含 `<br>` 标签）。

在 `renderMessages()` 函数中，assistant 消息的 content div 改为：

```javascript
              <div class="message-content">${msg.role === 'assistant' ? msg.content : escapeHtml(msg.content)}</div>
```

**替换 8：** 修改页面初始化，替换 `renderKbList(); renderModelList();` 为：

```javascript
    // --- 初始化 ---
    loadInitialData();
    // 页面加载时自动创建一个新会话
    newChat();
```

**替换 9：** 修改 `selectModel()` 同时更新当前模型显示名。

不需要改动，已经是正确的。

- [ ] **Step 3: 手动验证 — 启动 Web 服务**

Run: `uv run uvicorn web:app --host 0.0.0.0 --port 8000 --reload`

打开浏览器访问 `http://localhost:8000/chat.html`，验证：
1. 页面正常加载
2. 侧边栏显示空会话列表
3. 发送消息后能看到流式回复
4. 刷新后能恢复历史会话

- [ ] **Step 4: Commit**

```bash
git add frontend/chat.html frontend/js/api.js
git commit -m "feat: 前端 chat.html 对接真实 API"
```

---

## 阶段 2：文档管理 + 设置

### Task 9: 创建 api/documents.py

**Files:**
- Create: `api/documents.py`
- Modify: `web.py`（注册路由）

- [ ] **Step 1: 创建 api/documents.py**

```python
"""文档管理路由 — 上传、列表、删除、统计。"""

import os
import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

import config
import db

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("")
async def list_documents(status: str | None = None, type: str | None = None):
    docs = await db.list_documents(status=status, file_type=type)
    for d in docs:
        if isinstance(d.get("tags"), str):
            import json
            d["tags"] = json.loads(d["tags"])
    return docs


@router.get("/stats")
async def document_stats():
    return await db.get_document_stats()


@router.get("/{doc_id}")
async def get_document(doc_id: int):
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    if isinstance(doc.get("tags"), str):
        import json
        doc["tags"] = json.loads(doc["tags"])
    return doc


@router.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...), category: str = Form("report")):
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    results = []
    for f in files:
        filename = f.filename or "untitled"
        filepath = os.path.join(config.UPLOAD_DIR, filename)
        # 避免重名
        if os.path.exists(filepath):
            name, ext = os.path.splitext(filename)
            filepath = os.path.join(config.UPLOAD_DIR, f"{name}_{id(f)}{ext}")

        content = await f.read()
        with open(filepath, "wb") as out:
            out.write(content)

        file_type = os.path.splitext(filename)[1].lstrip(".").lower()
        title = os.path.splitext(filename)[0]
        doc_id = await db.add_document(
            title=title, filename=filename, filepath=filepath,
            file_size=len(content), file_type=file_type or "txt",
            source="upload", tags=[category],
        )

        # 后台触发 RAG ingest（简化：同步执行）
        try:
            await db.update_document(doc_id, status="processing")
            import rag
            count = rag.ingest(config.UPLOAD_DIR)
            chunks_approx = max(1, len(content) // 500)
            await db.update_document(doc_id, status="ready", chunks=chunks_approx)
        except Exception as e:
            await db.update_document(doc_id, status="error", error_message=str(e))

        results.append({"id": doc_id, "filename": filename, "status": "ready" if os.path.exists(filepath) else "error"})

    return results


@router.delete("/{doc_id}")
async def delete_document(doc_id: int):
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    # 删除文件
    if doc.get("filepath") and os.path.exists(doc["filepath"]):
        os.remove(doc["filepath"])
    ok = await db.delete_document(doc_id)
    return {"ok": ok}
```

- [ ] **Step 2: 在 web.py 注册 documents 路由**

在 `web.py` 的 import 区域添加：

```python
from api.documents import router as documents_router
```

在 `app.include_router(sessions_router)` 后添加：

```python
app.include_router(documents_router)
```

- [ ] **Step 3: Commit**

```bash
git add api/documents.py web.py
git commit -m "feat: 添加文档管理 API（上传/列表/删除）"
```

---

### Task 10: 创建 api/settings.py

**Files:**
- Create: `api/settings.py`
- Modify: `web.py`（注册路由）

- [ ] **Step 1: 创建 api/settings.py**

```python
"""设置管理路由。"""

import json

import httpx

from fastapi import APIRouter

import config
import db

router = APIRouter(prefix="/api/settings", tags=["settings"])

# 默认配置
_DEFAULTS = {
    "chat": {
        "provider": "ali", "apiUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "apiKey": "", "model": "qwen-max", "temperature": 0.7, "maxTokens": 4096,
    },
    "parse": {
        "provider": "ali", "apiUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "apiKey": "", "summaryModel": "qwen-plus", "embedModel": "text-embedding-v3",
        "chunkSize": 512, "chunkOverlap": 64,
    },
    "general": {
        "vectorDb": "chroma", "storagePath": "./data", "httpsProxy": "",
    },
}


@router.get("")
async def get_settings():
    stored = await db.get_settings()
    # 合并默认值
    result = {}
    for key, default_val in _DEFAULTS.items():
        result[key] = {**default_val, **stored.get(key, {})}
    return result


@router.put("")
async def save_settings(body: dict):
    await db.save_settings(body)
    return {"ok": True}


@router.post("/test-connection")
async def test_connection(body: dict):
    api_url = body.get("apiUrl", "")
    api_key = body.get("apiKey", "")
    if not api_url or not api_key:
        return {"ok": False, "message": "缺少 API 地址或 Key"}

    # 尝试调用 models 接口验证
    try:
        url = api_url.rstrip("/") + "/models"
        resp = httpx.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10.0)
        if resp.status_code == 200:
            return {"ok": True, "message": "连接成功"}
        return {"ok": False, "message": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"ok": False, "message": str(e)}
```

- [ ] **Step 2: 在 web.py 注册 settings 路由**

在 `web.py` 的 import 区域添加：

```python
from api.settings import router as settings_router
```

在 documents 路由注册后添加：

```python
app.include_router(settings_router)
```

- [ ] **Step 3: Commit**

```bash
git add api/settings.py web.py
git commit -m "feat: 添加设置管理 API"
```

---

### Task 11: 改造前端 knowledge.html 对接 API

**Files:**
- Modify: `frontend/knowledge.html`

- [ ] **Step 1: 在 knowledge.html 引入 api.js**

在 `</head>` 前添加：

```html
  <script src="js/api.js"></script>
```

实际上由于 `api.js` 不依赖 DOM，可以放在 `<head>` 中。或者放在 `</body>` 前的 `<script>` 标签之前。

- [ ] **Step 2: 替换 mock `documents` 数组为 API 加载**

删除 `const documents = [...]`（第 607-752 行），替换为：

```javascript
    let documents = [];

    async function loadDocuments() {
      try {
        documents = await api('GET', '/documents');
        filterDocs();
        loadStats();
      } catch (e) {
        console.error('加载文档失败:', e);
      }
    }

    async function loadStats() {
      try {
        const stats = await api('GET', '/documents/stats');
        const cards = document.querySelectorAll('.stat-value');
        if (cards.length >= 4) {
          cards[0].textContent = stats.total;
          cards[1].textContent = stats.ready;
          cards[2].textContent = stats.pending + stats.processing;
          cards[3].textContent = documents.reduce((s, d) => s + (d.chunks || 0), 0);
        }
      } catch (e) {
        console.error('加载统计失败:', e);
      }
    }
```

- [ ] **Step 3: 修改 `deleteDoc()` 为 API 调用**

```javascript
    async function deleteDoc(id) {
      try {
        await api('DELETE', `/documents/${id}`);
        await loadDocuments();
      } catch (e) {
        console.error('删除失败:', e);
      }
    }
```

- [ ] **Step 4: 修改 `startUpload()` 为真实文件上传**

替换 `startUpload()` 函数：

```javascript
    async function startUpload() {
      if (selectedFiles.length === 0) {
        alert('请先选择文件');
        return;
      }

      closeUploadModal();
      const progressEl = document.getElementById('uploadProgress');
      progressEl.classList.add('active');

      const category = document.getElementById('uploadCategory').value;

      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        const itemEl = document.createElement('div');
        itemEl.className = 'upload-item';
        itemEl.innerHTML = `
          <div class="upload-item-icon">${getTypeIcon(category)}</div>
          <div class="upload-item-info">
            <div class="upload-item-name">${file.name}</div>
            <div class="upload-item-status" id="status-${i}">上传中...</div>
            <div class="upload-progress-bar"><div class="upload-progress-fill" id="progress-${i}" style="width: 0%"></div></div>
          </div>`;
        progressEl.appendChild(itemEl);

        try {
          const formData = new FormData();
          formData.append('files', file);
          formData.append('category', category);

          const xhr = new XMLHttpRequest();
          xhr.open('POST', '/api/documents/upload');

          xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
              const pct = Math.round(e.loaded / e.total * 100);
              document.getElementById(`progress-${i}`).style.width = pct + '%';
              document.getElementById(`status-${i}`).textContent = `上传中... ${pct}%`;
            }
          };

          xhr.onload = () => {
            document.getElementById(`progress-${i}`).style.width = '100%';
            document.getElementById(`progress-${i}`).classList.add('complete');
            document.getElementById(`status-${i}`).textContent = '已完成';
            loadDocuments();
          };

          xhr.onerror = () => {
            document.getElementById(`status-${i}`).textContent = '上传失败';
          };

          xhr.send(formData);
        } catch (e) {
          document.getElementById(`status-${i}`).textContent = '上传失败: ' + e.message;
        }
      }

      selectedFiles = [];
    }
```

- [ ] **Step 5: 修改初始化**

替换 `init()` 调用为：

```javascript
    loadDocuments();
```

- [ ] **Step 6: Commit**

```bash
git add frontend/knowledge.html
git commit -m "feat: 前端 knowledge.html 对接真实 API"
```

---

### Task 12: 改造前端 settings.html 对接 API

**Files:**
- Modify: `frontend/settings.html`

- [ ] **Step 1: 引入 api.js**

在 `</body>` 前的 `<script>` 之前添加：

```html
  <script src="js/api.js"></script>
```

- [ ] **Step 2: 替换 `saveSettings()` 为真实 API 调用**

```javascript
    async function saveSettings() {
      const config = {
        chat: {
          provider: document.getElementById('chatProvider').value,
          apiUrl: document.getElementById('chatApiUrl').value,
          apiKey: document.getElementById('chatApiKey').value,
          model: document.getElementById('chatModel').value,
          temperature: parseFloat(document.getElementById('chatTemp').value),
          maxTokens: parseInt(document.getElementById('chatMaxTokens').value),
        },
        parse: {
          provider: document.getElementById('parseProvider').value,
          apiUrl: document.getElementById('parseApiUrl').value,
          apiKey: document.getElementById('parseApiKey').value,
          summaryModel: document.getElementById('parseSummaryModel').value,
          embedModel: document.getElementById('parseEmbedModel').value,
          chunkSize: parseInt(document.getElementById('chunkSize').value),
          chunkOverlap: parseInt(document.getElementById('chunkOverlap').value),
        },
        general: {
          vectorDb: document.getElementById('vectorDb').value,
          storagePath: document.getElementById('storagePath').value,
          httpsProxy: document.getElementById('httpsProxy').value,
        },
      };

      try {
        await api('PUT', '/settings', config);
        hasUnsaved = false;
        document.getElementById('saveHint').textContent = '已保存';
        document.getElementById('saveHint').style.color = 'var(--color-success)';
        showToast('设置已保存', 'success');
        setTimeout(() => {
          document.getElementById('saveHint').textContent = '修改后需保存才会生效';
          document.getElementById('saveHint').style.color = '';
        }, 3000);
      } catch (e) {
        showToast('保存失败: ' + e.message, 'error');
      }
    }
```

- [ ] **Step 3: 替换 `testConnection()` 为真实 API 调用**

```javascript
    async function testConnection(section) {
      const btn = document.getElementById(`${section}TestBtn`);
      const statusEl = document.getElementById(`${section}Status`);

      btn.classList.add('testing');
      btn.textContent = '测试中...';
      statusEl.innerHTML = '<span class="status-dot disconnected"></span><span class="status-text">正在连接...</span>';

      try {
        const apiUrl = document.getElementById(`${section}ApiUrl`).value;
        const apiKey = document.getElementById(`${section}ApiKey`).value;
        const result = await api('POST', '/settings/test-connection', { apiUrl, apiKey });

        btn.classList.remove('testing');
        if (result.ok) {
          btn.classList.add('success');
          btn.textContent = '连接成功';
          statusEl.innerHTML = '<span class="status-dot connected"></span><span class="status-text connected">已连接</span>';
        } else {
          btn.classList.add('fail');
          btn.textContent = '连接失败';
          statusEl.innerHTML = `<span class="status-dot error"></span><span class="status-text error">${result.message}</span>`;
        }
      } catch (e) {
        btn.classList.remove('testing');
        btn.classList.add('fail');
        btn.textContent = '连接失败';
        statusEl.innerHTML = `<span class="status-dot error"></span><span class="status-text error">${e.message}</span>`;
      }

      setTimeout(() => {
        btn.classList.remove('success', 'fail');
        btn.textContent = '测试连接';
      }, 3000);
    }
```

- [ ] **Step 4: 添加页面加载时从 API 读取设置**

在 `<script>` 块末尾添加：

```javascript
    // 从 API 加载设置
    (async function loadSettings() {
      try {
        const settings = await api('GET', '/settings');
        if (settings.chat) {
          const c = settings.chat;
          if (c.provider) document.getElementById('chatProvider').value = c.provider;
          if (c.apiUrl) document.getElementById('chatApiUrl').value = c.apiUrl;
          if (c.apiKey) document.getElementById('chatApiKey').value = c.apiKey;
          if (c.model) document.getElementById('chatModel').value = c.model;
          if (c.temperature != null) { document.getElementById('chatTemp').value = c.temperature; document.getElementById('chatTempVal').textContent = c.temperature; }
          if (c.maxTokens != null) { document.getElementById('chatMaxTokens').value = c.maxTokens; document.getElementById('chatMaxTokensVal').textContent = c.maxTokens; }
        }
        if (settings.parse) {
          const p = settings.parse;
          if (p.provider) document.getElementById('parseProvider').value = p.provider;
          if (p.apiUrl) document.getElementById('parseApiUrl').value = p.apiUrl;
          if (p.apiKey) document.getElementById('parseApiKey').value = p.apiKey;
          if (p.summaryModel) document.getElementById('parseSummaryModel').value = p.summaryModel;
          if (p.embedModel) document.getElementById('parseEmbedModel').value = p.embedModel;
          if (p.chunkSize != null) { document.getElementById('chunkSize').value = p.chunkSize; document.getElementById('chunkSizeVal').textContent = p.chunkSize; }
          if (p.chunkOverlap != null) { document.getElementById('chunkOverlap').value = p.chunkOverlap; document.getElementById('chunkOverlapVal').textContent = p.chunkOverlap; }
        }
        if (settings.general) {
          const g = settings.general;
          if (g.vectorDb) document.getElementById('vectorDb').value = g.vectorDb;
          if (g.storagePath) document.getElementById('storagePath').value = g.storagePath;
          if (g.httpsProxy) document.getElementById('httpsProxy').value = g.httpsProxy;
        }
      } catch (e) {
        console.error('加载设置失败:', e);
      }
    })();
```

- [ ] **Step 5: Commit**

```bash
git add frontend/settings.html
git commit -m "feat: 前端 settings.html 对接真实 API"
```

---

## 阶段 3：FRA + CLI 改造

### Task 13: 创建 api/fra.py

**Files:**
- Create: `api/fra.py`
- Modify: `web.py`（注册路由）

- [ ] **Step 1: 创建 api/fra.py**

```python
"""财报分析 API — SSE 流式进度推送。"""

import datetime
import json
import logging
import os

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

import config
import db

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fra", tags=["fra"])


async def _run_fra_stream(query: str, session_id: str | None = None):
    """执行 FRA 流程，逐阶段 SSE 推送。"""
    import financial_report_analysis
    from financial_report_analysis.template import DIMENSIONS
    from financial_report_analysis.prompts import ANALYZE_PROMPT, REDUCE_PROMPT
    import rag
    import llm

    _TOP_K = 3

    # Map: 检索
    dim_data = []
    total = sum(len(d["sub_questions"]) for d in DIMENSIONS)
    done = 0

    yield f"event: progress\ndata: {json.dumps({'stage': 'retrieve', 'detail': f'开始检索 {total} 个子问题...'}, ensure_ascii=False)}\n\n"

    for dim in DIMENSIONS:
        seen_texts = set()
        chunks = []
        sources = set()

        for sq in dim["sub_questions"]:
            done += 1
            yield f"event: progress\ndata: {json.dumps({'stage': 'retrieve', 'detail': f'[{done}/{total}] {sq}'}, ensure_ascii=False)}\n\n"
            try:
                results = rag.query(sq, top_k=_TOP_K)
            except Exception:
                results = []
            for r in results:
                text = r.get("text", "")
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    chunks.append(text)
                source = r.get("source", "")
                if source:
                    sources.add(source)

        dim_data.append({"name": dim["name"], "chunks": chunks, "sources": sources})

    # Analyze: 逐维度分析
    summaries = {}
    for i, dd in enumerate(dim_data, 1):
        yield f"event: progress\ndata: {json.dumps({'stage': 'analyze', 'detail': f'[{i}/{len(dim_data)}] {dd["name"]}'}, ensure_ascii=False)}\n\n"
        if not dd["chunks"]:
            summaries[dd["name"]] = f"【{dd['name']}】该维度缺乏足够数据。"
        else:
            chunks_text = "\n\n---\n\n".join(dd["chunks"])
            prompt = ANALYZE_PROMPT.format(dimension_name=dd["name"], chunks=chunks_text)
            summaries[dd["name"]] = llm.call_llm("你是一个专业的金融分析师。", prompt)

    # 汇总来源
    all_sources = set()
    for dd in dim_data:
        all_sources.update(dd["sources"])

    # Reduce: 生成报告
    yield f"event: progress\ndata: {json.dumps({'stage': 'reduce', 'detail': '生成报告...'}, ensure_ascii=False)}\n\n"

    summaries_text = ""
    for dim in DIMENSIONS:
        summaries_text += f"\n## {dim['name']}\n{summaries[dim['name']]}\n"
    sources_text = "\n".join(f"· {s}" for s in sorted(all_sources))
    prompt = REDUCE_PROMPT.format(query=query, summaries=summaries_text, sources=sources_text)
    report = llm.call_llm("你是一个资深金融分析师。", prompt)

    # 保存报告
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(config.BASE_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    filepath = os.path.join(report_dir, f"financial_report_{ts}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    # 写入数据库
    import sqlite3
    db_path = config.DB_PATH
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "INSERT INTO fra_reports (session_id, query, content, filepath, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, query, report, filepath, datetime.datetime.now().isoformat()),
    )
    conn.commit()
    report_id = cursor.lastrowid
    conn.close()

    yield f"event: done\ndata: {json.dumps({'report_id': report_id, 'filepath': filepath}, ensure_ascii=False)}\n\n"


@router.post("")
async def run_fra(request: Request):
    body = await request.json()
    query = body.get("query", "")
    session_id = body.get("session_id")
    if not query:
        from fastapi import HTTPException
        raise HTTPException(400, "缺少 query 参数")

    return StreamingResponse(
        _run_fra_stream(query, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/reports")
async def list_fra_reports():
    import sqlite3
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT id, session_id, query, filepath, created_at FROM fra_reports ORDER BY created_at DESC LIMIT 20")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


@router.get("/reports/{report_id}")
async def get_fra_report(report_id: int):
    import sqlite3
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM fra_reports WHERE id=?", (report_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(404, "报告不存在")
    return dict(row)
```

- [ ] **Step 2: 在 web.py 注册 fra 路由**

添加 import 和 include_router。

- [ ] **Step 3: Commit**

```bash
git add api/fra.py web.py
git commit -m "feat: 添加 FRA 报告分析 API"
```

---

### Task 14: 改造 nanocode.py 使用 llm.py

**Files:**
- Modify: `nanocode.py`
- Modify: `financial_report_analysis/pipeline.py`

- [ ] **Step 1: 改造 nanocode.py 的 call_api 为使用 llm.py**

替换 `nanocode.py` 中的 `call_api` 函数和 `_run_agentic_loop` 函数：

```python
def call_api(messages, system_prompt):
    """调用 LLM API (非流式，兼容旧逻辑)。"""
    import llm
    body = llm._build_body(messages, system_prompt, tools=make_schema())
    import httpx
    resp = httpx.post(config.API_URL, json=body, headers={"Authorization": f"Bearer {config.API_KEY}", "Content-Type": "application/json"}, timeout=120.0)
    return resp.json()
```

删除 `import urllib.request`（第 8 行），因为不再使用。

- [ ] **Step 2: 改造 pipeline.py 的 _call_llm 使用 llm.call_llm**

替换 `financial_report_analysis/pipeline.py` 中的 `_call_llm` 函数：

```python
def _call_llm(system_prompt: str, user_content: str) -> str:
    """纯文本 LLM 调用，不携带工具。"""
    import llm
    return llm.call_llm(system_prompt, user_content)
```

删除 pipeline.py 中的 `import urllib.request`（第 6 行）。

- [ ] **Step 3: 验证 CLI 仍可运行**

Run: `uv run python nanocode.py`（手动 Ctrl+C 退出即可）

Expected: 启动正常，显示 `nanocode-financial | qwen3.5-flash`

- [ ] **Step 4: Commit**

```bash
git add nanocode.py financial_report_analysis/pipeline.py
git commit -m "refactor: CLI 和 FRA 改用 llm.py 调用层"
```

---

### Task 15: 最终验证

- [ ] **Step 1: 启动 Web 服务**

Run: `uv run uvicorn web:app --host 0.0.0.0 --port 8000 --reload`

- [ ] **Step 2: 验证 API 端点**

Run（另一个终端）:

```bash
# 健康检查
curl http://localhost:8000/api/health

# 创建会话
curl -X POST http://localhost:8000/api/sessions

# 列出会话
curl http://localhost:8000/api/sessions

# 模型列表
curl http://localhost:8000/api/models

# 文档统计
curl http://localhost:8000/api/documents/stats

# 设置
curl http://localhost:8000/api/settings
```

Expected: 所有端点返回 JSON

- [ ] **Step 3: 浏览器验证全部页面**

打开 `http://localhost:8000/chat.html` — 对话、会话管理
打开 `http://localhost:8000/knowledge.html` — 文档列表、上传
打开 `http://localhost:8000/settings.html` — 设置读写、连接测试

- [ ] **Step 4: 最终 Commit**

```bash
git add -A
git commit -m "feat: 完成后端-前端连接（阶段1-3）"
```
