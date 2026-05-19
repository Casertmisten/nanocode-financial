"""SQLite 数据库初始化 + CRUD 层（异步，基于 aiosqlite）"""

import json
import sqlite3
from datetime import datetime, timezone

import aiosqlite

import config
from utils import BaseLogger

log = BaseLogger.getLogger("db")

DB_PATH = config.DB_PATH


def _now() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串"""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 数据库连接与初始化
# ---------------------------------------------------------------------------

async def get_db() -> aiosqlite.Connection:
    """获取数据库连接，开启 WAL 模式和外键约束"""
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    """创建所有表（如果不存在）"""
    db = await get_db()
    try:
        await db.executescript("""
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

            CREATE TABLE IF NOT EXISTS stock_list (
                code TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                market TEXT DEFAULT '',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS research_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
                query TEXT NOT NULL,
                content TEXT NOT NULL,
                filepath TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
                model TEXT NOT NULL DEFAULT '',
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'chat',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_token_usage_created
                ON token_usage(created_at);
            CREATE INDEX IF NOT EXISTS idx_token_usage_model
                ON token_usage(model);

            CREATE TABLE IF NOT EXISTS web_search_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                answer TEXT DEFAULT '',
                results TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_web_search_query ON web_search_cache(query);
        """)
        await db.commit()
        log.info("数据库初始化完成: %s", DB_PATH)
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# 辅助：将 Row 转为 dict
# ---------------------------------------------------------------------------

def _row_to_dict(row: aiosqlite.Row) -> dict:
    return dict(row)


# ---------------------------------------------------------------------------
# Sessions CRUD
# ---------------------------------------------------------------------------

async def create_session(session_id: str, title: str = "新对话", model: str = "") -> dict:
    """创建新会话，返回会话字典"""
    now = _now()
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO sessions (id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, title, model, now, now),
        )
        await db.commit()
        log.info("创建会话: %s", session_id)
        return {"id": session_id, "title": title, "model": model,
                "knowledge_base_ids": "[]", "created_at": now, "updated_at": now}
    finally:
        await db.close()


async def list_sessions(limit: int = 50) -> list[dict]:
    """获取会话列表，按更新时间倒序"""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        await db.close()


async def get_session(session_id: str) -> dict | None:
    """获取单个会话"""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None
    finally:
        await db.close()


async def update_session(session_id: str, **fields) -> bool:
    """更新会话字段，自动更新 updated_at"""
    if not fields:
        return False
    fields["updated_at"] = _now()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [session_id]
    db = await get_db()
    try:
        cursor = await db.execute(f"UPDATE sessions SET {set_clause} WHERE id = ?", values)
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def delete_session(session_id: str) -> bool:
    """删除会话（级联删除消息）"""
    db = await get_db()
    try:
        cursor = await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()
        log.info("删除会话: %s, 影响=%d 行", session_id, cursor.rowcount)
        return cursor.rowcount > 0
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Messages CRUD
# ---------------------------------------------------------------------------

async def add_message(session_id: str, role: str, content: str,
                      tool_calls: str | None = None,
                      tool_result: str | None = None) -> int:
    """添加消息，返回消息 ID"""
    now = _now()
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, tool_result, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, tool_calls, tool_result, now),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def get_messages(session_id: str, limit: int = 100) -> list[dict]:
    """获取会话消息，按创建时间正序"""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Documents CRUD
# ---------------------------------------------------------------------------

async def add_document(title: str, filename: str, filepath: str,
                       file_size: int, file_type: str, source: str = "upload",
                       description: str = "", tags: str | None = None) -> int:
    """添加文档记录，返回文档 ID"""
    now = _now()
    if tags is None:
        tags = "[]"
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO documents (title, filename, filepath, file_size, file_type, source, "
            "tags, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (title, filename, filepath, file_size, file_type, source, tags, description, now, now),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def list_documents(status: str | None = None,
                         file_type: str | None = None,
                         limit: int = 100) -> list[dict]:
    """获取文档列表，支持按状态和类型筛选"""
    conditions = []
    params: list = []
    if status:
        conditions.append("status = ?")
        params.append(status)
    if file_type:
        conditions.append("file_type = ?")
        params.append(file_type)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    db = await get_db()
    try:
        cursor = await db.execute(
            f"SELECT * FROM documents {where} ORDER BY created_at DESC LIMIT ?", params + [limit]
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        await db.close()


async def get_document(doc_id: int) -> dict | None:
    """获取单个文档"""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None
    finally:
        await db.close()


async def update_document(doc_id: int, **fields) -> bool:
    """更新文档字段，自动更新 updated_at"""
    if not fields:
        return False
    fields["updated_at"] = _now()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [doc_id]
    db = await get_db()
    try:
        cursor = await db.execute(f"UPDATE documents SET {set_clause} WHERE id = ?", values)
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def delete_document(doc_id: int) -> bool:
    """删除文档记录"""
    db = await get_db()
    try:
        cursor = await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


async def get_document_stats() -> dict:
    """获取文档统计信息"""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT status, COUNT(*) as count FROM documents GROUP BY status"
        )
        rows = await cursor.fetchall()
        stats = {_row_to_dict(r)["status"]: _row_to_dict(r)["count"] for r in rows}
        # 补齐所有状态
        for s in ("pending", "processing", "ready", "error"):
            stats.setdefault(s, 0)
        # 总数
        cursor = await db.execute("SELECT COUNT(*) as total FROM documents")
        row = await cursor.fetchone()
        stats["total"] = _row_to_dict(row)["total"]
        return stats
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Settings CRUD
# ---------------------------------------------------------------------------

async def get_settings() -> dict:
    """获取所有设置项，value 从 JSON 反序列化"""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT key, value FROM settings")
        rows = await cursor.fetchall()
        result = {}
        for r in rows:
            d = _row_to_dict(r)
            try:
                result[d["key"]] = json.loads(d["value"])
            except (json.JSONDecodeError, TypeError):
                result[d["key"]] = d["value"]
        return result
    finally:
        await db.close()


async def save_settings(data: dict):
    """保存设置项，value 序列化为 JSON"""
    now = _now()
    db = await get_db()
    try:
        for key, value in data.items():
            await db.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, json.dumps(value, ensure_ascii=False), now),
            )
        await db.commit()
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# FRA Reports CRUD
# ---------------------------------------------------------------------------

async def add_fra_report(query: str, content: str, filepath: str,
                         session_id: str | None = None) -> int:
    """添加 FRA 报告，返回报告 ID"""
    now = _now()
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO fra_reports (session_id, query, content, filepath, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, query, content, filepath, now),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def list_fra_reports(limit: int = 20) -> list[dict]:
    """获取 FRA 报告列表"""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, session_id, query, filepath, created_at FROM fra_reports "
            "ORDER BY created_at DESC LIMIT ?", (limit,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        await db.close()


async def get_fra_report(report_id: int) -> dict | None:
    """获取单个 FRA 报告"""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM fra_reports WHERE id = ?", (report_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Stock List 缓存
# ---------------------------------------------------------------------------

async def sync_stock_list():
    """从数据源同步A股股票列表到数据库。"""
    import asyncio
    from datasource import stock as stock_mod

    log.info("开始同步A股股票列表...")
    try:
        stocks = await asyncio.to_thread(stock_mod.get_stock_list)
    except Exception:
        log.error("获取股票列表失败", exc_info=True)
        return 0

    if not stocks:
        log.warning("数据源返回空股票列表")
        return 0

    now = _now()
    db = await get_db()
    try:
        await db.execute("DELETE FROM stock_list")
        await db.executemany(
            "INSERT INTO stock_list (code, name, market, updated_at) VALUES (?, ?, ?, ?)",
            [
                (s.get("code", ""), s.get("name", ""), s.get("market", ""), now)
                for s in stocks
            ],
        )
        await db.commit()
        log.info("股票列表同步完成: %d 条", len(stocks))
        return len(stocks)
    finally:
        await db.close()


def get_cached_stock_list(keyword: str = "", limit: int = 50) -> list[dict]:
    """从本地缓存查询股票列表（同步，供工具调用）。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        if keyword:
            cursor = conn.execute(
                "SELECT code, name, market FROM stock_list "
                "WHERE code LIKE ? OR name LIKE ? LIMIT ?",
                (f"%{keyword}%", f"%{keyword}%", limit),
            )
        else:
            cursor = conn.execute(
                "SELECT code, name, market FROM stock_list LIMIT ?", (limit,),
            )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Research Reports CRUD
# ---------------------------------------------------------------------------

async def add_research_report(query: str, content: str, filepath: str,
                               session_id: str | None = None) -> int:
    """添加研究报告，返回报告 ID"""
    now = _now()
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO research_reports (session_id, query, content, filepath, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, query, content, filepath, now),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def list_research_reports(limit: int = 20) -> list[dict]:
    """获取研究报告列表"""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, session_id, query, filepath, created_at FROM research_reports "
            "ORDER BY created_at DESC LIMIT ?", (limit,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        await db.close()


async def get_research_report(report_id: int) -> dict | None:
    """获取单个研究报告"""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM research_reports WHERE id = ?", (report_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Token Usage CRUD
# ---------------------------------------------------------------------------

async def add_token_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    source: str = "chat",
    session_id: str | None = None,
) -> int:
    """记录一次 LLM 调用的 token 用量，返回插入行 id"""
    db = await get_db()
    try:
        cursor = await db.execute(
            "INSERT INTO token_usage (session_id, model, prompt_tokens, completion_tokens, "
            "total_tokens, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, model, prompt_tokens, completion_tokens, total_tokens, source, _now()),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def get_token_summary(days: int = 30) -> dict:
    """获取 token 用量汇总统计"""
    db = await get_db()
    try:
        # 总计
        cursor = await db.execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0) as total_prompt, "
            "COALESCE(SUM(completion_tokens), 0) as total_completion, "
            "COALESCE(SUM(total_tokens), 0) as total_tokens, "
            "COUNT(*) as total_calls "
            "FROM token_usage WHERE created_at >= datetime('now', ?)",
            (f"-{days} days",),
        )
        totals = _row_to_dict(await cursor.fetchone())

        # 按模型分组
        cursor = await db.execute(
            "SELECT model, SUM(prompt_tokens) as prompt_tokens, "
            "SUM(completion_tokens) as completion_tokens, "
            "SUM(total_tokens) as total_tokens, COUNT(*) as calls "
            "FROM token_usage WHERE created_at >= datetime('now', ?) "
            "GROUP BY model ORDER BY total_tokens DESC",
            (f"-{days} days",),
        )
        by_model = [_row_to_dict(r) for r in await cursor.fetchall()]

        # 按来源分组
        cursor = await db.execute(
            "SELECT source, SUM(total_tokens) as total_tokens, COUNT(*) as calls "
            "FROM token_usage WHERE created_at >= datetime('now', ?) "
            "GROUP BY source ORDER BY total_tokens DESC",
            (f"-{days} days",),
        )
        by_source = [_row_to_dict(r) for r in await cursor.fetchall()]

        return {
            "totals": totals,
            "by_model": by_model,
            "by_source": by_source,
            "days": days,
        }
    finally:
        await db.close()


async def get_token_daily(days: int = 30) -> list[dict]:
    """获取按天的 token 用量趋势"""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT DATE(created_at) as date, "
            "SUM(prompt_tokens) as prompt_tokens, "
            "SUM(completion_tokens) as completion_tokens, "
            "SUM(total_tokens) as total_tokens, COUNT(*) as calls "
            "FROM token_usage WHERE created_at >= datetime('now', ?) "
            "GROUP BY DATE(created_at) ORDER BY date",
            (f"-{days} days",),
        )
        return [_row_to_dict(r) for r in await cursor.fetchall()]
    finally:
        await db.close()


async def get_token_recent(limit: int = 50) -> list[dict]:
    """获取最近的 token 用量明细"""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, session_id, model, prompt_tokens, completion_tokens, "
            "total_tokens, source, created_at FROM token_usage "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [_row_to_dict(r) for r in await cursor.fetchall()]
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Web Search Cache CRUD
# ---------------------------------------------------------------------------

def save_web_search(query: str, answer: str, results: list[dict]) -> int:
    """保存一次 Web 搜索结果，返回记录 ID（同步，供工具调用）"""
    now = _now()
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(
            "INSERT INTO web_search_cache (query, answer, results, created_at) VALUES (?, ?, ?, ?)",
            (query, answer, json.dumps(results, ensure_ascii=False), now),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_web_search_cache(query: str, limit: int = 5) -> list[dict]:
    """按查询关键词模糊匹配已缓存的搜索结果（同步）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT * FROM web_search_cache WHERE query LIKE ? ORDER BY created_at DESC LIMIT ?",
            (f"%{query}%", limit),
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()
