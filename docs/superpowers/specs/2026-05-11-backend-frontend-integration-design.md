# 后端-前端连接设计

**日期：** 2026-05-11
**状态：** 已确认

## 背景

项目现有 CLI 入口 (`nanocode.py`) 和 3 个前端 HTML 页面 (`frontend/`)，两者完全独立。前端用 mock 数据，后端无 HTTP 服务。需要搭建 Web 服务层，将后端核心能力（对话、RAG、FRA、数据源）通过 API 暴露给前端，同时保持 CLI 继续可用。

## 决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 用户模型 | 单用户，无登录 | 个人工具 |
| Web 框架 | FastAPI + uvicorn | 已有依赖，原生 async/SSE |
| 数据库 | SQLite (aiosqlite) | 零配置，单用户足够 |
| LLM 调用 | httpx (流式) | 替换 urllib，支持 SSE |
| 前端托管 | FastAPI StaticFiles | 开发简单，无需 nginx |
| 前端 API | 原生 fetch + SSE | 无需引入框架 |
| CLI 共存 | 抽取 llm.py 共享层 | CLI 和 Web 共用核心逻辑 |

## 架构

```
用户 → localhost:8000
       ├── CLI 入口 (nanocode.py)
       └── Web 入口 (web.py / FastAPI)
              │
       ┌──────┴──────────────┐
       │     共享核心模块     │
       │  rag/  datasource/  │
       │  financial_report_  │
       │  analysis/          │
       └──────┬──────────────┘
              │
       ┌──────┴──────────────┐
       │     数据存储         │
       │  SQLite / ChromaDB  │
       │  文件系统            │
       └─────────────────────┘
```

## 数据库设计

SQLite 单文件 (`data/app.db`)，WAL 模式。

### sessions 表

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '新对话',
    model TEXT NOT NULL DEFAULT '',
    knowledge_base_ids TEXT DEFAULT '[]',  -- JSON 数组
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### messages 表

```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user','assistant','system','tool')),
    content TEXT NOT NULL DEFAULT '',
    tool_calls TEXT,               -- JSON，仅 assistant
    tool_result TEXT,              -- 仅 tool
    created_at TEXT NOT NULL
);
CREATE INDEX idx_messages_session ON messages(session_id, created_at);
```

### documents 表

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL UNIQUE,
    file_size INTEGER DEFAULT 0,
    file_type TEXT NOT NULL,        -- pdf/md/txt
    source TEXT DEFAULT 'upload',
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending','processing','ready','error')),
    chunks INTEGER DEFAULT 0,
    tags TEXT DEFAULT '[]',
    description TEXT DEFAULT '',
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_documents_status ON documents(status);
```

### settings 表

```sql
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,            -- JSON 字符串
    updated_at TEXT NOT NULL
);
```

### fra_reports 表

```sql
CREATE TABLE fra_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    query TEXT NOT NULL,
    content TEXT NOT NULL,
    filepath TEXT,
    created_at TEXT NOT NULL
);
```

## API 接口

### 对话

```
POST /api/chat
Body:   { session_id, message, model?, kb_ids? }
Resp:   SSE stream
        event: token      data: { content }
        event: tool_start data: { tool, args }
        event: tool_end   data: { result }
        event: done       data: { message_id }
```

后端流程：加载会话历史 → 拼接系统提示词 → 流式调用 LLM → SSE 推送 → 持久化消息。

### 会话管理

```
GET    /api/sessions          → 列表（最新在前）
GET    /api/sessions/:id      → 详情 + 消息
POST   /api/sessions          → 创建
DELETE /api/sessions/:id      → 删除
PATCH  /api/sessions/:id      → 更新标题/模型/知识库
```

### 文档管理

```
GET    /api/documents         → 列表（支持 ?status=&type= 筛选）
GET    /api/documents/stats   → 统计
POST   /api/documents/upload  → 上传（multipart/form-data）
DELETE /api/documents/:id     → 删除（文件 + 向量）
GET    /api/documents/:id     → 详情
```

上传流程：保存到 `data/uploads/` → 写元数据 → 触发 RAG ingest → 更新状态。

### 设置管理

```
GET    /api/settings          → 所有设置
PUT    /api/settings          → 保存
POST   /api/settings/test-connection → 测试连通性
```

### FRA 报告

```
POST   /api/fra               → 启动分析
Body:   { query, session_id? }
Resp:   SSE stream
        event: progress data: { stage, detail }
        event: done     data: { report_id }

GET    /api/fra/reports        → 报告列表
GET    /api/fra/reports/:id    → 报告详情
```

### 辅助接口

```
GET    /api/models             → 可用模型列表
GET    /api/knowledge-bases    → 知识库列表
GET    /api/health             → 健康检查
```

## 目录结构

```
nanocode-financial/
├── nanocode.py                    # CLI 入口（改造 LLM 调用）
├── web.py                         # Web 入口 (~50行)
├── config.py                      # 配置（小改）
├── llm.py                         # LLM 调用层（新增）
├── tools.py                       # Agent 工具（已有）
├── db.py                          # SQLite CRUD（新增）
├── api/                           # API 路由（新增）
│   ├── __init__.py
│   ├── chat.py
│   ├── sessions.py
│   ├── documents.py
│   ├── settings.py
│   └── fra.py
├── datasource/                    # 不变
├── rag/                           # 不变
├── financial_report_analysis/     # 不变
├── prompts/                       # 不变
├── frontend/                      # 改造 JS
│   ├── chat.html
│   ├── knowledge.html
│   ├── settings.html
│   ├── css/style.css
│   └── js/api.js                  # 新增
├── data/
│   ├── uploads/                   # 新增
│   ├── documents/                 # 已有
│   ├── chroma_db/                 # 已有
│   └── app.db                     # 新增
├── reports/                       # 已有
└── pyproject.toml                 # 加依赖
```

**新增文件：** `web.py`, `llm.py`, `db.py`, `api/` (5个), `frontend/js/api.js`, `data/uploads/`
**改造文件：** `nanocode.py`, `config.py`, `pyproject.toml`, 前端 3 个 HTML
**不变文件：** `datasource/`, `rag/`, `financial_report_analysis/`, `tools.py`

## LLM 调用改造

从 `nanocode.py` 抽取 LLM 调用逻辑到 `llm.py`：

- 用 `httpx` 替换 `urllib.request`，支持流式读取
- `stream_chat(messages, system_prompt, tools)` → 同步生成器（CLI 用）
- `async_stream_chat(messages, system_prompt, tools)` → 异步生成器（Web 用）
- 工具执行逻辑保持 `tools.py` 注册，`llm.py` 只负责调用和解析

`nanocode.py` 的 agentic loop 改为调用 `llm.stream_chat()` + `tools.py`，功能不变。

## 实现阶段

### 阶段 1 — 基础骨架（MVP）

- `web.py` + `db.py` + `api/chat.py` + `api/sessions.py`
- `llm.py` 从 nanocode.py 抽取
- SSE 对话流式输出
- 会话持久化
- 前端 chat.html 对接

### 阶段 2 — 文档 + 设置

- `api/documents.py` + `api/settings.py`
- 文件上传 + RAG ingest
- 前端 knowledge.html + settings.html 对接

### 阶段 3 — FRA + 优化

- `api/fra.py`
- FRA 流式进度推送
- CLI 改造用 llm.py
