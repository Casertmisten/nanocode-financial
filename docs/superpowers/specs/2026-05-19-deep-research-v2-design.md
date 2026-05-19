# Deep Research V2 — 开放性问题三 Agent 协作研究

## 概述

新增 Deep Research 功能，支持用户输入任意开放性研究问题，通过 Plan → Executor × N → Generator 三 Agent 协作，自动拆解问题、并行搜集信息、生成结构化研究报告。

与现有 FRA（固定 3 维度 33 子问题的财报分析）的区别：FRA 面向特定场景且模板固定，Deep Research 面向任意问题且由 LLM 动态规划。

## 架构

```
用户输入研究问题
        ↓
   Plan Agent（LLM 调用）
        ↓
  生成子任务列表（JSON）
        ↓
   前端展示 → 用户确认/编辑
        ↓
  asyncio.gather(*[Executor(task) for task in sub_tasks])
        ↓
   Generator Agent（LLM 调用）
        ↓
   结构化 Markdown 报告
```

三阶段顺序管道，Executor 阶段并行执行。

## 模块结构

```
deep_research/
├── __init__.py       # 对外暴露 async_run()
├── schemas.py        # 数据结构：SubTask, ExecutorResult
├── plan.py           # Plan Agent：问题拆解 + 工具分配
├── executor.py       # Executor Agent：单子任务多轮工具调用循环
├── generator.py      # Generator Agent：汇总生成报告
└── prompts.py        # 三个 Agent 的系统提示词
```

API 层新增 `api/research.py`，前端新增 `research.html`。

## Plan Agent

**输入**：用户研究问题

**实现**：一次非流式 LLM 调用，system prompt 要求输出 JSON 格式的子任务列表。

**输出结构**：

```json
{
  "research_outline": "简要研究框架描述",
  "sub_tasks": [
    {
      "id": 1,
      "title": "子任务标题",
      "description": "子任务详细描述，说明需要搜集什么信息",
      "tools": ["rag_query", "stock_financial", "web_search"],
      "search_queries": ["预设的搜索关键词1", "关键词2"]
    }
  ]
}
```

**工具推荐**：system prompt 中列出所有可用工具及适用场景，由 LLM 自主分配。约束：子任务数量 3-8 个。

**用户确认环节**：Plan 完成后通过 SSE `event: plan` 推送给前端，用户可删除/修改/新增子任务，确认后通过 `POST /api/research/execute` 继续。

### 可用工具清单

| 工具 | 适用场景 |
|------|---------|
| `rag_query` | 搜索本地知识库中的研报、新闻、分析文档 |
| `web_search` | Web 搜索获取最新外部信息 |
| `stock_list` | 查询 A 股股票列表 |
| `stock_basic_info` | 获取个股基本信息 |
| `stock_quotes` | 获取个股实时行情 |
| `batch_stock_quotes` | 批量获取多只股票行情 |
| `stock_historical` | 获取历史 K 线数据 |
| `stock_financial` | 获取财务报表数据 |
| `market_status` | 获取市场状态 |
| `market_news` | 获取市场新闻 |
| `stock_news` | 获取个股新闻 |

## Executor Agent

**执行模式**：每个子任务启动一个独立的 Executor，通过 `asyncio.gather` 并行。

**核心循环**（最多 8 轮工具调用）：

```
子任务描述 + 限定工具集的 schema → LLM
        ↓
  LLM 返回工具调用 → 执行 → 结果回传 LLM
        ↓
  重复直到无工具调用或达到上限
        ↓
  LLM 输出该子任务的信息摘要
```

**工具集**：仅使用 Plan 为该子任务分配的工具子集。工具 schema 从 `tools.py` 的 `make_schema()` 中筛选。

**输出**（`ExecutorResult`）：

```python
@dataclass
class ExecutorResult:
    task_id: int
    title: str
    summary: str          # LLM 生成的信息摘要
    sources: list[str]    # 引用来源列表
    raw_data: list[str]   # 原始检索数据
```

**超时保护**：单个 Executor 超时 120 秒，超时则返回已有结果并标记为不完整。

## Generator Agent

**输入**：用户原始问题 + 所有 `ExecutorResult`

**实现**：一次非流式 LLM 调用，生成完整 Markdown 报告。

**报告结构**（prompt 引导）：

```markdown
# 研究报告：{问题标题}

## 研究概述
（简要说明研究问题和主要发现）

## 1. {子任务1标题}
（基于 Executor 结果的详细分析，引用数据来源）

## 2. {子任务2标题}
...

## 综合分析与结论
（跨子任务的综合判断、核心观点、风险提示）

## 信息来源
- 本地文档：《xxx》
- 外部来源：xxx
```

**报告持久化**：新建 `research_reports` 表（结构与 `fra_reports` 一致），文件保存到 `reports/` 目录。

## Web 搜索工具

**实现**：使用 `duckduckgo-search` 库，无需额外 API Key。

**注册到 `tools.py`**：

```python
"web_search": (
    "使用搜索引擎搜索互联网信息。返回搜索结果的标题、摘要和链接。",
    {"query": "string", "max_results": "number?"},
    web_search_tool,
)
```

**返回格式**：搜索结果条目（标题 + 摘要 + 链接）的文本拼接。

## API 设计

新增 `api/research.py`，挂载到 `/api/research`。

### 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/research/plan` | 输入问题，返回子任务列表（同步 JSON 响应） |
| POST | `/api/research/execute` | 输入确认后的子任务列表，SSE 流式推送执行进度和报告 |
| GET | `/api/research/reports` | 列出历史研究报告 |
| GET | `/api/research/reports/{id}` | 获取单个报告 |

### SSE 事件

```
event: plan          → {research_outline, sub_tasks}
event: executor_start → {task_id, title}
event: executor_progress → {task_id, tool, args}
event: executor_done  → {task_id, title, summary_length}
event: generate_start → {}
event: generate_done  → {report_id, filepath}
event: done           → {report_id}
```

### 请求/响应

**POST /api/research/plan**
```json
// 请求
{"query": "分析比亚迪的竞争优势和投资价值"}

// 响应
{
  "research_outline": "...",
  "sub_tasks": [...]
}
```

**POST /api/research/execute**
```json
// 请求
{
  "query": "原始问题",
  "sub_tasks": [...],   // 用户确认/编辑后的子任务列表
  "session_id": "可选"
}

// 响应：SSE 流
```

## 前端设计

新增 `research.html`，三步式交互：

1. **输入阶段**：搜索框输入研究问题，点击"开始研究"
2. **Plan 阶段**：展示子任务卡片列表，支持编辑/删除/新增，确认后继续
3. **执行 & 报告阶段**：展示各 Executor 进度，完成后渲染 Markdown 报告

复用现有 `frontend/js/api.js` 的 SSE 和 REST 工具函数。

## 数据库

新建 `research_reports` 表：

```sql
CREATE TABLE IF NOT EXISTS research_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    report TEXT NOT NULL,
    filepath TEXT,
    session_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

`db.py` 新增对应 CRUD：`add_research_report`, `list_research_reports`, `get_research_report`。

## 配置

`config.py` 新增：

```python
RESEARCH_MAX_SUBTASKS = 8        # 最大子任务数
RESEARCH_EXECUTOR_MAX_TURNS = 8  # Executor 最大工具调用轮次
RESEARCH_EXECUTOR_TIMEOUT = 120  # 单个 Executor 超时（秒）
```

## 依赖

`pyproject.toml` 新增：

```
duckduckgo-search>=6.0.0
```

## 与现有系统的关系

- **复用**：`llm.py`（call_llm, async_stream_chat）、`tools.py`（工具注册表、run_tool、make_schema）、`rag/`（知识库检索）、`datasource/`（股票/新闻数据）、`db.py`（数据库）
- **独立**：`deep_research/` 包、`api/research.py`、`research.html`、新提示词
- **不改动**：现有 FRA 功能、聊天功能、知识库管理

## 关键设计决策

1. **Plan 后用户确认**：避免 LLM 规划偏差导致资源浪费，用户可以调整方向
2. **Executor 工具集限定**：Plan 为每个子任务预分配工具，减少无意义的工具调用
3. **并行 Executor**：`asyncio.gather` 并行执行，配合超时保护
4. **Generator 单次调用**：所有子任务结果收集完毕后一次性生成，保证报告连贯性
5. **web_search 作为工具**：注册到统一工具表，Executor 可按需调用，与 RAG 检索互补
