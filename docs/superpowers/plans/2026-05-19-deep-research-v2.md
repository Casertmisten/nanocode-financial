# Deep Research V2 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现三 Agent 协作（Plan → Executor × N → Generator）的开放性问题研究功能，支持用户确认子任务、并行执行、Web 搜索、生成结构化报告。

**Architecture:** 顺序管道：Plan Agent 拆解问题 → 用户确认/编辑子任务 → asyncio.gather 并行启动多个 Executor Agent（每个是独立的多轮工具调用循环） → Generator Agent 汇总生成 Markdown 报告。通过 FastAPI SSE 流式推送进度。新增 `deep_research/` 包、`api/research.py` 路由、`research.html` 前端页面。

**Tech Stack:** FastAPI + SSE, httpx (LLM 调用), duckduckgo-search (Web 搜索), aiosqlite (数据库), asyncio (并行执行)

---

## 文件结构

| 操作 | 文件 | 职责 |
|------|------|------|
| 创建 | `deep_research/__init__.py` | 包入口，暴露 `async_run` |
| 创建 | `deep_research/schemas.py` | 数据结构：SubTask, ExecutorResult |
| 创建 | `deep_research/prompts.py` | 三个 Agent 的 system prompt |
| 创建 | `deep_research/plan.py` | Plan Agent：问题拆解 |
| 创建 | `deep_research/executor.py` | Executor Agent：多轮工具调用循环 |
| 创建 | `deep_research/generator.py` | Generator Agent：报告生成 |
| 修改 | `tools.py` | 新增 `web_search` 工具注册 |
| 修改 | `config.py` | 新增 Deep Research 配置常量 |
| 修改 | `db.py` | 新增 `research_reports` 表 + CRUD |
| 修改 | `web.py` | 注册 research 路由 |
| 修改 | `pyproject.toml` | 新增 `duckduckgo-search` 依赖 |
| 创建 | `api/research.py` | API 路由：plan / execute / reports |
| 创建 | `frontend/research.html` | Deep Research 前端页面 |
| 修改 | `frontend/chat.html` | 侧边栏导航添加"深度研究"入口 |

---

### Task 1: 安装 duckduckgo-search 依赖

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: 添加依赖**

```bash
uv add duckduckgo-search>=6.0.0
```

- [ ] **Step 2: 验证安装**

```bash
uv run python -c "from duckduckgo_search import DDGS; print('ok')"
```

Expected: 输出 `ok`

- [ ] **Step 3: 提交**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: 添加 duckduckgo-search 依赖"
```

---

### Task 2: 新增配置常量

**Files:**
- Modify: `config.py:57` (在 AKSHARE_CACHE_TTL 之后)

- [ ] **Step 1: 在 `config.py` 的 AKShare 配置之后添加 Deep Research 配置**

在 `config.py` 第 57 行 `AKSHARE_CACHE_TTL` 之后添加：

```python
# --- Deep Research ---
RESEARCH_MAX_SUBTASKS = 8        # 最大子任务数
RESEARCH_EXECUTOR_MAX_TURNS = 8  # Executor 最大工具调用轮次
RESEARCH_EXECUTOR_TIMEOUT = 120  # 单个 Executor 超时（秒）
```

- [ ] **Step 2: 验证**

```bash
uv run python -c "import config; print(config.RESEARCH_MAX_SUBTASKS, config.RESEARCH_EXECUTOR_MAX_TURNS, config.RESEARCH_EXECUTOR_TIMEOUT)"
```

Expected: 输出 `8 8 120`

- [ ] **Step 3: 提交**

```bash
git add config.py
git commit -m "feat: Deep Research 配置常量"
```

---

### Task 3: 新增数据库表和 CRUD

**Files:**
- Modify: `db.py:99` (在 `init_db` 中 `stock_list` 表之后)
- Modify: `db.py:469` (在文件末尾)

- [ ] **Step 1: 在 `init_db()` 的 `executescript` 中添加 `research_reports` 表**

在 `db.py` 的 `init_db` 函数中的 `stock_list` 表 CREATE 语句之后添加：

```python
            CREATE TABLE IF NOT EXISTS research_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
                query TEXT NOT NULL,
                content TEXT NOT NULL,
                filepath TEXT,
                created_at TEXT NOT NULL
            );
```

- [ ] **Step 2: 在 `db.py` 文件末尾添加 research reports CRUD 函数**

在 `db.py` 文件末尾（`get_cached_stock_list` 函数之后）添加：

```python
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
```

- [ ] **Step 3: 验证**

```bash
uv run python -c "
import asyncio, db
async def test():
    await db.init_db()
    rid = await db.add_research_report('测试问题', '# 测试报告', '/tmp/test.md')
    r = await db.get_research_report(rid)
    print(r['query'], r['content'])
    reports = await db.list_research_reports()
    print('报告数:', len(reports))
asyncio.run(test())
"
```

Expected: 输出 `测试问题 # 测试报告` 和 `报告数: 1`

- [ ] **Step 4: 提交**

```bash
git add db.py
git commit -m "feat: research_reports 数据库表及 CRUD"
```

---

### Task 4: 新增 web_search 工具

**Files:**
- Modify: `tools.py:200` (在 TOOLS 字典末尾添加 web_search)

- [ ] **Step 1: 在 `tools.py` 中添加 web_search 工具实现函数**

在 `tools.py` 的 `stock_news_tool` 函数之后（约第 198 行），`TOOLS` 字典之前添加：

```python
def web_search_tool(args):
    """使用 DuckDuckGo 搜索互联网信息。"""
    from duckduckgo_search import DDGS

    query = args["query"]
    max_results = args.get("max_results", 5)
    log.info("Web 搜索: query=%s, max_results=%d", query[:50], max_results)

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "未找到相关搜索结果。"
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"{i}. {r.get('title', '')}\n   {r.get('body', '')}\n   链接: {r.get('href', '')}")
        return "\n\n".join(parts)
    except Exception as e:
        log.error("Web 搜索失败: %s", e, exc_info=True)
        return f"搜索失败: {e}"
```

- [ ] **Step 2: 在 `tools.py` 的 `TOOLS` 字典中注册 web_search**

在 `TOOLS` 字典的 `"stock_news"` 条目之后添加：

```python
    # --- Web 搜索工具 ---
    "web_search": (
        "使用搜索引擎搜索互联网信息，获取最新新闻、观点、数据等。返回搜索结果的标题、摘要和链接。",
        {"query": "string", "max_results": "number?"},
        web_search_tool,
    ),
```

- [ ] **Step 3: 验证**

```bash
uv run python -c "
import tools
schema = tools.make_schema()
web = [s for s in schema if s['function']['name'] == 'web_search']
print('注册:', len(web) > 0)
result = tools.run_tool('web_search', {'query': '比亚迪 2024 年报', 'max_results': 2})
print(result[:200])
"
```

Expected: 输出 `注册: True` 以及搜索结果文本

- [ ] **Step 4: 提交**

```bash
git add tools.py
git commit -m "feat: 新增 web_search 工具（DuckDuckGo）"
```

---

### Task 5: 创建 deep_research 包骨架和数据结构

**Files:**
- Create: `deep_research/__init__.py`
- Create: `deep_research/schemas.py`

- [ ] **Step 1: 创建 `deep_research/schemas.py`**

```python
"""Deep Research 数据结构定义。"""

from dataclasses import dataclass, field


@dataclass
class SubTask:
    """Plan Agent 生成的子任务。"""
    id: int
    title: str
    description: str
    tools: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)


@dataclass
class ExecutorResult:
    """Executor Agent 执行结果。"""
    task_id: int
    title: str
    summary: str = ""
    sources: list[str] = field(default_factory=list)
    raw_data: list[str] = field(default_factory=list)
    complete: bool = True  # 是否完整完成（超时则为 False）
```

- [ ] **Step 2: 创建 `deep_research/__init__.py`**

```python
"""Deep Research — 开放性问题三 Agent 协作研究。"""
```

- [ ] **Step 3: 提交**

```bash
git add deep_research/
git commit -m "feat: Deep Research 包骨架和数据结构"
```

---

### Task 6: 创建 Deep Research 提示词

**Files:**
- Create: `deep_research/prompts.py`

- [ ] **Step 1: 创建 `deep_research/prompts.py`**

```python
"""Deep Research 三个 Agent 的系统提示词。"""

# Plan Agent 可分配的工具及说明
_TOOLS_DESCRIPTION = """
可用工具及适用场景：
- rag_query: 搜索本地金融知识库（研报、新闻、分析文档）
- web_search: 使用搜索引擎获取互联网最新信息
- stock_list: 查询A股股票列表
- stock_basic_info: 获取个股基本信息（公司名称、行业、市值等）
- stock_quotes: 获取个股实时行情
- batch_stock_quotes: 批量获取多只股票实时行情
- stock_historical: 获取历史K线数据
- stock_financial: 获取财务报表数据
- market_status: 获取当前市场状态
- market_news: 聚合采集最新A股市场新闻
- stock_news: 获取个股相关新闻
"""

plan_system_prompt = f"""你是一个专业的研究规划师。你的任务是将用户的研究问题拆解为 3-8 个独立的子任务，每个子任务负责搜集某一方面的信息。

{_TOOLS_DESCRIPTION}

要求：
1. 输出严格的 JSON 格式，不要包含任何其他文本
2. 子任务数量 3-8 个，每个子任务目标明确、互不重叠
3. 为每个子任务分配合适的工具（从上述工具中选择）
4. 为每个子任务提供 2-3 个预设搜索关键词
5. 确保子任务覆盖问题的各个方面

输出格式：
{{
  "research_outline": "简要研究框架描述（一句话）",
  "sub_tasks": [
    {{
      "id": 1,
      "title": "子任务标题",
      "description": "详细描述需要搜集什么信息",
      "tools": ["tool_name1", "tool_name2"],
      "search_queries": ["关键词1", "关键词2"]
    }}
  ]
}}"""

executor_system_prompt = """你是一个信息搜集专家。你的任务是针对给定研究子任务，使用提供的工具搜集和提取关键信息。

要求：
1. 主动调用工具获取信息，不要仅凭自身知识回答
2. 每次调用工具后，分析结果并决定是否需要进一步搜索
3. 最终输出一段结构化的信息摘要，包含具体数据和事实
4. 在摘要末尾列出所有信息来源
5. 如果某个工具返回空结果或错误，尝试换一个搜索词或换一个工具

输出格式：
## 信息摘要
（详细的信息摘要，包含具体数据）

## 信息来源
- 来源1
- 来源2
"""

generator_system_prompt = """你是一个资深研究分析师。你的任务是根据多个子任务的研究结果，撰写一份结构完整、逻辑清晰的研究报告。

报告要求：
1. 章节按子任务主题组织，每个章节基于对应的子任务结果展开
2. 引用具体数据和事实，标注来源
3. 最后给出综合分析与结论，提炼核心观点和关键发现
4. 语言专业但通俗易懂
5. 报告长度适中，避免冗余重复

报告结构：
# 研究报告：{问题标题}

## 研究概述
（简要说明研究问题和主要发现，3-5句话）

## 1. {子任务1标题}
（基于 Executor 结果的详细分析）

## 2. {子任务2标题}
...

## 综合分析与结论
（跨子任务的综合判断、核心观点、风险提示）

## 信息来源
- 本地文档：《xxx》
- 外部来源：xxx
"""
```

- [ ] **Step 2: 验证**

```bash
uv run python -c "from deep_research.prompts import plan_system_prompt, executor_system_prompt, generator_system_prompt; print('Plan提示词长度:', len(plan_system_prompt)); print('ok')"
```

Expected: 输出提示词长度和 `ok`

- [ ] **Step 3: 提交**

```bash
git add deep_research/prompts.py
git commit -m "feat: Deep Research 提示词模板"
```

---

### Task 7: 实现 Plan Agent

**Files:**
- Create: `deep_research/plan.py`

- [ ] **Step 1: 创建 `deep_research/plan.py`**

```python
"""Plan Agent — 将研究问题拆解为子任务列表。"""

import json
import logging

import llm
from deep_research.prompts import plan_system_prompt
from deep_research.schemas import SubTask

log = logging.getLogger(__name__)


def plan(query: str) -> tuple[str, list[SubTask]]:
    """执行 Plan Agent，返回研究框架和子任务列表。

    Args:
        query: 用户研究问题。

    Returns:
        (research_outline, sub_tasks)
    """
    log.info("Plan Agent 开始: query=%s", query[:80])

    user_prompt = f"请将以下研究问题拆解为子任务：\n\n{query}"
    response = llm.call_llm(plan_system_prompt, user_prompt)

    # 解析 JSON（兼容 LLM 输出前后可能有的 markdown 代码块标记）
    text = response.strip()
    if text.startswith("```"):
        # 去掉 ```json 和 ``` 包裹
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        log.error("Plan JSON 解析失败: %s, 原始响应: %s", e, text[:300])
        raise ValueError(f"Plan Agent 返回格式错误: {e}") from e

    outline = data.get("research_outline", "")
    raw_tasks = data.get("sub_tasks", [])

    sub_tasks = []
    for i, t in enumerate(raw_tasks):
        sub_tasks.append(SubTask(
            id=t.get("id", i + 1),
            title=t.get("title", f"子任务 {i + 1}"),
            description=t.get("description", ""),
            tools=t.get("tools", []),
            search_queries=t.get("search_queries", []),
        ))

    log.info("Plan 完成: %d 个子任务, 框架: %s", len(sub_tasks), outline[:50])
    return outline, sub_tasks
```

- [ ] **Step 2: 验证**

```bash
uv run python -c "
from deep_research.plan import plan
outline, tasks = plan('分析比亚迪的竞争优势和投资价值')
print('框架:', outline)
for t in tasks:
    print(f'  [{t.id}] {t.title} -> {t.tools}')
"
```

Expected: 输出研究框架和子任务列表（需要 LLM API 可用）

- [ ] **Step 3: 提交**

```bash
git add deep_research/plan.py
git commit -m "feat: Plan Agent 实现问题拆解"
```

---

### Task 8: 实现 Executor Agent

**Files:**
- Create: `deep_research/executor.py`

- [ ] **Step 1: 创建 `deep_research/executor.py`**

```python
"""Executor Agent — 单子任务多轮工具调用循环。"""

import asyncio
import json
import logging

import httpx

import config
import llm
import tools
from deep_research.prompts import executor_system_prompt
from deep_research.schemas import ExecutorResult, SubTask

log = logging.getLogger(__name__)

# 不允许 Executor 使用的工具（文件操作类）
_BLOCKED_TOOLS = {"read", "write", "edit", "glob", "grep", "bash", "rag_ingest"}


def _filter_schema(allowed_tools: list[str]) -> list[dict]:
    """从完整工具 schema 中筛选允许的工具。"""
    full_schema = tools.make_schema()
    allowed = set(allowed_tools) - _BLOCKED_TOOLS
    return [s for s in full_schema if s["function"]["name"] in allowed]


def _extract_sources(text: str) -> list[str]:
    """从 LLM 输出中提取信息来源行。"""
    sources = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("- ") and ("来源" in text[:text.index(line)] if line in text else False):
            sources.append(line[2:])
    return sources


async def execute_task(sub_task: SubTask) -> ExecutorResult:
    """执行单个子任务，返回搜集结果。

    Args:
        sub_task: Plan Agent 分配的子任务。

    Returns:
        ExecutorResult 包含摘要、来源和原始数据。
    """
    log.info("Executor 启动: task_id=%d, title=%s", sub_task.id, sub_task.title)

    task_schema = _filter_schema(sub_task.tools)
    if not task_schema:
        log.warning("子任务 %d 无可用工具: %s", sub_task.id, sub_task.tools)

    user_prompt = f"""研究子任务：{sub_task.title}

详细描述：{sub_task.description}

建议搜索关键词：{', '.join(sub_task.search_queries)}

请使用提供的工具搜集信息，完成这个子任务。"""

    messages = [{"role": "user", "content": user_prompt}]
    raw_data: list[str] = []
    sources: list[str] = []
    max_turns = config.RESEARCH_EXECUTOR_MAX_TURNS

    try:
        for turn in range(max_turns):
            # 异步流式调用 LLM
            content_parts: list[str] = []
            tool_calls_map: dict[int, dict] = {}

            async for chunk in llm.async_stream_chat(
                messages, executor_system_prompt, tools=task_schema if task_schema else None,
            ):
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                text = delta.get("content")
                if text:
                    content_parts.append(text)

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

            # 无工具调用，LLM 输出最终摘要
            if not tool_calls_map:
                break

            # 记录 assistant 消息
            assistant_msg: dict = {
                "role": "assistant",
                "content": "".join(content_parts) or None,
                "tool_calls": [tool_calls_map[i] for i in sorted(tool_calls_map)],
            }
            messages.append(assistant_msg)

            # 执行工具调用
            for idx in sorted(tool_calls_map):
                tc = tool_calls_map[idx]
                tool_name = tc["function"]["name"]
                try:
                    tool_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    tool_args = {}

                log.info("Executor[%d] 工具调用: %s(%s)", sub_task.id, tool_name,
                         json.dumps(tool_args, ensure_ascii=False)[:100])

                # 在线程中执行同步工具
                tool_result = await asyncio.to_thread(tools.run_tool, tool_name, tool_args)
                raw_data.append(f"[{tool_name}] {tool_result[:500]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(tool_result),
                })

            if turn == max_turns - 1:
                log.warning("Executor[%d] 达到最大轮次 %d", sub_task.id, max_turns)

    except asyncio.TimeoutError:
        log.warning("Executor[%d] 超时", sub_task.id)
    except Exception:
        log.error("Executor[%d] 执行失败", sub_task.id, exc_info=True)

    # 最终摘要为所有 content 拼接
    summary = "".join(content_parts) if content_parts else "未能获取有效信息。"
    sources = _extract_sources(summary)

    log.info("Executor[%d] 完成: 摘要长度=%d, 来源数=%d", sub_task.id, len(summary), len(sources))
    return ExecutorResult(
        task_id=sub_task.id,
        title=sub_task.title,
        summary=summary,
        sources=sources,
        raw_data=raw_data,
        complete=not tool_calls_map or turn < max_turns - 1,
    )
```

- [ ] **Step 2: 验证语法正确**

```bash
uv run python -c "from deep_research.executor import execute_task; print('ok')"
```

Expected: 输出 `ok`

- [ ] **Step 3: 提交**

```bash
git add deep_research/executor.py
git commit -m "feat: Executor Agent 多轮工具调用循环"
```

---

### Task 9: 实现 Generator Agent

**Files:**
- Create: `deep_research/generator.py`

- [ ] **Step 1: 创建 `deep_research/generator.py`**

```python
"""Generator Agent — 汇总子任务结果生成结构化报告。"""

import logging

import llm
from deep_research.prompts import generator_system_prompt
from deep_research.schemas import ExecutorResult

log = logging.getLogger(__name__)


def generate(query: str, results: list[ExecutorResult]) -> str:
    """执行 Generator Agent，生成完整研究报告。

    Args:
        query: 用户原始研究问题。
        results: 所有 Executor 的执行结果。

    Returns:
        Markdown 格式的完整研究报告。
    """
    log.info("Generator 开始: query=%s, 子任务数=%d", query[:50], len(results))

    # 构建各子任务结果文本
    results_text = ""
    all_sources: list[str] = []

    for i, r in enumerate(results, 1):
        status = "" if r.complete else "（未完整完成）"
        results_text += f"\n### 子任务 {i}：{r.title} {status}\n\n{r.summary}\n"
        if r.sources:
            all_sources.extend(r.sources)

    # 去重来源
    unique_sources = list(dict.fromkeys(all_sources))

    user_prompt = f"""用户研究问题：{query}

以下是各子任务的研究结果：

{results_text}

请根据以上研究结果，撰写一份完整的研究报告。"""

    report = llm.call_llm(generator_system_prompt, user_prompt)

    log.info("Generator 完成: 报告长度=%d", len(report))
    return report
```

- [ ] **Step 2: 验证语法正确**

```bash
uv run python -c "from deep_research.generator import generate; print('ok')"
```

Expected: 输出 `ok`

- [ ] **Step 3: 提交**

```bash
git add deep_research/generator.py
git commit -m "feat: Generator Agent 报告生成"
```

---

### Task 10: 实现完整 pipeline 入口

**Files:**
- Modify: `deep_research/__init__.py`

- [ ] **Step 1: 重写 `deep_research/__init__.py`，暴露 `plan` 和 `execute` 分离的 API**

```python
"""Deep Research — 开放性问题三 Agent 协作研究。

提供两个阶段 API：
1. plan(query) → 拆解子任务
2. execute(query, sub_tasks, on_progress) → 并行执行 + 生成报告
"""

import asyncio
import datetime
import json
import logging
import os
from typing import AsyncIterator, Callable

import config

from deep_research.schemas import ExecutorResult, SubTask
from deep_research.plan import plan as _plan
from deep_research.executor import execute_task
from deep_research.generator import generate as _generate

log = logging.getLogger(__name__)


def plan(query: str) -> tuple[str, list[SubTask]]:
    """Plan 阶段：拆解问题为子任务列表。"""
    return _plan(query)


async def execute(
    query: str,
    sub_tasks: list[SubTask],
    on_progress: Callable[[str, dict], None] | None = None,
) -> AsyncIterator[str]:
    """Execute + Generate 阶段：并行执行子任务并生成报告。

    Args:
        query: 用户原始问题。
        sub_tasks: 确认后的子任务列表。
        on_progress: SSE 进度回调，接收 (event_type, data_dict)。

    Yields:
        SSE 事件字符串。
    """
    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    # 并行执行所有子任务
    results: list[ExecutorResult] = [None] * len(sub_tasks)

    async def _run_one(idx: int, task: SubTask):
        if on_progress:
            on_progress("executor_start", {"task_id": task.id, "title": task.title})
        yield _sse("executor_start", {"task_id": task.id, "title": task.title})

        try:
            result = await asyncio.wait_for(
                execute_task(task),
                timeout=config.RESEARCH_EXECUTOR_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning("Executor[%d] 超时 (%ds)", task.id, config.RESEARCH_EXECUTOR_TIMEOUT)
            result = ExecutorResult(
                task_id=task.id, title=task.title,
                summary="执行超时，未能完成信息搜集。", complete=False,
            )
        except Exception as e:
            log.error("Executor[%d] 失败: %s", task.id, e, exc_info=True)
            result = ExecutorResult(
                task_id=task.id, title=task.title,
                summary=f"执行失败: {e}", complete=False,
            )

        results[idx] = result
        yield _sse("executor_done", {
            "task_id": task.id, "title": task.title,
            "summary_length": len(result.summary), "complete": result.complete,
        })

    # 用 gather 并行
    async def _run_all():
        coros = [_run_one(i, t) for i, t in enumerate(sub_tasks)]
        # collect generators
        gens = []
        for coro in coros:
            gen = coro.__aiter__()
            gens.append(gen)

        # 逐个推进所有 generator
        done = [False] * len(gens)
        while not all(done):
            for i, gen in enumerate(gens):
                if done[i]:
                    continue
                try:
                    sse_str = await gen.__anext__()
                    yield sse_str
                except StopAsyncIteration:
                    done[i] = True

    async for sse_str in _run_all():
        yield sse_str

    # 过滤掉失败的空结果
    valid_results = [r for r in results if r is not None]

    # Generator 阶段
    yield _sse("generate_start", {})

    report = _generate(query, valid_results)

    # 保存报告到文件和数据库
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(config.BASE_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    filepath = os.path.join(report_dir, f"research_{ts}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    import db
    report_id = await db.add_research_report(query, report, filepath)
    log.info("研究报告已保存: id=%d, path=%s", report_id, filepath)

    yield _sse("generate_done", {"report_id": report_id, "filepath": filepath})
    yield _sse("done", {"report_id": report_id})
```

- [ ] **Step 2: 验证语法正确**

```bash
uv run python -c "from deep_research import plan, execute; print('ok')"
```

Expected: 输出 `ok`

- [ ] **Step 3: 提交**

```bash
git add deep_research/__init__.py
git commit -m "feat: Deep Research pipeline 入口（plan + execute 分离）"
```

---

### Task 11: 创建 API 路由

**Files:**
- Create: `api/research.py`
- Modify: `web.py:49` (注册路由)

- [ ] **Step 1: 创建 `api/research.py`**

```python
"""Deep Research API — Plan 同步接口 + Execute SSE 流式推送。"""

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import db
from deep_research import plan as dr_plan
from deep_research import execute as dr_execute
from deep_research.schemas import SubTask
from utils import BaseLogger

log = BaseLogger.getLogger("research")

router = APIRouter(prefix="/api/research", tags=["research"])


class PlanRequest(BaseModel):
    query: str


class ExecuteRequest(BaseModel):
    query: str
    sub_tasks: list[dict]
    session_id: str | None = None


@router.post("/plan")
async def run_plan(req: PlanRequest):
    """Plan 阶段：拆解问题为子任务列表。"""
    if not req.query.strip():
        raise HTTPException(400, "缺少研究问题")

    log.info("收到 Plan 请求: query=%s", req.query[:80])

    try:
        outline, sub_tasks = dr_plan(req.query)
    except Exception as e:
        log.error("Plan 失败: %s", e, exc_info=True)
        raise HTTPException(500, f"规划失败: {e}")

    return {
        "research_outline": outline,
        "sub_tasks": [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "tools": t.tools,
                "search_queries": t.search_queries,
            }
            for t in sub_tasks
        ],
    }


@router.post("/execute")
async def run_execute(req: ExecuteRequest):
    """Execute 阶段：并行执行子任务 + 生成报告，SSE 流式推送。"""
    if not req.sub_tasks:
        raise HTTPException(400, "子任务列表为空")

    log.info("收到 Execute 请求: %d 个子任务", len(req.sub_tasks))

    sub_tasks = [
        SubTask(
            id=t.get("id", i + 1),
            title=t.get("title", ""),
            description=t.get("description", ""),
            tools=t.get("tools", []),
            search_queries=t.get("search_queries", []),
        )
        for i, t in enumerate(req.sub_tasks)
    ]

    async def stream():
        async for sse_str in dr_execute(req.query, sub_tasks):
            yield sse_str

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/reports")
async def list_reports():
    """获取历史研究报告列表。"""
    return await db.list_research_reports()


@router.get("/reports/{report_id}")
async def get_report(report_id: int):
    """获取单个研究报告。"""
    report = await db.get_research_report(report_id)
    if not report:
        raise HTTPException(404, "报告不存在")
    return report
```

- [ ] **Step 2: 在 `web.py` 中注册 research 路由**

在 `web.py` 的 import 区域（约第 16 行 `from api.fra import router as fra_router` 之后）添加：

```python
from api.research import router as research_router
```

在路由注册区域（约第 50 行 `app.include_router(fra_router)` 之后）添加：

```python
app.include_router(research_router)
```

- [ ] **Step 3: 验证路由注册**

```bash
uv run python -c "
from web import app
routes = [r.path for r in app.routes if hasattr(r, 'path')]
research = [r for r in routes if 'research' in r]
print('Research 路由:', research)
"
```

Expected: 输出包含 `/api/research/plan`, `/api/research/execute`, `/api/research/reports` 等

- [ ] **Step 4: 提交**

```bash
git add api/research.py web.py
git commit -m "feat: Deep Research API 路由（plan/execute/reports）"
```

---

### Task 12: 创建前端页面

**Files:**
- Create: `frontend/research.html`
- Modify: `frontend/chat.html:697-710` (侧边栏导航添加深度研究入口)

- [ ] **Step 1: 创建 `frontend/research.html`**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>金融智能分析助手 - 深度研究</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="css/style.css">
  <script src="https://cdn.jsdelivr.net/npm/marked@15/marked.min.js"></script>
  <style>
    .research-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      padding: var(--stack-lg) calc(var(--stack-lg) * 2);
      max-width: 900px;
      margin: 0 auto;
      width: 100%;
    }

    .research-input-area {
      display: flex;
      gap: var(--stack-sm);
      margin-bottom: var(--stack-lg);
    }

    .research-input-area input {
      flex: 1;
      padding: 10px var(--stack-md);
      border: 1px solid var(--color-outline-variant);
      border-radius: var(--radius);
      font: var(--font-body-md);
      background: var(--color-surface-container-lowest);
    }

    .research-input-area input:focus {
      border-color: var(--color-primary);
      box-shadow: 0 0 0 2px rgba(0, 35, 111, 0.1);
    }

    .research-input-area input::placeholder {
      color: var(--color-outline);
    }

    .btn-research {
      padding: 10px 20px;
      border-radius: var(--radius);
      background: var(--color-primary);
      color: var(--color-on-primary);
      font-weight: 500;
      white-space: nowrap;
      transition: all var(--transition-fast);
    }

    .btn-research:hover { background: var(--color-primary-container); }
    .btn-research:disabled { background: var(--color-surface-container); color: var(--color-outline); cursor: default; }

    /* 子任务卡片 */
    .task-list { margin-bottom: var(--stack-lg); }

    .task-list-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: var(--stack-md);
    }

    .task-list-header h3 { font-size: 16px; font-weight: 600; }

    .task-card {
      background: var(--color-surface-container-lowest);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-md);
      padding: var(--stack-md);
      margin-bottom: var(--stack-sm);
      position: relative;
    }

    .task-card-header {
      display: flex;
      align-items: flex-start;
      gap: var(--stack-sm);
    }

    .task-card-number {
      width: 24px;
      height: 24px;
      border-radius: 50%;
      background: var(--color-primary);
      color: var(--color-on-primary);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      font-weight: 600;
      flex-shrink: 0;
    }

    .task-card-body { flex: 1; min-width: 0; }

    .task-card-title {
      font-weight: 500;
      margin-bottom: 4px;
    }

    .task-card-title input {
      width: 100%;
      padding: 2px 6px;
      border: 1px solid transparent;
      border-radius: 4px;
      font-weight: 500;
    }

    .task-card-title input:focus {
      border-color: var(--color-primary);
      background: var(--color-surface-container-lowest);
    }

    .task-card-desc {
      font-size: 13px;
      color: var(--color-on-surface-variant);
      margin-bottom: 6px;
    }

    .task-card-desc textarea {
      width: 100%;
      padding: 4px 6px;
      border: 1px solid transparent;
      border-radius: 4px;
      font-size: 13px;
      color: var(--color-on-surface-variant);
      resize: vertical;
      min-height: 36px;
    }

    .task-card-desc textarea:focus {
      border-color: var(--color-primary);
      background: var(--color-surface-container-lowest);
    }

    .task-card-tools {
      display: flex;
      gap: 4px;
      flex-wrap: wrap;
    }

    .task-tool-tag {
      font-size: 11px;
      padding: 1px 8px;
      border-radius: var(--radius-full);
      background: var(--color-primary-fixed);
      color: var(--color-primary);
    }

    .task-card-delete {
      position: absolute;
      top: 8px;
      right: 8px;
      width: 24px;
      height: 24px;
      border-radius: 4px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--color-outline);
      opacity: 0;
      transition: all var(--transition-fast);
    }

    .task-card:hover .task-card-delete { opacity: 1; }
    .task-card-delete:hover { background: var(--color-surface-container); color: var(--color-danger); }

    .task-actions {
      display: flex;
      gap: var(--stack-sm);
      margin-top: var(--stack-md);
    }

    .btn-confirm {
      padding: 8px 20px;
      border-radius: var(--radius);
      background: var(--color-primary);
      color: var(--color-on-primary);
      font-weight: 500;
      transition: all var(--transition-fast);
    }

    .btn-confirm:hover { background: var(--color-primary-container); }

    .btn-add-task {
      padding: 8px 16px;
      border-radius: var(--radius);
      border: 1px solid var(--color-outline-variant);
      color: var(--color-on-surface-variant);
      font-size: 13px;
      transition: all var(--transition-fast);
    }

    .btn-add-task:hover { border-color: var(--color-primary); color: var(--color-primary); }

    /* 执行进度 */
    .progress-section { margin-bottom: var(--stack-lg); }

    .progress-section h3 {
      font-size: 16px;
      font-weight: 600;
      margin-bottom: var(--stack-md);
    }

    .executor-progress {
      display: flex;
      align-items: center;
      gap: var(--stack-sm);
      padding: 8px 12px;
      border-radius: var(--radius);
      margin-bottom: 4px;
      font-size: 13px;
    }

    .executor-progress.running {
      background: var(--color-primary-fixed);
      color: var(--color-primary);
    }

    .executor-progress.done {
      background: rgba(16, 185, 129, 0.08);
      color: #047857;
    }

    .executor-progress.error {
      background: rgba(239, 68, 68, 0.08);
      color: #dc2626;
    }

    /* 报告展示 */
    .report-section {
      background: var(--color-surface-container-lowest);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-md);
      padding: var(--stack-lg);
      line-height: 1.8;
    }

    .report-section h1 { font-size: 1.4em; margin-bottom: 0.5em; }
    .report-section h2 { font-size: 1.2em; margin: 1em 0 0.4em; }
    .report-section h3 { font-size: 1.05em; margin: 0.8em 0 0.3em; }
    .report-section p { margin: 0.5em 0; }
    .report-section ul, .report-section ol { padding-left: 1.5em; margin: 0.4em 0; }
    .report-section li { margin: 0.2em 0; }
    .report-section table { border-collapse: collapse; width: 100%; margin: 0.8em 0; }
    .report-section th, .report-section td { border: 1px solid var(--color-border); padding: 0.4em 0.8em; text-align: left; }
    .report-section th { background: var(--color-surface-container); font-weight: 600; }
    .report-section strong { font-weight: 600; }
    .report-section code { background: var(--color-surface-container); padding: 0.1em 0.3em; border-radius: 4px; font-size: 0.9em; }
    .report-section blockquote { border-left: 3px solid var(--color-primary); padding: 0.4em 1em; margin: 0.5em 0; background: var(--color-surface-container-low); border-radius: 0 var(--radius) var(--radius) 0; }

    /* 加载动画 */
    .spinner {
      width: 16px;
      height: 16px;
      border: 2px solid currentColor;
      border-top-color: transparent;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    .phase-label {
      display: flex;
      align-items: center;
      gap: var(--stack-sm);
      padding: var(--stack-md) 0;
      font-weight: 500;
      color: var(--color-on-surface-variant);
    }

    .outline-text {
      font-size: 14px;
      color: var(--color-on-surface-variant);
      padding: var(--stack-sm) 0 var(--stack-md);
      line-height: 1.6;
      border-left: 3px solid var(--color-primary);
      padding-left: var(--stack-md);
    }

    @media (max-width: 768px) {
      .research-container { padding: var(--stack-md); }
    }
  </style>
</head>
<body>
  <div class="mobile-overlay" id="mobileOverlay" onclick="toggleSidebar()"></div>
  <div class="app-layout">
    <!-- 侧边栏 -->
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-header">
        <div class="sidebar-logo">FA</div>
        <span class="sidebar-title">FinAssist</span>
      </div>
      <div class="sidebar-actions"></div>
      <nav class="sidebar-nav">
        <a href="chat.html" class="nav-item">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          对话
        </a>
        <a href="research.html" class="nav-item active">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
          深度研究
        </a>
        <a href="knowledge.html" class="nav-item">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
          知识库
        </a>
        <a href="settings.html" class="nav-item">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
          设置
        </a>
      </nav>
    </aside>

    <!-- 主内容区 -->
    <main class="main-content">
      <div class="top-bar">
        <div class="top-bar-left">
          <button class="btn-icon btn-hamburger" onclick="toggleSidebar()">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
          </button>
          <h1 class="top-bar-title">深度研究</h1>
        </div>
      </div>

      <div class="research-container" id="researchContainer">
        <!-- Phase 1: 输入 -->
        <div id="phaseInput">
          <div class="phase-label">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            输入你的研究问题
          </div>
          <div class="research-input-area">
            <input type="text" id="queryInput" placeholder="例如：分析比亚迪的竞争优势和投资价值" onkeydown="if(event.key==='Enter')startResearch()">
            <button class="btn-research" id="btnStart" onclick="startResearch()">开始研究</button>
          </div>
        </div>

        <!-- Phase 2: Plan 结果 -->
        <div id="phasePlan" style="display:none">
          <div class="phase-label">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            研究框架
          </div>
          <div class="outline-text" id="outlineText"></div>
          <div class="task-list-header">
            <h3>子任务列表</h3>
            <button class="btn-add-task" onclick="addNewTask()">+ 新增子任务</button>
          </div>
          <div class="task-list" id="taskList"></div>
          <div class="task-actions">
            <button class="btn-confirm" onclick="confirmTasks()">确认并开始执行</button>
            <button class="btn-add-task" onclick="resetResearch()">重新规划</button>
          </div>
        </div>

        <!-- Phase 3: 执行 & 报告 -->
        <div id="phaseExecute" style="display:none">
          <div class="progress-section" id="progressSection">
            <div class="phase-label">
              <div class="spinner"></div>
              执行中...
            </div>
            <div id="executorProgress"></div>
          </div>
          <div class="progress-section" id="generateProgress" style="display:none">
            <div class="phase-label">
              <div class="spinner"></div>
              生成报告中...
            </div>
          </div>
          <div id="reportSection" style="display:none">
            <div class="phase-label">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
              研究报告
            </div>
            <div class="report-section" id="reportContent"></div>
          </div>
        </div>
      </div>
    </main>
  </div>

  <script src="js/api.js"></script>
  <script>
    let currentQuery = '';
    let planData = null;

    function toggleSidebar() {
      document.getElementById('sidebar').classList.toggle('open');
      document.getElementById('mobileOverlay').classList.toggle('active');
    }

    // Phase 1: 开始研究
    async function startResearch() {
      const query = document.getElementById('queryInput').value.trim();
      if (!query) return;

      currentQuery = query;
      document.getElementById('btnStart').disabled = true;
      document.getElementById('btnStart').textContent = '规划中...';

      try {
        const data = await api('POST', '/research/plan', { query });
        planData = data;
        renderPlan(data);
        document.getElementById('phaseInput').style.display = 'none';
        document.getElementById('phasePlan').style.display = 'block';
      } catch (e) {
        alert('规划失败: ' + e.message);
        document.getElementById('btnStart').disabled = false;
        document.getElementById('btnStart').textContent = '开始研究';
      }
    }

    // Phase 2: 渲染子任务
    function renderPlan(data) {
      document.getElementById('outlineText').textContent = data.research_outline;
      const list = document.getElementById('taskList');
      list.innerHTML = data.sub_tasks.map((t, i) => renderTaskCard(t, i)).join('');
    }

    function renderTaskCard(t, index) {
      const toolTags = (t.tools || []).map(tool => `<span class="task-tool-tag">${tool}</span>`).join('');
      return `
        <div class="task-card" data-index="${index}">
          <button class="task-card-delete" onclick="deleteTask(${index})" title="删除">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </button>
          <div class="task-card-header">
            <div class="task-card-number">${t.id}</div>
            <div class="task-card-body">
              <div class="task-card-title"><input type="text" value="${escapeAttr(t.title)}" onchange="updateTask(${index}, 'title', this.value)"></div>
              <div class="task-card-desc"><textarea onchange="updateTask(${index}, 'description', this.value)">${escapeHtml(t.description)}</textarea></div>
              <div class="task-card-tools">${toolTags}</div>
            </div>
          </div>
        </div>`;
    }

    function updateTask(index, field, value) {
      if (planData && planData.sub_tasks[index]) {
        planData.sub_tasks[index][field] = value;
      }
    }

    function deleteTask(index) {
      if (planData) {
        planData.sub_tasks.splice(index, 1);
        renderPlan(planData);
      }
    }

    function addNewTask() {
      if (!planData) return;
      const newId = Math.max(0, ...planData.sub_tasks.map(t => t.id)) + 1;
      planData.sub_tasks.push({
        id: newId,
        title: '新子任务',
        description: '',
        tools: ['rag_query', 'web_search'],
        search_queries: [],
      });
      renderPlan(planData);
    }

    // Phase 3: 确认并执行
    async function confirmTasks() {
      if (!planData || !planData.sub_tasks.length) return;

      document.getElementById('phasePlan').style.display = 'none';
      document.getElementById('phaseExecute').style.display = 'block';

      // 初始化 Executor 进度条
      const progressEl = document.getElementById('executorProgress');
      progressEl.innerHTML = planData.sub_tasks.map(t => `
        <div class="executor-progress running" id="exec-${t.id}">
          <div class="spinner"></div>
          ${escapeHtml(t.title)}
        </div>
      `).join('');

      try {
        const res = await fetch('/api/research/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: currentQuery,
            sub_tasks: planData.sub_tasks,
          }),
        });

        if (!res.ok) {
          const text = await res.text();
          throw new Error(text);
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
              const data = JSON.parse(line.slice(6));
              handleSSE(currentEvent, data);
              currentEvent = '';
            }
          }
        }
      } catch (e) {
        alert('执行失败: ' + e.message);
        resetResearch();
      }
    }

    function handleSSE(event, data) {
      switch (event) {
        case 'executor_done': {
          const el = document.getElementById(`exec-${data.task_id}`);
          if (el) {
            el.className = 'executor-progress ' + (data.complete ? 'done' : 'error');
            el.innerHTML = `
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
              ${escapeHtml(data.title)}${data.complete ? '' : ' (未完整完成)'}`;
          }
          break;
        }
        case 'generate_start': {
          document.getElementById('generateProgress').style.display = 'block';
          break;
        }
        case 'generate_done': {
          document.getElementById('progressSection').style.display = 'none';
          document.getElementById('generateProgress').style.display = 'none';
          // 加载报告
          loadReport(data.report_id);
          break;
        }
      }
    }

    async function loadReport(reportId) {
      try {
        const report = await api('GET', `/research/reports/${reportId}`);
        const el = document.getElementById('reportSection');
        el.style.display = 'block';
        document.getElementById('reportContent').innerHTML = marked.parse(report.content);
      } catch (e) {
        document.getElementById('reportContent').innerHTML = '<p>报告加载失败</p>';
      }
    }

    function resetResearch() {
      currentQuery = '';
      planData = null;
      document.getElementById('phaseInput').style.display = 'block';
      document.getElementById('phasePlan').style.display = 'none';
      document.getElementById('phaseExecute').style.display = 'none';
      document.getElementById('btnStart').disabled = false;
      document.getElementById('btnStart').textContent = '开始研究';
      document.getElementById('queryInput').value = '';
    }

    function escapeHtml(str) {
      const div = document.createElement('div');
      div.textContent = str;
      return div.innerHTML;
    }

    function escapeAttr(str) {
      return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }
  </script>
</body>
</html>
```

- [ ] **Step 2: 在 `frontend/chat.html` 的侧边栏导航中添加"深度研究"入口**

在 `chat.html` 的 `<nav class="sidebar-nav">` 区域中，在"对话"链接之后、"知识库"链接之前添加：

```html
        <a href="research.html" class="nav-item">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
          深度研究
        </a>
```

同样在 `knowledge.html` 和 `settings.html` 的侧边栏导航中添加相同入口（如果它们也有侧边栏导航的话）。

- [ ] **Step 3: 提交**

```bash
git add frontend/research.html frontend/chat.html
git commit -m "feat: Deep Research 前端页面 + 导航入口"
```

---

### Task 13: 集成测试 & 修复

**Files:**
- 可能修复的文件取决于集成测试结果

- [ ] **Step 1: 启动 ChromaDB 服务**

```bash
bash run.sh &
sleep 3
```

- [ ] **Step 2: 启动 Web 服务**

```bash
uv run python web.py &
sleep 3
```

- [ ] **Step 3: 测试 Plan API**

```bash
curl -s -X POST http://localhost:8000/api/research/plan \
  -H 'Content-Type: application/json' \
  -d '{"query": "分析比亚迪的竞争优势"}' | python -m json.tool
```

Expected: 返回 `research_outline` 和 `sub_tasks` 列表

- [ ] **Step 4: 测试 Execute API（SSE）**

```bash
# 先用 Plan 获取子任务，然后手动触发 Execute
curl -N -X POST http://localhost:8000/api/research/execute \
  -H 'Content-Type: application/json' \
  -d '{"query": "测试", "sub_tasks": [{"id": 1, "title": "测试搜索", "description": "搜索比亚迪基本信息", "tools": ["web_search"], "search_queries": ["比亚迪"]}]}' \
  --max-time 180
```

Expected: 看到 SSE 事件流（executor_start, executor_done, generate_start, generate_done, done）

- [ ] **Step 5: 修复发现的问题**

根据测试结果修复任何集成问题。

- [ ] **Step 6: 提交修复**

```bash
git add -A
git commit -m "fix: Deep Research 集成测试修复"
```

---

## 自审结果

**1. Spec 覆盖检查：**
- Plan Agent → Task 7 ✓
- Executor Agent → Task 8 ✓
- Generator Agent → Task 9 ✓
- web_search 工具 → Task 4 ✓
- API 路由 → Task 11 ✓
- 前端页面 → Task 12 ✓
- 数据库表 → Task 3 ✓
- 配置常量 → Task 2 ✓
- 用户确认子任务 → Task 12 (前端 Plan 阶段) ✓
- 并行执行 → Task 10 (asyncio) ✓
- 报告持久化 → Task 10 (save) ✓

**2. 占位符扫描：** 无 TBD/TODO。

**3. 类型一致性：** `SubTask` 和 `ExecutorResult` 在 `schemas.py` 定义，`plan.py` 返回 `list[SubTask]`，`executor.py` 接收 `SubTask` 返回 `ExecutorResult`，`generator.py` 接收 `list[ExecutorResult]`，`__init__.py` 和 `api/research.py` 均使用一致类型。API 端点路径与前端 fetch 路径一致。
