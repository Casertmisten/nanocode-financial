# Deep Research 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 Map-Reduce 架构的财报深度分析功能，3 维度 33 子问题检索 + LLM 分析 + 结构化报告输出。

**Architecture:** 固定模板定义 3 个维度共 33 个子问题，逐子问题调用现有 RAG 管线检索，维度内去重后由 LLM 分析生成小结，最终汇总为完整报告。纯文本 LLM 调用，不使用 function calling。

**Tech Stack:** Python 3, urllib, 现有 rag 模块, config 模块

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `deep_research/template.py` | 3 维度 33 子问题的固定模板定义 |
| `deep_research/prompts.py` | Analyze 阶段和 Reduce 阶段的 prompt 模板 |
| `deep_research/pipeline.py` | Map-Reduce 流程编排（检索、去重、LLM 分析、汇总） |
| `deep_research/__init__.py` | 公共接口 `run(query)` |
| `nanocode.py` | 修改 `_deep_research_loop` 调用 `deep_research.run(query)` |

---

### Task 1: 创建维度模板

**Files:**
- Create: `deep_research/template.py`

- [ ] **Step 1: 创建 `deep_research/` 目录和 `template.py`**

```python
"""固定维度模板：3 个维度，33 个子问题。"""

DIMENSIONS = [
    {
        "name": "财务指标分析",
        "sub_questions": [
            "营业收入及同比变化",
            "营业成本及毛利率变化",
            "净利润及净利率趋势",
            "扣除非经常性损益净利润",
            "销售费用率",
            "管理费用率",
            "研发费用率",
            "ROE和ROA",
            "基本每股收益和稀释每股收益",
            "非经常性损益项目",
            "综合收益总额",
            "经营活动现金流净额",
            "经营现金流与净利润匹配度",
            "投资活动现金流及资本开支",
            "筹资活动现金流及融资情况",
        ],
        "web_search_queries": [],
    },
    {
        "name": "资产负债分析",
        "sub_questions": [
            "总资产规模及结构",
            "总负债规模及结构",
            "资产负债率变动趋势",
            "流动资产和非流动资产构成",
            "流动负债和非流动负债构成",
            "流动比率和速动比率",
            "应收账款及坏账准备",
            "存货及跌价准备",
            "商誉及减值",
            "股东权益及变动",
        ],
        "web_search_queries": [],
    },
    {
        "name": "行业与业务分析",
        "sub_questions": [
            "主营业务构成及收入结构",
            "各业务板块毛利率",
            "前五大客户和供应商",
            "行业发展趋势和市场规模",
            "竞争格局和主要竞争对手",
            "核心竞争力和技术优势",
            "在研项目和新业务布局",
            "主要风险因素",
        ],
        "web_search_queries": [],
    },
]
```

- [ ] **Step 2: 验证**

Run: `uv run python -c "from deep_research.template import DIMENSIONS; print(len(DIMENSIONS), sum(len(d['sub_questions']) for d in DIMENSIONS))"`
Expected: `3 33`

- [ ] **Step 3: 提交**

```bash
git add deep_research/template.py
git commit -m "feat(deep-research): add fixed dimension template with 33 sub-questions"
```

---

### Task 2: 创建 prompt 模板

**Files:**
- Create: `deep_research/prompts.py`

- [ ] **Step 1: 创建 `prompts.py`**

```python
"""Deep Research 各阶段的 prompt 模板。"""

ANALYZE_PROMPT = """你是专业金融分析师。根据以下检索到的财报数据，撰写【{dimension_name}】分析。

要求：
- 数据驱动，引用具体数字和同比变化
- 指出关键趋势和异常变动
- 不超过 800 字
- 如果检索数据中没有相关信息，明确说明"该维度缺乏足够数据"

检索数据：
{chunks}"""

REDUCE_PROMPT = """你是资深金融分析师。根据以下三个维度的分析小结，结合用户的研究需求，生成完整的深度财报分析报告。

报告格式要求：
- 使用以下章节结构：一、财务指标分析 / 二、资产负债分析 / 三、行业与业务分析 / 四、综合评价与风险提示 / 五、数据来源
- 各维度分析之间要有过渡和呼应
- 综合评价部分要给出明确的结论性判断
- 数据来源章节只列出文档名称，不列具体片段
- 使用 Markdown 格式，关键数字加粗

用户研究需求：{query}

{summaries}

数据来源：
{sources}"""
```

- [ ] **Step 2: 提交**

```bash
git add deep_research/prompts.py
git commit -m "feat(deep-research): add analyze and reduce prompt templates"
```

---

### Task 3: 创建 Map-Reduce 流水线

**Files:**
- Create: `deep_research/pipeline.py`

- [ ] **Step 1: 创建 `pipeline.py`**

```python
"""Map-Reduce 流水线：检索 → 去重 → 维度分析 → 汇总报告。"""

import json
import logging
import urllib.request

import config
import rag
from deep_research.prompts import ANALYZE_PROMPT, REDUCE_PROMPT
from deep_research.template import DIMENSIONS

log = logging.getLogger(__name__)

_TOP_K_PER_QUESTION = 3


def _call_llm(system_prompt: str, user_content: str) -> str:
    """纯文本 LLM 调用，不携带工具。"""
    request = urllib.request.Request(
        config.API_URL,
        data=json.dumps(
            {
                "model": config.MODEL,
                "max_tokens": 4096,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            }
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.API_KEY}",
        },
    )
    response = urllib.request.urlopen(request)
    result = json.loads(response.read())
    return result["choices"][0]["message"]["content"]


def _retrieve_all():
    """Map 阶段：逐子问题检索，维度内去重。

    Returns:
        list[dict]: 每个元素包含 name, chunks, sources。
    """
    dim_data = []
    total = sum(len(d["sub_questions"]) for d in DIMENSIONS)
    done = 0

    for dim in DIMENSIONS:
        seen_texts = set()
        chunks = []
        sources = set()

        for sq in dim["sub_questions"]:
            done += 1
            print(
                f"  {config.DIM}⏺ 检索 [{done}/{total}] {sq}{config.RESET}",
                end="\r",
                flush=True,
            )
            try:
                results = rag.query(sq, top_k=_TOP_K_PER_QUESTION)
            except Exception:
                log.warning("检索子问题失败: %s", sq, exc_info=True)
                results = []

            for r in results:
                text = r.get("text", "")
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    chunks.append(text)
                source = r.get("source", "")
                if source:
                    sources.add(source)

        print(f"  {config.DIM}⏺ {dim['name']}：检索到 {len(chunks)} 条去重结果" + " " * 20)
        dim_data.append(
            {"name": dim["name"], "chunks": chunks, "sources": sources}
        )

    return dim_data


def _analyze_dimension(name: str, chunks: list[str]) -> str:
    """Analyze 阶段：对一个维度生成分析小结。"""
    if not chunks:
        return f"【{name}】该维度缺乏足够数据，无法进行分析。"

    chunks_text = "\n\n---\n\n".join(chunks)
    prompt = ANALYZE_PROMPT.format(dimension_name=name, chunks=chunks_text)
    return _call_llm("你是一个专业的金融分析师。", prompt)


def _reduce(query: str, summaries: dict[str, str], sources: set[str]) -> str:
    """Reduce 阶段：汇总各维度小结为完整报告。"""
    summaries_text = ""
    for dim in DIMENSIONS:
        summaries_text += f"\n## {dim['name']}\n{summaries[dim['name']]}\n"

    sources_text = "\n".join(f"· {s}" for s in sorted(sources))
    prompt = REDUCE_PROMPT.format(
        query=query, summaries=summaries_text, sources=sources_text
    )
    return _call_llm("你是一个资深金融分析师。", prompt)


def run(query: str) -> str:
    """执行完整的 Deep Research 流程。

    Args:
        query: 用户研究需求。

    Returns:
        Markdown 格式的完整报告。
    """
    # Map: 检索
    dim_data = _retrieve_all()

    # Analyze: 逐维度分析
    summaries = {}
    for i, dd in enumerate(dim_data, 1):
        print(f"  {config.DIM}⏺ 分析 [{i}/{len(dim_data)}] {dd['name']}{config.RESET}")
        summaries[dd["name"]] = _analyze_dimension(dd["name"], dd["chunks"])

    # 汇总来源
    all_sources = set()
    for dd in dim_data:
        all_sources.update(dd["sources"])

    # Reduce: 生成报告
    print(f"  {config.DIM}⏺ 生成报告...{config.RESET}")
    report = _reduce(query, summaries, all_sources)

    return report
```

- [ ] **Step 2: 验证导入**

Run: `uv run python -c "from deep_research.pipeline import run; print('OK')"`
Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add deep_research/pipeline.py
git commit -m "feat(deep-research): add Map-Reduce pipeline with retrieval, analysis, and report generation"
```

---

### Task 4: 创建公共接口

**Files:**
- Create: `deep_research/__init__.py`

- [ ] **Step 1: 创建 `__init__.py`**

```python
"""Deep Research 模块 —— 财报深度分析。

Usage:
    from deep_research import run
    report = run("华邦健康2025年财报分析")
"""

from deep_research.pipeline import run

__all__ = ["run"]
```

- [ ] **Step 2: 提交**

```bash
git add deep_research/__init__.py
git commit -m "feat(deep-research): add public interface"
```

---

### Task 5: 连接主程序

**Files:**
- Modify: `nanocode.py:98-102`

- [ ] **Step 1: 修改 `_deep_research_loop`**

将 `nanocode.py` 中的 `_deep_research_loop` 从占位实现改为调用 `deep_research.run()`：

```python
def _deep_research_loop(query: str):
    """深度研究循环。"""
    import deep_research

    print(f"\n{config.YELLOW}⏺ 进入深度研究模式{config.RESET}")
    print(f"{config.DIM}  研究课题: {query}{config.RESET}\n")

    report = deep_research.run(query)

    print(f"\n{separator()}")
    print(render_markdown(report))
    print(separator())
```

- [ ] **Step 2: 手动验证**

Run: `uv run nanocode.py`，然后输入：
```
/deep_research 华邦健康2025年财报分析
```

验证：
1. 终端逐行显示检索进度 `[1/33]`、`[2/33]`...
2. 显示维度分析进度
3. 最终输出完整 Markdown 报告，包含 5 个章节

- [ ] **Step 3: 提交**

```bash
git add nanocode.py
git commit -m "feat: connect deep research to main REPL via /deep_research command"
```

---

### Task 6: 端到端验证

- [ ] **Step 1: 确认 RAG 服务可用**

Run: `uv run python -c "import rag; print(len(rag.query('华邦健康营收')))"` 
Expected: 输出一个非零数字（检索结果条数）

- [ ] **Step 2: 运行完整 Deep Research**

Run: `uv run nanocode.py`，输入：
```
/deep_research 华邦健康2025年财报分析
```

检查：
- 33 个子问题全部检索完成
- 3 个维度分析完成
- 报告包含：财务指标分析、资产负债分析、行业与业务分析、综合评价与风险提示、数据来源
- 数据来源只列文档名

- [ ] **Step 3: 最终提交**

如有修复，提交所有变更。
