"""Generator Agent — 汇总子任务结果生成结构化报告。"""

import logging

import llm
from deep_research.prompts import generator_system_prompt
from deep_research.schemas import ExecutorResult

log = logging.getLogger(__name__)


def generate(query: str, results: list[ExecutorResult], usage_out: dict | None = None) -> str:
    """执行 Generator Agent，生成完整研究报告。"""
    log.info("Generator 开始: query=%s, 子任务数=%d", query[:50], len(results))

    results_text = ""
    all_sources: list[str] = []

    for i, r in enumerate(results, 1):
        status = "" if r.complete else "（未完整完成）"
        results_text += f"\n### 子任务 {i}：{r.title} {status}\n\n{r.summary}\n"
        if r.sources:
            all_sources.extend(r.sources)

    user_prompt = f"""用户研究问题：{query}

以下是各子任务的研究结果：

{results_text}

请根据以上研究结果，撰写一份完整的研究报告。"""

    report = llm.call_llm(generator_system_prompt, user_prompt, usage_out=usage_out)

    log.info("Generator 完成: 报告长度=%d", len(report))
    return report
