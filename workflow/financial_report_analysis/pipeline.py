"""Map-Reduce 流水线：检索 → 去重 → 维度分析（并行） → 汇总报告。"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import config
import rag
import llm as llm_mod
from prompts.financial_report_analysis import (
    analyze_prompt,
    analyze_system_prompt,
    reduce_prompt,
    reduce_system_prompt,
)
from workflow.financial_report_analysis.template import DIMENSIONS

log = logging.getLogger(__name__)

_TOP_K_PER_QUESTION = 3


async def _call_llm(system_prompt: str, user_content: str) -> str:
    """异步 LLM 调用。"""
    return await llm_mod.async_call_llm(system_prompt, user_content)


async def _retrieve_all():
    """Map 阶段：逐子问题检索，维度内去重。"""
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
                results = await asyncio.to_thread(rag.query, sq, top_k=_TOP_K_PER_QUESTION)
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


async def _analyze_dimension(name: str, chunks: list[str]) -> str:
    """Analyze 阶段：对一个维度生成分析小结。"""
    if not chunks:
        return f"【{name}】该维度缺乏足够数据，无法进行分析。"

    chunks_text = "\n\n---\n\n".join(chunks)
    prompt = analyze_prompt.format(dimension_name=name, chunks=chunks_text)
    return await _call_llm(analyze_system_prompt, prompt)


async def _reduce(query: str, summaries: dict[str, str], sources: set[str]) -> str:
    """Reduce 阶段：汇总各维度小结为完整报告。"""
    summaries_text = ""
    for dim in DIMENSIONS:
        summaries_text += f"\n## {dim['name']}\n{summaries[dim['name']]}\n"

    sources_text = "\n".join(f"· {s}" for s in sorted(sources))
    prompt = reduce_prompt.format(
        query=query, summaries=summaries_text, sources=sources_text
    )
    return await _call_llm(reduce_system_prompt, prompt)


async def run(query: str, progress_cb=None) -> str:
    """执行完整的财报分析流程（异步并行）。

    Args:
        query: 用户研究需求。
        progress_cb: 进度回调。

    Returns:
        Markdown 格式的完整报告。
    """
    # Map: 检索
    dim_data = await _retrieve_all()

    # Analyze: 各维度并行分析
    async def _analyze_one(dd, idx):
        print(f"  {config.DIM}⏺ 分析 [{idx}/{len(dim_data)}] {dd['name']}{config.RESET}")
        return dd["name"], await _analyze_dimension(dd["name"], dd["chunks"])

    results = await asyncio.gather(*[
        _analyze_one(dd, i) for i, dd in enumerate(dim_data, 1)
    ])
    summaries = {name: result for name, result in results}

    # 汇总来源
    all_sources = set()
    for dd in dim_data:
        all_sources.update(dd["sources"])

    # Reduce: 生成报告（流式输出）
    print(f"  {config.DIM}⏺ 生成报告...{config.RESET}")
    summaries_text = ""
    for dim in DIMENSIONS:
        summaries_text += f"\n## {dim['name']}\n{summaries[dim['name']]}\n"
    sources_text = "\n".join(f"· {s}" for s in sorted(all_sources))
    reduce_user = reduce_prompt.format(query=query, summaries=summaries_text, sources=sources_text)

    def _on_token(text):
        if progress_cb:
            progress_cb({"type": "token", "content": text})

    report = await llm_mod.async_stream_llm(reduce_system_prompt, reduce_user, on_token=_on_token)

    return report
