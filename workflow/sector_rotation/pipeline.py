"""行业轮动与机会发现流水线 — 异步并行版。

步骤 1(市场) 和 3(新闻) 并行 → 步骤 2(依赖1) → 步骤 4(依赖1+2+3) → 5/6 串行。
"""

import asyncio
import logging

import llm
import tools as tools_module
from workflow.sector_rotation.prompts import (
    MARKET_SYSTEM, MARKET_PROMPT,
    HEAT_SYSTEM, HEAT_PROMPT,
    NEWS_SYSTEM, NEWS_PROMPT,
    FLOW_SYSTEM, FLOW_PROMPT,
    RANK_SYSTEM, RANK_PROMPT,
    REPORT_SYSTEM, REPORT_PROMPT,
    STEP_NAMES,
)

log = logging.getLogger(__name__)

# 工具白名单
_ALLOWED_TOOLS = {
    "stock_list", "stock_basic_info", "stock_quotes", "batch_stock_quotes",
    "stock_historical", "stock_financial", "market_status",
    "market_news", "stock_news", "web_search",
}

# 主要指数代码（用于市场概览）
_INDEX_CODES = {
    "上证指数": "000001",
    "深证成指": "399001",
    "创业板指": "399006",
    "沪深300": "000300",
}


async def _call_tool(name: str, args: dict) -> str:
    """异步安全调用工具。"""
    if name not in _ALLOWED_TOOLS:
        return f"工具 {name} 不在工作流白名单中"
    return str(await tools_module.run_tool_async(name, args))


async def run(query: str, entities: dict | None = None, progress_cb=None) -> str:
    """执行行业轮动与机会发现工作流（异步并行）。

    Args:
        query: 用户原始问题。
        entities: 意图识别提取的实体。
        progress_cb: 进度回调。

    Returns:
        Markdown 格式的行业机会报告。
    """
    entities = entities or {}
    total_steps = len(STEP_NAMES)

    def _progress(step_key: str, idx: int):
        if progress_cb:
            progress_cb(STEP_NAMES[step_key], idx, total_steps)

    async def _tool(name: str, args: dict) -> str:
        if progress_cb:
            progress_cb({"type": "tool_start", "tool": name, "args": args})
        result = await _call_tool(name, args)
        if progress_cb:
            progress_cb({"type": "tool_end", "tool": name})
        return result

    # ── 并行组：步骤 1(市场) + 3(新闻) ──

    async def _step_market():
        _progress("market", 1)
        log.info("行业轮动 [1/%d] 市场数据分析", total_steps)
        market_status = await _tool("market_status", {})
        index_codes = ",".join(_INDEX_CODES.values())
        index_quotes = await _tool("batch_stock_quotes", {"codes": index_codes})
        return await llm.async_call_llm(
            MARKET_SYSTEM,
            MARKET_PROMPT.format(
                query=query, market_status=market_status, index_quotes=index_quotes,
            ),
        )

    async def _step_news():
        _progress("news", 3)
        log.info("行业轮动 [3/%d] 新闻聚合", total_steps)
        market_news = await _tool("market_news", {})
        web_info = await _tool("web_search", {
            "query": "A股 行业板块 热点 机会 轮动",
            "max_results": 5,
        })
        return await llm.async_call_llm(
            NEWS_SYSTEM,
            NEWS_PROMPT.format(
                query=query, market_news=market_news, web_info=web_info,
            ),
        )

    market_analysis, news_analysis = await asyncio.gather(
        _step_market(),
        _step_news(),
    )

    # ── 步骤 2：行业热度分析（依赖步骤 1） ──
    _progress("heat", 2)
    log.info("行业轮动 [2/%d] 行业热度分析", total_steps)

    sector_leaders = "600519,000858,601318,600036,000001,600900,601012,300750"
    sector_data = await _tool("batch_stock_quotes", {"codes": sector_leaders})

    heat_analysis = await llm.async_call_llm(
        HEAT_SYSTEM,
        HEAT_PROMPT.format(
            query=query, market_analysis=market_analysis, sector_data=sector_data,
        ),
    )

    # ── 步骤 4：资金流向分析 ──
    _progress("flow", 4)
    log.info("行业轮动 [4/%d] 资金流向分析", total_steps)

    leader_data = await _tool("stock_financial", {"code": "600519"})

    flow_analysis = await llm.async_call_llm(
        FLOW_SYSTEM,
        FLOW_PROMPT.format(
            query=query,
            market_analysis=market_analysis,
            heat_analysis=heat_analysis,
            news_analysis=news_analysis,
            leader_data=leader_data,
        ),
    )

    # ── 步骤 5：候选行业排序 ──
    _progress("rank", 5)
    log.info("行业轮动 [5/%d] 候选行业排序", total_steps)

    ranking = await llm.async_call_llm(
        RANK_SYSTEM,
        RANK_PROMPT.format(
            query=query,
            market_analysis=market_analysis,
            heat_analysis=heat_analysis,
            news_analysis=news_analysis,
            flow_analysis=flow_analysis,
        ),
    )

    # ── 步骤 6：生成机会报告（流式输出） ──
    _progress("report", 6)
    log.info("行业轮动 [6/%d] 机会报告生成", total_steps)

    def _on_token(text):
        if progress_cb:
            progress_cb({"type": "token", "content": text})

    report = await llm.async_stream_llm(
        REPORT_SYSTEM,
        REPORT_PROMPT.format(
            query=query,
            market_analysis=market_analysis,
            heat_analysis=heat_analysis,
            news_analysis=news_analysis,
            flow_analysis=flow_analysis,
            ranking=ranking,
        ),
        on_token=_on_token,
    )

    return report
