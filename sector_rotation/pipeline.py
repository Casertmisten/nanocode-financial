"""行业轮动与机会发现流水线 — 6 步同步 pipeline。"""

import logging

import config
import llm
import tools as tools_module
from sector_rotation.prompts import (
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


def _call_tool(name: str, args: dict) -> str:
    """安全调用工具。"""
    if name not in _ALLOWED_TOOLS:
        return f"工具 {name} 不在工作流白名单中"
    return str(tools_module.run_tool(name, args))


def _call_llm(system: str, prompt: str) -> str:
    """LLM 调用封装。"""
    return llm.call_llm(system, prompt)


def run(query: str, entities: dict | None = None, progress_cb=None) -> str:
    """执行行业轮动与机会发现工作流。

    Args:
        query: 用户原始问题。
        entities: 意图识别提取的实体（此工作流一般无特定实体）。
        progress_cb: 进度回调函数，签名 (step_name: str, step_idx: int, total: int)。

    Returns:
        Markdown 格式的行业机会报告。
    """
    entities = entities or {}
    total_steps = len(STEP_NAMES)

    def _progress(step_key: str, idx: int):
        if progress_cb:
            progress_cb(STEP_NAMES[step_key], idx, total_steps)

    # ── 步骤 1：市场数据分析 ──
    _progress("market", 1)
    log.info("行业轮动 [1/%d] 市场数据分析", total_steps)

    market_status = _call_tool("market_status", {})
    # 获取主要指数行情
    index_codes = ",".join(_INDEX_CODES.values())
    index_quotes = _call_tool("batch_stock_quotes", {"codes": index_codes})

    market_analysis = _call_llm(
        MARKET_SYSTEM,
        MARKET_PROMPT.format(
            query=query, market_status=market_status, index_quotes=index_quotes,
        ),
    )

    # ── 步骤 2：行业热度分析 ──
    _progress("heat", 2)
    log.info("行业轮动 [2/%d] 行业热度分析", total_steps)

    # 获取一些代表性板块龙头股行情，辅助判断板块热度
    sector_leaders = "600519,000858,601318,600036,000001,600900,601012,300750"
    sector_data = _call_tool("batch_stock_quotes", {"codes": sector_leaders})

    heat_analysis = _call_llm(
        HEAT_SYSTEM,
        HEAT_PROMPT.format(
            query=query, market_analysis=market_analysis, sector_data=sector_data,
        ),
    )

    # ── 步骤 3：新闻聚合分析 ──
    _progress("news", 3)
    log.info("行业轮动 [3/%d] 新闻聚合", total_steps)

    market_news = _call_tool("market_news", {})
    web_info = _call_tool("web_search", {
        "query": "A股 行业板块 热点 机会 轮动",
        "max_results": 5,
    })

    news_analysis = _call_llm(
        NEWS_SYSTEM,
        NEWS_PROMPT.format(
            query=query, market_news=market_news, web_info=web_info,
        ),
    )

    # ── 步骤 4：资金流向分析 ──
    _progress("flow", 4)
    log.info("行业轮动 [4/%d] 资金流向分析", total_steps)

    # 获取几个行业龙头的财务数据辅助判断
    leader_data = _call_tool("stock_financial", {"code": "600519"})

    flow_analysis = _call_llm(
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

    ranking = _call_llm(
        RANK_SYSTEM,
        RANK_PROMPT.format(
            query=query,
            market_analysis=market_analysis,
            heat_analysis=heat_analysis,
            news_analysis=news_analysis,
            flow_analysis=flow_analysis,
        ),
    )

    # ── 步骤 6：生成机会报告 ──
    _progress("report", 6)
    log.info("行业轮动 [6/%d] 机会报告生成", total_steps)

    report = _call_llm(
        REPORT_SYSTEM,
        REPORT_PROMPT.format(
            query=query,
            market_analysis=market_analysis,
            heat_analysis=heat_analysis,
            news_analysis=news_analysis,
            flow_analysis=flow_analysis,
            ranking=ranking,
        ),
    )

    return report
