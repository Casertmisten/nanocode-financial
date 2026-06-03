"""个股投资决策流水线 — 异步并行版。

步骤 1(基本面)、2(估值)、3(新闻)、5(用户画像) 互不依赖，asyncio.gather 并行。
步骤 4(风险) 依赖 1+2+3，步骤 6(最终) 依赖全部，串行。
"""

import asyncio
import logging
from datetime import datetime, timedelta

import llm
import tools as tools_module
import memory.profile as profile_mod
from workflow.stock_investment.prompts import (
    FUNDAMENTAL_SYSTEM, FUNDAMENTAL_PROMPT,
    VALUATION_SYSTEM, VALUATION_PROMPT,
    NEWS_SYSTEM, NEWS_PROMPT,
    RISK_SYSTEM, RISK_PROMPT,
    FINAL_SYSTEM, FINAL_PROMPT,
    STEP_NAMES,
)

log = logging.getLogger(__name__)

# 工作流可调用的工具白名单
_ALLOWED_TOOLS = {
    "stock_list", "stock_basic_info", "stock_quotes", "batch_stock_quotes",
    "stock_historical", "stock_financial", "market_status",
    "market_news", "stock_news", "web_search",
}


async def _call_tool(name: str, args: dict) -> str:
    """异步安全调用工具。"""
    if name not in _ALLOWED_TOOLS:
        return f"工具 {name} 不在工作流白名单中"
    return str(await tools_module.run_tool_async(name, args))


async def _resolve_stock_code(query: str, entities: dict) -> str | None:
    """从实体或用户问题中解析股票代码。"""
    code = entities.get("stock_code")
    if code and len(code) == 6 and code.isdigit():
        return code

    import re
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", query)
    if match:
        return match.group(1)

    stock_name = entities.get("stock_name", "")
    if not stock_name:
        for keyword in ["分析", "评估", "看看", "了解", "研究"]:
            if keyword in query:
                parts = query.split(keyword)
                if len(parts) > 1:
                    candidate = parts[-1].strip().rstrip("的投资价值怎么样")
                    if candidate:
                        stock_name = candidate
                        break

    if stock_name:
        result = await _call_tool("stock_list", {"keyword": stock_name})
        for line in result.split("\n"):
            match = re.search(r"(\d{6})", line)
            if match and stock_name in line:
                return match.group(1)

    return None


async def run(query: str, entities: dict | None = None, progress_cb=None) -> str:
    """执行个股投资决策工作流（异步并行）。

    Args:
        query: 用户原始问题。
        entities: 意图识别提取的实体。
        progress_cb: 进度回调。

    Returns:
        Markdown 格式的投资建议报告。
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

    # 解析股票代码
    stock_code = await _resolve_stock_code(query, entities)
    if not stock_code:
        return "无法识别您提到的股票。请提供股票名称或代码（如 600519）。"

    stock_name = entities.get("stock_name", stock_code)

    # ── 并行组：步骤 1/2/3/5 ──

    async def _step_fundamental():
        _progress("fundamental", 1)
        log.info("个股投资决策 [1/%d] 公司基本面分析: %s", total_steps, stock_code)
        basic_info = await _tool("stock_basic_info", {"code": stock_code})
        financial_data = await _tool("stock_financial", {"code": stock_code})
        return await llm.async_call_llm(
            FUNDAMENTAL_SYSTEM,
            FUNDAMENTAL_PROMPT.format(
                query=query, basic_info=basic_info, financial_data=financial_data,
            ),
        )

    async def _step_valuation():
        _progress("valuation", 2)
        log.info("个股投资决策 [2/%d] 估值分析: %s", total_steps, stock_code)
        quotes = await _tool("stock_quotes", {"code": stock_code})
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        historical = await _tool("stock_historical", {
            "code": stock_code,
            "start_date": start_date.strftime("%Y%m%d"),
            "end_date": end_date.strftime("%Y%m%d"),
        })
        return await llm.async_call_llm(
            VALUATION_SYSTEM,
            VALUATION_PROMPT.format(
                query=query, quotes=quotes, historical=historical,
            ),
        )

    async def _step_news():
        _progress("news", 3)
        log.info("个股投资决策 [3/%d] 新闻事件分析: %s", total_steps, stock_code)
        stock_news = await _tool("stock_news", {"symbol": stock_code, "limit": 10})
        web_info = await _tool("web_search", {
            "query": f"{stock_name} 股票 最新消息 利好利空",
            "max_results": 5,
        })
        return await llm.async_call_llm(
            NEWS_SYSTEM,
            NEWS_PROMPT.format(
                query=query, stock_news=stock_news, web_info=web_info,
            ),
        )

    async def _step_profile():
        _progress("profile", 5)
        log.info("个股投资决策 [5/%d] 结合用户画像", total_steps)
        profile = profile_mod.load_profile()
        return profile_mod.render_profile_markdown(profile) or "未提供用户画像，请给出通用建议。"

    fundamental_analysis, valuation_analysis, news_analysis, user_profile = await asyncio.gather(
        _step_fundamental(),
        _step_valuation(),
        _step_news(),
        _step_profile(),
    )

    # ── 步骤 4：风险分析 ──
    _progress("risk", 4)
    log.info("个股投资决策 [4/%d] 风险分析: %s", total_steps, stock_code)

    risk_analysis = await llm.async_call_llm(
        RISK_SYSTEM,
        RISK_PROMPT.format(
            query=query,
            fundamental_analysis=fundamental_analysis,
            valuation_analysis=valuation_analysis,
            news_analysis=news_analysis,
        ),
    )

    # ── 步骤 6：生成最终投资建议（流式输出） ──
    _progress("final", 6)
    log.info("个股投资决策 [6/%d] 生成投资建议", total_steps)

    def _on_token(text):
        if progress_cb:
            progress_cb({"type": "token", "content": text})

    report = await llm.async_stream_llm(
        FINAL_SYSTEM,
        FINAL_PROMPT.format(
            query=query,
            user_profile=user_profile,
            fundamental_analysis=fundamental_analysis,
            valuation_analysis=valuation_analysis,
            news_analysis=news_analysis,
            risk_analysis=risk_analysis,
            stock_name=stock_name,
        ),
        on_token=_on_token,
    )

    return report
