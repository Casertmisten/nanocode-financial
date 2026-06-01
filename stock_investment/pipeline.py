"""个股投资决策流水线 — 6 步同步 pipeline。"""

import json
import logging

import config
import llm
import tools as tools_module
import memory.profile as profile_mod
from stock_investment.prompts import (
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


def _call_tool(name: str, args: dict) -> str:
    """安全调用工具，限制为白名单内的数据工具。"""
    if name not in _ALLOWED_TOOLS:
        return f"工具 {name} 不在工作流白名单中"
    return str(tools_module.run_tool(name, args))


def _resolve_stock_code(query: str, entities: dict) -> str | None:
    """从实体或用户问题中解析股票代码。如果已有 code 直接用，否则通过 stock_list 查询。"""
    code = entities.get("stock_code")
    if code and len(code) == 6 and code.isdigit():
        return code

    # 尝试从问题中提取6位数字
    import re
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", query)
    if match:
        return match.group(1)

    # 没有 code，尝试从 stock_name 查询
    stock_name = entities.get("stock_name", "")
    if not stock_name:
        # 尝试从问题中提取股票名称（启发式）
        for keyword in ["分析", "评估", "看看", "了解", "研究"]:
            if keyword in query:
                parts = query.split(keyword)
                if len(parts) > 1:
                    candidate = parts[-1].strip().rstrip("的投资价值怎么样")
                    if candidate:
                        stock_name = candidate
                        break

    if stock_name:
        result = _call_tool("stock_list", {"keyword": stock_name})
        # stock_list 返回的可能是 DataFrame 字符串，尝试解析代码
        for line in result.split("\n"):
            # 格式通常是 "代码  名称"，找6位数字
            match = re.search(r"(\d{6})", line)
            if match and stock_name in line:
                return match.group(1)

    return None


def _call_llm(system: str, prompt: str) -> str:
    """LLM 调用封装。"""
    return llm.call_llm(system, prompt)


def run(query: str, entities: dict | None = None, progress_cb=None) -> str:
    """执行个股投资决策工作流。

    Args:
        query: 用户原始问题。
        entities: 意图识别提取的实体（stock_code, stock_name 等）。
        progress_cb: 进度回调函数，签名 (step_name: str, step_idx: int, total: int)。

    Returns:
        Markdown 格式的投资建议报告。
    """
    entities = entities or {}
    total_steps = len(STEP_NAMES)

    def _progress(step_key: str, idx: int):
        if progress_cb:
            progress_cb(STEP_NAMES[step_key], idx, total_steps)

    # 解析股票代码
    stock_code = _resolve_stock_code(query, entities)
    if not stock_code:
        return "无法识别您提到的股票。请提供股票名称或代码（如 600519）。"

    # 获取股票基本信息用于报告标题
    stock_name = entities.get("stock_name", stock_code)

    # ── 步骤 1：公司基本面分析 ──
    _progress("fundamental", 1)
    log.info("个股投资决策 [1/%d] 公司基本面分析: %s", total_steps, stock_code)

    basic_info = _call_tool("stock_basic_info", {"code": stock_code})
    financial_data = _call_tool("stock_financial", {"code": stock_code})

    fundamental_analysis = _call_llm(
        FUNDAMENTAL_SYSTEM,
        FUNDAMENTAL_PROMPT.format(
            query=query, basic_info=basic_info, financial_data=financial_data,
        ),
    )

    # ── 步骤 2：估值分析 ──
    _progress("valuation", 2)
    log.info("个股投资决策 [2/%d] 估值分析: %s", total_steps, stock_code)

    quotes = _call_tool("stock_quotes", {"code": stock_code})

    # 获取近3个月历史数据
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    historical = _call_tool("stock_historical", {
        "code": stock_code,
        "start_date": start_date.strftime("%Y%m%d"),
        "end_date": end_date.strftime("%Y%m%d"),
    })

    valuation_analysis = _call_llm(
        VALUATION_SYSTEM,
        VALUATION_PROMPT.format(
            query=query, quotes=quotes, historical=historical,
        ),
    )

    # ── 步骤 3：新闻事件分析 ──
    _progress("news", 3)
    log.info("个股投资决策 [3/%d] 新闻事件分析: %s", total_steps, stock_code)

    stock_news = _call_tool("stock_news", {"symbol": stock_code, "limit": 10})
    web_info = _call_tool("web_search", {
        "query": f"{stock_name} 股票 最新消息 利好利空",
        "max_results": 5,
    })

    news_analysis = _call_llm(
        NEWS_SYSTEM,
        NEWS_PROMPT.format(
            query=query, stock_news=stock_news, web_info=web_info,
        ),
    )

    # ── 步骤 4：风险分析 ──
    _progress("risk", 4)
    log.info("个股投资决策 [4/%d] 风险分析: %s", total_steps, stock_code)

    risk_analysis = _call_llm(
        RISK_SYSTEM,
        RISK_PROMPT.format(
            query=query,
            fundamental_analysis=fundamental_analysis,
            valuation_analysis=valuation_analysis,
            news_analysis=news_analysis,
        ),
    )

    # ── 步骤 5：结合用户画像 ──
    _progress("profile", 5)
    log.info("个股投资决策 [5/%d] 结合用户画像", total_steps)

    profile = profile_mod.load_profile()
    user_profile = profile_mod.render_profile_markdown(profile) or "未提供用户画像，请给出通用建议。"

    # ── 步骤 6：生成最终投资建议 ──
    _progress("final", 6)
    log.info("个股投资决策 [6/%d] 生成投资建议", total_steps)

    report = _call_llm(
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
    )

    return report
