"""Tool implementations and registry for the agentic loop."""

import glob as globlib
import json
import os
import re
import subprocess

from utils import BaseLogger

log = BaseLogger.getLogger("tools")


# --- Original tool implementations ---


def read(args):
    lines = open(args["path"]).readlines()
    offset = args.get("offset", 0)
    limit = args.get("limit", len(lines))
    selected = lines[offset : offset + limit]
    return "".join(f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected))


def write(args):
    with open(args["path"], "w") as f:
        f.write(args["content"])
    return "ok"


def edit(args):
    text = open(args["path"]).read()
    old, new = args["old"], args["new"]
    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all=true)"
    replacement = text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
    with open(args["path"], "w") as f:
        f.write(replacement)
    return "ok"


def glob(args):
    pattern = (args.get("path", ".") + "/" + args["pat"]).replace("//", "/")
    files = globlib.glob(pattern, recursive=True)
    files = sorted(
        files,
        key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0,
        reverse=True,
    )
    return "\n".join(files) or "none"


def grep(args):
    pattern = re.compile(args["pat"])
    hits = []
    for filepath in globlib.glob(args.get("path", ".") + "/**", recursive=True):
        try:
            for line_num, line in enumerate(open(filepath), 1):
                if pattern.search(line):
                    hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
        except Exception:
            pass
    return "\n".join(hits[:50]) or "none"


def bash(args):
    proc = subprocess.Popen(
        args["cmd"],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_lines = []
    try:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                print(f"  \033[2m│ {line.rstrip()}\033[0m", flush=True)
                output_lines.append(line)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        output_lines.append("\n(timed out after 30s)")
    return "".join(output_lines).strip() or "(empty)"


# --- RAG tool implementations ---


def rag_query_tool(args):
    """RAG 查询工具，用于查询知识库中的文档。仅在调用此工具时才触发查询改写。"""
    import rag as rag_module

    question = args["question"]
    top_k = args.get("top_k", 5)
    file_paths = args.get("file_paths")
    log.info("RAG 查询: question=%s, top_k=%d, file_paths=%s", question[:50], top_k, len(file_paths) if file_paths else "全部")

    # 查询改写：仅 RAG 工具调用时执行
    queries = [question]
    try:
        import llm
        variants = llm.rewrite_queries(question)
        queries.extend(variants)
        log.info("查询改写完成: %d 路并行（原问题 + %d 变体）", len(queries), len(queries) - 1)
    except Exception:
        log.warning("查询改写失败，使用原始问题", exc_info=True)

    return rag_module.query_formatted(question, top_k, file_paths=file_paths, queries=queries)


def rag_ingest_tool(args):
    """RAG 导入工具，用于将文档导入知识库。"""
    import rag as rag_module

    doc_path = args.get("path") or None
    log.info("RAG 导入: path=%s", doc_path)
    count = rag_module.ingest(doc_path)
    if count == 0:
        return "没有新文档需要导入（所有文档已在知识库中）。"
    return f"成功导入 {count} 个新文档到知识库。"


# --- 数据源工具实现 ---


def _fmt(data) -> str:
    """统一将数据源结果序列化为字符串。"""
    import pandas as pd

    if data is None:
        return "未获取到数据。"
    if isinstance(data, pd.DataFrame):
        return data.to_string()
    return json.dumps(data, ensure_ascii=False, default=str)


def stock_list_tool(args):
    import db
    keyword = args.get("keyword", "")
    try:
        stocks = db.get_cached_stock_list(keyword)
    except Exception:
        stocks = []
    if not stocks:
        return "本地缓存为空，股票列表尚未同步。"
    return _fmt(stocks)


def stock_basic_info_tool(args):
    from datasource import stock
    return _fmt(stock.get_stock_basic_info(args["code"]))


def stock_quotes_tool(args):
    from datasource import stock
    return _fmt(stock.get_stock_quotes(args["code"]))


def batch_stock_quotes_tool(args):
    from datasource import stock
    codes = [c.strip() for c in args["codes"].split(",") if c.strip()]
    return _fmt(stock.get_batch_stock_quotes(codes))


def stock_historical_tool(args):
    from datasource import stock
    return _fmt(stock.get_historical_data(
        args["code"], args["start_date"], args["end_date"],
        args.get("period", "daily"),
    ))


def stock_financial_tool(args):
    from datasource import stock
    return _fmt(stock.get_financial_data(args["code"]))


def market_status_tool(args):
    from datasource import stock
    return _fmt(stock.get_market_status())


def market_news_tool(args):
    from datasource import news
    return _fmt(news.collect_market_news())


def stock_news_tool(args):
    from datasource import news
    return _fmt(news.get_stock_news(args["symbol"], args.get("limit", 10)))


# --- Tool registry ---
# (description, param_schema, function)

TOOLS = {
    "read": (
        "Read file with line numbers (file path, not directory)",
        {"path": "string", "offset": "number?", "limit": "number?"},
        read,
    ),
    "write": (
        "Write content to file",
        {"path": "string", "content": "string"},
        write,
    ),
    "edit": (
        "Replace old with new in file (old must be unique unless all=true)",
        {"path": "string", "old": "string", "new": "string", "all": "boolean?"},
        edit,
    ),
    "glob": (
        "Find files by pattern, sorted by mtime",
        {"pat": "string", "path": "string?"},
        glob,
    ),
    "grep": (
        "Search files for regex pattern",
        {"pat": "string", "path": "string?"},
        grep,
    ),
    "bash": (
        "Run shell command",
        {"cmd": "string"},
        bash,
    ),
    "rag_query": (
        "搜索本地金融知识库（研报、新闻、分析文档）。当用户询问金融、股票、市场相关问题时使用此工具获取相关信息。",
        {"question": "string", "top_k": "number?"},
        rag_query_tool,
    ),
    "rag_ingest": (
        "将文档导入本地金融知识库。支持 PDF 和 Markdown 文件。当用户要求导入或更新文档时使用。",
        {"path": "string"},
        rag_ingest_tool,
    ),
    # --- 股票数据工具 ---
    "stock_list": (
        "查询A股股票列表，支持按代码或名称关键词筛选。不传keyword返回前50条。",
        {"keyword": "string?"},
        stock_list_tool,
    ),
    "stock_basic_info": (
        "获取个股基本信息（公司名称、行业、市值等）。参数code为股票代码，如600000。",
        {"code": "string"},
        stock_basic_info_tool,
    ),
    "stock_quotes": (
        "获取个股实时行情（最新价、涨跌幅、成交量等）。参数code为股票代码。",
        {"code": "string"},
        stock_quotes_tool,
    ),
    "batch_stock_quotes": (
        "批量获取多只股票实时行情。参数codes为逗号分隔的股票代码，如'600000,000001'。",
        {"codes": "string"},
        batch_stock_quotes_tool,
    ),
    "stock_historical": (
        "获取个股历史K线数据。参数：code(股票代码)、start_date(起始日期YYYYMMDD)、end_date(结束日期YYYYMMDD)、period(周期: daily/weekly/monthly，可选)。",
        {"code": "string", "start_date": "string", "end_date": "string", "period": "string?"},
        stock_historical_tool,
    ),
    "stock_financial": (
        "获取个股财务数据（利润表、资产负债表、现金流量表等）。参数code为股票代码。",
        {"code": "string"},
        stock_financial_tool,
    ),
    "market_status": (
        "获取当前A股市场状态（开/闭市、交易时段等）。",
        {},
        market_status_tool,
    ),
    # --- 新闻数据工具 ---
    "market_news": (
        "聚合采集最新A股市场新闻，来源于东方财富、新浪等多个财经网站。",
        {},
        market_news_tool,
    ),
    "stock_news": (
        "获取个股相关新闻。参数symbol为股票代码，limit为返回条数(可选，默认10)。",
        {"symbol": "string", "limit": "number?"},
        stock_news_tool,
    ),
}


def run_tool(name, args):
    """Execute a tool by name with given arguments."""
    try:
        result = TOOLS[name][2](args)
        log.info("工具执行成功: %s", name)
        return result
    except Exception as err:
        log.error("工具执行失败: %s, error=%s", name, err, exc_info=True)
        return f"error: {err}"


def make_schema():
    """Convert TOOLS registry into OpenAI function calling format."""
    result = []
    for name, (description, params, _fn) in TOOLS.items():
        properties = {}
        required = []
        for param_name, param_type in params.items():
            is_optional = param_type.endswith("?")
            base_type = param_type.rstrip("?")
            properties[param_name] = {
                "type": "integer" if base_type == "number" else base_type
            }
            if not is_optional:
                required.append(param_name)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return result
