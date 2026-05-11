"""内部工具函数：类型转换、市场判断等。"""

from datetime import datetime
from typing import Any, Optional

import pandas as pd


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value) or value is None:
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def safe_int(value: Any) -> int:
    try:
        if pd.isna(value) or value is None:
            return 0
        return int(float(value))
    except (ValueError, TypeError):
        return 0


def safe_str(value: Any) -> str:
    try:
        if pd.isna(value) or value is None:
            return ""
        return str(value)
    except Exception:
        return ""


def determine_market(code: str) -> str:
    """根据股票代码判断市场名称。"""
    if code.startswith(("60", "68")):
        return "上海证券交易所"
    elif code.startswith(("00", "30")):
        return "深圳证券交易所"
    elif code.startswith("8"):
        return "北京证券交易所"
    return "未知市场"


def get_full_symbol(code: str) -> str:
    """6位股票代码 → 标准代码（如 600000.SH）。"""
    if not code:
        return ""
    code = str(code).strip()
    if code.startswith(("60", "68", "90")):
        return f"{code}.SS"
    elif code.startswith(("00", "30", "20")):
        return f"{code}.SZ"
    elif code.startswith(("8", "4")):
        return f"{code}.BJ"
    return code


def get_market_info(code: str) -> dict:
    """根据股票代码返回市场元数据。"""
    if code.startswith(("60", "68")):
        exchange, name = "SSE", "上海证券交易所"
    elif code.startswith(("00", "30")):
        exchange, name = "SZSE", "深圳证券交易所"
    elif code.startswith("8"):
        exchange, name = "BSE", "北京证券交易所"
    else:
        exchange, name = "UNKNOWN", "未知交易所"
    return {
        "market_type": "CN",
        "exchange": exchange,
        "exchange_name": name,
        "currency": "CNY",
        "timezone": "Asia/Shanghai",
    }


def parse_news_time(time_str: str) -> Optional[str]:
    """解析新闻时间字符串，返回 ISO 格式。"""
    if not time_str:
        return None
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%m-%d %H:%M",
        "%m/%d %H:%M",
    ]
    for fmt in formats:
        try:
            t = datetime.strptime(str(time_str), fmt)
            if fmt in ("%m-%d %H:%M", "%m/%d %H:%M"):
                t = t.replace(year=datetime.now().year)
            return t.isoformat()
        except ValueError:
            continue
    return None
