"""股票数据公共 API — 自动委托给注册的 StockProvider。"""

from typing import Optional

import pandas as pd

# 延迟导入避免循环依赖
_provider = None


def _get_provider():
    global _provider
    if _provider is None:
        from datasource import _stock_provider
        _provider = _stock_provider
    return _provider


def is_available() -> bool:
    return _get_provider().is_available()


def get_stock_list() -> list[dict]:
    """获取 A 股列表。"""
    return _get_provider().get_stock_list()


def get_stock_basic_info(code: str) -> Optional[dict]:
    """获取个股基本信息。"""
    return _get_provider().get_stock_basic_info(code)


def get_stock_quotes(code: str) -> Optional[dict]:
    """获取个股实时行情。"""
    return _get_provider().get_stock_quotes(code)


def get_batch_stock_quotes(codes: list[str]) -> dict[str, dict]:
    """批量获取实时行情。"""
    return _get_provider().get_batch_stock_quotes(codes)


def get_historical_data(
    code: str, start_date: str, end_date: str, period: str = "daily"
) -> Optional[pd.DataFrame]:
    """获取历史 K 线数据。"""
    return _get_provider().get_historical_data(code, start_date, end_date, period)


def get_financial_data(code: str) -> dict:
    """获取四大财务报表。"""
    return _get_provider().get_financial_data(code)


def get_market_status() -> dict:
    """获取市场状态。"""
    return _get_provider().get_market_status()
