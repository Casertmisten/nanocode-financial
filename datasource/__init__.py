"""数据源模块 — 统一的 A 股数据与新闻接口。

使用方法：
    from datasource import stock, news

    # 股票数据
    quotes = stock.get_stock_quotes("600519")
    hist = stock.get_historical_data("600519", "2025-01-01", "2025-12-31")

    # 新闻
    market_news = news.collect_market_news()
    sentiment = news.analyze_sentiment("大涨", "贵州茅台涨停")

注册新数据源：
    from datasource import register_stock_provider, register_news_provider
    register_stock_provider("tushare", TushareStockProvider())
    register_news_provider("tushare", TushareNewsProvider())
"""

from datasource import news, stock
from datasource.base import NewsProvider, StockProvider

__all__ = ["stock", "news", "StockProvider", "NewsProvider",
           "register_stock_provider", "register_news_provider",
           "set_active_stock_provider", "set_active_news_provider"]

# ---- Provider 注册表 ----

_stock_providers: dict[str, StockProvider] = {}
_news_providers: dict[str, NewsProvider] = {}

_active_stock: str = ""
_active_news: str = ""

# 模块级 provider 对象，供 stock.py / news.py 委托调用
_stock_provider: StockProvider | None = None
_news_provider: NewsProvider | None = None


def register_stock_provider(name: str, provider: StockProvider):
    """注册股票数据源。"""
    global _stock_provider
    _stock_providers[name] = provider
    if not _active_stock:
        set_active_stock_provider(name)


def register_news_provider(name: str, provider: NewsProvider):
    """注册新闻数据源。"""
    global _news_provider
    _news_providers[name] = provider
    if not _active_news:
        set_active_news_provider(name)


def set_active_stock_provider(name: str):
    """切换当前使用的股票数据源。"""
    global _stock_provider, _active_stock
    if name not in _stock_providers:
        raise ValueError(f"未注册的股票数据源: {name}，可用: {list(_stock_providers.keys())}")
    _active_stock = name
    _stock_provider = _stock_providers[name]


def set_active_news_provider(name: str):
    """切换当前使用的新闻数据源。"""
    global _news_provider, _active_news
    if name not in _news_providers:
        raise ValueError(f"未注册的新闻数据源: {name}，可用: {list(_news_providers.keys())}")
    _active_news = name
    _news_provider = _news_providers[name]


# ---- 注册默认 AKShare 数据源 ----

def _register_defaults():
    from datasource._akshare_stock import AKShareStockProvider
    from datasource._akshare_news import AKShareNewsProvider
    register_stock_provider("akshare", AKShareStockProvider())
    register_news_provider("akshare", AKShareNewsProvider())

_register_defaults()
