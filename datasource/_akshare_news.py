"""AKShare 新闻数据实现。"""

import logging
import time
from typing import Optional

from datasource._helpers import parse_news_time

log = logging.getLogger(__name__)


def _get_ak():
    """获取 akshare 实例（复用 stock 模块的初始化）。"""
    from datasource._akshare_stock import _ensure_initialized, _ak
    _ensure_initialized()
    return _ak


class AKShareNewsProvider:
    """基于 AKShare 的新闻数据实现。"""

    def collect_market_news(self) -> list[dict]:
        """从 6 个来源聚合市场新闻。"""
        ak = _get_ak()
        if ak is None:
            return []

        all_news: list[dict] = []
        sources = [
            ("财经早餐-东财", lambda: ak.stock_info_cjzc_em()),
            ("全球财经快讯-东财", lambda: ak.stock_info_global_em()),
            ("全球财经快讯-新浪", lambda: ak.stock_info_global_sina()),
            ("快讯-富途", lambda: ak.stock_info_global_futu()),
            ("全球财经直播-同花顺", lambda: ak.stock_info_global_ths()),
            ("电报-财联社", lambda: ak.stock_info_global_cls()),
        ]
        for name, fn in sources:
            try:
                df = fn()
                if df is not None and not df.empty:
                    for item in df.to_dict("records"):
                        all_news.append(_normalize(item, name))
                    log.debug("%s: %d 条", name, len(df))
            except Exception as e:
                log.debug("%s 采集失败: %s", name, e)

        if not all_news:
            log.warning("所有新闻源采集失败")
        return all_news

    def get_stock_news(self, symbol: str, limit: int = 10) -> list[dict]:
        """获取个股新闻（东方财富）。"""
        ak = _get_ak()
        if ak is None:
            return []

        symbol = symbol.zfill(6)
        news_df = None

        for attempt in range(3):
            try:
                news_df = ak.stock_news_em(symbol=symbol)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
                else:
                    log.warning("获取 %s 新闻失败: %s", symbol, e)
                    return []

        if news_df is None or news_df.empty:
            return []

        results = []
        for _, row in news_df.head(limit).iterrows():
            title = str(row.get("新闻标题", "") or row.get("标题", ""))
            content = str(row.get("新闻内容", "") or row.get("内容", ""))
            if not title:
                continue
            results.append({
                "symbol": symbol,
                "title": title,
                "content": content,
                "summary": str(row.get("新闻摘要", "") or row.get("摘要", "")),
                "url": str(row.get("新闻链接", "") or row.get("链接", "")),
                "source": str(row.get("文章来源", "") or row.get("来源", "") or "东方财富"),
                "publish_time": parse_news_time(
                    row.get("发布时间", "") or row.get("时间", "")
                ),
            })
        return results


# ---- 内部工具 ----

_FIELD_MAP = {
    "标题": "title", "摘要": "content", "发布时间": "time",
    "时间": "time", "链接": "link", "内容": "content",
}


def _normalize(item: dict, source: str) -> dict:
    normalized = {}
    for key, value in item.items():
        normalized[_FIELD_MAP.get(key, key)] = value
    normalized["source"] = source
    return normalized
