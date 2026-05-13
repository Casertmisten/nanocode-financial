"""AKShare 股票数据实现。"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from datasource._helpers import (
    determine_market,
    get_full_symbol,
    get_market_info,
    safe_float,
    safe_int,
)

log = logging.getLogger(__name__)

_initialized = False
_ak = None

# 股票列表缓存
_stock_list_cache: Optional[pd.DataFrame] = None
_cache_time: Optional[datetime] = None
_CACHE_TTL = timedelta(hours=1)


def _ensure_initialized():
    """延迟初始化：导入 akshare 并修补请求头。"""
    global _initialized, _ak
    if _initialized:
        return
    try:
        import akshare as ak
        import requests

        try:
            from curl_cffi import requests as curl_requests

            use_curl = True
        except ImportError:
            use_curl = False

        if not hasattr(requests, "_akshare_headers_patched"):
            original_get = requests.get
            last_req = {"time": 0}

            def patched_get(url, **kwargs):
                is_em = "eastmoney.com" in url

                if is_em:
                    elapsed = time.time() - last_req["time"]
                    if elapsed < 0.5:
                        time.sleep(0.5 - elapsed)
                    last_req["time"] = time.time()

                if use_curl and is_em:
                    try:
                        kw = {
                            "timeout": kwargs.get("timeout", 10),
                            "impersonate": "chrome120",
                        }
                        for k in ("params", "data", "json"):
                            if k in kwargs:
                                kw[k] = kwargs[k]
                        return curl_requests.get(url, **kw)
                    except Exception:
                        pass

                if is_em:
                    em_headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                      "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
                        "Referer": "https://www.eastmoney.com/",
                    }
                    if "headers" not in kwargs or kwargs["headers"] is None:
                        kwargs["headers"] = em_headers
                    elif isinstance(kwargs["headers"], dict):
                        kwargs["headers"].setdefault("User-Agent", em_headers["User-Agent"])
                        kwargs["headers"].setdefault("Referer", em_headers["Referer"])

                for attempt in range(3):
                    try:
                        return original_get(url, **kwargs)
                    except Exception as e:
                        is_ssl = "SSL" in str(e) or "UNEXPECTED_EOF" in str(e)
                        if is_ssl and attempt < 2:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                        raise

            requests.get = patched_get
            requests._akshare_headers_patched = True

        _ak = ak
        _initialized = True
        log.info("AKShare 初始化成功")
    except ImportError:
        log.warning("akshare 未安装")
    except Exception as e:
        log.warning("AKShare 初始化失败: %s", e)


class AKShareStockProvider:
    """基于 AKShare 的股票数据实现。"""

    def is_available(self) -> bool:
        _ensure_initialized()
        return _ak is not None

    def get_stock_list(self) -> list[dict]:
        global _stock_list_cache, _cache_time
        _ensure_initialized()
        if _ak is None:
            return []
        if _stock_list_cache is not None and _cache_time and datetime.now() - _cache_time < _CACHE_TTL:
            return _stock_list_cache.to_dict("records")
        try:
            df = _ak.stock_info_a_code_name()
            if df is None or df.empty:
                return []
            _stock_list_cache = df
            _cache_time = datetime.now()
            return df.to_dict("records")
        except Exception as e:
            log.warning("获取股票列表失败: %s", e)
            return []

    def get_stock_basic_info(self, code: str) -> Optional[dict]:
        _ensure_initialized()
        if _ak is None:
            return None
        try:
            info_df = _ak.stock_individual_info_em(symbol=code)
            if info_df is not None and not info_df.empty:
                info = {"code": code}
                for item_name, key in [
                    ("股票简称", "name"), ("所属行业", "industry"),
                    ("所属地区", "area"), ("上市时间", "list_date"),
                ]:
                    row = info_df[info_df["item"] == item_name]
                    if not row.empty:
                        info[key] = str(row["value"].iloc[0])
                info.setdefault("name", f"股票{code}")
                info.setdefault("industry", "未知")
                info.setdefault("area", "未知")
                info["market"] = determine_market(code)
                info["full_symbol"] = get_full_symbol(code)
                info["market_info"] = get_market_info(code)
                return info
        except Exception as e:
            log.warning("获取 %s 基本信息失败: %s", code, e)
        for s in self.get_stock_list():
            if s.get("code") == code:
                return {
                    "code": code, "name": s.get("name", f"股票{code}"),
                    "industry": "未知", "area": "未知",
                    "market": determine_market(code),
                }
        return None

    def get_stock_quotes(self, code: str) -> Optional[dict]:
        _ensure_initialized()
        if _ak is None:
            return None
        # 级别 1: stock_bid_ask_em
        try:
            df = _ak.stock_bid_ask_em(symbol=code)
            if df is not None and not df.empty:
                d = dict(zip(df["item"], df["value"]))
                vol = int(safe_float(d.get("总手", 0))) * 100
                cn_now = datetime.now(timezone(timedelta(hours=8)))
                return _build_quotes(code, {
                    "name": f"股票{code}",
                    "price": safe_float(d.get("最新", 0)),
                    "change": safe_float(d.get("涨跌", 0)),
                    "change_percent": safe_float(d.get("涨幅", 0)),
                    "volume": vol,
                    "amount": safe_float(d.get("金额", 0)),
                    "open": safe_float(d.get("今开", 0)),
                    "high": safe_float(d.get("最高", 0)),
                    "low": safe_float(d.get("最低", 0)),
                    "pre_close": safe_float(d.get("昨收", 0)),
                    "turnover_rate": safe_float(d.get("换手", 0)),
                    "volume_ratio": safe_float(d.get("量比", 0)),
                    "quote_source": "stock_bid_ask_em",
                    "trade_date": cn_now.strftime("%Y-%m-%d"),
                    "updated_at": cn_now.isoformat(),
                })
        except Exception:
            pass
        raw = _get_realtime_fallback(code)
        if raw:
            return _build_quotes(code, raw)
        return None

    def get_batch_stock_quotes(self, codes: list[str]) -> dict[str, dict]:
        _ensure_initialized()
        if _ak is None:
            return {}
        for attempt in range(2):
            try:
                time.sleep(0.3)
                try:
                    spot_df = _ak.stock_zh_a_spot()
                except Exception:
                    time.sleep(0.5)
                    spot_df = _ak.stock_zh_a_spot_em()
                if spot_df is None or spot_df.empty:
                    continue
                codes_set = set(codes)
                result = {}
                for _, row in spot_df.iterrows():
                    rc = str(row.get("代码", ""))
                    if rc in codes_set:
                        result[rc] = _build_quotes(rc, _row_to_raw(row, rc, "spot"))
                return result
            except Exception as e:
                if attempt == 0:
                    time.sleep(1)
                else:
                    log.warning("批量获取行情失败: %s", e)
        return {}

    def get_historical_data(
        self, code: str, start_date: str, end_date: str, period: str = "daily"
    ) -> Optional[pd.DataFrame]:
        _ensure_initialized()
        if _ak is None:
            return None
        try:
            df = _ak.stock_zh_a_hist(
                symbol=code, period=period,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
            if df is None or df.empty:
                return None
            col_map = {
                "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                "最低": "low", "成交量": "volume", "成交额": "amount", "振幅": "amplitude",
                "涨跌幅": "change_percent", "涨跌额": "change", "换手率": "turnover",
            }
            df = df.rename(columns=col_map)
            df["code"] = code
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            for col in ("open", "close", "high", "low", "volume", "amount"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            return df
        except Exception as e:
            log.warning("获取 %s 历史数据失败: %s", code, e)
            return None

    def get_financial_data(self, code: str) -> dict:
        _ensure_initialized()
        if _ak is None:
            return {}
        result = {}
        fetchers = [
            ("main_indicators", lambda: _ak.stock_financial_abstract(symbol=code)),
            ("balance_sheet", lambda: _ak.stock_balance_sheet_by_report_em(symbol=code)),
            ("income_statement", lambda: _ak.stock_profit_sheet_by_report_em(symbol=code)),
            ("cash_flow", lambda: _ak.stock_cash_flow_sheet_by_report_em(symbol=code)),
        ]
        for name, fn in fetchers:
            try:
                df = fn()
                if df is not None and not df.empty:
                    result[name] = df.to_dict("records")
            except Exception as e:
                log.debug("获取 %s %s 失败: %s", code, name, e)
        return result

    def get_market_status(self) -> dict:
        now = datetime.now()
        is_trading = now.weekday() < 5 and (
            (9 <= now.hour < 12) or (13 <= now.hour < 15)
        )
        return {
            "market_status": "open" if is_trading else "closed",
            "current_time": now.isoformat(),
            "trading_day": now.weekday() < 5,
        }


# ---- 内部函数 ----


def _get_realtime_fallback(code: str) -> Optional[dict]:
    """全市场快照 → 历史日线的回退链。"""
    for fn_name, source in [
        (lambda: _ak.stock_zh_a_spot(), "stock_zh_a_spot"),
        (lambda: _ak.stock_zh_a_spot_em(), "stock_zh_a_spot_em"),
    ]:
        try:
            spot_df = fn_name()
            if spot_df is not None and not spot_df.empty:
                row_df = spot_df[spot_df["代码"] == code]
                if not row_df.empty:
                    return _row_to_raw(row_df.iloc[0], code, source)
        except Exception:
            pass
    # 历史日线兜底
    try:
        hist_df = _ak.stock_zh_a_hist(symbol=code, period="daily", adjust="")
        if hist_df is not None and not hist_df.empty:
            r = hist_df.iloc[-1]
            return {
                "name": f"股票{code}",
                "price": safe_float(r.get("收盘", 0)),
                "change": 0,
                "change_percent": safe_float(r.get("涨跌幅", 0)),
                "volume": safe_int(r.get("成交量", 0)),
                "amount": safe_float(r.get("成交额", 0)),
                "open": safe_float(r.get("开盘", 0)),
                "high": safe_float(r.get("最高", 0)),
                "low": safe_float(r.get("最低", 0)),
                "pre_close": safe_float(r.get("收盘", 0)),
                "quote_source": "stock_zh_a_hist",
            }
    except Exception:
        pass
    return None


def _row_to_raw(r, code: str, source: str) -> dict:
    return {
        "name": str(r.get("名称", f"股票{code}")),
        "price": safe_float(r.get("最新价", 0)),
        "change": safe_float(r.get("涨跌额", 0)),
        "change_percent": safe_float(r.get("涨跌幅", 0)),
        "volume": safe_int(r.get("成交量", 0)),
        "amount": safe_float(r.get("成交额", 0)),
        "open": safe_float(r.get("今开", 0)),
        "high": safe_float(r.get("最高", 0)),
        "low": safe_float(r.get("最低", 0)),
        "pre_close": safe_float(r.get("昨收", 0)),
        "turnover_rate": safe_float(r.get("换手率", None)),
        "volume_ratio": safe_float(r.get("量比", None)),
        "pe": safe_float(r.get("市盈率-动态", None)),
        "pb": safe_float(r.get("市净率", None)),
        "total_mv": safe_float(r.get("总市值", None)),
        "circ_mv": safe_float(r.get("流通市值", None)),
        "quote_source": source,
    }


def _build_quotes(code: str, raw: dict) -> dict:
    cn_now = datetime.now(timezone(timedelta(hours=8)))
    total_mv = raw.get("total_mv")
    circ_mv = raw.get("circ_mv")
    return {
        "code": code,
        "name": raw.get("name", f"股票{code}"),
        "price": safe_float(raw.get("price", 0)),
        "change": safe_float(raw.get("change", 0)),
        "change_percent": safe_float(raw.get("change_percent", 0)),
        "volume": safe_int(raw.get("volume", 0)),
        "amount": safe_float(raw.get("amount", 0)),
        "open": safe_float(raw.get("open", 0)),
        "high": safe_float(raw.get("high", 0)),
        "low": safe_float(raw.get("low", 0)),
        "pre_close": safe_float(raw.get("pre_close", 0)),
        "turnover_rate": raw.get("turnover_rate"),
        "volume_ratio": raw.get("volume_ratio"),
        "pe": raw.get("pe"),
        "pb": raw.get("pb"),
        "total_mv": total_mv / 1e8 if total_mv else None,
        "circ_mv": circ_mv / 1e8 if circ_mv else None,
        "trade_date": raw.get("trade_date", cn_now.strftime("%Y-%m-%d")),
        "updated_at": raw.get("updated_at", cn_now.isoformat()),
        "full_symbol": get_full_symbol(code),
        "market_info": get_market_info(code),
        "quote_source": raw.get("quote_source", "unknown"),
    }
