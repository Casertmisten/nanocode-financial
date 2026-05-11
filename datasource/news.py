"""新闻数据公共 API — 自动委托给注册的 NewsProvider。

新闻分析函数（analyze_sentiment 等）是纯函数，不依赖 provider。"""

# ---- 新闻采集 ----


def collect_market_news() -> list[dict]:
    """从多源聚合市场新闻。"""
    from datasource import _news_provider
    return _news_provider.collect_market_news()


def get_stock_news(symbol: str, limit: int = 10) -> list[dict]:
    """获取个股新闻。"""
    from datasource import _news_provider
    return _news_provider.get_stock_news(symbol, limit)


# ---- 新闻分析（纯函数，不依赖 provider）----


def analyze_sentiment(content: str, title: str) -> str:
    """情感分析。返回 'positive'、'negative' 或 'neutral'。"""
    text = f"{title} {content}"

    positive = [
        "利好", "上涨", "增长", "盈利", "突破", "创新高", "买入", "推荐",
        "看好", "乐观", "强势", "大涨", "飙升", "暴涨", "涨停", "涨幅",
        "业绩增长", "营收增长", "净利润增长", "扭亏为盈", "超预期",
        "获批", "中标", "签约", "合作", "并购", "重组", "分红", "回购",
    ]
    negative = [
        "利空", "下跌", "亏损", "风险", "暴跌", "卖出", "警告", "下调",
        "看空", "悲观", "弱势", "大跌", "跳水", "跌停", "跌幅",
        "业绩下滑", "营收下降", "净利润下降", "低于预期",
        "被查", "违规", "处罚", "诉讼", "退市", "停牌", "商誉减值",
    ]
    p = sum(1 for kw in positive if kw in text)
    n = sum(1 for kw in negative if kw in text)
    if p > n:
        return "positive"
    elif n > p:
        return "negative"
    return "neutral"


def calculate_sentiment_score(content: str, title: str) -> float:
    """情感评分，范围 -1.0 到 1.0。"""
    text = f"{title} {content}"
    positive = {
        "涨停": 1.0, "暴涨": 0.9, "大涨": 0.8, "飙升": 0.8,
        "创新高": 0.7, "突破": 0.6, "上涨": 0.5, "增长": 0.4,
        "利好": 0.6, "看好": 0.5, "推荐": 0.5, "买入": 0.6,
    }
    negative = {
        "跌停": -1.0, "暴跌": -0.9, "大跌": -0.8, "跳水": -0.8,
        "创新低": -0.7, "破位": -0.6, "下跌": -0.5, "下滑": -0.4,
        "利空": -0.6, "看空": -0.5, "卖出": -0.6, "警告": -0.5,
    }
    score = sum(w for kw, w in positive.items() if kw in text)
    score += sum(w for kw, w in negative.items() if kw in text)
    return max(-1.0, min(1.0, score / 3.0))


def extract_keywords(content: str, title: str) -> list[str]:
    """提取财经关键词，最多 10 个。"""
    text = f"{title} {content}"
    keywords = [
        "股票", "公司", "市场", "投资", "业绩", "财报", "政策", "行业",
        "分析", "预测", "涨停", "跌停", "上涨", "下跌", "盈利", "亏损",
        "并购", "重组", "分红", "回购", "增持", "减持", "融资", "IPO",
        "监管", "央行", "利率", "汇率", "GDP", "通胀", "经济", "贸易",
        "科技", "互联网", "新能源", "医药", "房地产", "金融", "制造业",
    ]
    return [kw for kw in keywords if kw in text][:10]


def assess_importance(content: str, title: str) -> str:
    """评估新闻重要性：high / medium / low。"""
    text = f"{title} {content}"
    high = [
        "业绩", "财报", "年报", "季报", "重大", "公告", "监管", "政策",
        "并购", "重组", "退市", "停牌", "涨停", "跌停", "暴涨", "暴跌",
        "央行", "证监会", "交易所", "违规", "处罚", "立案", "调查",
    ]
    medium = [
        "分析", "预测", "观点", "建议", "行业", "市场", "趋势", "机会",
        "研报", "评级", "目标价", "增持", "减持", "买入", "卖出",
        "合作", "签约", "中标", "获批", "分红", "回购",
    ]
    if any(kw in text for kw in high):
        return "high"
    if any(kw in text for kw in medium):
        return "medium"
    return "low"


def classify_news(content: str, title: str) -> str:
    """新闻分类。"""
    text = f"{title} {content}"
    if any(kw in text for kw in ["公告", "业绩", "财报", "年报", "季报"]):
        return "company_announcement"
    if any(kw in text for kw in ["政策", "监管", "央行", "证监会", "国务院"]):
        return "policy_news"
    if any(kw in text for kw in ["行业", "板块", "产业", "领域"]):
        return "industry_news"
    if any(kw in text for kw in ["市场", "指数", "大盘", "沪指", "深成指"]):
        return "market_news"
    if any(kw in text for kw in ["研报", "分析", "评级", "目标价", "机构"]):
        return "research_report"
    return "general"
