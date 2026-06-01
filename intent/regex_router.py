"""正则路由层 — 通过关键词模式匹配用户意图。"""

import re

from intent.schemas import IntentResult

# 各意图的正则模式列表，按 (pattern, entity_extractor) 组织
# entity_extractor 是一个函数，接收 match 对象，返回 entities dict

_STOCK_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def _extract_stock_entities(text: str, match: re.Match) -> dict:
    """从文本中提取股票代码。"""
    entities = {}
    code_match = _STOCK_CODE_RE.search(text)
    if code_match:
        entities["stock_code"] = code_match.group(1)
    return entities


# 意图 -> [(正则模式, 提取函数), ...]
_PATTERNS: dict[str, list[tuple[re.Pattern, object]]] = {
    # ==========================================
    # 个股投资决策 Workflow
    # ==========================================
    "stock_investment": [
        # 买卖决策
        (re.compile(r"值得买吗"), None),
        (re.compile(r"可以买(吗)?"), None),
        (re.compile(r"能买(吗)?"), None),
        (re.compile(r"该买(吗)?"), None),

        # 投资价值
        (re.compile(r"投资建议"), None),
        (re.compile(r"投资价值"), None),
        (re.compile(r"是否值得投资"), None),
        (re.compile(r"值得长期持有"), None),
        (re.compile(r"长期持有"), None),

        # 个股比较（投资角度）
        (re.compile(r".+和.+谁更值得投资"), None),
        (re.compile(r".+和.+哪个更值得买"), None),
    ],

    # ==========================================
    # 行业轮动 Workflow
    # ==========================================
    "sector_rotation": [
        (re.compile(r"行业轮动"), None),
        (re.compile(r"板块轮动"), None),

        (re.compile(r"哪些行业有机会"), None),
        (re.compile(r"哪个行业值得关注"), None),
        (re.compile(r"哪些板块有机会"), None),
        (re.compile(r"哪个板块值得关注"), None),

        (re.compile(r"热点行业"), None),
        (re.compile(r"热点板块"), None),

        (re.compile(r"行业前景"), None),
        (re.compile(r"行业比较"), None),
        (re.compile(r"板块比较"), None),

        (re.compile(r"赛道分析"), None),
        (re.compile(r"热门赛道"), None),
    ],

    # ==========================================
    # 财报分析 Workflow
    # ==========================================
    "fra": [
        # 财报分析
        (re.compile(r"财报分析"), None),
        (re.compile(r"年报分析"), None),
        (re.compile(r"季报分析"), None),
        (re.compile(r"半年报分析"), None),

        # 财报解读
        (re.compile(r"解读.*财报"), None),
        (re.compile(r"分析.*财报"), None),
        (re.compile(r"拆解.*财报"), None),

        (re.compile(r"解读.*年报"), None),
        (re.compile(r"分析.*年报"), None),
        (re.compile(r"拆解.*年报"), None),

        (re.compile(r"解读.*季报"), None),
        (re.compile(r"分析.*季报"), None),
        (re.compile(r"拆解.*季报"), None),

        # Q1/Q2/Q3/Q4 分析
        (re.compile(r"Q[1-4].*分析"), None),
        (re.compile(r"分析.*Q[1-4]"), None),
    ],
}

def classify(user_input: str) -> IntentResult | None:
    """正则匹配意图。命中返回 IntentResult，未命中返回 None。"""
    for intent, patterns in _PATTERNS.items():
        for pattern, extractor in patterns:
            match = pattern.search(user_input)
            if match:
                entities = {}
                if extractor:
                    entities = extractor(user_input, match)
                # 即使没有提取器，也尝试提取股票代码
                if "stock_code" not in entities:
                    code_match = _STOCK_CODE_RE.search(user_input)
                    if code_match:
                        entities["stock_code"] = code_match.group(1)
                return IntentResult(intent=intent, confidence=0.9, entities=entities)
    return None
