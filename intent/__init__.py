"""意图识别模块 — 两层路由：正则优先，LLM 兜底。

Usage:
    from intent import classify_intent
    result = classify_intent("帮我分析贵州茅台的投资价值")
    print(result.intent)      # "stock_investment"
    print(result.entities)    # {"stock_name": "贵州茅台"}
"""

import config
from intent.schemas import IntentResult
from intent.regex_router import classify as regex_classify
from intent.llm_router import classify as llm_classify


def classify_intent(user_input: str) -> IntentResult:
    """对用户输入进行意图识别。

    第一层：正则匹配（快速、零成本）
    第二层：LLM 路由（正则未命中时启用，需要 INTENT_USE_LLM=true）
    """
    # 第一层：正则匹配
    result = regex_classify(user_input)
    if result:
        return result

    # 第二层：LLM 路由
    if config.INTENT_USE_LLM:
        return llm_classify(user_input)

    # 兜底：通用对话
    return IntentResult(intent="general", confidence=0.3)
