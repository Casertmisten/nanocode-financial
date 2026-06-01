"""LLM 路由层 — 正则未命中时，通过 LLM 对用户输入做意图分类。"""

import json
import llm
from intent.schemas import IntentResult
from utils import BaseLogger

log = BaseLogger.getLogger("intent.llm_router")

_SYSTEM_PROMPT = """
你是一个金融助手的意图识别模块。

你的任务是判断：
用户的问题是否需要调用专业分析工作流。

重要原则：

- general 是默认分类。
- 只有当用户明确要求进行深度分析、投资决策、多维度比较、综合研究时，才进入对应工作流。
- 对于事实查询、知识问答、公司介绍、财务指标查询、新闻查询、概念解释等问题，即使涉及股票、行业或财报，也应归类为 general。
- 宁可归类为 general，也不要轻易进入工作流。

可选意图如下：

1. stock_investment
个股投资决策工作流。

仅当用户明确要求：
- 是否值得投资
- 是否值得买入或卖出
- 投资价值评估
- 未来上涨空间分析
- 买卖建议
- 个股对比并给出投资建议
- 风险收益分析

时使用。

示例：
- 现在可以买英伟达吗？
- 特斯拉未来还有上涨空间吗？
- AMD和英伟达谁更值得投资？
- 帮我做一下腾讯的投资分析

输出：
{"intent": "stock_investment", "stock_name": "股票名称（如有）", "stock_code": "股票代码（如有）"}

--------------------------------------------------

2. sector_rotation
行业轮动与机会发现工作流。

仅当用户明确要求：
- 行业机会分析
- 板块轮动分析
- 资金流向分析
- 行业比较
- 热点赛道研究
- 市场机会挖掘

时使用。

示例：
- 最近哪些板块值得关注？
- AI和新能源哪个赛道更有机会？
- 当前市场热点行业有哪些？
- 哪个行业未来一年成长空间最大？

输出：
{"intent": "sector_rotation", "stock_name": "股票名称（如有）", "stock_code": "股票代码（如有）"}

--------------------------------------------------

3. fra
财报深度分析工作流。

仅当用户明确要求：
- 财报分析
- 财报解读
- 年报分析
- 季报分析
- 财务状况评估
- 财务指标深度拆解

时使用。

示例：
- 分析一下腾讯2025Q1财报
- 解读英伟达最新财报
- 帮我拆解平安银行年报
- 评价一下苹果最新财务表现

输出：
{"intent": "fra", "stock_name": "股票名称（如有）", "stock_code": "股票代码（如有）"}

--------------------------------------------------

4. general

以下情况全部归类为 general：

- 股票基础信息查询
- 公司介绍
- 财务数据查询
- 分红数据查询
- 新闻查询
- 市场知识问答
- 概念解释
- 行情查询
- 非金融问题
- 无法确定用户是否需要深度分析

示例：

- 英伟达主营业务是什么？
- 腾讯市值是多少？
- 华邦健康2025年分红多少钱？
- 苹果CEO是谁？
- 什么是市盈率？
- AI行业是什么？
- 今天A股涨了吗？
- 贵州茅台有哪些产品？
- 帮我介绍一下宁德时代
- 今天天气怎么样？

以上全部输出：

{"intent": "general", "stock_name": "股票名称（如有）", "stock_code": "股票代码（如有）"}

--------------------------------------------------

特别注意：

对于以下问题：

- 帮我分析贵州茅台
- 介绍一下英伟达
- 腾讯怎么样
- 苹果公司如何

由于用户没有明确表达投资决策、财报分析或行业研究需求，

统一归类为：

{"intent": "general", "stock_name": "股票名称（如有）", "stock_code": "股票代码（如有）"}

--------------------------------------------------

请严格按照以下 JSON 格式输出，不允许输出任何解释：

{
  "intent": "意图类型",
  "stock_name": "股票名称（如有）",
  "stock_code": "股票代码（如有）"
}
"""


def classify(user_input: str) -> IntentResult:
    """用 LLM 对用户输入做意图分类。返回 IntentResult。"""
    try:
        response = llm.call_llm(_SYSTEM_PROMPT, user_input, enable_thinking=False)
        log.info("LLM 响应: %s", response)
        response = response.strip()
        log.info("LLM 响应: %s", response)

        # 提取 JSON（可能包裹在 ```json ... ``` 中）
        if response.startswith("```"):
            response = response.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        data = json.loads(response)
        intent = data.get("intent", "general")

        # 校验意图类型
        valid_intents = {"stock_investment", "sector_rotation", "fra", "general"}
        if intent not in valid_intents:
            intent = "general"

        entities = {}
        if data.get("stock_name"):
            entities["stock_name"] = data["stock_name"]
        if data.get("stock_code"):
            entities["stock_code"] = data["stock_code"]

        return IntentResult(intent=intent, confidence=0.8, entities=entities)

    except Exception as e:
        log.warning("LLM 路由失败: %s", e, exc_info=True)
        return IntentResult(intent="general", confidence=0.5)
