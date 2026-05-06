"""路由层：根据用户查询判断走 RAG QA 还是 Deep Research。"""

import json
import urllib.request

import config

# 关键词直出规则：包含这些词直接走深度研究
_DEEP_RESEARCH_KEYWORDS = ["深度分析"]

ROUTER_PROMPT = """## Role

你是一个高精度的金融意图分流器（Router）。你的唯一任务是将用户的问题分类到两个垂直处理路径之一。


## Classification Criteria

- **A [简单问答]**:
  - 特征：客观事实、专有名词定义、实时行情查询、基础金融常识、公开财报数据摘录。
  - 逻辑：无需多步推理，仅需检索或提取。

- **B [深度研究]**:
  - 特征：多维度对比、投资策略建议、风险评估、未来走势预测、需综合多方数据的逻辑推演。
  - 逻辑：需要深度思考、多步推理。


## Constraints

- 必须遵循"非 A 即 B"的原则。
- 严禁输出类别标签以外的任何字符（包括空格、标点、解释、前缀）。


## User Query

{query}


## Output (A or B)"""


def route_query(query: str) -> str:
    """判断查询应该走 RAG QA 还是 Deep Research。

    Returns:
        "rag" 或 "deep_research"
    """
    # 关键词直出
    for kw in _DEEP_RESEARCH_KEYWORDS:
        if kw in query:
            return "deep_research"

    # 调用模型分类
    prompt = ROUTER_PROMPT.format(query=query)
    request = urllib.request.Request(
        config.API_URL,
        data=json.dumps(
            {
                "model": config.MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.API_KEY}",
        },
    )
    response = urllib.request.urlopen(request)
    result = json.loads(response.read())
    answer = result["choices"][0]["message"]["content"].strip()

    return "deep_research" if "B" in answer else "rag"
