"""意图识别结果数据类。"""

from dataclasses import dataclass, field


@dataclass
class IntentResult:
    """意图识别结果。"""

    intent: str  # stock_investment | sector_rotation | fra | general
    confidence: float = 1.0
    # 提取的实体，如 {"stock_name": "贵州茅台", "stock_code": "600519"}
    entities: dict = field(default_factory=dict)
