"""个股投资决策工作流。

Usage:
    from workflow.stock_investment import run
    report = run("帮我分析贵州茅台的投资价值", {"stock_code": "600519"})
"""

from workflow.stock_investment.pipeline import run

__all__ = ["run"]
