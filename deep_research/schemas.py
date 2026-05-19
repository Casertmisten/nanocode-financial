"""Deep Research 数据结构定义。"""

from dataclasses import dataclass, field


@dataclass
class SubTask:
    """Plan Agent 生成的子任务。"""
    id: int
    title: str
    description: str
    tools: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)


@dataclass
class ExecutorResult:
    """Executor Agent 执行结果。"""
    task_id: int
    title: str
    summary: str = ""
    sources: list[str] = field(default_factory=list)
    raw_data: list[str] = field(default_factory=list)
    complete: bool = True  # 是否完整完成（超时则为 False）
    usage: dict = field(default_factory=dict)  # 累积 token 用量
