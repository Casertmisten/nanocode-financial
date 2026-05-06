"""Deep Research 各阶段的 prompt 模板。"""

import os

_PROMPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "prompts",
    "deep_research",
)


def _load(name: str) -> str:
    with open(os.path.join(_PROMPT_DIR, name), "r", encoding="utf-8") as f:
        return f.read().strip()


ANALYZE_PROMPT = _load("analyze.txt")
REDUCE_PROMPT = _load("reduce.txt")
