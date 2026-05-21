"""L1: 用户画像管理 — JSON 读写、Markdown 渲染、候选池管理、LLM 合并更新。"""

import json
import os
from datetime import datetime, timezone

import config
import llm
from memory.prompts import MERGE_PROFILE_PROMPT
from utils import BaseLogger

log = BaseLogger.getLogger("memory.profile")

_DEFAULT_PROFILE = {
    "preferred_markets": [],
    "focus_sectors": [],
    "watched_stocks": [],
    "risk_tolerance": "",
    "report_style": "",
    "language": "中文",
}


def load_profile() -> dict | None:
    """加载用户画像 JSON，文件不存在返回 None。"""
    path = config.MEMORY_PROFILE_PATH
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("profile", _DEFAULT_PROFILE)
    except Exception:
        log.warning("加载用户画像失败", exc_info=True)
        return None


def render_profile_markdown(profile: dict | None) -> str:
    """将用户画像渲染为 Markdown 片段。画像为空时返回空字符串。"""
    if not profile:
        return ""
    lines = ["## 用户画像"]
    for key, label in [
        ("preferred_markets", "关注市场"),
        ("focus_sectors", "关注行业"),
        ("watched_stocks", "关注个股"),
        ("risk_tolerance", "风险偏好"),
        ("report_style", "报告风格"),
    ]:
        val = profile.get(key)
        if val and val != [] and val != "":
            if isinstance(val, list):
                display = "、".join(str(v) for v in val)
            else:
                display = str(val)
            lines.append(f"- {label}：{display}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def add_candidate(session_id: str, fields: dict):
    """将一轮会话提取的画像候选写入候选池。"""
    path = config.MEMORY_CANDIDATES_PATH
    candidates_data = {"candidates": [], "session_count": 0}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                candidates_data = json.load(f)
        except Exception:
            log.warning("读取候选池失败，重新初始化", exc_info=True)

    candidates_data["candidates"].append({
        "session_id": session_id,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "fields": {k: v for k, v in fields.items() if v and v != [] and v != ""},
    })
    candidates_data["session_count"] += 1

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(candidates_data, f, ensure_ascii=False, indent=2)

    log.info("画像候选已添加: session=%s, 候选池=%d 轮", session_id, candidates_data["session_count"])

    # 检查是否触发画像更新
    if candidates_data["session_count"] >= config.MEMORY_PROFILE_UPDATE_INTERVAL:
        _update_profile(candidates_data)


def _update_profile(candidates_data: dict):
    """用 LLM 合并候选并更新 profile.json。"""
    current_profile = load_profile() or _DEFAULT_PROFILE
    candidates_text = json.dumps(candidates_data["candidates"], ensure_ascii=False, indent=2)

    prompt = MERGE_PROFILE_PROMPT.format(
        current_profile=json.dumps(current_profile, ensure_ascii=False, indent=2),
        candidates=candidates_text,
    )

    try:
        result = llm.call_llm("你是一个用户画像管理助手。", prompt)
        # 提取 JSON（可能包裹在 ```json ... ``` 中）
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        new_profile = json.loads(result)
    except Exception:
        log.warning("LLM 合并画像失败，跳过本次更新", exc_info=True)
        # 清空候选池避免无限累积
        _clear_candidates()
        return

    # 写入 profile.json
    profile_data = {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "profile": new_profile,
    }
    path = config.MEMORY_PROFILE_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, ensure_ascii=False, indent=2)

    log.info("用户画像已更新: %s", json.dumps(new_profile, ensure_ascii=False)[:200])
    _clear_candidates()


def _clear_candidates():
    """清空候选池。"""
    path = config.MEMORY_CANDIDATES_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"candidates": [], "session_count": 0}, f)
