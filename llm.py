"""LLM 流式调用层 — 同步/异步流式 + 同步非流式接口。"""

import json
from typing import AsyncIterator, Iterator

import httpx

import config
from utils import BaseLogger
from prompts.main_prompts import rewrite_queries_prompt

log = BaseLogger.getLogger("llm")

_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {config.API_KEY}",
}
_TIMEOUT = 120.0


def _build_body(
    messages: list[dict],
    system_prompt: str,
    tools: list[dict] | None = None,
    model: str | None = None,
    stream: bool = False,
) -> dict:
    """构造请求体。"""
    all_messages = [{"role": "system", "content": system_prompt}] + messages
    body: dict = {
        "model": model or config.MODEL,
        "max_tokens": 8192,
        "messages": all_messages,
        "stream": stream,
    }
    if tools is not None:
        body["tools"] = tools
    return body


def stream_chat(
    messages: list[dict],
    system_prompt: str,
    tools: list[dict] | None = None,
    model: str | None = None,
    usage_out: dict | None = None,
) -> Iterator[dict]:
    """同步流式迭代器，用于 CLI。"""
    body = _build_body(messages, system_prompt, tools, model, stream=True)
    log.info("同步流式调用: model=%s, 消息数=%d", body["model"], len(messages))
    with httpx.Client(timeout=_TIMEOUT, proxy=None) as client:
        with client.stream("POST", config.API_URL, headers=_HEADERS, json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    # 流式响应的最后一个 chunk 通常携带 usage
                    if usage_out is not None and chunk.get("usage"):
                        usage_out.update(chunk["usage"])
                    yield chunk
                except json.JSONDecodeError:
                    log.warning("SSE JSON 解析失败: %s", payload[:200])


async def async_stream_chat(
    messages: list[dict],
    system_prompt: str,
    tools: list[dict] | None = None,
    model: str | None = None,
    usage_out: dict | None = None,
) -> AsyncIterator[dict]:
    """异步流式迭代器，用于 Web SSE。"""
    body = _build_body(messages, system_prompt, tools, model, stream=True)
    log.info("异步流式调用: model=%s, 消息数=%d", body["model"], len(messages))
    async with httpx.AsyncClient(timeout=_TIMEOUT, proxy=None) as client:
        async with client.stream("POST", config.API_URL, headers=_HEADERS, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    if usage_out is not None and chunk.get("usage"):
                        usage_out.update(chunk["usage"])
                    yield chunk
                except json.JSONDecodeError:
                    log.warning("SSE JSON 解析失败: %s", payload[:200])


def rewrite_queries(
    question: str,
    n: int = 3,
    model: str | None = None,
    usage_out: dict | None = None,
) -> list[str]:
    """用 LLM 将用户问题改写为 n 个语义相近的检索变体。"""
    system = rewrite_queries_prompt
    user = f"原问题：{question}\n\n请生成 {n} 个改写查询："
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    body = _build_body(messages, "", model=model, stream=False)
    body["max_tokens"] = 512
    body["chat_template_kwargs"] = {"enable_thinking": False}
    log.info("查询改写: question=%s", question[:50])
    resp = httpx.post(config.API_URL, headers=_HEADERS, json=body, timeout=30.0, proxy=None)
    resp.raise_for_status()
    data = resp.json()
    if usage_out is not None and "usage" in data:
        usage_out.update(data["usage"])
    content = data["choices"][0]["message"]["content"]
    variants = [line.strip() for line in content.strip().split("\n") if line.strip()]
    import re
    cleaned = [re.sub(r"^\d+[\.\)、]\s*", "", v) for v in variants]
    log.info("查询改写完成: 生成 %d 个变体", len(cleaned))
    log.info("改写后的查询：%s", cleaned[:n])
    return cleaned[:n]


def call_llm(
    system_prompt: str,
    user_content: str,
    model: str | None = None,
    usage_out: dict | None = None,
) -> str:
    """同步非流式调用，用于 FRA pipeline 等场景。返回 response text content。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    body = _build_body(messages, "", model=model, stream=False)
    log.info("同步非流式调用: model=%s, 内容长度=%d", body["model"], len(user_content))
    resp = httpx.post(config.API_URL, headers=_HEADERS, json=body, timeout=_TIMEOUT, proxy=None)
    resp.raise_for_status()
    data = resp.json()
    if usage_out is not None and "usage" in data:
        usage_out.update(data["usage"])
    content = data["choices"][0]["message"]["content"]
    log.info("同步调用完成: 响应长度=%d", len(content))
    return content
