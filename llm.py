"""LLM 流式调用层 — 同步/异步流式 + 同步非流式接口。"""

import json
import logging
from typing import AsyncIterator, Iterator

import httpx

import config

log = logging.getLogger(__name__)

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
) -> Iterator[dict]:
    """同步流式迭代器，用于 CLI。"""
    body = _build_body(messages, system_prompt, tools, model, stream=True)
    with httpx.Client(timeout=_TIMEOUT) as client:
        with client.stream("POST", config.API_URL, headers=_HEADERS, json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    break
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    log.warning("SSE JSON 解析失败: %s", payload[:200])


async def async_stream_chat(
    messages: list[dict],
    system_prompt: str,
    tools: list[dict] | None = None,
    model: str | None = None,
) -> AsyncIterator[dict]:
    """异步流式迭代器，用于 Web SSE。"""
    body = _build_body(messages, system_prompt, tools, model, stream=True)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        async with client.stream("POST", config.API_URL, headers=_HEADERS, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    break
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    log.warning("SSE JSON 解析失败: %s", payload[:200])


def call_llm(
    system_prompt: str,
    user_content: str,
    model: str | None = None,
) -> str:
    """同步非流式调用，用于 FRA pipeline 等场景。返回 response text content。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    body = _build_body(messages, "", model=model, stream=False)
    resp = httpx.post(config.API_URL, headers=_HEADERS, json=body, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
