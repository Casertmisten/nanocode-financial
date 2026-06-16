"""LLM 流式调用层 — 同步/异步流式 + 同步非流式接口。"""

import json
from typing import AsyncIterator, Iterator
import httpx
from langfuse import get_client, observe
import config
from utils import BaseLogger
from prompts.main_prompts import rewrite_queries_prompt

log = BaseLogger.getLogger("llm")

_langfuse = get_client()

_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {config.API_KEY}",
}
_TIMEOUT = float(config.LLM_TIMEOUT)


def get_api_base_url(url: str = config.API_URL) -> str:
    """从 chat completions endpoint 反推 API base URL。

    API_URL 经过规范化后形如 .../v1/chat/completions，
    这里去掉 /chat/completions（及 /v1/chat/completions）后缀，返回 .../v1。
    用于拼接 /models 等非 chat 端点。
    """
    base = url.rstrip("/")
    for suffix in ("/chat/completions",):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base.rstrip("/")


def check_llm_connection() -> tuple[bool, str, list[str]]:
    """启动时探测 LLM 是否可连通。

    GET {base}/models 一次，返回 (ok, message, model_ids)：
    - ok=True：连通且响应正常；message 含模型名校验结果。
    - ok=False：message 给出分类后的中文原因（404/401/超时等），model_ids 为空。
    超时设短（10s），不阻塞启动。

    如需完整 item（含 owned_by 等字段），请改用 fetch_models()。
    """
    ok, message, items = fetch_models()
    model_ids = [m.get("id", "") for m in items if m.get("id")]
    return (ok, message, model_ids)


def fetch_models() -> tuple[bool, str, list[dict]]:
    """拉取 /models 完整 item 列表 + 连通性判定。

    返回 (ok, message, items)：items 为 data 数组（含 id/owned_by 等原始字段）。
    ok=False 时 items 为空，message 为分类后的中文原因。
    """
    base = get_api_base_url()
    models_url = f"{base}/models"
    try:
        resp = httpx.get(
            models_url,
            headers={"Authorization": f"Bearer {config.API_KEY}"},
            timeout=10.0,
            proxy=None,
        )
    except httpx.TimeoutException:
        return (False, f"连接超时（检查网络/代理）: {models_url}", [])
    except httpx.ConnectError as e:
        return (False, f"无法连接到 API 服务器（检查网络/代理）: {models_url}\n  {e}", [])
    except httpx.HTTPError as e:
        return (False, f"请求异常: {type(e).__name__}: {e}", [])

    if resp.status_code == 404:
        return (
            False,
            f"API 地址不正确（HTTP 404）: {models_url}\n"
            "  请检查 ALI_API_URL 是否指向 .../v1 或 .../v1/chat/completions",
            [],
        )
    if resp.status_code in (401, 403):
        return (
            False,
            f"API Key 无效或无权限（HTTP {resp.status_code}）: {models_url}\n"
            "  请检查 ALI_API_KEY 是否正确",
            [],
        )
    if resp.status_code != 200:
        return (False, f"HTTP {resp.status_code}: {models_url}", [])

    try:
        data = resp.json()
    except Exception as e:
        return (False, f"响应解析失败（非合法 JSON）: {e}", [])
    items = data.get("data", []) if isinstance(data, dict) else []

    # 校验配置的模型名是否在列表中（软警告，不影响 ok）
    model_ids = [m.get("id", "") for m in items if isinstance(m, dict) and m.get("id")]
    if config.MODEL and model_ids and config.MODEL not in model_ids:
        return (
            True,
            f"连通正常，但配置的模型 {config.MODEL!r} 不在可用列表中（共 {len(model_ids)} 个）"
            "——模型名可能拼错，LLM 调用时可能失败",
            items,
        )
    return (True, f"连接正常，模型 {config.MODEL!r} 可用（共 {len(model_ids)} 个）", items)


def _usage_to_langfuse(usage: dict | None) -> dict | None:
    """将 OpenAI 格式的 usage 转为 Langfuse 格式。"""
    if not usage:
        return None
    return {
        "input": usage.get("prompt_tokens", 0),
        "output": usage.get("completion_tokens", 0),
    }


def _build_body(
    messages: list[dict],
    system_prompt: str,
    tools: list[dict] | None = None,
    model: str | None = None,
    stream: bool = False,
    max_tokens: int | None = None,
) -> dict:
    """构造请求体。"""
    all_messages = [{"role": "system", "content": system_prompt}] + messages
    body: dict = {
        "model": model or config.MODEL,
        "max_tokens": max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
        "messages": all_messages,
        "stream": stream,
    }
    if tools is not None:
        body["tools"] = tools
    if stream:
        body["stream_options"] = {"include_usage": True}
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
    obs = _langfuse.start_observation(
        name="stream-chat",
        as_type="generation",
        model=body["model"],
        input={"message_count": len(messages)},
    )
    try:
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
        obs.update(output="(streamed)", usage_details=_usage_to_langfuse(usage_out))
    finally:
        obs.end()


async def async_stream_chat(
    messages: list[dict],
    system_prompt: str,
    tools: list[dict] | None = None,
    model: str | None = None,
    usage_out: dict | None = None,
    max_tokens: int | None = None,
) -> AsyncIterator[dict]:
    """异步流式迭代器，用于 Web SSE。"""
    # 构造请求体
    body = _build_body(messages, system_prompt, tools, model, stream=True, max_tokens=max_tokens)
    log.info("异步流式调用: model=%s, 消息数=%d", body["model"], len(messages))
    obs = _langfuse.start_observation(
        name="async-stream-chat",
        as_type="generation",
        model=body["model"],
        input={"message_count": len(messages)},
    )
    try:
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
        obs.update(output="(streamed)", usage_details=_usage_to_langfuse(usage_out))
    finally:
        obs.end()


@observe(as_type="generation", name="rewrite-queries", capture_input=False)
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
    usage = data.get("usage")
    if usage_out is not None and usage:
        usage_out.update(usage)
    content = data["choices"][0]["message"]["content"]
    variants = [line.strip() for line in content.strip().split("\n") if line.strip()]
    import re
    cleaned = [re.sub(r"^\d+[\.\)、]\s*", "", v) for v in variants]
    log.info("查询改写完成: 生成 %d 个变体", len(cleaned))
    log.info("改写后的查询：%s", cleaned[:n])
    _langfuse.update_current_generation(
        input={"question": question, "n": n},
        output=cleaned[:n],
        model=body["model"],
        usage_details=_usage_to_langfuse(usage),
    )
    return cleaned[:n]


@observe(as_type="generation", name="call-llm", capture_input=False)
def call_llm(
    system_prompt: str,
    user_content: str,
    model: str | None = None,
    usage_out: dict | None = None,
    enable_thinking: bool = True,
) -> str:
    """同步非流式调用，用于 FRA pipeline 等场景。返回 response text content。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    body = _build_body(messages, "", model=model, stream=False)
    if not enable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
        log.info("禁用思考模式")
    else:
        log.info("启用思考模式")
    log.info("同步非流式调用: model=%s, 内容长度=%d, 思考模式=%s", body["model"], len(user_content), enable_thinking)
    resp = httpx.post(config.API_URL, headers=_HEADERS, json=body, timeout=_TIMEOUT, proxy=None)
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage")
    if usage_out is not None and usage:
        usage_out.update(usage)
    content = data["choices"][0]["message"]["content"]
    log.info("同步调用完成: 响应长度=%d", len(content))
    _langfuse.update_current_generation(
        input={"content_length": len(user_content)},
        output=content[:500],
        model=body["model"],
        usage_details=_usage_to_langfuse(usage),
    )
    return content


@observe(as_type="generation", name="async-call-llm", capture_input=False)
async def async_call_llm(
    system_prompt: str,
    user_content: str,
    model: str | None = None,
    usage_out: dict | None = None,
    enable_thinking: bool = True,
) -> str:
    """异步非流式调用，用于工作流 pipeline 并行场景。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    body = _build_body(messages, "", model=model, stream=False)
    if not enable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    log.info("异步非流式调用: model=%s, 内容长度=%d", body["model"], len(user_content))
    async with httpx.AsyncClient(timeout=_TIMEOUT, proxy=None) as client:
        resp = await client.post(config.API_URL, headers=_HEADERS, json=body)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage")
        if usage_out is not None and usage:
            usage_out.update(usage)
        content = data["choices"][0]["message"]["content"]
        log.info("异步调用完成: 响应长度=%d", len(content))
        _langfuse.update_current_generation(
            input={"content_length": len(user_content)},
            output=content[:500],
            model=body["model"],
            usage_details=_usage_to_langfuse(usage),
        )
        return content


@observe(as_type="generation", name="async-stream-llm", capture_input=False)
async def async_stream_llm(
    system_prompt: str,
    user_content: str,
    model: str | None = None,
    on_token=None,
) -> str:
    """异步流式调用，逐 token 通过 on_token(text) 回调，返回完整内容。"""
    messages = [{"role": "user", "content": user_content}]
    body = _build_body(messages, system_prompt, model=model, stream=True)
    log.info("异步流式调用: model=%s, 内容长度=%d", body["model"], len(user_content))
    content_parts = []
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
                    choices = chunk.get("choices", [])
                    if choices:
                        text = choices[0].get("delta", {}).get("content")
                        if text:
                            content_parts.append(text)
                            if on_token:
                                on_token(text)
                except json.JSONDecodeError:
                    pass
    full = "".join(content_parts)
    log.info("异步流式调用完成: 响应长度=%d", len(full))
    _langfuse.update_current_generation(
        input={"content_length": len(user_content)},
        output=full[:500],
        model=body["model"],
    )
    return full
