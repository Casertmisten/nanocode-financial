"""MinerU PDF 解析 — 轻量级优先，精准解析兜底，超长文档分段处理。"""

import io
import math
import os
import time
import zipfile

import fitz  # pymupdf
import requests

import config
from utils import BaseLogger

log = BaseLogger.getLogger("rag.pdf_parser")

# 精准解析单次最大页数
_ACC_MAX_PAGES = 200


class MinerUParseError(Exception):
    """MinerU 解析异常。"""


def _get_page_count(file_path: str) -> int:
    """获取 PDF 页数。"""
    doc = fitz.open(file_path)
    count = doc.page_count
    doc.close()
    return count


def parse_pdf_to_markdown(file_path: str) -> str:
    """将 PDF 文件解析为 Markdown 文本。

    优先使用轻量级解析（≤10MB、≤20页、无需 Token），
    失败则回退到精准解析（需 Token）。
    超过 200 页时分段调用精准解析后拼接结果。
    """
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)
    page_count = _get_page_count(file_path)
    log.info("开始解析 PDF: %s (%.1fMB, %d页)", file_name, file_size / 1024 / 1024, page_count)

    # 轻量级解析（≤10MB 且 ≤20页）
    if file_size <= config.MINERU_LIGHT_MAX_SIZE and page_count <= config.MINERU_LIGHT_MAX_PAGES:
        try:
            md = _parse_lightweight(file_path, file_name)
            log.info("轻量级解析成功: %s", file_name)
            return md
        except MinerUParseError as e:
            log.warning("轻量级解析失败，回退精准解析: %s", e)
    else:
        log.info("跳过轻量级解析（文件=%.1fMB, 页数=%d），直接使用精准解析",
                 file_size / 1024 / 1024, page_count)

    # 精准解析（超长文档分段）
    md = _parse_precise_chunked(file_path, file_name, page_count)
    log.info("精准解析成功: %s", file_name)
    return md


def _parse_precise_chunked(file_path: str, file_name: str, page_count: int) -> str:
    """精准解析，超过 _ACC_MAX_PAGES 自动分段。"""
    if page_count <= _ACC_MAX_PAGES:
        return _parse_precise(file_path, file_name)

    chunks = math.ceil(page_count / _ACC_MAX_PAGES)
    log.info("文档共 %d 页，分 %d 段解析（每段≤%d页）", page_count, chunks, _ACC_MAX_PAGES)

    parts = []
    for i in range(chunks):
        start_page = i * _ACC_MAX_PAGES + 1  # 1-indexed
        end_page = min((i + 1) * _ACC_MAX_PAGES, page_count)
        page_ranges = f"{start_page}-{end_page}"
        log.info("解析第 %d/%d 段: 页码 %s", i + 1, chunks, page_ranges)

        md = _parse_precise(file_path, file_name, page_ranges=page_ranges)
        parts.append(md)

    return "\n\n".join(parts)


# ── 轻量级解析 ──────────────────────────────────────────────


def _parse_lightweight(file_path: str, file_name: str) -> str:
    """Agent 轻量解析 API（无需 Token，文件上传模式）。"""
    # 1. 获取签名上传 URL
    api_url = config.MINERU_LIGHT_FILE_URL
    resp = requests.post(api_url, json={
        "file_name": file_name,
        "language": "ch",
    }, timeout=30)
    result = resp.json()
    if result.get("code") != 0:
        raise MinerUParseError(f"轻量级任务创建失败: {result.get('msg')}")

    task_id = result["data"]["task_id"]
    file_url = result["data"]["file_url"]
    log.info("轻量级任务已创建: task_id=%s", task_id)

    # 2. PUT 上传文件
    with open(file_path, "rb") as f:
        put_resp = requests.put(file_url, data=f, timeout=60)
    if put_resp.status_code not in (200, 201):
        raise MinerUParseError(f"文件上传失败: HTTP {put_resp.status_code}")

    # 3. 轮询结果
    return _poll_lightweight(task_id)


def _poll_lightweight(task_id: str) -> str:
    """轮询轻量级解析结果，返回 Markdown 文本。"""
    url = f"https://mineru.net/api/v1/agent/parse/{task_id}"
    start = time.time()

    while time.time() - start < config.MINERU_TIMEOUT:
        resp = requests.get(url, timeout=30)
        result = resp.json()
        state = result["data"]["state"]

        if state == "done":
            md_url = result["data"]["markdown_url"]
            md_resp = requests.get(md_url, timeout=60)
            return md_resp.text

        if state == "failed":
            err_code = result["data"].get("err_code", 0)
            err_msg = result["data"].get("err_msg", "未知错误")
            raise MinerUParseError(f"[{err_code}] {err_msg}")

        time.sleep(config.MINERU_POLL_INTERVAL)

    raise MinerUParseError(f"轮询超时 ({config.MINERU_TIMEOUT}s)")


# ── 精准解析 ────────────────────────────────────────────────


def _parse_precise(file_path: str, file_name: str, page_ranges: str | None = None) -> str:
    """精准解析 API（需 Token，文件上传模式）。"""
    if not config.MINERU_ACC_KEY:
        raise MinerUParseError("未配置 MinerU Token，无法使用精准解析")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.MINERU_ACC_KEY}",
    }

    # 1. 申请文件上传链接
    file_info: dict = {"name": file_name}
    if page_ranges:
        file_info["page_ranges"] = page_ranges

    api_url = config.MINERU_ACC_FILE_URL
    resp = requests.post(api_url, headers=headers, json={
        "files": [file_info],
        "model_version": config.MINERU_MODEL_VERSION,
    }, timeout=30)
    result = resp.json()
    if result.get("code") != 0:
        raise MinerUParseError(f"精准解析任务创建失败: {result.get('msg')}")

    batch_id = result["data"]["batch_id"]
    upload_url = result["data"]["file_urls"][0]
    log.info("精准解析任务已创建: batch_id=%s", batch_id)

    # 2. PUT 上传文件
    with open(file_path, "rb") as f:
        put_resp = requests.put(upload_url, data=f, timeout=120)
    if put_resp.status_code not in (200, 201):
        raise MinerUParseError(f"文件上传失败: HTTP {put_resp.status_code}")

    # 3. 轮询结果
    return _poll_precise(batch_id, headers)


def _poll_precise(batch_id: str, headers: dict) -> str:
    """轮询精准解析结果，下载 zip 并提取 Markdown。"""
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    start = time.time()

    while time.time() - start < config.MINERU_TIMEOUT:
        resp = requests.get(url, headers=headers, timeout=30)
        result = resp.json()
        extract_results = result["data"]["extract_result"]

        for item in extract_results:
            state = item["state"]

            if state == "done" and item.get("full_zip_url"):
                return _extract_markdown_from_zip(item["full_zip_url"])

            if state == "failed":
                raise MinerUParseError(f"精准解析失败: {item.get('err_msg', '未知错误')}")

        time.sleep(config.MINERU_POLL_INTERVAL)

    raise MinerUParseError(f"轮询超时 ({config.MINERU_TIMEOUT}s)")


def _extract_markdown_from_zip(zip_url: str) -> str:
    """从 MinerU 精准解析的 zip 包中提取 Markdown。"""
    resp = requests.get(zip_url, timeout=120)
    if resp.status_code != 200:
        raise MinerUParseError(f"下载解析结果失败: HTTP {resp.status_code}")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # 精准解析结果中 Markdown 文件名格式: **/full.md
        md_files = [n for n in zf.namelist() if n.endswith("full.md")]
        if not md_files:
            raise MinerUParseError("zip 包中未找到 Markdown 文件")
        return zf.read(md_files[0]).decode("utf-8")
