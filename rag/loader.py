"""Document loading — PDF 通过 MinerU 转 Markdown 后加载，其他走 LlamaIndex。"""

import os
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from rag.pdf_parser import parse_pdf_to_markdown
from utils import BaseLogger

log = BaseLogger.getLogger("rag.loader")

# 需要通过 MinerU 解析的文件类型
_MINERU_EXTS = {".pdf"}
# 直接由 LlamaIndex 加载的文件类型
_DIRECT_EXTS = [".md", ".txt", ".markdown"]


def load_documents(doc_dir: str) -> list:
    """Load all supported documents from a directory.

    PDF 文件先通过 MinerU API 转为 Markdown 再加载。
    """
    path = Path(doc_dir)
    if not path.exists():
        raise FileNotFoundError(f"Document directory not found: {doc_dir}")

    # 先处理 PDF 文件（转为 Markdown）
    md_dir = os.path.join(doc_dir, ".mineru_cache")
    os.makedirs(md_dir, exist_ok=True)

    for f in path.rglob("*"):
        if f.suffix.lower() in _MINERU_EXTS and f.is_file():
            _convert_pdf_to_md(str(f), md_dir)

    # 统一用 LlamaIndex 加载（包含转换后的 Markdown）
    supported_ext = _DIRECT_EXTS
    reader = SimpleDirectoryReader(
        input_dir=str(path),
        required_exts=supported_ext,
        recursive=True,
    )
    documents = reader.load_data()
    log.info("从目录加载文档: %s, 数量=%d", doc_dir, len(documents))
    return documents


def load_single_file(file_path: str) -> list:
    """Load a single file into LlamaIndex Document objects.

    PDF 文件先通过 MinerU 转为 Markdown。
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if p.suffix.lower() in _MINERU_EXTS:
        md_path = _convert_pdf_to_md(file_path, os.path.dirname(file_path))
        p = Path(md_path)

    reader = SimpleDirectoryReader(input_files=[str(p)])
    docs = reader.load_data()
    log.info("加载单文件: %s, 文档数=%d", file_path, len(docs))
    return docs


def _convert_pdf_to_md(pdf_path: str, cache_dir: str) -> str:
    """将 PDF 通过 MinerU 转为 Markdown，缓存到 cache_dir。"""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    md_path = os.path.join(cache_dir, f"{pdf_name}.md")

    # 已有缓存则直接返回
    if os.path.exists(md_path):
        log.info("使用缓存的 Markdown: %s", md_path)
        return md_path

    log.info("通过 MinerU 解析 PDF: %s → %s", pdf_path, md_path)
    md_content = parse_pdf_to_markdown(pdf_path)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    log.info("Markdown 已保存: %s (%d 字符)", md_path, len(md_content))
    return md_path
