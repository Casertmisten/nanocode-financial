"""RAG module - document ingestion and semantic retrieval.

Usage:
    import rag

    # Import documents into knowledge base
    count = rag.ingest("/path/to/docs")

    # Query the knowledge base
    results = rag.query("苹果公司财报表现如何")
"""

import os

import config
from rag.indexer import add_to_index, build_fresh_index, get_index
from rag.loader import load_single_file
from rag.loader import load_documents
from rag.retriever import format_results, retrieve
from utils import BaseLogger

log = BaseLogger.getLogger("rag")


def ingest(doc_path: str | None = None) -> int:
    """Import documents into the financial knowledge base.

    Args:
        doc_path: Directory containing documents. Defaults to config.DOCUMENTS_DIR.

    Returns:
        Number of new documents processed.
    """
    doc_dir = doc_path or config.DOCUMENTS_DIR
    log.info("开始导入文档目录: %s", doc_dir)
    documents = load_documents(doc_dir)
    if not documents:
        log.info("目录中无支持的文档: %s", doc_dir)
        return 0

    chroma_dir = config.CHROMA_PERSIST_DIR

    # Check if index already exists
    tracker_file = os.path.join(os.path.dirname(chroma_dir), ".ingested.json")
    if os.path.exists(tracker_file) or os.path.exists(chroma_dir):
        # Incremental update
        index, _ = get_index(
            chroma_dir,
            config.EMBEDDING_API_URL,
            config.EMBEDDING_API_KEY,
            config.EMBEDDING_MODEL,
        )
        return add_to_index(index, documents, chroma_dir, doc_dir)
    else:
        # Fresh build
        _, _, count = build_fresh_index(
            documents,
            chroma_dir,
            config.EMBEDDING_API_URL,
            config.EMBEDDING_API_KEY,
            config.EMBEDDING_MODEL,
        )
        log.info("全新索引构建完成: %d 个文档", count)
        return count


def query(question: str, top_k: int = 5) -> list[dict]:
    """Search the knowledge base for relevant information.

    Args:
        question: Natural language question.
        top_k: Number of results to return.

    Returns:
        List of {"text": str, "source": str, "score": float}.
    """
    index, _ = get_index(
        config.CHROMA_PERSIST_DIR,
        config.EMBEDDING_API_URL,
        config.EMBEDDING_API_KEY,
        config.EMBEDDING_MODEL,
    )
    return retrieve(index, question, top_k)


def query_formatted(question: str, top_k: int = 5) -> str:
    """Search and return formatted results string (for tool output)."""
    results = query(question, top_k)
    return format_results(results)


def ingest_file(file_path: str) -> int:
    """将单个文件导入知识库（增量索引）。"""
    log.info("导入单文件: %s", file_path)
    documents = load_single_file(file_path)
    if not documents:
        log.warning("文件加载失败或无内容: %s", file_path)
        return 0

    index, _ = get_index(
        config.CHROMA_PERSIST_DIR,
        config.EMBEDDING_API_URL,
        config.EMBEDDING_API_KEY,
        config.EMBEDDING_MODEL,
    )
    count = add_to_index(index, documents, config.CHROMA_PERSIST_DIR, os.path.dirname(file_path))
    log.info("单文件导入完成: %s, 新增=%d", file_path, count)
    return count
