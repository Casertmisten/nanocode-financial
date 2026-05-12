"""Document loading via LlamaIndex SimpleDirectoryReader."""

from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from utils import BaseLogger

log = BaseLogger.getLogger("rag.loader")


def load_documents(doc_dir: str) -> list:
    """Load all supported documents from a directory.

    Returns list of LlamaIndex Document objects with metadata
    (file_path, file_name, file_type, file_size).
    """
    path = Path(doc_dir)
    if not path.exists():
        raise FileNotFoundError(f"Document directory not found: {doc_dir}")

    supported_ext = [".pdf", ".md", ".txt", ".markdown"]
    reader = SimpleDirectoryReader(
        input_dir=str(path),
        required_exts=supported_ext,
        recursive=True,
    )
    documents = reader.load_data()
    log.info("从目录加载文档: %s, 支持=%s, 数量=%d", doc_dir, supported_ext, len(documents))
    return documents


def load_single_file(file_path: str) -> list:
    """Load a single file into LlamaIndex Document objects."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = SimpleDirectoryReader(input_files=[str(p)])
    docs = reader.load_data()
    log.info("加载单文件: %s, 文档数=%d", file_path, len(docs))
    return docs
