"""Chroma vector index management with LlamaIndex."""

import hashlib
import json
import os

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag.chunker import split_documents
from utils import BaseLogger

log = BaseLogger.getLogger("rag.indexer")

# Track which custom model names have been registered with LlamaIndex enums.
_registered_models: set = set()


def _register_custom_model(model_name: str):
    """Register a non-standard model name with LlamaIndex's embedding enums.

    LlamaIndex's OpenAIEmbedding validates model names against a fixed enum
    that only contains OpenAI models.  DashScope (Ali) uses different model
    names (e.g. ``text-embedding-v3``).  This function monkey-patches the enum
    and lookup dicts so that custom model names are accepted.

    Safe to call multiple times -- each model name is registered only once.
    """
    if model_name in _registered_models:
        return

    from llama_index.embeddings.openai.base import (
        OpenAIEmbeddingMode,
        OpenAIEmbeddingModeModel,
        OpenAIEmbeddingModelType,
        _QUERY_MODE_MODEL_DICT,
        _TEXT_MODE_MODEL_DICT,
    )

    # Add a _missing_ hook so unknown values produce valid enum members
    # instead of raising ValueError.
    def _make_missing(enum_base):
        def _missing(cls, value):
            if isinstance(value, str):
                member = str.__new__(cls, value)
                member._name_ = value
                member._value_ = value
                return member
            return None
        return classmethod(_missing)

    if not hasattr(OpenAIEmbeddingModelType, "_custom_models_patched"):
        OpenAIEmbeddingModelType._missing_ = _make_missing(OpenAIEmbeddingModelType)
        OpenAIEmbeddingModelType._custom_models_patched = True

    if not hasattr(OpenAIEmbeddingModeModel, "_custom_models_patched"):
        OpenAIEmbeddingModeModel._missing_ = _make_missing(OpenAIEmbeddingModeModel)
        OpenAIEmbeddingModeModel._custom_models_patched = True

    custom_model = OpenAIEmbeddingModelType(model_name)
    custom_mode_model = OpenAIEmbeddingModeModel(model_name)

    for mode in (OpenAIEmbeddingMode.SIMILARITY_MODE, OpenAIEmbeddingMode.TEXT_SEARCH_MODE):
        key = (mode, custom_model)
        _QUERY_MODE_MODEL_DICT.setdefault(key, custom_mode_model)
        _TEXT_MODE_MODEL_DICT.setdefault(key, custom_mode_model)

    _registered_models.add(model_name)
    log.info("注册自定义嵌入模型: %s", model_name)


def _get_embed_model(api_url: str, api_key: str, model_name: str):
    """Create OpenAI-compatible embedding model."""
    _register_custom_model(model_name)
    return OpenAIEmbedding(
        model=model_name,
        api_key=api_key,
        api_base=api_url,
    )


def _ingested_tracker_path(chroma_dir: str) -> str:
    """Path to the JSON file tracking which files have been ingested."""
    return os.path.join(os.path.dirname(chroma_dir), ".ingested.json")


def _load_ingested(tracker_path: str) -> dict:
    """Load the ingested files tracker. Returns {filepath: md5hash}."""
    if not os.path.exists(tracker_path):
        return {}
    with open(tracker_path, "r") as f:
        return json.load(f)


def _save_ingested(tracker_path: str, data: dict):
    """Save the ingested files tracker."""
    os.makedirs(os.path.dirname(tracker_path), exist_ok=True)
    with open(tracker_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _file_md5(file_path: str) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_index(
    chroma_dir: str,
    embedding_api_url: str,
    embedding_api_key: str,
    embedding_model_name: str,
):
    """Get or create a VectorStoreIndex backed by Chroma.

    Returns (index, embed_model) tuple.
    """
    embed_model = _get_embed_model(embedding_api_url, embedding_api_key, embedding_model_name)

    os.makedirs(chroma_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    collection = chroma_client.get_or_create_collection("financial_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    log.info("获取索引完成: chroma_dir=%s, 集合文档数=%d", chroma_dir, collection.count())
    return index, embed_model


def add_to_index(
    index,
    documents: list,
    chroma_dir: str,
    doc_dir: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
) -> int:
    """Add new/modified documents to the index. Returns count of docs added.

    Uses an ingested tracker (.ingested.json) to skip files already processed
    with the same content hash.
    """
    tracker_path = _ingested_tracker_path(chroma_dir)
    ingested = _load_ingested(tracker_path)

    # Filter to only new or modified documents
    new_docs = []
    for doc in documents:
        file_path = doc.metadata.get("file_path", "")
        if not file_path or not os.path.exists(file_path):
            continue
        current_hash = _file_md5(file_path)
        if ingested.get(file_path) == current_hash:
            continue
        new_docs.append(doc)
        ingested[file_path] = current_hash

    if not new_docs:
        log.info("无新文档需要索引（共检查 %d 个）", len(documents))
        return 0

    # Split and insert
    nodes = split_documents(new_docs, chunk_size, chunk_overlap)
    index.insert_nodes(nodes)
    log.info("增量索引完成: %d 个文档, %d 个节点", len(new_docs), len(nodes))

    # Update tracker
    _save_ingested(tracker_path, ingested)
    return len(new_docs)


def build_fresh_index(
    documents: list,
    chroma_dir: str,
    embedding_api_url: str,
    embedding_api_key: str,
    embedding_model_name: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
):
    """Build a fresh index from documents. Used when no index exists yet.

    Returns (index, embed_model, doc_count) tuple.
    """
    log.info("开始构建全新索引: %d 个文档", len(documents))
    embed_model = _get_embed_model(embedding_api_url, embedding_api_key, embedding_model_name)

    os.makedirs(chroma_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    collection = chroma_client.get_or_create_collection("financial_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    nodes = split_documents(documents, chunk_size, chunk_overlap)

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # Save tracker
    tracker_path = _ingested_tracker_path(chroma_dir)
    ingested = {}
    for doc in documents:
        file_path = doc.metadata.get("file_path", "")
        if file_path and os.path.exists(file_path):
            ingested[file_path] = _file_md5(file_path)
    _save_ingested(tracker_path, ingested)

    log.info("全新索引构建完成: %d 个文档, %d 个节点", len(documents), len(nodes))
    return index, embed_model, len(documents)
