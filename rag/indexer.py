"""使用LlamaIndex进行Chroma向量索引管理。"""

import hashlib
import json
import os

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from rag.chunker import split_documents
from utils import BaseLogger

log = BaseLogger.getLogger("rag.indexer")

# 跟踪哪些自定义模型名称已经注册到LlamaIndex枚举中。
_registered_models: set = set()


def _register_custom_model(model_name: str):
    """将非标准模型名称注册到LlamaIndex的嵌入枚举中。

    LlamaIndex的OpenAIEmbedding会根据固定的枚举验证模型名称，
    该枚举只包含OpenAI模型。DashScope（阿里）使用不同的模型名称
    （例如 ``text-embedding-v3``）。此函数通过猴子补丁修改枚举
    和查找字典，使自定义模型名称被接受。

    可以安全地多次调用——每个模型名称只会注册一次。
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
    """创建OpenAI兼容的嵌入模型。"""
    _register_custom_model(model_name)
    return OpenAIEmbedding(
        model=model_name,
        api_key=api_key,
        api_base=api_url,
    )


def _ingested_tracker_path(chroma_dir: str) -> str:
    """跟踪已摄入文件的JSON文件路径。"""
    return os.path.join(os.path.dirname(chroma_dir), ".ingested.json")


def _parents_map_path(chroma_dir: str) -> str:
    """父 chunk 映射文件路径。"""
    return os.path.join(os.path.dirname(chroma_dir), ".parents.json")


def _load_parents(path: str) -> dict:
    """加载父 chunk 映射。"""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _save_parents(path: str, data: dict):
    """保存父 chunk 映射。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def _load_ingested(tracker_path: str) -> dict:
    """加载已摄入文件跟踪器。返回 {文件路径: md5哈希}。"""
    if not os.path.exists(tracker_path):
        return {}
    with open(tracker_path, "r") as f:
        return json.load(f)


def _save_ingested(tracker_path: str, data: dict):
    """保存已摄入文件跟踪器。"""
    os.makedirs(os.path.dirname(tracker_path), exist_ok=True)
    with open(tracker_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _file_md5(file_path: str) -> str:
    """计算文件的MD5哈希值。"""
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
    """获取或创建由Chroma支持的VectorStoreIndex。

    返回 (index, embed_model) 元组。
    """
    embed_model = _get_embed_model(embedding_api_url, embedding_api_key, embedding_model_name)

    chroma_client = config.get_chroma_client()
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
    embed_model=None,
) -> int:
    """将新文档/修改后的文档添加到索引中。返回添加的文档数量。

    使用摄入跟踪器（.ingested.json）跳过已处理且内容哈希相同的文件。
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
    child_nodes, parent_map = split_documents(new_docs, chunk_size, chunk_overlap, embed_model=embed_model)
    index.insert_nodes(child_nodes)

    # 合并并保存父 chunk 映射
    pp = _parents_map_path(chroma_dir)
    all_parents = _load_parents(pp)
    all_parents.update(parent_map)
    _save_parents(pp, all_parents)

    log.info("增量索引完成: %d 个文档, %d 个子节点, %d 个父节点", len(new_docs), len(child_nodes), len(parent_map))

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
    """从文档构建新索引。当索引不存在时使用。

    返回 (index, embed_model, doc_count) 元组。
    """
    log.info("开始构建全新索引: %d 个文档", len(documents))
    embed_model = _get_embed_model(embedding_api_url, embedding_api_key, embedding_model_name)

    chroma_client = config.get_chroma_client()
    collection = chroma_client.get_or_create_collection("financial_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    child_nodes, parent_map = split_documents(documents, chunk_size, chunk_overlap, embed_model=embed_model)

    index = VectorStoreIndex(
        nodes=child_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    # 保存父 chunk 映射
    pp = _parents_map_path(chroma_dir)
    _save_parents(pp, parent_map)

    # Save tracker
    tracker_path = _ingested_tracker_path(chroma_dir)
    ingested = {}
    for doc in documents:
        file_path = doc.metadata.get("file_path", "")
        if file_path and os.path.exists(file_path):
            ingested[file_path] = _file_md5(file_path)
    _save_ingested(tracker_path, ingested)

    log.info("全新索引构建完成: %d 个文档, %d 个子节点, %d 个父节点", len(documents), len(child_nodes), len(parent_map))
    return index, embed_model, len(documents)