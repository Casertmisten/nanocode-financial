"""Hybrid retrieval: async multi-query vector + BM25 recall, RRF fusion, optional reranking.
   混合检索：多路异步向量召回 + BM25 召回 → 合并去重 → RRF 融合 → Rerank。
   查询改写由调用方负责。
"""

import asyncio
import json
import os
import urllib.request

import jieba
import numpy as np
import config
from rank_bm25 import BM25Okapi

from llama_index.core.retrievers import VectorIndexRetriever
from utils import BaseLogger

log = BaseLogger.getLogger("rag.retriever")


def _load_parent_map() -> dict:
    """加载父 chunk 映射。"""
    pp = os.path.join(os.path.dirname(config.CHROMA_PERSIST_DIR), ".parents.json")
    if not os.path.exists(pp):
        return {}
    with open(pp, "r") as f:
        return json.load(f)


# 每路独立召回数量
_RECALL_PER_CHANNEL = 200
# RRF 融合后保留条数
_RRF_TOP = 50
# Rerank 输入条数
_RERANK_TOP = 50


def _tokenize(text: str) -> list[str]:
    """中文分词，用于 BM25。"""
    return list(jieba.cut(text))


def _bm25_recall(question: str, top_k: int) -> dict[str, float]:
    """独立 BM25 召回：从 ChromaDB 全量文档建索引，返回 top_k 个 node_id → score。"""
    client = config.get_chroma_client()
    collection = client.get_collection("financial_docs")
    all_docs = collection.get(include=["documents", "metadatas"])

    if not all_docs["ids"]:
        return {}

    corpus = all_docs["documents"]
    ids = all_docs["ids"]
    tokenized_corpus = [_tokenize(doc) for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(question))

    # 按 BM25 分数取 top_k
    top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    log.info("BM25 召回: 语料=%d 条, 返回 %d 条", len(corpus), len(top_indices))
    return {ids[i]: float(bm25_scores[i]) for i in top_indices}


async def _async_vector_recall(index, question: str, top_k: int) -> dict[str, float]:
    """异步向量召回，返回 top_k 个 node_id → score。"""
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    nodes = await retriever.aretrieve(question)
    result = {}
    for nws in nodes:
        result[nws.node.node_id] = nws.score or 0.0
    log.info("异步向量召回: query=%s, 返回 %d 条", question[:30], len(result))
    return result


def _rrf_fuse(vector_results: dict, bm25_results: dict, k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion 融合两路结果。"""
    # 向量排名
    vector_ranked = sorted(vector_results, key=vector_results.get, reverse=True)
    vector_rank = {nid: rank for rank, nid in enumerate(vector_ranked)}

    # BM25 排名
    bm25_ranked = sorted(bm25_results, key=bm25_results.get, reverse=True)
    bm25_rank = {nid: rank for rank, nid in enumerate(bm25_ranked)}

    all_ids = set(vector_results) | set(bm25_results)
    fused = {}
    for nid in all_ids:
        v_rank = vector_rank.get(nid, 9999)
        b_rank = bm25_rank.get(nid, 9999)
        fused[nid] = 0.7 / (k + v_rank) + 0.3 / (k + b_rank)
    log.info("RRF 融合: 向量=%d, BM25=%d, 融合=%d", len(vector_results), len(bm25_results), len(fused))
    return fused


def _map_to_parents(results: list[dict]) -> list[dict]:
    """将子 chunk 映射到父 chunk：按 parent_id 去重，返回父 chunk 文本。

    有 parent_id 的子 chunk → 替换为父 chunk 文本（同 parent_id 仅保留得分最高的一个）。
    无 parent_id 的（非 Markdown）→ 保持不变。
    """
    parent_ids = {r["parent_id"] for r in results if r.get("parent_id")}
    if not parent_ids:
        return results

    parent_map = _load_parent_map()
    if not parent_map:
        return results

    final = []
    seen = set()
    for r in results:
        pid = r.get("parent_id", "")
        if pid and pid in parent_map:
            if pid in seen:
                continue
            seen.add(pid)
            final.append({
                "text": parent_map[pid]["text"],
                "source": r["source"],
                "score": r["score"],
            })
        else:
            final.append(r)

    log.info("父子映射: %d 条子 chunk → %d 条（含 %d 个父 chunk）", len(results), len(final), len(seen))
    return final


def _rerank(query: str, documents: list[str], top_n: int) -> list[dict]:
    """调用本地 reranker API。成功返回排序结果，失败返回空列表。"""
    if not config.RERANK_API_URL:
        return []

    url = config.RERANK_API_URL.rstrip("/") + "/rerank"
    payload = json.dumps({
        "model": config.RERANK_MODEL or "/model",
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }).encode()

    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            results = json.loads(resp.read().decode())
            log.info("Rerank 成功: 输入=%d, 输出=%d", len(documents), len(results))
            return results
    except Exception:
        log.warning("Reranker 调用失败，降级使用混合分数", exc_info=True)
        return []


async def async_retrieve(index, question: str, top_k: int = 20, file_paths: list[str] | None = None, queries: list[str] | None = None) -> list[dict]:
    """异步并行混合检索 + RRF 融合 + Rerank。

    1. 每路向量召回各取 top 200，异步并行；合并去重取最高分（同一 node 取 max）
    2. BM25 召回 top 200
    3. 向量 + BM25 结果 RRF 融合，取 top 50
    4. Reranker 精排输出 top_k（默认 20）；无 reranker 时直接用融合分数取 top_k

    file_paths: 若提供，仅返回这些文件路径下的结果（匹配 ChromaDB file_name 元数据）。
    queries: 若提供，异步并行向量召回（含原问题）；否则仅用 question 单路召回。
    """
    log.info("检索开始: question=%s, top_k=%d, queries=%d", question[:50], top_k, len(queries) if queries else 1)

    vec_queries = queries or [question]

    # 异步并行向量召回
    tasks = [_async_vector_recall(index, q, _RECALL_PER_CHANNEL) for q in vec_queries]
    recall_results = await asyncio.gather(*tasks)

    all_vector_results: dict[str, float] = {}
    for vr in recall_results:
        for nid, score in vr.items():
            all_vector_results[nid] = max(all_vector_results.get(nid, 0), score)

    bm25_results = _bm25_recall(question, _RECALL_PER_CHANNEL)

    if not all_vector_results and not bm25_results:
        log.info("检索结果为空")
        return []

    # RRF 融合，取 top 50
    fused = _rrf_fuse(all_vector_results, bm25_results)
    sorted_ids = sorted(fused, key=fused.get, reverse=True)[:_RRF_TOP]

    # 获取 node 文本（从 ChromaDB）
    client = config.get_chroma_client()
    collection = client.get_collection("financial_docs")

    # 有文件过滤时需要获取全部元数据；无过滤时取 top 50
    metadata_limit = len(sorted_ids) if file_paths else _RRF_TOP
    stored = collection.get(ids=sorted_ids[:metadata_limit], include=["documents", "metadatas"])

    id_to_text = {}
    id_to_source = {}
    id_to_parent_id = {}
    for nid, doc, meta in zip(stored["ids"], stored["documents"], stored["metadatas"]):
        id_to_text[nid] = doc
        id_to_source[nid] = meta.get("file_name", "") if meta else ""
        id_to_parent_id[nid] = meta.get("parent_id", "") if meta else ""

    # 按文件路径过滤（比较不含扩展名的文件名，兼容 PDF→MD 转换）
    if file_paths:
        allowed_stems = {os.path.splitext(os.path.basename(p))[0] for p in file_paths}
        filtered_ids = [
            nid for nid in sorted_ids
            if nid in id_to_source
            and os.path.splitext(id_to_source[nid])[0] in allowed_stems
        ]
        if filtered_ids:
            sorted_ids = filtered_ids
            log.info("文件过滤: 候选 %d, 匹配 %d, 允许的文件名=%s", metadata_limit, len(filtered_ids), allowed_stems)

    # Rerank：取 top 50 进入精排
    candidate_ids = sorted_ids[:_RERANK_TOP]
    candidate_docs = [id_to_text.get(nid, "") for nid in candidate_ids]
    rerank_results = _rerank(question, candidate_docs, top_n=top_k)

    if rerank_results:
        results = []
        for item in rerank_results:
            idx = item["index"]
            if idx < len(candidate_ids):
                nid = candidate_ids[idx]
                results.append({
                    "text": id_to_text.get(nid, ""),
                    "source": id_to_source.get(nid, "unknown"),
                    "score": round(item.get("score", 0), 4),
                    "parent_id": id_to_parent_id.get(nid, ""),
                })
        log.info("Rerank 检索完成: %d 条子 chunk", len(results))
        return _map_to_parents(results)

    # 无 reranker，用融合分数
    results = []
    for nid in sorted_ids[:top_k]:
        results.append({
            "text": id_to_text.get(nid, ""),
            "source": id_to_source.get(nid, "unknown"),
            "score": round(fused[nid], 4),
            "parent_id": id_to_parent_id.get(nid, ""),
        })
    log.info("混合检索完成（无 rerank）: %d 条子 chunk", len(results))
    return _map_to_parents(results)


def retrieve(index, question: str, top_k: int = 5, file_paths: list[str] | None = None, queries: list[str] | None = None) -> list[dict]:
    """同步封装：在已有事件循环中复用，否则新建。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 已在异步上下文中（如 FastAPI），用线程桥接
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, async_retrieve(index, question, top_k, file_paths, queries)).result()
    else:
        return asyncio.run(async_retrieve(index, question, top_k, file_paths, queries))


def format_results(results: list[dict]) -> str:
    """Format retrieval results into a readable string for LLM consumption."""
    if not results:
        return "未在本地知识库中找到相关信息。"

    parts = [f"从本地知识库检索到 {len(results)} 条相关信息：\n"]
    for r in results:
        parts.append(f"[来源: {r['source']}，相关度: {r['score']:.4f}]")
        parts.append(f"{r['text']}\n")
    return "\n".join(parts)
