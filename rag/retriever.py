"""Hybrid retrieval: query rewrite, parallel vector + BM25 recall, RRF fusion, optional reranking.
   混合检索：查询改写 → 多路并行向量召回 + BM25 召回 → 合并去重 → RRF 融合 → Rerank。
"""

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

# 每路独立召回数量
_RECALL_PER_CHANNEL = 40


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


def _vector_recall(index, question: str, top_k: int) -> dict[str, float]:
    """独立向量召回，返回 top_k 个 node_id → score。"""
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    nodes = retriever.retrieve(question)
    result = {}
    for nws in nodes:
        result[nws.node.node_id] = nws.score or 0.0
    log.info("向量召回: 返回 %d 条", len(result))
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


def retrieve(index, question: str, top_k: int = 5, file_paths: list[str] | None = None) -> list[dict]:
    """查询改写 + 并行混合检索 + RRF 融合 + Rerank。

    1. LLM 将问题改写为 3 个变体
    2. 原问题 + 变体并行向量召回，BM25 仅用原问题
    3. 多路结果合并去重，RRF 融合
    4. Reranker 精排输出 top_k

    file_paths: 若提供，仅返回这些文件路径下的结果（匹配 ChromaDB file_name 元数据）。
    """
    log.info("检索开始: question=%s, top_k=%d", question[:50], top_k)

    # 查询改写
    queries = [question]
    try:
        import llm
        variants = llm.rewrite_queries(question)
        queries.extend(variants)
    except Exception:
        log.warning("查询改写失败，使用原始问题", exc_info=True)
    log.info("检索查询: %d 个（原问题 + %d 变体）", len(queries), len(queries) - 1)

    # 多路并行向量召回 + BM25 召回
    all_vector_results: dict[str, float] = {}
    for q in queries:
        vr = _vector_recall(index, q, _RECALL_PER_CHANNEL)
        for nid, score in vr.items():
            # 同一 node 取最高分
            all_vector_results[nid] = max(all_vector_results.get(nid, 0), score)
    bm25_results = _bm25_recall(question, _RECALL_PER_CHANNEL)

    if not all_vector_results and not bm25_results:
        log.info("检索结果为空")
        return []

    # RRF 融合
    fused = _rrf_fuse(all_vector_results, bm25_results)
    sorted_ids = sorted(fused, key=fused.get, reverse=True)

    # 获取 node 文本（从 ChromaDB）
    client = config.get_chroma_client()
    collection = client.get_collection("financial_docs")

    # 有文件过滤时需要获取更多元数据；无过滤时只需 top 20
    metadata_limit = len(sorted_ids) if file_paths else 20
    stored = collection.get(ids=sorted_ids[:metadata_limit], include=["documents", "metadatas"])

    id_to_text = {}
    id_to_source = {}
    for nid, doc, meta in zip(stored["ids"], stored["documents"], stored["metadatas"]):
        id_to_text[nid] = doc
        id_to_source[nid] = meta.get("file_name", "") if meta else ""

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

    # Rerank
    candidate_ids = sorted_ids[:20]
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
                })
        log.info("Rerank 检索完成: %d 条结果", len(results))
        return results

    # 无 reranker，用融合分数
    results = []
    for nid in sorted_ids[:top_k]:
        results.append({
            "text": id_to_text.get(nid, ""),
            "source": id_to_source.get(nid, "unknown"),
            "score": round(fused[nid], 4),
        })
    log.info("混合检索完成（无 rerank）: %d 条结果", len(results))
    return results


def format_results(results: list[dict]) -> str:
    """Format retrieval results into a readable string for LLM consumption."""
    if not results:
        return "未在本地知识库中找到相关信息。"

    parts = [f"从本地知识库检索到 {len(results)} 条相关信息：\n"]
    for i, r in enumerate(results, 1):
        parts.append(f"[来源: {r['source']}，相关度: {r['score']:.4f}]")
        parts.append(f"{r['text']}\n")
    return "\n".join(parts)
