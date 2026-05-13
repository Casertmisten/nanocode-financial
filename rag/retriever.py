"""Hybrid retrieval: parallel vector + BM25 recall, RRF fusion, optional reranking."""

import json
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


def retrieve(index, question: str, top_k: int = 5) -> list[dict]:
    """并行混合检索 + RRF 融合 + Rerank。

    1. 向量、BM25 各独立召回 40 条（并行）
    2. RRF 融合两路结果
    3. Reranker 精排输出 top_k
    """
    log.info("检索开始: question=%s, top_k=%d", question[:50], top_k)

    # 两路并行召回
    vector_results = _vector_recall(index, question, _RECALL_PER_CHANNEL)
    bm25_results = _bm25_recall(question, _RECALL_PER_CHANNEL)

    if not vector_results and not bm25_results:
        log.info("检索结果为空")
        return []

    # RRF 融合
    fused = _rrf_fuse(vector_results, bm25_results)
    sorted_ids = sorted(fused, key=fused.get, reverse=True)

    # 获取 node 文本（从 ChromaDB）
    client = config.get_chroma_client()
    collection = client.get_collection("financial_docs")
    stored = collection.get(ids=sorted_ids[:20], include=["documents", "metadatas"])

    id_to_text = {}
    id_to_source = {}
    for nid, doc, meta in zip(stored["ids"], stored["documents"], stored["metadatas"]):
        id_to_text[nid] = doc
        id_to_source[nid] = meta.get("file_name", "unknown") if meta else "unknown"

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
