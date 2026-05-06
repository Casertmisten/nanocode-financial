"""Similarity-based retrieval from Chroma vector index."""


def retrieve(index, question: str, top_k: int = 5) -> list[dict]:
    """Search the index for relevant document chunks.

    Returns list of dicts:
        [{"text": "...", "source": "filename", "score": 0.85}, ...]
    sorted by relevance (highest first).
    """
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes_with_scores = retriever.retrieve(question)

    results = []
    for nws in nodes_with_scores:
        results.append({
            "text": nws.node.text,
            "source": nws.node.metadata.get("file_name", "unknown"),
            "score": round(nws.score, 4) if nws.score is not None else 0.0,
        })
    return results


def format_results(results: list[dict]) -> str:
    """Format retrieval results into a readable string for LLM consumption."""
    if not results:
        return "未在本地知识库中找到相关信息。"

    parts = [f"从本地知识库检索到 {len(results)} 条相关信息：\n"]
    for i, r in enumerate(results, 1):
        parts.append(f"[来源: {r['source']}，相关度: {r['score']:.2f}]")
        parts.append(f"{r['text']}\n")
    return "\n".join(parts)
