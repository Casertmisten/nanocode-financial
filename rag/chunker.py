"""Text chunking — parent-child for Markdown, SentenceSplitter fallback for others.
Markdown: 先按标题层级切出大 chunk（父），再用句子分割切成小 chunk（子），用于「小 chunk 检索、大 chunk 生成」。
"""

import config
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter, SemanticSplitterNodeParser

# 父 chunk 超长时的兜底字符阈值
_MAX_PARENT_CHARS = 3000


def _build_parent_map(parent_nodes: list) -> dict:
    """将父节点列表映射为 {node_id: {text, metadata}}。"""
    parent_map = {}
    for node in parent_nodes:
        parent_map[node.node_id] = {
            "text": node.text,
            "metadata": dict(node.metadata),
        }
    return parent_map


def split_documents(
    documents: list,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    embed_model=None,
) -> tuple[list, dict]:
    """按文件类型选择分块策略。

    Markdown: 标题层级 → 父 chunk → 句子分割 → 子 chunk（带 parent_id）
    其他文件: SentenceSplitter 常规分块（无父子关系）

    Returns:
        (child_nodes, parent_map) 元组
        - child_nodes: 用于向量检索的小 chunk
        - parent_map: {parent_id: {text, metadata}}，用于生成时提供上下文
    """
    md_parser = MarkdownNodeParser()

    # 父 chunk 兜底分割器（超大段落进一步拆分）
    parent_splitter = SentenceSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE, #  1000
        chunk_overlap=config.PARENT_CHUNK_OVERLAP, #  100
    )

    # 子 chunk 分割器（语义块）
    if embed_model is None:
        raise ValueError("SemanticSplitterNodeParser 需要 embed_model，请传入嵌入模型")
    child_splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=1,
        breakpoint_percentile_threshold=88,
        chunk_size=config.CHILD_CHUNK_SIZE, # 200
        chunk_overlap=config.CHILD_CHUNK_OVERLAP, # 20
    )

    # 非 Markdown 分块器
    other_splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=128,
        separator=" ",
        paragraph_separator="\n\n\n",
        secondary_chunking_regex="[^,.;]+[,.;]?",
    )

    md_docs = [d for d in documents if _is_markdown(d)]
    other_docs = [d for d in documents if not _is_markdown(d)]

    all_children = []
    parent_map = {}

    # ── Markdown: 父子切割 ──
    if md_docs:
        # 1. 按标题层级切出父 chunk
        parent_nodes = md_parser.get_nodes_from_documents(md_docs)

        # 2. 超长父 chunk 兜底拆分
        if any(len(n.text) > _MAX_PARENT_CHARS for n in parent_nodes):
            parent_nodes = parent_splitter(parent_nodes)

        # 3. 建立父 chunk 映射
        parent_map = _build_parent_map(parent_nodes)

        # 4. 将每个父 chunk 句子分割成子 chunk
        for parent_node in parent_nodes:
            children = child_splitter([parent_node])
            for child in children:
                child.metadata["parent_id"] = parent_node.node_id
            all_children.extend(children)

    # ── 非 Markdown: 常规分块（无父子关系）──
    if other_docs:
        all_children.extend(other_splitter.get_nodes_from_documents(other_docs))

    return all_children, parent_map


def _is_markdown(doc) -> bool:
    """判断文档是否为 Markdown 格式。"""
    ext = doc.metadata.get("file_type", "") or doc.metadata.get("filename", "")
    return ext.lower() in (".md", ".markdown")
