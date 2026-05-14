"""Text chunking — heading-aware + semantic for Markdown, SentenceSplitter fallback."""

from llama_index.core.node_parser import MarkdownNodeParser, SemanticSplitterNodeParser, SentenceSplitter

# 超过此字符数的标题段落会二次拆分
_MAX_CHUNK_SIZE = 1500


def _needs_sub_split(nodes: list) -> bool:
    """检查是否有节点超过最大块大小。"""
    return any(len(n.text) > _MAX_CHUNK_SIZE for n in nodes)


def split_documents(
    documents: list,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    embed_model=None,
) -> list:
    """按文件类型选择分块策略。

    Markdown → MarkdownNodeParser（按标题层级切分）
              → SemanticSplitterNodeParser（同层级内语义分块，需 embed_model）
              → SentenceSplitter 回退（无 embed_model 或单节点超长时）
    其他文件 → SentenceSplitter（按句子切分）
    """
    md_parser = MarkdownNodeParser()
    sentence_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    md_docs = [d for d in documents if _is_markdown(d)]
    other_docs = [d for d in documents if not _is_markdown(d)]

    nodes = []

    # Markdown: 标题层级分块 + 语义二次分块
    if md_docs:
        md_nodes = md_parser.get_nodes_from_documents(md_docs)

        if embed_model is not None:
            # 同层级内按语义相似度二次切分
            semantic_parser = SemanticSplitterNodeParser(
                embed_model=embed_model,
                buffer_size=1,
                breakpoint_percentile_threshold=95,
            )
            md_nodes = semantic_parser.build_semantic_nodes_from_nodes(md_nodes)
        elif _needs_sub_split(md_nodes):
            # 无嵌入模型时回退到句子切分
            md_nodes = sentence_splitter(md_nodes)

        # 语义分块后仍可能有超长节点，兜底拆分
        if _needs_sub_split(md_nodes):
            md_nodes = sentence_splitter(md_nodes)

        nodes.extend(md_nodes)

    # 非 Markdown: 原有策略
    if other_docs:
        nodes.extend(
            sentence_splitter.get_nodes_from_documents(other_docs)
        )

    return nodes


def _is_markdown(doc) -> bool:
    """判断文档是否为 Markdown 格式。"""
    ext = doc.metadata.get("file_type", "") or doc.metadata.get("filename", "")
    return ext.lower() in (".md", ".markdown")
