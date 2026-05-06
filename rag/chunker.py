"""Text chunking — heading-aware for Markdown, SentenceSplitter fallback."""

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

# 超过此字符数的标题段落会二次拆分
_MAX_CHUNK_SIZE = 1500


def _needs_sub_split(nodes: list) -> bool:
    """检查是否有节点超过最大块大小。"""
    return any(len(n.text) > _MAX_CHUNK_SIZE for n in nodes)


def split_documents(
    documents: list,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
) -> list:
    """按文件类型选择分块策略。

    Markdown → MarkdownNodeParser（按标题层级切分，再对超长段落二次拆分）
    其他文件 → SentenceSplitter
    """
    md_parser = MarkdownNodeParser()
    sentence_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    md_docs = [d for d in documents if _is_markdown(d)]
    other_docs = [d for d in documents if not _is_markdown(d)]

    nodes = []

    # Markdown: 标题层级分块
    if md_docs:
        md_nodes = md_parser.get_nodes_from_documents(md_docs)
        # 对超长段落用 SentenceSplitter 二次拆分
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
