"""Text chunking via LlamaIndex SentenceSplitter."""

from llama_index.core.node_parser import SentenceSplitter


def create_splitter(chunk_size: int = 1024, chunk_overlap: int = 100):
    """Create a SentenceSplitter instance.

    chunk_size=1024 and chunk_overlap=100 are tuned for Chinese financial
    reports where characters are information-dense.
    """
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def split_documents(documents: list, chunk_size: int = 1024, chunk_overlap: int = 100) -> list:
    """Split documents into nodes (chunks).

    Returns list of LlamaIndex TextNode objects, each with
    text content and inherited metadata.
    """
    splitter = create_splitter(chunk_size, chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes
