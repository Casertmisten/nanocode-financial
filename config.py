"""Configuration management - API, Embedding, and path settings."""

import os

import dotenv

dotenv.load_dotenv()

# --- LLM API ---
API_URL = os.environ.get("ALI_API_URL", "")
API_KEY = os.environ.get("ALI_API_KEY", "")
MODEL = os.environ.get("ALI_MODEL", "qwen3.5-flash")

# --- Embedding API ---
EMBEDDING_API_URL = os.environ.get(
    "EMBEDDING_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", API_KEY)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")

# --- Reranker API ---
RERANK_API_URL = os.environ.get("RERANK_API_URL", "")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "")

# --- RAG Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.environ.get(
    "DOCUMENTS_DIR", os.path.join(BASE_DIR, "data", "documents")
)
CHROMA_PERSIST_DIR = os.environ.get(
    "CHROMA_PERSIST_DIR", os.path.join(BASE_DIR, "data", "chroma_db")
)
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8001"))


def get_chroma_client():
    """获取 ChromaDB HttpClient（连接独立服务）。"""
    import chromadb
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# --- AKShare 数据源 ---
AKSHARE_CACHE_TTL = int(os.environ.get("AKSHARE_CACHE_TTL", "3600"))

# --- Web 服务 ---
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "data", "app.db"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "data", "uploads"))

# --- ANSI colors ---
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, YELLOW, RED = (
    "\033[34m",
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
)
