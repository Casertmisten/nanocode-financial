"""Configuration management - API, Embedding, and path settings."""

import os

import dotenv

dotenv.load_dotenv()

# --- LLM API ---
def _normalize_chat_url(url: str) -> str:
    """规范化 LLM Chat Completions endpoint。

    兼容两种配置写法：
    - base URL（如 .../v1）：自动补全为 .../v1/chat/completions
    - 完整 endpoint（已含 /chat/completions）：原样使用（去除多余尾斜杠）
    空值原样返回，交由调用层报错。
    """
    if not url:
        return url
    stripped = url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


API_URL = _normalize_chat_url(os.environ.get("ALI_API_URL", ""))
API_KEY = os.environ.get("ALI_API_KEY", "")
MODEL = os.environ.get("ALI_MODEL", "deepseek-v4-flash")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "8192"))
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))

# --- Agentic Loop ---
# Agentic Loop 最大执行轮次
AGENT_MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "10"))
# 工具批处理大小
TOOL_BATCH_SIZE = int(os.environ.get("TOOL_BATCH_SIZE", "5"))

# --- Embedding API ---
EMBEDDING_API_URL = os.environ.get(
    "EMBEDDING_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", API_KEY)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")

# --- Reranker API ---
RERANK_API_URL = os.environ.get("RERANK_API_URL", "")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "")

# --- RAG Chunking ---
PARENT_CHUNK_SIZE = int(os.environ.get("PARENT_CHUNK_SIZE", "1000"))
PARENT_CHUNK_OVERLAP = int(os.environ.get("PARENT_CHUNK_OVERLAP", "100"))
CHILD_CHUNK_SIZE = int(os.environ.get("CHILD_CHUNK_SIZE", "200"))
CHILD_CHUNK_OVERLAP = int(os.environ.get("CHILD_CHUNK_OVERLAP", "20"))

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

# --- MinerU PDF 解析 ---
MINERU_ACC_URL = os.environ.get("MinerU_ACC_URL", "https://mineru.net/api/v4/extract/task")
MINERU_ACC_KEY = os.environ.get("MinerU_ACC_KEY", "")
MINERU_MODEL_VERSION = os.environ.get("MinerU_MODEL_VERSION", "vlm")
MINERU_LIGHT_URL = os.environ.get("MinerU_LIGHT_URL", "https://mineru.net/api/v1/agent/parse/url")
MINERU_LIGHT_FILE_URL = "https://mineru.net/api/v1/agent/parse/file"
MINERU_ACC_FILE_URL = "https://mineru.net/api/v4/file-urls/batch"
# MinerU 解析超时（秒）和轮询间隔（秒）
MINERU_TIMEOUT = int(os.environ.get("MINERU_TIMEOUT", "300"))
MINERU_POLL_INTERVAL = int(os.environ.get("MINERU_POLL_INTERVAL", "3"))
# 轻量级解析限制
MINERU_LIGHT_MAX_SIZE = 10 * 1024 * 1024  # 10MB
MINERU_LIGHT_MAX_PAGES = 20

# --- AKShare 数据源 ---
AKSHARE_CACHE_TTL = int(os.environ.get("AKSHARE_CACHE_TTL", "3600"))

# --- Deep Research ---
# 最大子任务数
RESEARCH_MAX_SUBTASKS = 8        
# Executor 最大工具调用轮次
RESEARCH_EXECUTOR_MAX_TURNS = 8  
# 单个 Executor 超时（秒）
RESEARCH_EXECUTOR_TIMEOUT = 120  

# --- Memory ---
# 用户画像路径
MEMORY_PROFILE_PATH = os.path.join(BASE_DIR, "data", "profile.json")
# 用户画像候选池路径
MEMORY_CANDIDATES_PATH = os.path.join(BASE_DIR, "data", "profile_candidates.json")
# 会话记忆集合名称
MEMORY_SESSION_COLLECTION = "session_memory"
# 会话记忆最大 token 数量
MEMORY_SESSION_MAX_TOKENS = int(os.environ.get("MEMORY_SESSION_MAX_TOKENS", "6000"))
# 会话记忆压缩轮次（超过预算时压缩最早的消息，最多压缩多少轮）
MEMORY_COMPRESS_ROUNDS = 3
# 记忆时间衰减 lambda（用于计算记忆重要性时，时间越久远的记忆衰减越多）
MEMORY_TIME_DECAY_LAMBDA = float(os.environ.get("MEMORY_TIME_DECAY_LAMBDA", "0.05"))
# 会话记忆跨会话 top-k 选择（用于从所有会话中选择最相关的 k 条记忆）
MEMORY_CROSS_SESSION_TOP_K = int(os.environ.get("MEMORY_CROSS_SESSION_TOP_K", "5"))
# 会话记忆跨会话注入 k 条记忆（用于从所有会话中注入 k 条记忆）
MEMORY_CROSS_SESSION_INJECT_K = int(os.environ.get("MEMORY_CROSS_SESSION_INJECT_K", "3"))
# 会话记忆用户画像更新间隔（秒）
MEMORY_PROFILE_UPDATE_INTERVAL = int(os.environ.get("MEMORY_PROFILE_UPDATE_INTERVAL", "3"))
# 会话记忆最小保留轮次（用于控制记忆的保留时间）
MEMORY_MIN_KEEP_ROUNDS = 5

# --- 意图识别 ---
# 是否启用 LLM 路由层（正则层始终启用）
INTENT_USE_LLM = os.environ.get("INTENT_USE_LLM", "true").lower() in ("1", "true", "yes")

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

# --- langfuse ---
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "https://api.langfuse.com")
# 是否启用 Langfuse 监控（关闭后所有追踪/上报变为 no-op，不影响业务逻辑）
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "true").lower() in ("1", "true", "yes")
# Langfuse SDK 读取 LANGFUSE_HOST 环境变量
os.environ.setdefault("LANGFUSE_HOST", LANGFUSE_BASE_URL)
# Langfuse SDK 原生开关：读取 LANGFUSE_TRACING_ENABLED 环境变量，为 false 时
# 客户端整体进入 no-op 模式（start_observation 返回空对象，所有 update 自动跳过），
# 因此 llm.py / rag/retriever.py 中的追踪代码无需任何改动。
os.environ["LANGFUSE_TRACING_ENABLED"] = "true" if LANGFUSE_ENABLED else "false"
