# RAG 金融问答模块设计

## 背景

基于 nanocode（单文件 Python agentic coding assistant）构建个人金融智能分析助手。本文档覆盖 V1 的第一个功能：RAG 金融问答。后续 Deep Research 和每日新闻总结将在独立的设计文档中规划。

## 决策记录

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 框架 | LlamaIndex | 专门为 RAG 设计，内置文档加载、分块、索引 |
| 向量数据库 | Chroma | 轻量级嵌入式，无需启动服务，适合个人项目 |
| Embedding | OpenAI 兼容 API | 灵活切换本地部署或第三方服务 |
| 集成方式 | 作为工具注册到 agentic loop | 用户对话中自然触发 RAG |
| LLM 合成 | 自写 context + prompt，走现有 call_api | 保持 LLM 调用链路统一 |
| 代码结构 | 拆分多文件模块 | 加入 RAG 后代码量增加，单文件不可维护 |
| 数据格式 | PDF 研报 + 文本/Markdown | 初始批量导入，后续爬虫定期更新 |

## 项目结构

```
nanocode-financial/
├── nanocode.py              # 主程序入口（agentic loop）
├── config.py                # 配置管理（API、Embedding、路径等）
├── tools.py                 # 现有 6 个工具 + RAG 工具
├── rag/                     # RAG 模块
│   ├── __init__.py          # 导出 ingest, query
│   ├── loader.py            # 文档加载（PDF、Markdown、纯文本）
│   ├── chunker.py           # 文本分块策略
│   ├── indexer.py           # 索引构建与管理（Chroma + LlamaIndex）
│   └── retriever.py         # 检索逻辑（top-k 相似度搜索）
├── data/
│   ├── documents/           # 原始文档（PDF/MD/TXT）
│   └── chroma_db/           # Chroma 向量数据库持久化
├── prompts/
│   └── rag_system.txt       # RAG 问答的系统提示词
├── .env                     # API 配置
├── pyproject.toml           # uv 项目配置
└── docs/
    └── 金融助手项目PRD.md
```

## 数据流

### 离线：数据摄入

```
用户文档（PDF/MD/TXT） → loader.py → chunker.py → indexer.py → Chroma DB
```

- **PDF 解析**：LlamaIndex `SimpleDirectoryReader` + PyMuPDF
- **分块策略**：LlamaIndex `SentenceSplitter`，chunk_size=512, overlap=50（中文研报经验值）
- **Embedding**：OpenAI 兼容 API，配置在 `.env`
- **Chroma 持久化**：`data/chroma_db/`，增量更新（只处理新增/修改文档）

### 在线：查询

```
用户问题 → agentic loop → rag_query 工具 → retriever.py → 拼装 context → call_api() → 输出回答+引用
```

1. 用户输入金融相关问题
2. LLM 判断需要调用 `rag_query` 工具
3. `retriever.py` 对问题做 Embedding，在 Chroma 中 top-k 检索
4. Python 层拼装 context + question 为完整 prompt
5. 通过 `call_api()` 发送给 LLM
6. 输出回答，附带引用来源

## 核心接口

```python
# rag/__init__.py

def ingest(doc_path: str) -> int:
    """将文档目录下的文件加载、分块、存入 Chroma。返回处理文档数。"""

def query(question: str, top_k: int = 5) -> list[dict]:
    """检索相关文档片段。返回 [{"text": "...", "source": "文件名", "score": 0.85}, ...]"""
```

## 工具注册

在 `TOOLS` 字典中新增两个工具：

```python
"rag_query": (
    "Search local financial knowledge base for relevant information. Use when user asks about financial topics.",
    {"question": "string", "top_k": "number?"},
    rag_query_tool,
),
"rag_ingest": (
    "Import documents into the financial knowledge base. Supports PDF and Markdown.",
    {"path": "string"},
    rag_ingest_tool,
),
```

## 增量索引策略

- Chroma 存储已有文档的 hash（在 metadata 中记录）
- `ingest()` 对比 hash，只处理新增或修改过的文档
- 避免每次全量重建索引

## 依赖新增

```
llama-index-core
llama-index-readers-file
llama-index-vector-stores-chroma
chromadb
pymupdf
python-dotenv（已有）
```

## 系统提示词

```
你是一个金融分析助手。当用户询问金融、股票、市场相关问题时，
使用 rag_query 工具从本地知识库检索相关信息来回答。
当用户要求导入文档时，使用 rag_ingest 工具。
回答时请附带引用来源。
```

## 不做的事

- 不做实时交易系统
- 不做量化回测
- 不做用户资产管理
- Deep Research 和每日新闻总结在后续设计文档中规划
