# Nanocode Financial - 个人金融智能分析助手

基于国内大模型（通义千问 / DeepSeek / 智谱 GLM）的个人金融分析助手，支持 RAG 问答、专业模式工作流（个股投资决策、行业轮动、财报深度分析）、A 股实时数据查询、三层记忆系统和意图识别，提供 CLI 和 Web 两种交互方式。

## 功能特性

### 专业模式 & 意图识别
- 两层意图路由：正则快速匹配（零成本）→ LLM 精准分类（兜底）
- 自动识别投资决策、行业轮动、财报分析等意图，路由到对应工作流
- 未命中专业意图时自动回退到通用 RAG 对话

### 个股投资决策
- 多步工作流：行业分析 → 基本面评估 → 技术面分析 → 综合投资建议
- 基于 AKShare 获取实时行情、财务指标、历史数据
- 生成结构化投资报告，包含评分、风险提示和操作建议

### 行业轮动与机会发现
- 自动扫描全市场行业板块资金流向与涨跌幅
- 结合政策面、资金面、技术面多维度分析
- 输出行业热度排行与轮动建议

### RAG 金融问答
- 支持上传 PDF / Markdown / 纯文本文档，自动解析并向量化入库
- PDF 解析接入 MinerU，轻量级与精准两种模式自动切换
- Markdown 标题感知分块 + 语义分割，保证上下文完整性
- 混合检索：多路向量召回 + BM25 稀疏召回 + RRF 融合 + Rerank 精排
- LLM 查询改写，提升检索召回率

### 财报深度分析（FRA）
- Map-Reduce 架构：3 大维度、33 个子问题模板
- 自动从知识库中检索相关内容，逐维度分析后生成结构化投资报告
- 报告包含财务指标、资产负债、行业与经营、综合评价四个章节

### A 股数据与新闻
- 基于 AKShare 获取实时行情、历史 K 线、财务报表等数据
- 聚合 6 大新闻源的市场与个股资讯
- 新闻情感分析、关键词提取、重要性评估、自动分类

### 交互方式
- **CLI 模式**：终端 REPL，支持 16 种工具调用，ANSI 彩色输出
- **Web 模式**：FastAPI + SSE 流式输出，五页面前端（对话、知识库、深度研究、数据统计、设置）

### 三层记忆系统
- **L1 用户画像**：JSON 存储，Markdown 注入，候选池累积 3 轮后 LLM 自动合并更新
- **L2 跨会话记忆**：LLM 摘要 + 向量化存入 ChromaDB，RAG 检索 + 时间衰减权重注入
- **L3 会话压缩**：Token 预算控制，超限时自动压缩为摘要标记（保留原始消息，前端显示压缩提示，传模型时以摘要为起点）

### 错误恢复
- **输出截断**：max_tokens 用完时自动翻倍续写（上限 4 倍）
- **上下文超限**：HTTP 400 时自动触发压缩重试
- **限流重试**：HTTP 429 间隔 2 秒重试，最多 5 次
- **认证/权限**：401 提示 API Key 错误，403 提示模型欠费，立即终止

## 技术架构

```
┌─────────────────────────────────────────────────────┐
│                   前端 (HTML/JS/CSS)                  │
│     chat / knowledge / research / stats / settings   │
├─────────────────────────────────────────────────────┤
│                FastAPI / CLI 入口                     │
│           SSE 流式 / REPL agentic loop                │
├──────────┬──────────┬──────────┬──────────┬─────────┤
│  LLM 调用  │  RAG 系统  │  数据源   │  意图识别   │ 工具注册 │
│ OpenAI 兼容 │ 混合检索   │ AKShare  │ 正则+LLM  │ 16 个工具│
│ 流式/同步   │ ChromaDB  │ 行情/新闻 │ 两层路由   │ rag/stock│
│ 查询改写    │ LlamaIndex│          │           │ news/bash│
├──────────┼──────────┼──────────┼──────────┴─────────┤
│ 三层记忆   │ Deep      │  财报分析  │  专业工作流        │
│ L1 画像    │ Research  │ Map-Reduce │  个股投资决策      │
│ L2 跨会话  │ 3-Agent   │  3×33     │  行业轮动发现      │
│ L3 压缩    │ 协作研究   │          │                   │
├──────────┴──────────┴──────────┴──────────────────┤
│              SQLite + ChromaDB + 文件存储              │
│    会话/消息/文档/设置/报告/记忆/股票列表缓存/Token用量    │
└─────────────────────────────────────────────────────┘
```

## 项目结构

```
nanocode-financial/
├── cli.py                       # CLI 入口（REPL + agentic loop）
├── config.py                    # 配置加载（API 密钥、路径、参数）
├── db.py                        # 异步 SQLite CRUD
├── llm.py                       # LLM 调用层（同步/异步流式）
├── tools.py                     # 工具注册表（16 个工具）
├── web.py                       # FastAPI 应用入口
│
├── intent/                      # 意图识别
│   ├── __init__.py              # 统一接口：正则优先 → LLM 兜底
│   ├── schemas.py               # IntentResult 数据模型
│   ├── regex_router.py          # 正则快速路由
│   └── llm_router.py            # LLM 精准分类
│
├── stock_investment/             # 个股投资决策工作流
│   ├── __init__.py              # 入口
│   ├── pipeline.py              # 多步分析流水线
│   └── prompts.py               # 提示词模板
│
├── sector_rotation/              # 行业轮动与机会发现
│   ├── __init__.py              # 入口
│   ├── pipeline.py              # 行业扫描 + 轮动分析
│   └── prompts.py               # 提示词模板
│
├── rag/                         # RAG 系统
│   ├── indexer.py               # ChromaDB 向量索引（增量入库）
│   ├── retriever.py             # 混合检索（多路召回 + RRF + Rerank）
│   ├── loader.py                # 文档加载
│   ├── chunker.py               # Markdown 标题感知 + 语义分块
│   └── pdf_parser.py            # MinerU PDF 解析
│
├── api/                         # API 路由
│   ├── chat.py                  # SSE 流式对话（含错误恢复）
│   ├── sessions.py              # 会话管理
│   ├── documents.py             # 文档上传与管理
│   ├── settings.py              # 设置管理
│   └── fra.py                   # 财报分析 SSE 接口
│
├── datasource/                  # 数据源
│   ├── stock.py / news.py       # 股票 & 新闻数据接口
│   ├── _akshare_stock.py        # AKShare 股票数据实现
│   ├── _akshare_news.py         # AKShare 新闻数据实现
│   └── _helpers.py              # 类型转换、市场识别等工具
│
├── financial_report_analysis/   # 财报深度分析
│   ├── pipeline.py              # Map-Reduce 流水线
│   └── template.py              # 3 维度 33 子问题模板
│
├── deep_research/               # 开放性深度研究
│   ├── __init__.py              # 入口：plan → execute → generate
│   ├── plan.py                  # Planner：问题拆解为子任务
│   ├── executor.py              # Executor：并行多源检索
│   ├── generator.py             # Generator：综合生成报告
│   ├── schemas.py               # 数据模型
│   └── prompts.py               # 提示词模板
│
├── memory/                      # 三层记忆系统
│   ├── __init__.py              # 统一接口：inject / save
│   ├── profile.py               # L1 用户画像（JSON + Markdown）
│   ├── session_memory.py        # L2 跨会话记忆（摘要 + 向量化 + RAG）
│   ├── context.py               # L3 会话压缩（标记模式，保留原始消息）
│   └── prompts.py               # 记忆相关提示词模板
│
├── prompts/                     # 提示词模板
├── frontend/                    # Web 前端（HTML/CSS/JS）
│   ├── chat.html                # 对话页面
│   ├── knowledge.html           # 知识库页面
│   ├── research.html            # 深度研究页面
│   ├── stats.html               # 数据统计页面
│   ├── settings.html            # 设置页面
│   ├── css/                     # 样式文件
│   └── js/
│       └── api.js               # 统一 API 工具层（SSE 流式）
├── utils/                       # 日志 & RAG 评估工具
└── docs/                        # 设计文档 & PRD
```

## 快速开始

### 环境要求

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) 包管理器
- 大模型 API Key（阿里 DashScope / 智谱 GLM / DeepSeek 任选其一）
- （可选）MinerU API Key，用于精准 PDF 解析

### 安装

```bash
# 克隆仓库
git clone https://github.com/Casertmisten/nanocode-financial.git
cd nanocode-financial

# 安装依赖（uv 自动创建虚拟环境）
uv sync

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key 和模型配置
```

### 配置说明

编辑 `.env` 文件，至少配置以下项：

```env
# LLM API（三选一或按需组合）
ALI_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ALI_API_KEY=sk-xxx
ALI_MODEL=qwen-plus

# Embedding（默认使用 DashScope）
EMBEDDING_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3

# Reranker（可选，提升检索精度）
RERANKER_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RERANKER_MODEL=gte-rerank
```

### 运行

```bash
# 启动 ChromaDB 服务（首次需要）
bash run_db.sh

# Web 模式（推荐）
bash run_web.sh
# 或直接
uv run python web.py

# CLI 模式
uv run python cli.py
```

Web 模式启动后访问 `http://localhost:8000`。

### CLI 快捷命令

| 命令 | 说明 |
|------|------|
| `/fra <股票名>` | 触发财报深度分析 |
| `/clear` | 清空当前对话 |
| `/q` | 退出 |

## 核心依赖

| 依赖 | 用途 |
|------|------|
| FastAPI + Uvicorn | Web 框架 & ASGI 服务器 |
| LlamaIndex | RAG 框架 |
| ChromaDB | 向量数据库 |
| AKShare | A 股行情 & 新闻数据 |
| rank-bm25 + jieba | BM25 稀疏检索 + 中文分词 |
| httpx | 异步 HTTP 客户端 |
| aiosqlite | 异步 SQLite |
| PyMuPDF | PDF 页数统计 |

## RAG 检索流程

```
用户提问
  ↓
LLM 查询改写（生成 N 个语义变体）
  ↓
┌─────────────┬─────────────┐
│ 多路向量召回  │  BM25 稀疏召回 │
│ (每路 Top200) │  (Top200)    │
└──────┬──────┴──────┬──────┘
       ↓             ↓
    RRF 融合 (0.7:0.3 加权)
       ↓
    Top50 候选
       ↓
  Rerank 精排（可选）
       ↓
    最终结果 → LLM 生成回答
```

## 专业模式流程

```
用户输入（勾选专业模式）
  ↓
意图识别（正则匹配 → LLM 兜底）
  ↓
┌──────────────┬──────────────┬──────────────┐
│ 个股投资决策    │ 行业轮动发现   │  财报深度分析   │
│ 多步分析流水线  │ 行业扫描+排行  │ Map-Reduce    │
│ 评分+建议      │ 轮动建议      │ 3×33 子问题   │
└──────┬───────┴──────┬───────┴──────┬───────┘
       ↓              ↓              ↓
   未命中专业意图 → 通用 RAG 对话（agentic loop + 工具调用）
```

## 三层记忆流程

```
每轮对话开始                        每轮对话结束
     ↓                                ↓
┌─────────────┐              ┌──────────────────┐
│ L3 会话压缩  │              │ L2 摘要生成+向量化  │
│ Token 预算检查│              │ 主题+画像候选提取   │
│ 超限→摘要标记 │              └────────┬─────────┘
│ (保留原始消息) │                       ↓
└──────┬──────┘              ┌──────────────────┐
       ↓                     │ ChromaDB 存储      │
┌─────────────┐              │ session_memory    │
│ L1 画像注入  │              │ (独立 collection)  │
│ Markdown 渲染 │              └──────────────────┘
└──────┬──────┘                       ↑
       ↓                     ┌──────────────────┐
┌─────────────┐              │ L1 候选池累积      │
│ L2 跨会话检索 │              │ 3轮→LLM合并更新画像 │
│ RAG + 时间衰减│              └──────────────────┘
└──────┬──────┘
       ↓
  注入 System Prompt
```

## License

MIT
