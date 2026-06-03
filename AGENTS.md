# AGENTS.md

本文件为 Codex (Codex.ai/code) 提供项目指导。

## 项目概述

基于 nanocode 改造的**个人金融智能分析助手**。核心是一个单文件 Python agentic 编码助手，通过 OpenAI 兼容格式接入国内大模型（阿里 DashScope / 智谱 GLM），具备完整的工具调用、对话历史和终端彩色输出能力。

项目当前处于 `feat/RAG` 分支，正在开发 RAG 问答、Deep Research 和每日新闻总结三大功能（详见 `docs/金融助手项目PRD.md`）。

## 环境与依赖管理

统一使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境和依赖。

```bash
# 安装依赖并运行
uv run nanocode.py

# 添加新依赖
uv add <package>

# 手动同步环境
uv sync
```

## 运行方式

```bash
# 1. 配置 .env 文件（已有模板，填入 API Key 即可）
#    支持 Ali DashScope 和 智谱两套 API

# 2. 运行（uv 自动管理虚拟环境和依赖）
uv run nanocode.py
```

## 项目结构

```
nanocode-financial/
├── nanocode.py          # 主程序，全部逻辑所在（~300行）
├── .env                 # API 配置（已在 .gitignore 中）
├── docs/
│   └── 金融助手项目PRD.md  # 产品需求文档
└── .venv/               # Python 虚拟环境（已在 .gitignore 中）
```

## 测试与规范

目前无测试套件、lint 或 CI/CD。通过运行 `python nanocode.py` 交互式验证改动。

## 架构说明

全部代码在 `nanocode.py` 中，结构如下：

1. **配置与环境变量**（第1-16行）：通过 `dotenv` 加载 `.env`，读取 `ALI_API_URL`、`ALI_API_KEY`、`ALI_MODEL` 等环境变量。支持阿里 DashScope 和智谱 GLM 两套 API。
2. **ANSI 颜色常量**（第18-26行）：终端输出的颜色控制。
3. **工具实现**（第32-100行）：六个工具函数——`read`（读文件）、`write`（写文件）、`edit`（编辑文件）、`glob`（文件查找）、`grep`（内容搜索）、`bash`（执行命令）。
4. **工具注册表**（第106-137行）：`TOOLS` 字典，将工具名映射为 `(描述, 参数schema, 函数)` 三元组。参数使用简写格式：`"string"` 表示必填，`"number?"` 表示可选。
5. **Schema 生成**（第149-180行）：`make_schema()` 将工具注册表转换为 OpenAI 兼容的 function calling 格式。
6. **API 调用**（第183-202行）：`call_api()` 通过 `urllib.request` 调用 OpenAI 兼容的 Chat Completions 接口。使用 `Authorization: Bearer` 认证。
7. **主循环**（第215-298行）：`main()` 驱动 REPL 和 agentic loop——用户输入 → API 调用 → 处理响应（文本输出 + 工具调用）→ 执行工具并回传结果 → 循环直到无工具调用。

### Agentic Loop

核心流程：用户消息 → API 调用 → 处理响应块 → 如有工具调用则执行并将结果回传 → 重复直到无工具调用。使用 OpenAI Chat Completions 的 function calling 格式。

### 关键设计细节

- API 格式：OpenAI 兼容的 Chat Completions（`choices[0].message`、`tool_calls`、function calling）
- 工具参数使用自定义简写 schema（`"string"`、`"number?"`），调用时转换为 OpenAI function calling 格式
- `edit` 工具要求匹配字符串唯一，除非传入 `all=true` 替换所有匹配
- `bash` 工具实时流式输出，30 秒超时
- `grep` 工具限制最多 50 条结果
- 认证方式：`Authorization: Bearer <API_KEY>`

## PRD 核心功能（规划中）

详见 `docs/金融助手项目PRD.md`：

1. **RAG 金融问答**：基于本地知识库（新闻、研报、金融文档）进行语义检索 + LLM 生成带引用的回答
2. **个股深度研究（Deep Research）**：多步推理 + Web Search，自动拆解问题、多源检索、生成结构化投资报告
3. **每日金融新闻总结**：自动聚合多源金融新闻，生成结构化市场总结报告

## 设计约束

- 不做实时交易系统、量化回测、多智能体协作
- 优先低复杂度、pipeline 清晰、输出结构化、可演示
- 注释全部用中文
