# 三层记忆系统设计文档

> 日期：2026-05-21
> 状态：已确认
> 方案：A — 最小侵入式

---

## 1. 概述

为金融智能分析助手设计三层记忆系统，使 LLM 具备用户偏好感知、跨会话记忆和长对话压缩能力。

| 层 | 名称 | 作用 | 存储 | 注入时机 |
|---|---|---|---|---|
| L1 | 用户画像 | 记录用户偏好和习惯 | `data/profile.json` | 每次构建 system prompt |
| L2 | 跨会话记忆 | 保留历史会话摘要 | ChromaDB `session_memory` collection | 每轮对话前 RAG 检索 |
| L3 | 当前会话记忆 | 压缩超长对话 | SQLite `session_summaries` 表 | 发送给 LLM 前检查压缩 |

设计原则：**最小侵入**，改动集中在 `api/chat.py` 的消息准备阶段和新增 `memory/` 模块，不动 RAG/Deep Research 核心代码。

---

## 2. 第一层：用户画像（L1）

### 2.1 存储结构

文件路径：`data/profile.json`

```json
{
  "version": 1,
  "updated_at": "2026-05-21T10:00:00Z",
  "profile": {
    "preferred_markets": ["A股", "港股"],
    "focus_sectors": ["新能源", "半导体"],
    "watched_stocks": ["600519", "00700"],
    "risk_tolerance": "中等",
    "report_style": "结构化分析报告",
    "language": "中文"
  }
}
```

### 2.2 Markdown 模板

读取 JSON 后填充模板，追加到 system prompt 尾部：

```markdown
## 用户画像
- 关注市场：{preferred_markets}
- 关注行业：{focus_sectors}
- 关注个股：{watched_stocks}
- 风险偏好：{risk_tolerance}
- 报告风格：{report_style}
```

首次运行时 `profile.json` 不存在，使用空模板（不注入任何画像内容）。

### 2.3 更新机制

- **候选池**：`data/profile_candidates.json`，存储每次会话提取的画像候选
- **提取时机**：每轮对话结束时，在 L2 的摘要生成 prompt 中附带「提取用户画像候选」指令
- **触发更新**：候选池累积 3 轮会话后，调用 LLM 合并候选 → 更新 `profile.json`，清空候选池
- **手动编辑**：用户可直接编辑 `profile.json`，下次对话立即生效

### 2.4 候选池结构

```json
{
  "candidates": [
    {
      "session_id": "abc123",
      "extracted_at": "2026-05-21T10:00:00Z",
      "fields": {
        "focus_sectors": ["AI芯片"],
        "watched_stocks": ["002049"]
      }
    }
  ],
  "session_count": 1
}
```

---

## 3. 第二层：跨会话记忆（L2）

### 3.1 存储

复用现有 ChromaDB，新建 collection `session_memory`。

每条记录结构：

```python
{
    "document": "会话摘要文本...",
    "metadata": {
        "session_id": "abc123",
        "created_at": "2026-05-21T10:00:00Z",
        "session_type": "chat",           # chat | deep_research
        "topics": ["特斯拉", "新能源"],
    }
}
```

### 3.2 写入流程（会话结束时）

1. 收集本次会话全部 messages
2. 调用 LLM 生成摘要（含主题标签提取和用户画像候选提取）
3. 调用现有 embedding API 生成向量
4. 写入 ChromaDB `session_memory` collection

### 3.3 摘要生成 Prompt

```
你是一个对话摘要助手。请对以下对话进行摘要，要求：
1. 提炼用户的核心问题和助手的回答要点
2. 提取 3-5 个主题关键词
3. 摘要长度控制在 200 字以内
4. 如果对话中暴露了用户的投资偏好、关注行业/个股、报告风格偏好，请额外列出

对话内容：
{messages}
```

### 3.4 读取流程（每轮对话前）

1. 对当前用户消息 embedding
2. 在 `session_memory` 中向量检索 top-K（默认 K=5）
3. 应用**时间衰减权重**：`adjusted_score = raw_score * exp(-λ * days_since)`
   - λ 默认 0.05（约 14 天半衰期）
   - `days_since` = (当前时间 - 记录创建时间).days
4. 取加权后 top-3 注入 system prompt

### 3.5 注入格式

追加到 system prompt 尾部：

```markdown
## 历史对话记忆
以下是与你之前对话相关的记忆片段：
- [2026-05-20] 用户询问了特斯拉Q1财报，助手分析了营收增长和利润率变化...
- [2026-05-18] 用户关注新能源板块，助手整理了光伏产业链分析...
```

### 3.6 Deep Research 特殊处理

Deep Research 生成的报告全文也作为一条记录存入 `session_memory`，metadata 中 `session_type` 标记为 `deep_research`。检索时与其他记录平等参与 RAG 检索。

---

## 4. 第三层：当前会话记忆（L3）

### 4.1 目标

当对话消息超过 token 预算时，自动压缩最早的消息为摘要，避免超出 LLM 上下文窗口。

### 4.2 配置

| 参数 | 默认值 | 说明 |
|---|---|---|
| `SESSION_MAX_TOKENS` | 6000 | 发送给 LLM 的消息 token 预算（不含 system prompt） |
| `COMPRESS_ROUNDS` | 3 | 每次压缩的对话轮数（1 轮 = 1 对 user + assistant） |

### 4.3 存储新增

SQLite 新增 `session_summaries` 表：

```sql
CREATE TABLE IF NOT EXISTS session_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    start_msg_id INTEGER,
    end_msg_id INTEGER,
    summary TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_summaries_session
    ON session_summaries(session_id);
```

messages 表不变。被压缩的消息**不删除**，仅在不发送给 LLM 时用摘要替代。

### 4.4 压缩流程

在 `_messages_to_llm_format` 之前触发：

1. **Token 估算**：遍历 messages，按中文约 1.5 token/字符估算总 token 数
2. **判断是否压缩**：若总 token > `SESSION_MAX_TOKENS`，进入压缩
3. **选择压缩范围**：取最早的 3 轮对话（user + assistant 对），跳过中间的 tool messages
4. **生成摘要**：调用 LLM 生成摘要（同步非流式，使用 `call_llm`）
5. **保存摘要**：写入 `session_summaries` 表
6. **替换消息**：构造给 LLM 的消息列表时，用摘要消息替代被压缩的原始消息
7. **循环检查**：重复步骤 1-6 直到 token 数在预算内

### 4.5 摘要消息格式

替代被压缩的原始消息：

```python
{
    "role": "system",
    "content": "[对话摘要] 用户询问了特斯拉的财报表现，助手分析了Q1营收同比增长47%..."
}
```

### 4.6 保护规则

- **不压缩最近的对话**：始终保留最近 5 轮对话的原始消息
- **不压缩进行中的工具调用**：如果最早的对话中包含 tool_calls，跳过该轮
- **已有摘要复用**：如果 `session_summaries` 中已有某段消息的摘要，直接复用不重复生成

---

## 5. 整体数据流

### 5.1 请求阶段（发送给 LLM 前）

```
get_messages(session_id)
    │
    ▼
L3: 计算 token → 超限则压缩（生成/复用摘要）
    │
    ▼
_messages_to_llm_format(messages)
    │
    ▼
L1: 加载 profile.json → 渲染 Markdown
L2: 当前消息 embedding → ChromaDB RAG 检索 → 时间衰减 → top-3
    │
    ▼
合并 L1 + L2 → 追加到 system prompt
    │
    ▼
_agentic_loop(messages, system_prompt)
```

### 5.2 响应阶段（对话结束后）

```
save assistant message
    │
    ▼
L2: 会话摘要生成 → embedding → 写入 ChromaDB
    │
    ▼
L1: 提取画像候选 → 写入候选池 → 累积 ≥ 3 则触发画像更新
    │
    ▼
done
```

---

## 6. 新增模块结构

```
memory/
├── __init__.py        # 对外统一接口：load_profile, inject_memory, compress_if_needed, save_session_memory
├── profile.py         # L1: profile.json 读写 + Markdown 模板渲染 + 候选池管理 + LLM 合并更新
├── session_memory.py  # L2: 摘要生成 + ChromaDB 存取 + 时间衰减检索
├── context.py         # L3: token 估算 + 压缩触发 + 摘要管理
└── prompts.py         # 三层记忆相关的 prompt 模板
```

---

## 7. 新增配置项

添加到 `config.py`：

```python
# --- Memory ---
MEMORY_PROFILE_PATH = os.path.join(BASE_DIR, "data", "profile.json")
MEMORY_CANDIDATES_PATH = os.path.join(BASE_DIR, "data", "profile_candidates.json")
MEMORY_SESSION_COLLECTION = "session_memory"
MEMORY_SESSION_MAX_TOKENS = int(os.environ.get("MEMORY_SESSION_MAX_TOKENS", "6000"))
MEMORY_COMPRESS_ROUNDS = 3
MEMORY_TIME_DECAY_LAMBDA = float(os.environ.get("MEMORY_TIME_DECAY_LAMBDA", "0.05"))
MEMORY_CROSS_SESSION_TOP_K = 5
MEMORY_CROSS_SESSION_INJECT_K = 3
MEMORY_PROFILE_UPDATE_INTERVAL = 3
```

---

## 8. 对现有代码的改动点

| 文件 | 改动内容 |
|---|---|
| `config.py` | 新增 Memory 相关配置项 |
| `db.py` | 新增 `session_summaries` 表 + CRUD |
| `api/chat.py` | 在 `_agentic_loop` 前注入 L1/L2/L3；对话结束后触发 L2 写入和 L1 候选提取 |
| `deep_research/` | 研究完成后写入 L2 的 `session_memory` |

---

## 9. 不做的事情

- 不做多用户支持（单用户 JSON 文件足够）
- 不做记忆的手动管理界面（用户可直接编辑 JSON 文件）
- 不做摘要的定期清理（ChromaDB 按自然增长，后续可按需清理旧记录）
- 不修改 RAG 核心检索逻辑和 Deep Research pipeline
