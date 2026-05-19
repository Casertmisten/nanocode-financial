import hashlib
import os
import json
import random
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from llama_index.core import Document as LlamaDocument
from tqdm import tqdm

from rag.chunker import split_documents as _llama_split


# =========================================================
# 1. 本地模型
# =========================================================

local_llm = ChatOpenAI(
    model="qwen3-6-35b-a3b",
    base_url="http://58.251.255.50:8229/v1",
    api_key="EMPTY",
    temperature=0.3,
    max_tokens=8192,
    model_kwargs={
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "qa_list",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "answer": {"type": "string"},
                            },
                            "required": ["question", "answer"],
                            "additionalProperties": False,
                        }
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            },
        },
    }},
)

local_embeddings = OpenAIEmbeddings(
    model="Qwen3-Embedding-0.6B",
    base_url="http://localhost:30000/v1",
    api_key="EMPTY",
)


# =========================================================
# 2. 配置
# =========================================================

MARKDOWN_DIR = "/home/caser/文档/data/markdown-test"
OUTPUT_FILE = "/home/caser/文档/data/rag_eval_dataset.jsonl"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOTAL_QUESTIONS = 300
QA_PER_CHUNK = 2
MIN_CHUNK_LENGTH = 200


# =========================================================
# 3. Markdown 加载
# =========================================================

def load_markdown_files(markdown_dir: str) -> List[Document]:
    all_docs = []
    for root, _, files in os.walk(markdown_dir):
        for file in files:
            if not file.endswith(".md"):
                continue
            file_path = os.path.join(root, file)
            try:
                docs = TextLoader(file_path, encoding="utf-8").load()
                for doc in docs:
                    doc.metadata["source"] = file
                all_docs.extend(docs)
                print(f"Loaded: {file}")
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    return all_docs


# =========================================================
# 4. 文档切块
# =========================================================

def split_documents(docs: List[Document]) -> List[Document]:
    # 转换为 llama_index Document，复用项目统一的切块策略
    llama_docs = [LlamaDocument(text=d.page_content, metadata=d.metadata) for d in docs]
    nodes = _llama_split(llama_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # 转换回 langchain Document
    chunks = [
        Document(page_content=n.text, metadata=n.metadata)
        for n in nodes if len(n.text.strip()) >= MIN_CHUNK_LENGTH
    ]
    print(f"Total chunks: {len(chunks)}")
    return chunks


# =========================================================
# 5. Prompt
# =========================================================

QA_PROMPT = ChatPromptTemplate.from_template("""
你是专业的中文RAG评测集构建专家。
请根据给定文档内容，生成高质量中文问答数据。

要求：
1. 生成 {qa_count} 个问题
2. 问题必须能从文档中直接回答
3. 不允许脱离文档内容
4. 不要生成开放性问题
5. 不要生成"根据文档"这种表达
6. 问题必须真实、自然、像用户会问的
7. 问题类型尽量多样化：
   - 事实类
   - 流程类
   - 对比类
   - 条件类
   - 参数类
8. 答案必须准确且完整
9. 返回 JSON 数组
10. 不要输出 markdown

文档内容：
{context}

输出格式：

[
    {{
        "question": "...",
        "answer": "..."
    }}
]
""")


# =========================================================
# 6. QA 生成
# =========================================================

def generate_qa_from_chunk(chunk_text: str, qa_count: int = 3, max_retries: int = 2):
    chain = QA_PROMPT | local_llm
    for attempt in range(max_retries + 1):
        try:
            response = chain.invoke({"context": chunk_text, "qa_count": qa_count})
            text = response.content.strip()
            data = json.loads(text)
            return data.get("items", data if isinstance(data, list) else [])
        except Exception as e:
            if attempt < max_retries:
                print(f"  retry {attempt + 1}/{max_retries}: {e}")
            else:
                print(f"QA generation failed after {max_retries + 1} attempts: {e}")
    return []


# =========================================================
# 7. 增量构建评测数据
# =========================================================

def _chunk_hash(text: str) -> str:
    """chunk 文本的哈希 ID，用于断点续传。"""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _load_processed_ids(output_file: str) -> set:
    """从已有 jsonl 读取已处理的 chunk_id。"""
    if not os.path.exists(output_file):
        return set()
    ids = set()
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                cid = json.loads(line).get("metadata", {}).get("chunk_id")
                if cid:
                    ids.add(cid)
            except (json.JSONDecodeError, AttributeError):
                continue
    return ids


def _append_jsonl(items: list, output_file: str):
    """追加写入 jsonl。"""
    with open(output_file, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_rag_eval_dataset(chunks: List[Document], output_file: str):
    processed_ids = _load_processed_ids(output_file)
    # 筛出未处理的 chunk
    unprocessed = [c for c in chunks
                   if _chunk_hash(c.page_content.strip()) not in processed_ids]
    skipped = len(chunks) - len(unprocessed)
    print(f"已处理 {len(processed_ids)} 个 chunk，本次跳过 {skipped} 个")

    # 按目标总数计算需要采样多少个 chunk
    n_needed = (TOTAL_QUESTIONS + QA_PER_CHUNK - 1) // QA_PER_CHUNK
    n_sample = min(n_needed, len(unprocessed))
    sampled = random.sample(unprocessed, n_sample)
    print(f"目标 {TOTAL_QUESTIONS} 题，采样 {n_sample} 个 chunk，每个生成 {QA_PER_CHUNK} 题")

    new_count = 0
    for chunk in tqdm(sampled, desc="Generating QA"):
        chunk_text = chunk.page_content.strip()
        cid = _chunk_hash(chunk_text)
        if cid in processed_ids:
            continue

        qas = generate_qa_from_chunk(chunk_text, qa_count=QA_PER_CHUNK)
        new_items = []
        for qa in qas:
            try:
                question = qa["question"].strip()
                answer = qa["answer"].strip()
                if len(question) < 5 or len(answer) < 5:
                    continue
                new_items.append({
                    "question": question,
                    "ground_truth": answer,
                    "contexts": [chunk_text],
                    "metadata": {
                        "source": chunk.metadata.get("source", ""),
                        "chunk_id": cid,
                    },
                })
            except Exception as e:
                print(e)

        if new_items:
            _append_jsonl(new_items, output_file)
            new_count += len(new_items)

    return new_count


# =========================================================
# 8. Embedding 去重
# =========================================================

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def deduplicate_questions(dataset, threshold=0.92):
    print("Embedding deduplication...")
    vectors = local_embeddings.embed_documents([x["question"] for x in dataset])
    unique_dataset, embeddings = [], []
    for idx, item in enumerate(dataset):
        if any(cosine_similarity(vectors[idx], ev) >= threshold for ev in embeddings):
            continue
        unique_dataset.append(item)
        embeddings.append(vectors[idx])
    print(f"Before dedup: {len(dataset)}, After dedup: {len(unique_dataset)}")
    return unique_dataset


# =========================================================
# 9. 保存 jsonl
# =========================================================

def save_jsonl(dataset, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved to: {output_file}")


# =========================================================
# 10. 主函数
# =========================================================

def main():
    print("=" * 60)
    print("Loading markdown files...")
    docs = load_markdown_files(MARKDOWN_DIR)

    print("Splitting documents...")
    chunks = split_documents(docs)
    random.shuffle(chunks)

    print("Generating QA dataset...")
    new_count = build_rag_eval_dataset(chunks, OUTPUT_FILE)
    print(f"本次新增: {new_count} 条")

    # 加载全量数据进行去重
    print("Deduplicating...")
    all_items = []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                all_items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    dataset = deduplicate_questions(all_items)

    save_jsonl(dataset, OUTPUT_FILE)
    print("DONE")


if __name__ == "__main__":
    main()
