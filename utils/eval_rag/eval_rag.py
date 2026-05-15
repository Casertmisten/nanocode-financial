"""使用 RAGAS 框架评估 RAG 系统（兼容 Qwen / vLLM / OpenAI-Compatible API）。"""

import json
import os
import sys
import traceback
import logging
import urllib.request
import urllib.error

import asyncio
from openai import AsyncOpenAI
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
from ragas.metrics.collections.context_recall import ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import BaseLogger
import dotenv

dotenv.load_dotenv()

log = BaseLogger.getLogger("eval_rag")
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# =========================================================
# 评估用 LLM / Embedding 配置
# =========================================================

EVAL_LLM_BASE_URL = os.environ.get("LOCAL_API_URL", "")
EVAL_LLM_API_KEY = os.environ.get("LOCAL_API_KEY", "")
EVAL_LLM_MODEL = os.environ.get("LOCAL_MODEL", "qwen3.5-plus")

EVAL_EMBED_BASE_URL = os.environ.get("EVAL_EMBED_BASE_URL", config.EMBEDDING_API_URL)
EVAL_EMBED_API_KEY = os.environ.get("EVAL_EMBED_API_KEY", config.EMBEDDING_API_KEY)
EVAL_EMBED_MODEL = os.environ.get("EVAL_EMBED_MODEL", config.EMBEDDING_MODEL)


def evaluator_llm_client():
    """构建评估用 LLM"""

    base_url = EVAL_LLM_BASE_URL
    if base_url.endswith("/chat/completions"):
        base_url = base_url.replace("/chat/completions", "")

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=EVAL_LLM_API_KEY,
    )
    return llm_factory(
        EVAL_LLM_MODEL,
        provider="openai",
        client=client,
        temperature=0.5,
        max_tokens=8192,
        n=3,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def evaluator_embeddings_client():
    """构建评估用 Embedding。"""

    client = AsyncOpenAI(
        base_url=EVAL_EMBED_BASE_URL,
        api_key=EVAL_EMBED_API_KEY,
    )
    return OpenAIEmbeddings(client=client, model=EVAL_EMBED_MODEL)


# =========================================================
# 调用 RAG 系统
# =========================================================

def call_rag_system(question: str) -> dict:
    """检索 + 生成，返回 response 和 retrieved_contexts。"""
    from rag.retriever import retrieve
    from rag.indexer import get_index

    index, _ = get_index(
        chroma_dir=config.CHROMA_PERSIST_DIR,
        embedding_api_url=config.EMBEDDING_API_URL,
        embedding_api_key=config.EMBEDDING_API_KEY,
        embedding_model_name=config.EMBEDDING_MODEL,
    )

    results = retrieve(index, question, top_k=10)
    retrieved_contexts = [r["text"].strip() for r in results if r.get("text", "").strip()]
    if not retrieved_contexts:
        retrieved_contexts = ["未检索到相关内容"]

    # 生成回答
    context_text = "\n\n".join(retrieved_contexts)
    prompt = (
        "请严格基于参考资料回答问题，不允许编造。"
        "如果资料中没有答案，直接回答不知道。"
        "回答必须简洁准确，以 JSON 格式输出，键名为 answer。\n\n"
        f"参考资料：\n{context_text}\n\n"
        f"问题：\n{question}"
    )

    payload = json.dumps({
        "model": config.MODEL,
        "messages": [
            {"role": "system", "content": "你是专业知识库问答助手，输出 JSON 格式。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()

    req = urllib.request.Request(
        config.API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.API_KEY}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw_response = json.loads(resp.read().decode())["choices"][0]["message"]["content"]

        # 尝试从 JSON 中提取 answer 字段
        try:
            parsed = json.loads(raw_response)
            if isinstance(parsed, dict):
                response = parsed.get("answer", parsed.get("response", raw_response))
            else:
                response = raw_response
        except (json.JSONDecodeError, TypeError):
            response = raw_response

    except urllib.error.HTTPError as e:
        print(f"LLM调用失败 [{e.code}]: {e.read().decode('utf-8', errors='replace')}")
        response = "不知道"
    except Exception as e:
        print(f"LLM调用失败: {e}")
        response = "不知道"

    return {"response": str(response), "retrieved_contexts": retrieved_contexts}


# =========================================================
# 加载数据集 + 调用 RAG（带缓存）
# =========================================================

def _load_cached_rag(cache_path: str):
    """从缓存文件加载已完成的 RAG 结果。"""
    if not os.path.exists(cache_path):
        return []
    results = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except (json.JSONDecodeError, TypeError):
                continue
    return results


def _append_cache(items: list, cache_path: str):
    """追加写入缓存。"""
    with open(cache_path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_and_run_rag(dataset_path: str, cache_path: str) -> list[dict]:
    """加载评测数据集，逐条调用 RAG，返回结果列表。已有缓存则跳过。"""
    with open(dataset_path, encoding="utf-8") as f:
        lines = f.readlines()

    # 加载已缓存的 RAG 结果
    cached = _load_cached_rag(cache_path)
    cached_questions = {r["user_input"] for r in cached}
    print(f"加载 {len(lines)} 条评测样本，已有缓存 {len(cached)} 条")

    new_cache_items = []

    for i, line in enumerate(lines):
        try:
            item = json.loads(line.strip())
            question = item["question"]
            ground_truth = item["ground_truth"]

            # 跳过已缓存的
            if question in cached_questions:
                continue

            print(f"\n[{i + 1}/{len(lines)}] Question: {question}")

            rag_result = call_rag_system(question)

            contexts = [str(x) for x in rag_result["retrieved_contexts"] if str(x).strip()]
            if not contexts:
                contexts = ["空context"]

            # 写入缓存
            cache_item = {
                "user_input": str(question),
                "response": str(rag_result["response"]),
                "reference": str(ground_truth),
                "retrieved_contexts": contexts,
            }
            _append_cache([cache_item], cache_path)
            new_cache_items.append(cache_item)

        except Exception as e:
            print(f"\n样本处理失败: {e}")
            traceback.print_exc()

    all_results = cached + new_cache_items
    print(f"\n总计 {len(all_results)} 条 RAG 结果（缓存 {len(cached)} + 新增 {len(new_cache_items)}）")
    return all_results


# =========================================================
# 运行 RAGAS 评估
# =========================================================

def run_evaluation(dataset_path: str = "rag_eval_dataset.jsonl"):

    evaluator_llm = evaluator_llm_client()
    evaluator_embeddings = evaluator_embeddings_client()

    # RAG 结果缓存路径
    cache_path = dataset_path.replace(".jsonl", "_rag_cache.jsonl")
    samples = load_and_run_rag(dataset_path, cache_path)
    print(f"\n{'=' * 60}\nDataset Loaded ({len(samples)} samples)\n{'=' * 60}")

    faithfulness = Faithfulness(llm=evaluator_llm)
    answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    context_recall = ContextRecall(llm=evaluator_llm)

    print(f"\n{'=' * 60}\n开始 RAGAS 评估\n{'=' * 60}")

    async def _evaluate_all():
        results = []
        for i, s in enumerate(samples):
            row = {
                "user_input": s["user_input"],
                "response": s["response"],
                "ground_truth": s["reference"],
                "retrieved_contexts": s["retrieved_contexts"],
            }

            for name, metric, keys in [
                ("faithfulness", faithfulness, ("user_input", "response", "retrieved_contexts")),
                ("answer_relevancy", answer_relevancy, ("user_input", "response")),
                ("context_recall", context_recall, ("user_input", "retrieved_contexts", "reference")),
            ]:
                try:
                    inp = {k: s[k] for k in keys}
                    r = await metric.ascore(**inp)
                    row[name] = r.value
                except Exception as e:
                    print(f"\n[{i + 1}] {name} 评估失败: {e}")
                    row[name] = float("nan")

            results.append(row)
            print(f"\r  [{i + 1}/{len(samples)}] faithfulness={row['faithfulness']:.4f}  answer_relevancy={row['answer_relevancy']:.4f}  context_recall={row['context_recall']:.4f}", end="", flush=True)

        print()
        return results

    results = asyncio.run(_evaluate_all())

    import pandas as pd
    df = pd.DataFrame(results)
    print(f"\n{'=' * 60}\nRAGAS 评估结果\n{'=' * 60}\n{df.to_string()}")

    # 平均分
    print("\n各指标平均分：")
    for col in ("faithfulness", "answer_relevancy", "context_recall"):
        try:
            print(f"  {col}: {df[col].dropna().mean():.4f}")
        except Exception:
            print(f"  {col}: FAILED")

    # 保存
    output_path = dataset_path.replace(".jsonl", "_ragas_result.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到: {output_path}")

    return df


if __name__ == "__main__":
    run_evaluation("/home/caser/文档/code/nanocode-financial/utils/eval_rag/rag_eval_dataset.jsonl")
