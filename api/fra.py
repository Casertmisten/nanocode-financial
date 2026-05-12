"""财报分析 API — SSE 流式进度推送。"""

import datetime
import json
import os
import sqlite3

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

import config
from utils import BaseLogger

log = BaseLogger.getLogger("fra")

router = APIRouter(prefix="/api/fra", tags=["fra"])


async def _run_fra_stream(query: str, session_id: str | None = None):
    """执行 FRA 流程，逐阶段 SSE 推送。"""
    from financial_report_analysis.template import DIMENSIONS
    from financial_report_analysis.prompts import ANALYZE_PROMPT, REDUCE_PROMPT
    import rag
    import llm

    _TOP_K = 3
    log.info("开始 FRA 分析: query=%s, session=%s", query[:50], session_id)

    # Map: 检索
    dim_data = []
    total = sum(len(d["sub_questions"]) for d in DIMENSIONS)
    done = 0

    yield f"event: progress\ndata: {json.dumps({'stage': 'retrieve', 'detail': f'开始检索 {total} 个子问题...'}, ensure_ascii=False)}\n\n"

    for dim in DIMENSIONS:
        seen_texts = set()
        chunks = []
        sources = set()

        for sq in dim["sub_questions"]:
            done += 1
            yield f"event: progress\ndata: {json.dumps({'stage': 'retrieve', 'detail': f'[{done}/{total}] {sq}'}, ensure_ascii=False)}\n\n"
            try:
                results = rag.query(sq, top_k=_TOP_K)
            except Exception:
                results = []
            for r in results:
                text = r.get("text", "")
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    chunks.append(text)
                source = r.get("source", "")
                if source:
                    sources.add(source)

        dim_data.append({"name": dim["name"], "chunks": chunks, "sources": sources})
        log.info("维度 [%s] 检索完成: %d 个 chunks, %d 个来源", dim["name"], len(chunks), len(sources))

    # Analyze: 逐维度分析
    summaries = {}
    for i, dd in enumerate(dim_data, 1):
        detail = f'[{i}/{len(dim_data)}] {dd["name"]}'
        yield f"event: progress\ndata: {json.dumps({'stage': 'analyze', 'detail': detail}, ensure_ascii=False)}\n\n"
        if not dd["chunks"]:
            summaries[dd["name"]] = f"【{dd['name']}】该维度缺乏足够数据。"
        else:
            chunks_text = "\n\n---\n\n".join(dd["chunks"])
            prompt = ANALYZE_PROMPT.format(dimension_name=dd["name"], chunks=chunks_text)
            summaries[dd["name"]] = llm.call_llm("你是一个专业的金融分析师。", prompt)
            log.info("维度 [%s] 分析完成, 结果长度=%d", dd["name"], len(summaries[dd["name"]]))

    # 汇总来源
    all_sources = set()
    for dd in dim_data:
        all_sources.update(dd["sources"])

    # Reduce: 生成报告
    yield f"event: progress\ndata: {json.dumps({'stage': 'reduce', 'detail': '生成报告...'}, ensure_ascii=False)}\n\n"

    summaries_text = ""
    for dim in DIMENSIONS:
        summaries_text += f"\n## {dim['name']}\n{summaries[dim['name']]}\n"
    sources_text = "\n".join(f"· {s}" for s in sorted(all_sources))
    prompt = REDUCE_PROMPT.format(query=query, summaries=summaries_text, sources=sources_text)
    report = llm.call_llm("你是一个资深金融分析师。", prompt)
    log.info("报告生成完成, 长度=%d", len(report))

    # 保存报告
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(config.BASE_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    filepath = os.path.join(report_dir, f"financial_report_{ts}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    # 写入数据库
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.execute(
        "INSERT INTO fra_reports (session_id, query, content, filepath, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, query, report, filepath, datetime.datetime.now().isoformat()),
    )
    conn.commit()
    report_id = cursor.lastrowid
    conn.close()
    log.info("报告已保存: id=%d, path=%s", report_id, filepath)

    yield f"event: done\ndata: {json.dumps({'report_id': report_id, 'filepath': filepath}, ensure_ascii=False)}\n\n"


@router.post("")
async def run_fra(request: Request):
    body = await request.json()
    query = body.get("query", "")
    session_id = body.get("session_id")
    if not query:
        raise HTTPException(400, "缺少 query 参数")

    log.info("收到 FRA 请求: session=%s, query=%s", session_id, query[:50])

    return StreamingResponse(
        _run_fra_stream(query, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/reports")
async def list_fra_reports():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT id, session_id, query, filepath, created_at FROM fra_reports ORDER BY created_at DESC LIMIT 20")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


@router.get("/reports/{report_id}")
async def get_fra_report(report_id: int):
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM fra_reports WHERE id=?", (report_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "报告不存在")
    return dict(row)
