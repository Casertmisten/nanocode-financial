"""Deep Research API — Plan 同步接口 + Execute SSE 流式推送。"""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import db
import deep_research
from deep_research.schemas import SubTask
from utils import BaseLogger

import config as _config

log = BaseLogger.getLogger("research")

router = APIRouter(prefix="/api/research", tags=["research"])


class PlanRequest(BaseModel):
    query: str


class ExecuteRequest(BaseModel):
    query: str
    sub_tasks: list[dict]
    session_id: str | None = None


def _sse(event: str, data: dict) -> str:
    """构造 SSE 事件字符串。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/plan")
async def run_plan(req: PlanRequest):
    """Plan 阶段：拆解问题为子任务列表。"""
    if not req.query.strip():
        raise HTTPException(400, "缺少研究问题")

    log.info("收到 Plan 请求: query=%s", req.query[:80])

    try:
        plan_usage: dict = {}
        outline, sub_tasks = deep_research.plan(req.query, usage_out=plan_usage)
    except Exception as e:
        log.error("Plan 失败: %s", e, exc_info=True)
        raise HTTPException(500, f"规划失败: {e}")

    # 记录 plan 阶段 token 用量
    if plan_usage.get("total_tokens"):
        try:
            await db.add_token_usage(
                model=_config.MODEL,
                prompt_tokens=plan_usage.get("prompt_tokens", 0),
                completion_tokens=plan_usage.get("completion_tokens", 0),
                total_tokens=plan_usage.get("total_tokens", 0),
                source="research",
            )
        except Exception as e:
            log.warning("记录 Plan token 用量失败: %s", e)

    return {
        "research_outline": outline,
        "sub_tasks": [
            {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "tools": t.tools,
                "search_queries": t.search_queries,
            }
            for t in sub_tasks
        ],
    }


@router.post("/execute")
async def run_execute(req: ExecuteRequest):
    """Execute 阶段：并行执行子任务 + 生成报告，SSE 流式推送。"""
    if not req.sub_tasks:
        raise HTTPException(400, "子任务列表为空")

    log.info("收到 Execute 请求: %d 个子任务", len(req.sub_tasks))

    sub_tasks = [
        SubTask(
            id=t.get("id", i + 1),
            title=t.get("title", ""),
            description=t.get("description", ""),
            tools=t.get("tools", []),
            search_queries=t.get("search_queries", []),
        )
        for i, t in enumerate(req.sub_tasks)
    ]

    async def stream():
        # 推送各 executor 启动事件
        for t in sub_tasks:
            yield _sse("executor_start", {"task_id": t.id, "title": t.title})

        # 执行 pipeline
        results, report, report_id, filepath = await deep_research.execute(req.query, sub_tasks)

        # 记录各 executor 的 token 用量
        for r in results:
            if r.usage.get("total_tokens"):
                try:
                    await db.add_token_usage(
                        model=_config.MODEL,
                        prompt_tokens=r.usage.get("prompt_tokens", 0),
                        completion_tokens=r.usage.get("completion_tokens", 0),
                        total_tokens=r.usage.get("total_tokens", 0),
                        source="research",
                        session_id=req.session_id,
                    )
                except Exception as e:
                    log.warning("记录 Executor token 用量失败: %s", e)

        # 推送各 executor 完成事件
        for r in results:
            yield _sse("executor_done", {
                "task_id": r.task_id, "title": r.title,
                "summary_length": len(r.summary), "complete": r.complete,
            })

        # 推送报告生成事件
        yield _sse("generate_start", {})
        yield _sse("generate_done", {"report_id": report_id, "filepath": filepath})
        yield _sse("done", {"report_id": report_id})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/reports")
async def list_reports():
    """获取历史研究报告列表。"""
    return await db.list_research_reports()


@router.get("/reports/{report_id}")
async def get_report(report_id: int):
    """获取单个研究报告。"""
    report = await db.get_research_report(report_id)
    if not report:
        raise HTTPException(404, "报告不存在")
    return report
