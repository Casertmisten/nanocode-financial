"""Deep Research — 开放性问题三 Agent 协作研究。"""

import asyncio
import datetime
import logging
import os

import config

from deep_research.schemas import ExecutorResult, SubTask
from deep_research.plan import plan as _plan
from deep_research.executor import execute_task
from deep_research.generator import generate as _generate

log = logging.getLogger(__name__)


def plan(query: str, usage_out: dict | None = None) -> tuple[str, list[SubTask]]:
    """Plan 阶段：拆解问题为子任务列表。"""
    return _plan(query, usage_out=usage_out)


async def execute(query: str, sub_tasks: list[SubTask]) -> tuple[list[ExecutorResult], str, int, str]:
    """Execute + Generate 阶段：并行执行子任务并生成报告。

    Returns:
        (executor_results, report_text, report_id, filepath)
    """
    # 并行执行所有子任务
    async def _run_one(task: SubTask) -> ExecutorResult:
        try:
            return await asyncio.wait_for(
                execute_task(task),
                timeout=config.RESEARCH_EXECUTOR_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning("Executor[%d] 超时 (%ds)", task.id, config.RESEARCH_EXECUTOR_TIMEOUT)
            return ExecutorResult(
                task_id=task.id, title=task.title,
                summary="执行超时，未能完成信息搜集。", complete=False,
            )
        except Exception as e:
            log.error("Executor[%d] 失败: %s", task.id, e, exc_info=True)
            return ExecutorResult(
                task_id=task.id, title=task.title,
                summary=f"执行失败: {e}", complete=False,
            )

    results = await asyncio.gather(*[_run_one(t) for t in sub_tasks])
    results_list = list(results)

    # Generator 阶段
    gen_usage: dict = {}
    report = _generate(query, results_list, usage_out=gen_usage)

    # 保存报告
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(config.BASE_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    filepath = os.path.join(report_dir, f"research_{ts}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    import db
    report_id = await db.add_research_report(query, report, filepath)
    log.info("研究报告已保存: id=%d, path=%s", report_id, filepath)

    # 保存到 L2 跨会话记忆
    try:
        from memory.session_memory import save_session_memory
        topics = [query[:20]]
        await save_session_memory(
            f"research_{report_id}",
            report[:1000],
            topics,
            session_type="deep_research",
        )
    except Exception:
        log.warning("Deep Research 报告保存到跨会话记忆失败", exc_info=True)

    return results_list, report, report_id, filepath
