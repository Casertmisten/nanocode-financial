"""Token 用量统计 API。"""

from fastapi import APIRouter, Query

import db
from utils import BaseLogger

log = BaseLogger.getLogger("token_stats")

router = APIRouter(prefix="/api/token-stats", tags=["token-stats"])


@router.get("/summary")
async def get_summary(days: int = Query(default=30, ge=1, le=365)):
    """获取 token 用量汇总统计。"""
    return await db.get_token_summary(days)


@router.get("/daily")
async def get_daily(days: int = Query(default=30, ge=1, le=365)):
    """获取按天的 token 用量趋势。"""
    return await db.get_token_daily(days)


@router.get("/recent")
async def get_recent(limit: int = Query(default=50, ge=1, le=500)):
    """获取最近的 token 用量明细。"""
    return await db.get_token_recent(limit)
