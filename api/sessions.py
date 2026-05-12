"""会话管理路由。"""

import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException

import db
from utils import BaseLogger

log = BaseLogger.getLogger("sessions")

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
async def list_sessions():
    sessions = await db.list_sessions()
    for s in sessions:
        if isinstance(s.get("knowledge_base_ids"), str):
            s["knowledge_base_ids"] = json.loads(s["knowledge_base_ids"])
    log.info("查询会话列表: %d 条", len(sessions))
    return sessions


@router.post("")
async def create_session():
    session_id = uuid4().hex[:16]
    session = await db.create_session(session_id)
    log.info("创建会话: %s", session_id)
    return session


@router.get("/{session_id}")
async def get_session(session_id: str):
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    messages = await db.get_messages(session_id)
    session["messages"] = messages
    if isinstance(session.get("knowledge_base_ids"), str):
        session["knowledge_base_ids"] = json.loads(session["knowledge_base_ids"])
    log.info("获取会话详情: %s, 消息数=%d", session_id, len(messages))
    return session


@router.patch("/{session_id}")
async def update_session(session_id: str, body: dict):
    allowed = {"title", "model", "knowledge_base_ids"}
    fields = {k: v for k, v in body.items() if k in allowed}
    if "knowledge_base_ids" in fields and not isinstance(fields["knowledge_base_ids"], str):
        fields["knowledge_base_ids"] = json.dumps(fields["knowledge_base_ids"], ensure_ascii=False)
    ok = await db.update_session(session_id, **fields)
    if not ok:
        raise HTTPException(404, "会话不存在")
    log.info("更新会话: %s, fields=%s", session_id, list(fields.keys()))
    return {"ok": True}


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    ok = await db.delete_session(session_id)
    if not ok:
        raise HTTPException(404, "会话不存在")
    log.info("删除会话: %s", session_id)
    return {"ok": True}
