"""文档管理路由 — 上传、列表、删除、统计。"""

import json
import os

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import config
import db
from utils import BaseLogger

log = BaseLogger.getLogger("documents")

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("")
async def list_documents(status: str | None = None, type: str | None = None):
    docs = await db.list_documents(status=status, file_type=type)
    for d in docs:
        if isinstance(d.get("tags"), str):
            d["tags"] = json.loads(d["tags"])
    log.info("查询文档列表: status=%s, type=%s, 共 %d 条", status, type, len(docs))
    return docs


@router.get("/stats")
async def document_stats():
    stats = await db.get_document_stats()
    log.info("文档统计: %s", stats)
    return stats


@router.get("/{doc_id}")
async def get_document(doc_id: int):
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    if isinstance(doc.get("tags"), str):
        doc["tags"] = json.loads(doc["tags"])
    return doc


@router.post("/upload")
async def upload_documents(
    files: list[UploadFile] = File(...), category: str = Form("report"),
):
    log.info("收到上传请求: %d 个文件, category=%s", len(files), category)
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    results = []
    for f in files:
        filename = f.filename or "untitled"
        filepath = os.path.join(config.UPLOAD_DIR, filename)
        if os.path.exists(filepath):
            name, ext = os.path.splitext(filename)
            import time
            filepath = os.path.join(config.UPLOAD_DIR, f"{name}_{int(time.time())}{ext}")

        content = await f.read()
        with open(filepath, "wb") as out:
            out.write(content)
        log.info("文件已保存: %s (%d bytes)", filepath, len(content))

        file_type = os.path.splitext(filename)[1].lstrip(".").lower() or "txt"
        title = os.path.splitext(filename)[0]
        doc_id = await db.add_document(
            title=title, filename=filename, filepath=filepath,
            file_size=len(content), file_type=file_type,
            source="upload", tags=json.dumps([category]),
        )
        log.info("文档记录已创建: id=%d, title=%s, type=%s", doc_id, title, file_type)

        # 触发 RAG ingest — 只索引当前上传的文件
        try:
            await db.update_document(doc_id, status="processing")
            import rag
            count = rag.ingest_file(filepath)
            chunks_approx = max(1, len(content) // 500)
            await db.update_document(doc_id, status="ready", chunks=chunks_approx)
            log.info("文档索引完成: id=%d, ingest=%d, chunks≈%d", doc_id, count, chunks_approx)
        except Exception as e:
            log.error("文档索引失败: id=%d, error=%s", doc_id, e, exc_info=True)
            await db.update_document(doc_id, status="error", error_message=str(e))

        results.append({"id": doc_id, "filename": filename, "status": "ready"})

    return results


@router.delete("/{doc_id}")
async def delete_document(doc_id: int):
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "文档不存在")
    if doc.get("filepath") and os.path.exists(doc["filepath"]):
        os.remove(doc["filepath"])
        log.info("已删除文件: %s", doc["filepath"])
    await db.delete_document(doc_id)
    log.info("已删除文档记录: id=%d", doc_id)
    return {"ok": True}
