"""文档管理路由 — 上传、列表、删除、统计。"""

import json
import os

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import config
import db

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("")
async def list_documents(status: str | None = None, type: str | None = None):
    docs = await db.list_documents(status=status, file_type=type)
    for d in docs:
        if isinstance(d.get("tags"), str):
            d["tags"] = json.loads(d["tags"])
    return docs


@router.get("/stats")
async def document_stats():
    return await db.get_document_stats()


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

        file_type = os.path.splitext(filename)[1].lstrip(".").lower() or "txt"
        title = os.path.splitext(filename)[0]
        doc_id = await db.add_document(
            title=title, filename=filename, filepath=filepath,
            file_size=len(content), file_type=file_type,
            source="upload", tags=[category],
        )

        # 触发 RAG ingest
        try:
            await db.update_document(doc_id, status="processing")
            import rag
            rag.ingest(config.UPLOAD_DIR)
            chunks_approx = max(1, len(content) // 500)
            await db.update_document(doc_id, status="ready", chunks=chunks_approx)
        except Exception as e:
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
    await db.delete_document(doc_id)
    return {"ok": True}
