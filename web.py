"""Web 入口 — FastAPI 应用，托管前端 + API 路由。"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import db
from api.chat import router as chat_router
from api.sessions import router as sessions_router
from api.documents import router as documents_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="FinAssist", version="0.1.0")


@app.on_event("startup")
async def startup():
    await db.init_db()


# API 路由
app.include_router(chat_router)
app.include_router(sessions_router)
app.include_router(documents_router)


# 健康检查
@app.get("/api/health")
async def health():
    return {"status": "ok"}


# 模型列表
@app.get("/api/models")
async def list_models():
    return [
        {"id": "qwen-max", "name": "Qwen-Max", "provider": "阿里"},
        {"id": "qwen-plus", "name": "Qwen-Plus", "provider": "阿里"},
        {"id": "qwen-turbo", "name": "Qwen-Turbo", "provider": "阿里"},
        {"id": "glm-4-plus", "name": "GLM-4-Plus", "provider": "智谱"},
        {"id": "glm-4-flash", "name": "GLM-4-Flash", "provider": "智谱"},
        {"id": "deepseek-v3", "name": "DeepSeek-V3", "provider": "DeepSeek"},
    ]


# 知识库列表（从文档表聚合）
@app.get("/api/knowledge-bases")
async def list_knowledge_bases():
    docs = await db.list_documents(status="ready")
    return [
        {"id": "default", "name": "默认知识库", "docs": len(docs), "updated": docs[0]["updated_at"][:10] if docs else ""},
    ]


# 前端静态文件（放在最后，避免覆盖 API 路由）
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
