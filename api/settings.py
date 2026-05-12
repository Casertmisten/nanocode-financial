"""设置管理路由。"""

import httpx
from fastapi import APIRouter

import db

router = APIRouter(prefix="/api/settings", tags=["settings"])

_DEFAULTS = {
    "chat": {
        "provider": "ali", "apiUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "apiKey": "", "model": "qwen-max", "temperature": 0.7, "maxTokens": 4096,
    },
    "parse": {
        "provider": "ali", "apiUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "apiKey": "", "summaryModel": "qwen-plus", "embedModel": "text-embedding-v3",
        "chunkSize": 512, "chunkOverlap": 64,
    },
    "general": {
        "vectorDb": "chroma", "storagePath": "./data", "httpsProxy": "",
    },
}


@router.get("")
async def get_settings():
    stored = await db.get_settings()
    result = {}
    for key, default_val in _DEFAULTS.items():
        result[key] = {**default_val, **stored.get(key, {})}
    return result


@router.put("")
async def save_settings(body: dict):
    await db.save_settings(body)
    return {"ok": True}


@router.post("/test-connection")
async def test_connection(body: dict):
    api_url = body.get("apiUrl", "")
    api_key = body.get("apiKey", "")
    if not api_url or not api_key:
        return {"ok": False, "message": "缺少 API 地址或 Key"}
    try:
        url = api_url.rstrip("/") + "/models"
        resp = httpx.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10.0)
        if resp.status_code == 200:
            return {"ok": True, "message": "连接成功"}
        return {"ok": False, "message": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"ok": False, "message": str(e)}
