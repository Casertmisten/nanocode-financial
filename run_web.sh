#!/usr/bin/env bash
set -e
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

echo "启动 Web 服务..."
# 不同步
uv run uvicorn web:app --host 127.0.0.1 --port 8000

# 启动时同步股票列表:
# SYNC_STOCKS=1 uv run uvicorn web:app --host 127.0.0.1 --port 8000
