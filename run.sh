#!/usr/bin/env bash
set -e
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

echo "启动 ChromaDB 服务 (host=${CHROMA_HOST:-localhost}, port=${CHROMA_PORT:-8001})..."
uv run chroma run \
  --host "${CHROMA_HOST:-localhost}" \
  --port "${CHROMA_PORT:-8001}" \
  --path "${CHROMA_PERSIST_DIR:-./data/chroma_db}"
