#!/bin/bash
# Обёртка периодической пересборки vector-индекса graphiti (systemd user oneshot).
# Запускает rebuild_vector_index.py в venv сервера. Лог: logs/rebuild_vector_index.log
set -euo pipefail

MCP_DIR="$HOME/graphiti/mcp_server"
cd "$MCP_DIR"
source .venv/bin/activate
export PYTHONPATH=src
exec python3 scripts/rebuild_vector_index.py "$@"
