#!/bin/bash
# Скрипт запуска Graphiti MCP как systemd-сервиса (HTTP-транспорт)
# Этот скрипт НЕ использует flock — systemd сам гарантирует единственность процесса.

set -e

LOCKFILE="/tmp/graphiti-mcp.lock"
MCP_LOG_DIR="$HOME/graphiti/mcp_server/logs"
mkdir -p "$MCP_LOG_DIR"
MCP_LOG_FILE="$MCP_LOG_DIR/service.log"

# Очистка lock-файла при любом завершении
cleanup() {
  local code="${1:-$?}"
  rm -f "$LOCKFILE"
  echo "[run-mcp-service] $(date '+%Y-%m-%d %H:%M:%S') Service stopped (exit=$code)" >> "$MCP_LOG_FILE"
  exit $code
}
trap cleanup EXIT SIGTERM SIGINT

# Пишем PID в lock-файл для совместимости со старым скриптом
echo $$ > "$LOCKFILE"

# Проверка/запуск Neo4j
if ! docker ps --filter name=graphiti-neo4j --format '{{.Names}}' | grep -q graphiti-neo4j; then
  if docker ps -a --filter name=graphiti-neo4j --format '{{.Names}}' | grep -q graphiti-neo4j; then
    docker start graphiti-neo4j
  else
    docker run -d \
      --name graphiti-neo4j \
      --restart unless-stopped \
      -p 7687:7687 \
      -p 7474:7474 \
      -e NEO4J_AUTH=neo4j/password \
      -e NEO4J_dbms_memory_heap_max__size=1G \
      -v graphiti-neo4j-data:/data \
      neo4j:5
  fi
  for i in $(seq 1 30); do
    if curl -s -u neo4j:password http://localhost:7474/ > /dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

cd ~/graphiti/mcp_server || exit 1
source .venv/bin/activate
export OPENAI_API_KEY=$(cat ~/.o3_llm_token)
export CONFIG_PATH=config/config.ozon.yaml
export PYTHONPATH=src
# Cap graphiti's parallel extraction/dedupe LLM fan-out. Default SEMAPHORE_LIMIT=20
# fires up to 20 concurrent /chat/completions bursts, which trips the gateway's
# concurrent-request quota (429) on rich episodes. Lower it so bursts stay well
# under the quota; retries + backoff then handle the rare 429.
export SEMAPHORE_LIMIT=4

echo "[run-mcp-service] $(date '+%Y-%m-%d %H:%M:%S') Starting Graphiti MCP HTTP on 127.0.0.1:8123 (PID $$)" >> "$MCP_LOG_FILE"

set +e  # отключаем set -e чтобы сохранить exit code Python-процесса
python3 -m graphiti_mcp_server \
  --transport http \
  --host 127.0.0.1 \
  --port 8123 \
  --config config/config.ozon.yaml
EXIT_CODE=$?
cleanup $EXIT_CODE
