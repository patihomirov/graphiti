#!/bin/bash

# Директория для логов
MCP_LOG_DIR="$HOME/graphiti/mcp_server/logs"
mkdir -p "$MCP_LOG_DIR"
MCP_LOG_FILE="$MCP_LOG_DIR/server.log"

# Защита от дублирования процесса
LOCKFILE="/tmp/graphiti-mcp.lock"

# Функция очистки — гарантированно снимает lock и удаляет lock-файл
cleanup() {
  local exit_code=$?
  # Снимаем flock с дескриптора 200
  flock -u 200 2>/dev/null
  # Удаляем lock-файл
  rm -f "$LOCKFILE"
  echo "[run-mcp] $(date '+%Y-%m-%d %H:%M:%S') Cleanup done (exit=$exit_code)" >> "$MCP_LOG_FILE"
  exit $exit_code
}
trap cleanup EXIT

# Трапим сигналы, чтобы гарантированно вызвать cleanup
trap 'echo "[run-mcp] $(date '+%Y-%m-%d %H:%M:%S') Received SIGTERM" >> "$MCP_LOG_FILE"; exit 0' SIGTERM
trap 'echo "[run-mcp] $(date '+%Y-%m-%d %H:%M:%S') Received SIGINT" >> "$MCP_LOG_FILE"; exit 0' SIGINT
trap 'echo "[run-mcp] $(date '+%Y-%m-%d %H:%M:%S') Received SIGQUIT" >> "$MCP_LOG_FILE"; exit 0' SIGQUIT
trap 'echo "[run-mcp] $(date '+%Y-%m-%d %H:%M:%S') Received SIGHUP" >> "$MCP_LOG_FILE"; exit 0' SIGHUP

exec 200>"$LOCKFILE"

# Проверяем, жив ли процесс, который держит lock
if ! flock -n 200; then
  # Lock занят — читаем PID из lock-файла
  LOCKED_PID=$(cat "$LOCKFILE" 2>/dev/null)
  if [ -n "$LOCKED_PID" ] && kill -0 "$LOCKED_PID" 2>/dev/null; then
    echo "[run-mcp] $(date '+%Y-%m-%d %H:%M:%S') Graphiti MCP already running (PID $LOCKED_PID), skipping" >> "$MCP_LOG_FILE"
    echo "[run-mcp] Graphiti MCP already running (PID $LOCKED_PID), skipping" >&2
    exit 0
  fi
  # PID мёртв или lock-файл пустой — берём lock принудительно
  echo "[run-mcp] $(date '+%Y-%m-%d %H:%M:%S') Stale lock from PID ${LOCKED_PID:-unknown}, removing" >> "$MCP_LOG_FILE"
  flock -x 200
fi

# Пишем свой PID в lock-файл
echo $$ > "$LOCKFILE"

# Авто-запуск Neo4j, если контейнер не запущен
if ! docker ps --filter name=graphiti-neo4j --format '{{.Names}}' | grep -q graphiti-neo4j; then
  if docker ps -a --filter name=graphiti-neo4j --format '{{.Names}}' | grep -q graphiti-neo4j; then
    # Контейнер есть, но остановлен — запускаем
    docker start graphiti-neo4j
  else
    # Контейнера нет — создаём
    docker run -d \
      --name graphiti-neo4j \
      --restart unless-stopped \
      -p 7687:7687 \
      -p 7474:7474 \
      -e NEO4J_AUTH=neo4j/password \
      -v graphiti-neo4j-data:/data \
      neo4j:5
  fi
  # Ждём пока Neo4j поднимется
  for i in $(seq 1 30); do
    if curl -s -u neo4j:password http://localhost:7474/ > /dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

cd ~/graphiti/mcp_server
source .venv/bin/activate
export OPENAI_API_KEY=$(cat ~/.o3_llm_token)
export CONFIG_PATH=config/config.ozon.yaml
export PYTHONPATH=src
echo "[run-mcp] $(date '+%Y-%m-%d %H:%M:%S') Starting Graphiti MCP server (PID $$)" >> "$MCP_LOG_FILE"
exec python3 -m graphiti_mcp_server --transport stdio --config config/config.ozon.yaml
