# Ozon-форк с resilience-стеком (durable SQLite-журнал + multi-model failover + parallel pool)

## Что это

Личный форк **patihomirov/graphiti** поверх `getzep/graphiti`.

Ветка **`release/graphiti-journal`** — стабильный срез (дерево `df6eac8`) с фазами resilience-стека:

- **durable SQLite-журнал** как primary write-path (`journal_enabled: true`);
- **graceful drain** — штатное завершение с обработкой очереди;
- **circuit breaker** — защита от каскадных отказов LLM-провайдера;
- **backpressure** — отказ приёма новых эпизодов при перегрузке вместо тихой потери;
- **disk-спул** — принятые эпизоды пишутся на диск и догоняются ретраями (идемпотентно по uuid);
- **multi-model failover** через `model_fallbacks` — автоматическое переключение моделей;
- **параллельный воркер-пул** с per-group FIFO и serial-зонами;
- **инструмент `search_raw_episodes`** — чтение raw-эпизодов из durable журнала.

## Быстрый старт

```bash
git clone -b release/graphiti-journal https://github.com/patihomirov/graphiti.git
cd graphiti
uv sync --extra dev
```

Запуск MCP-сервера: `mcp_server` (см. `mcp/graphiti/README` в opencode-config; пример systemd-юнита — скилл `graphiti-resilience` в opencode-config).

## Конфиг

Секция resilience (`mcp_server/config/config.ozon.yaml`):

```yaml
resilience:
  journal_enabled: true
  journal_path: ~/.graphiti/journal.db
  journal_workers: 1
  semaphore_limit: 10
  model_fallbacks:
    - "AgentLLM-default"
```

`max_tokens: 32768` — для reasoning-моделей (важно: reasoning съедает budget контекста → пустой `content` при `8192`).

## NOTE

Полный runbook / Probleme — в скилле `graphiti-resilience` репозитория `ptikhomirovan/opencode-config`.
