"""Configuration schemas with pydantic-settings and YAML support."""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source for loading from YAML files."""

    def __init__(self, settings_cls: type[BaseSettings], config_path: Path | None = None):
        super().__init__(settings_cls)
        self.config_path = config_path or Path('config.yaml')

    def _expand_env_vars(self, value: Any) -> Any:
        """Recursively expand environment variables in configuration values."""
        if isinstance(value, str):
            # Support ${VAR} and ${VAR:default} syntax
            import re

            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(3) if match.group(3) is not None else ''
                return os.environ.get(var_name, default_value)

            pattern = r'\$\{([^:}]+)(:([^}]*))?\}'

            # Check if the entire value is a single env var expression
            full_match = re.fullmatch(pattern, value)
            if full_match:
                result = replacer(full_match)
                # Convert boolean-like strings to actual booleans
                if isinstance(result, str):
                    lower_result = result.lower().strip()
                    if lower_result in ('true', '1', 'yes', 'on'):
                        return True
                    elif lower_result in ('false', '0', 'no', 'off'):
                        return False
                    elif lower_result == '':
                        # Empty string means env var not set - return None for optional fields
                        return None
                return result
            else:
                # Otherwise, do string substitution (keep as strings for partial replacements)
                return re.sub(pattern, replacer, value)
        elif isinstance(value, dict):
            return {k: self._expand_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._expand_env_vars(item) for item in value]
        return value

    def get_field_value(self, field_name: str, field_info: Any) -> Any:
        """Get field value from YAML config."""
        return None

    def __call__(self) -> dict[str, Any]:
        """Load and parse YAML configuration."""
        if not self.config_path.exists():
            return {}

        with open(self.config_path) as f:
            raw_config = yaml.safe_load(f) or {}

        # Expand environment variables
        return self._expand_env_vars(raw_config)


class ServerConfig(BaseModel):
    """Server configuration."""

    transport: str = Field(
        default='http',
        description='Transport type: http (default, recommended), stdio, or sse (deprecated)',
    )
    host: str = Field(default='0.0.0.0', description='Server host')
    port: int = Field(default=8000, description='Server port')


class OpenAIProviderConfig(BaseModel):
    """OpenAI provider configuration."""

    api_key: str | None = None
    api_url: str = 'https://api.openai.com/v1'
    organization_id: str | None = None


class AzureOpenAIProviderConfig(BaseModel):
    """Azure OpenAI provider configuration."""

    api_key: str | None = None
    api_url: str | None = None
    api_version: str = '2024-10-21'
    deployment_name: str | None = None
    use_azure_ad: bool = False


class AnthropicProviderConfig(BaseModel):
    """Anthropic provider configuration."""

    api_key: str | None = None
    api_url: str = 'https://api.anthropic.com'
    max_retries: int = 3


class GeminiProviderConfig(BaseModel):
    """Gemini provider configuration."""

    api_key: str | None = None
    project_id: str | None = None
    location: str = 'us-central1'


class GroqProviderConfig(BaseModel):
    """Groq provider configuration."""

    api_key: str | None = None
    api_url: str = 'https://api.groq.com/openai/v1'


class VoyageProviderConfig(BaseModel):
    """Voyage AI provider configuration."""

    api_key: str | None = None
    api_url: str = 'https://api.voyageai.com/v1'
    model: str = 'voyage-3'


class LLMProvidersConfig(BaseModel):
    """LLM providers configuration."""

    openai: OpenAIProviderConfig | None = None
    azure_openai: AzureOpenAIProviderConfig | None = None
    anthropic: AnthropicProviderConfig | None = None
    gemini: GeminiProviderConfig | None = None
    groq: GroqProviderConfig | None = None
    zai: OpenAIProviderConfig | None = None


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: str = Field(default='openai', description='LLM provider')
    model: str = Field(default='gpt-5.5', description='Model name')
    temperature: float | None = Field(
        default=None, description='Temperature (optional, defaults to None for reasoning models)'
    )
    max_tokens: int = Field(default=4096, description='Max tokens')
    structured_output_mode: Literal['json_schema', 'json_object'] = Field(
        default='json_schema',
        description=(
            'Structured output mode for OpenAIGenericClient: json_schema requests native '
            'schema enforcement; json_object injects the schema into the prompt for '
            'OpenAI-compatible providers that do not reliably honor json_schema.'
        ),
    )
    providers: LLMProvidersConfig = Field(default_factory=LLMProvidersConfig)


class EmbedderProvidersConfig(BaseModel):
    """Embedder providers configuration."""

    openai: OpenAIProviderConfig | None = None
    azure_openai: AzureOpenAIProviderConfig | None = None
    gemini: GeminiProviderConfig | None = None
    voyage: VoyageProviderConfig | None = None


class EmbedderConfig(BaseModel):
    """Embedder configuration."""

    provider: str = Field(default='openai', description='Embedder provider')
    model: str = Field(default='text-embedding-3-small', description='Model name')
    dimensions: int = Field(default=1536, description='Embedding dimensions')
    providers: EmbedderProvidersConfig = Field(default_factory=EmbedderProvidersConfig)


class Neo4jProviderConfig(BaseModel):
    """Neo4j provider configuration."""

    uri: str = 'bolt://localhost:7687'
    username: str = 'neo4j'
    password: str | None = None
    database: str = 'neo4j'
    use_parallel_runtime: bool = False


class FalkorDBProviderConfig(BaseModel):
    """FalkorDB provider configuration."""

    uri: str = 'redis://localhost:6379'
    username: str | None = None
    password: str | None = None
    database: str = 'default_db'


class DatabaseProvidersConfig(BaseModel):
    """Database providers configuration."""

    neo4j: Neo4jProviderConfig | None = None
    falkordb: FalkorDBProviderConfig | None = None


class DatabaseConfig(BaseModel):
    """Database configuration."""

    provider: str = Field(default='falkordb', description='Database provider')
    providers: DatabaseProvidersConfig = Field(default_factory=DatabaseProvidersConfig)


class EntityTypeConfig(BaseModel):
    """Entity type configuration.

    If ``name`` matches a model registered in ``models.entity_types.ENTITY_TYPES``,
    the rich Pydantic model (with its attributes and extraction instructions) is
    registered with graphiti-core. Otherwise a documentation-only model is built
    from ``name`` + ``description`` for backward compatibility.
    """

    name: str
    description: str


class EdgeTypeConfig(BaseModel):
    """Edge (fact) type configuration.

    Mirrors :class:`EntityTypeConfig`. If ``name`` matches a model registered in
    ``models.edge_types.EDGE_TYPES``, the rich Pydantic model is registered with
    graphiti-core. Otherwise a documentation-only model is built from
    ``name`` + ``description``.
    """

    name: str
    description: str


class EdgeTypeMapEntry(BaseModel):
    """Maps an ordered (source entity type, target entity type) pair to the edge
    type names permitted between them.

    Use ``'Entity'`` as a wildcard for either endpoint, matching graphiti-core's
    ``edge_type_map`` convention.
    """

    source: str = Field(default='Entity', description='Source entity type name')
    target: str = Field(default='Entity', description='Target entity type name')
    edge_types: list[str] = Field(
        default_factory=list, description='Edge type names allowed for this pair'
    )


class GraphitiAppConfig(BaseModel):
    """Graphiti-specific configuration."""

    group_id: str = Field(default='main', description='Group ID')
    episode_id_prefix: str | None = Field(default='', description='Episode ID prefix')
    user_id: str = Field(default='mcp_user', description='User ID')
    entity_types: list[EntityTypeConfig] = Field(default_factory=list)
    edge_types: list[EdgeTypeConfig] = Field(default_factory=list)
    edge_type_map: list[EdgeTypeMapEntry] = Field(default_factory=list)

    def model_post_init(self, __context) -> None:
        """Convert None to empty string for episode_id_prefix."""
        if self.episode_id_prefix is None:
            self.episode_id_prefix = ''


class ResilienceConfig(BaseModel):
    """Resilience configuration for the episode write path.

    Provides fail-fast backpressure via a circuit breaker and durable on-disk
    spooling of episodes that fail to ingest, so writes survive 429 waves and
    process restarts instead of being silently lost.
    """

    enabled: bool = Field(default=True, description='Enable resilience (breaker + spool)')
    max_queue_depth: int = Field(
        default=20, description='Max total queued episodes across all group_ids before fail-fast'
    )
    journal_max_pending: int = Field(
        default=500,
        ge=0,
        description='Soft ceiling for durable-journal intake (pending+processing), '
        'decoupled from max_queue_depth (the legacy in-memory limit). Journal rows '
        'live on disk and survive restarts, so a wider ceiling is safe: backpressure '
        'triggers at pending+processing >= journal_max_pending (plus '
        'enqueue_breakthrough_max_pending while the circuit is open) instead of the '
        'RAM-protecting 20 of the in-memory queue.',
    )
    failure_threshold: int = Field(
        default=3, description='Consecutive transient failures to trip the breaker open'
    )
    open_timeout_seconds: float = Field(
        default=30.0, description='Cooldown before the breaker tries a half-open probe'
    )
    probe_timeout_seconds: float = Field(
        default=60.0,
        description='A half-open probe not resolved within this time is re-granted '
        '(guards against a consumed-but-forgotten probe stranding the breaker)',
    )
    spool_enabled: bool = Field(default=True, description='Enable disk spooling of failed episodes')
    spool_dir: str = Field(
        default='~/.graphiti/spool',
        description='Directory for the episode spool (created on service start when spooling is enabled)',
    )
    retryer_interval_seconds: float = Field(
        default=15.0, description='Interval between spool scan ticks'
    )
    spool_backoff_base_seconds: float = Field(
        default=30.0, description='Base (in seconds) for exponential retry backoff'
    )
    retry_backoff_cap_seconds: float = Field(
        default=480.0,
        ge=1.0,
        description='Hard cap (seconds) on the exponential retry backoff applied to '
        'a journal row retry (and the spool retryer due-window). Without a cap a 30s '
        'base grows past 1920s (~32 min) by attempt 7, so a degraded backend leaves '
        'rows sleeping for hours while done stays frozen. The cap keeps episodes '
        'returning to work in minutes; StormThrottle and the per-channel rate '
        'limiter still gate the actual request rate. It is applied AFTER the storm '
        'multiplier so even a storming model never sleeps past the cap.',
    )
    max_spool_attempts: int = Field(
        default=10, description='Max retry attempts before an episode is moved to failed/'
    )
    permanent_fail_fast_attempts: int = Field(
        default=2,
        ge=1,
        description='RCA C1 fast-fail: after this many PERMANENT (non-transient) '
        'failure attempts a journal row is surrendered to ``failed`` immediately, '
        'instead of exhausting ``max_spool_attempts``. Permanent classes: pydantic '
        'ValidationError, neo4j client-side data errors (not network/auth), openai '
        'SDK 4xx (except 429), graphiti validation errors. Gives a malformed '
        'episode at most ``permanent_fail_fast_attempts`` attempts before it stops '
        'burning worker-slots (the 375/537/540 poison-zombie cycle).',
    )
    journal_poison_repeat_threshold: int = Field(
        default=3,
        ge=2,
        description='RCA C2 poison-detection: how many consecutive failures with the '
        'SAME normalized error signature (``repeat_signature``) before the row is '
        'quarantined to ``failed`` ("poison: repeated signature xN") - independent of '
        '``max_spool_attempts``. Catches deterministic timeouts / stream timeouts '
        'that would otherwise spin through all 10 attempts. The counter resets to 1 '
        'whenever the error signature changes.',
    )
    episode_wall_deadline_seconds: float = Field(
        default=21600.0,
        ge=0.0,
        description='RCA C3 wall-clock deadline per episode: an episode that has '
        'already started processing (attempt >= 1) and whose ``created_at`` is older '
        'than this is surrendered to ``failed`` ("deadline exceeded") at claim time '
        'and by the per-row watchdog - a hard finite guarantee regardless of attempt '
        'count. 0 disables the deadline. Rows that never attempted (attempt=0, e.g. '
        'the deep FIFO backlog) are NOT subject, so legitimate backlog is not killed '
        'on arrival.',
    )
    journal_enabled: bool = Field(
        default=False,
        description='Enable the durable SQLite write-ahead journal as the source of truth '
        'for the episode write path. When enabled, every episode plan is written to the '
        'journal DB before processing and workers claim rows from it, so episodes survive '
        'hard kills (SIGKILL) instead of being lost with the in-memory queue.',
    )
    journal_path: str | None = Field(
        default=None,
        description='Path to the SQLite journal DB. Defaults to ~/.graphiti/journal.db '
        'when journal_enabled.',
    )
    journal_lease_seconds: float = Field(
        default=300.0,
        description='Worker lease duration for claimed journal rows. A processing row '
        'whose lease_until has passed is considered stale and requeued on startup '
        '(heals hard-killed in-flight episodes).',
    )
    journal_workers: int = Field(
        default=1,
        ge=1,
        description='Size of the global journal worker pool. Workers claim rows from '
        'any group; a per-group processing lock keeps each group strictly FIFO, so the '
        'pool size bounds cross-group concurrency without ever parallelizing within a '
        'group (which would break previous-episode context ordering). With '
        'serial-leader + parallel-follower mode (queue_service), non-serial rows '
        '(explicit previous_episode_uuids) may run in parallel within a group.',
    )
    journal_follower_limit: int = Field(
        default=0,
        ge=0,
        description='Per-group cap on parallel in-flight NON-SERIAL rows processed by '
        'follower workers (0 = auto-derive as max(1, semaphore_limit // 2)). The '
        'group leader processes serial rows strictly FIFO under the group lock; '
        'followers drain requires_serial=0 rows in parallel up to this limit. The '
        'global semaphore_limit still bounds total in-flight across all groups.',
    )
    auto_previous_episode_uuid: bool = Field(
        default=False,
        description='Opt-in: when True and add_memory is called WITHOUT an explicit '
        'previous_episode_uuids (and without saga / update_communities), the server '
        'auto-substitutes previous_episode_uuids = [uuid of the newest materialized '
        'episode of the group] before enqueuing. This makes the episode parallel-safe '
        '(requires_serial=0) at the cost of narrower context (exactly one predecessor '
        'instead of auto-retrieval of the recent history). Defaults to False to keep '
        'the full-context auto-retrieval behaviour for every agent.',
    )
    semaphore_limit: int = Field(
        default=10,
        ge=1,
        description='Global cap on concurrent graphiti.add_episode calls (the LLM '
        'extraction + graph write). Shared by the journal worker pool and the direct '
        'add_memory path via the server-wide semaphore.',
    )
    journal_steward_interval_seconds: float = Field(
        default=15.0,
        ge=1.0,
        description='Interval at which the lease steward scans for processing rows '
        'whose owner vanished. The steal guard never touches rows owned by the '
        'current worker, so long in-flight LLM calls are never double-processed.',
    )
    journal_grace_seconds: float = Field(
        default=60.0,
        ge=1.0,
        description='How far past an expired lease a processing row must be before '
        'another worker may reclaim it.',
    )
    journal_processing_timeout_seconds: float = Field(
        default=1500.0,
        ge=60.0,
        description='Absolute wall-clock budget for a single graphiti.add_episode call '
        'inside the journal worker (per model attempt). A hung LLM call past this '
        'boundary is aborted and the row is released to retry (exponential backoff), '
        'so a wedged external call cannot strand a serial group for hours. 1500s '
        'sits above the worst observed add_episode (~1350s); the per-row watchdog '
        '(lease_seconds + journal_grace_seconds) still covers the heartbeat path.',
    )
    model_fallbacks: list[str] = Field(
        default_factory=list,
        description=(
            'Ordered fallback LLM models (gateway model IDs) tried by the journal '
            'worker when the active extraction model (llm.model) fails transiently '
            '(RateLimitError / EmptyResponseError / transport error). Order matters: '
            'the first not-yet-attempted model wins. Fallbacks use the same '
            'max_tokens as llm.max_tokens. Empty list = failover off.'
        ),
    )
    zai_fallback_enabled: bool = Field(
        default=False,
        description=(
            'Enable the zai (Z.AI via local proxy) overflow extraction channel. '
            'When True (and a providers.zai block is configured) the server builds '
            'a second OpenAIGenericClient whose traffic goes to a separate provider '
            'quota, and the journal worker inserts the zai channel between the '
            'active ozon model and resilience.model_fallbacks. One episode is '
            'always processed by a single model end-to-end (graph consistency). '
            'Default OFF: flipping it changes production failover behaviour and is '
            'done by rollout, not by default.'
        ),
    )
    zai_loss_strikes: int = Field(
        default=3,
        ge=1,
        description=(
            'Consecutive zai proxy failures (502 / 5xx / connection / authorization) '
            'after which the zai channel is auto-rolled-back: marked unavailable on '
            'a cooldown so only the ozon channel (and model_fallbacks) serve '
            'episodes. A 429 or any success resets the strike counter.'
        ),
    )
    zai_model: str = Field(
        default='zai/glm-5.3',
        description=(
            'Model id served by the zai overflow provider (providers.zai), in the '
            'provider-qualified opencode convention (e.g. zai/glm-5.3). Doubles as '
            "the (provider, model) channel key for per-channel reputation."
        ),
    )
    zai_overflow_pending_threshold: int = Field(
        default=0,
        ge=0,
        description=(
            'Backlog gate: the zai overflow channel participates in failover '
            'candidates only while the journal is backed up (unfinished '
            'pending+processing >= threshold). A value of 0 keeps the legacy '
            'behaviour: zai is offered whenever it is enabled. With a value >0 '
            'the zai channel is excluded from the candidate list *before* the '
            'per-model health reorder, so a reputation-driven reorder can never '
            'promote it to first while the backlog is under control.'
        ),
    )
    enqueue_breakthrough_max_pending: int = Field(
        default=0,
        ge=0,
        description='How many episodes may be enqueued past max_queue_depth while '
        'the circuit breaker is OPEN (bounded breakthrough intake). A value of 0 '
        'keeps the current behaviour: an open breaker rejects every new episode. '
        'With a value >0 and a configured model_fallbacks, canaries and critical '
        'episodes still reach the durable journal during a 429 storm and drain on '
        'a live fallback channel instead of being rejected with '
        'graphiti_backpressure:.',
    )
    zai_first_timeout_window_seconds: float = Field(
        default=300.0,
        ge=1.0,
        description='Sliding-window length (seconds) over which recent transient '
        'timeout / stream-timeout / connection errors on the ACTIVE ozon channel '
        'are counted for the zai-first balance. When the active channel shows at '
        'least ``zai_first_timeout_threshold`` such events inside the window, the '
        'zai overflow channel is promoted to the FIRST failover candidate (before '
        'the active model), so a formally "healthy" (by 429 reputation) but '
        'actually timeouting ozon line stops dominating the queue. 0 keeps the '
        'legacy failover order.',
    )
    zai_first_timeout_threshold: int = Field(
        default=2,
        ge=1,
        description='Minimum number of recent timeout / stream-timeout / '
        'connection-error events on the active ozon channel (inside '
        '``zai_first_timeout_window_seconds``) that flips failover to zai-first. '
        'A single isolated timeout is noise; a steady stream of them is the '
        'degradation signal this balance reacts to.',
    )
    model_reputation_window_seconds: float = Field(
        default=300.0,
        ge=1.0,
        description='Sliding-window length (seconds) for per-model reputation '
        'tracking. 429 / empty / transient / success events older than the window '
        'are pruned.',
    )
    model_reputation_429_per_min_threshold: float = Field(
        default=5.0,
        ge=0.0,
        description='A model is treated as unhealthy when its 429 rate over the '
        'reputation window exceeds this many events per minute.',
    )
    model_reputation_429_cooldown_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description='A 429 seen within this many seconds marks the model unhealthy '
        'regardless of 429 rate (fresh rate-limit avoidance).',
    )
    # --- Per-model storm throttle (adaptive pacing on 429 storms) ---
    # Defaults are active but conservative: storm entry needs a real burst
    # (>=6 429 within 60s) so isolated 429s never slow anything; only the
    # storming model is affected (per-model isolation), other models keep their
    # normal tempo.
    storm_min_request_interval_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description='Minimum pause (seconds) between consecutive LLM calls to a '
        'model in storm state. 0 = no inter-request pacing (Retry-After and the '
        'concurrency ceiling still apply). Only the storming model is slowed; '
        'healthy models keep their normal rate. For the always-active pacing '
        'feature (2026-10-21, config-driven) the same knob doubles as the '
        'inter-episode pause: after finishing an episode, a journal worker '
        'sleeps ``value * number of active channels`` before the next claim, '
        'so an idle pool never starts a whole batch of episodes at once '
        '(which is what burst the gateway into 429).',
    )
    storm_detection_429_window_seconds: float = Field(
        default=60.0,
        ge=1.0,
        description='Sliding-window length (seconds) over which 429 events are '
        'counted for per-model storm detection (reacts faster than the 300s '
        'reputation window).',
    )
    storm_detection_429_threshold: int = Field(
        default=6,
        ge=1,
        description='A model enters storm state when it records at least this many '
        '429 events within storm_detection_429_window_seconds (or its reputation '
        'per-minute 429 rate meets model_reputation_429_per_min_threshold). '
        '1-2 isolated 429s per hour are NOT a storm.',
    )
    storm_concurrency_ceiling: int = Field(
        default=1,
        ge=1,
        description='Maximum concurrent LLM calls a storming model may serve '
        '(per model). A storming model is capped at this many in-flight calls; '
        'other models are unaffected. The global semaphore_limit still bounds '
        'total in-flight across all models.',
    )
    respect_retry_after: bool = Field(
        default=True,
        description='When True, a Retry-After hint carried by a 429 error is '
        'honoured: the affected model is skipped/hard-paused (per model, capped) '
        'instead of being re-contacted immediately. The client retry backoff also '
        'respects the hint.',
    )
    channel_max_calls_per_minute: int = Field(
        default=0,
        ge=0,
        description='Per-(provider, model) channel always-active rate limit: the '
        'maximum number of LLM chat-completions calls a channel may make within a '
        'sliding 60s window. The journal worker waits for a free slot before each '
        'call instead of bursting packets at the gateway (which cuts whole batches '
        'with 429 and no Retry-After). Uniform stream beats a reactive storm '
        'throttle: the throttle only reacts once the 429s start, the limiter '
        'prevents them. 0 = OFF (legacy: reactive storm throttling only). Config '
        'enables it (e.g. 60).',
    )
    channel_max_embeddings_per_minute: int = Field(
        default=60,
        ge=0,
        description='Per-(base_url, embedding_model) channel always-active rate '
        'limit for the graph embedder, enforced by a SEPARATE ChannelRateLimiter '
        'instance (2026-10-22). Embeddings are the largest request source at the '
        'gateway and share the gateway bucket with chat, so a dedicated sliding-'
        'window ceiling smooths embedding peaks (measured 187/min) without gating '
        'the chat path. The chat limiter and this one never borrow slots from '
        'each other. 0 = OFF.',
    )
    channel_wait_abort_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description='When the per-channel rate limiter estimates a wait longer than '
        'this before a slot frees up, the journal worker skips to the next failover '
        'candidate (or lets the normal backoff take over) instead of queueing behind '
        'a saturated channel.',
    )


class GraphitiConfig(BaseSettings):
    """Graphiti configuration with YAML and environment support."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    graphiti: GraphitiAppConfig = Field(default_factory=GraphitiAppConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)

    # Additional server options
    destroy_graph: bool = Field(default=False, description='Clear graph on startup')

    model_config = SettingsConfigDict(
        env_prefix='',
        env_nested_delimiter='__',
        case_sensitive=False,
        extra='ignore',
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources to include YAML."""
        config_path = Path(os.environ.get('CONFIG_PATH', 'config/config.yaml'))
        yaml_settings = YamlSettingsSource(settings_cls, config_path)
        # Priority: CLI args (init) > env vars > yaml > defaults
        return (init_settings, env_settings, yaml_settings, dotenv_settings)

    def apply_cli_overrides(self, args) -> None:
        """Apply CLI argument overrides to configuration."""
        # Override server settings
        if hasattr(args, 'transport') and args.transport:
            self.server.transport = args.transport
        if hasattr(args, 'host') and args.host:
            self.server.host = args.host
        if hasattr(args, 'port') and args.port is not None:
            self.server.port = args.port

        # Override LLM settings
        if hasattr(args, 'llm_provider') and args.llm_provider:
            self.llm.provider = args.llm_provider
        if hasattr(args, 'model') and args.model:
            self.llm.model = args.model
        if hasattr(args, 'temperature') and args.temperature is not None:
            self.llm.temperature = args.temperature

        # Override embedder settings
        if hasattr(args, 'embedder_provider') and args.embedder_provider:
            self.embedder.provider = args.embedder_provider
        if hasattr(args, 'embedder_model') and args.embedder_model:
            self.embedder.model = args.embedder_model

        # Override database settings
        if hasattr(args, 'database_provider') and args.database_provider:
            self.database.provider = args.database_provider

        # Override Graphiti settings
        if hasattr(args, 'group_id') and args.group_id:
            self.graphiti.group_id = args.group_id
        if hasattr(args, 'user_id') and args.user_id:
            self.graphiti.user_id = args.user_id
