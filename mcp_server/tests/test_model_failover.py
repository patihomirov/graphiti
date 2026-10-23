"""Unit tests for the worker-level multi-model failover (Phase 3, variant 'b').

Covers the journal worker choosing one model per episode and failing over to the
next ``resilience.model_fallbacks`` candidate on a transient error (rate limit /
empty response), the ``attempted_models`` anti-loop column, the exhaustion path
falling through to the standard bump_retry + breaker flow, and the schema
migration. Everything runs against a tmp_path SQLite DB with fake graphiti/LLM
clients - no network, no graphiti-core data.
"""

import asyncio
import json
import sqlite3
import time

import pytest
from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError

from config.schema import ResilienceConfig
from services.model_reputation import ModelReputationTracker
from services.queue_journal import QueueJournal
from services.queue_service import QueueService


def make_plan(**over):
    plan = {
        'group_id': 'g1',
        'name': 'ep',
        'episode_body': 'some body',
        'source_description': 'desc',
        'source': 'text',
        'uuid': 'u-1',
        'reference_time': '2024-06-01T00:00:00+00:00',
        'excluded_entity_types': None,
        'previous_episode_uuids': None,
        'custom_extraction_instructions': None,
        'update_communities': False,
        'saga': None,
        'saga_previous_episode_uuid': None,
    }
    plan.update(over)
    return plan


def make_config(db_path, **over) -> ResilienceConfig:
    cfg = dict(
        journal_enabled=True,
        journal_path=str(db_path),
        journal_lease_seconds=300.0,
        spool_enabled=False,
        spool_backoff_base_seconds=0.01,
    )
    cfg.update(over)
    return ResilienceConfig(**cfg)


def make_builder():
    """Forward a subset of kwargs, mirroring the production episode_builder."""

    def _builder(plan: dict) -> dict:
        return {
            'name': plan['name'],
            'episode_body': plan['episode_body'],
            'source_description': plan['source_description'],
            'source': plan['source'],
            'group_id': plan['group_id'],
            'uuid': plan.get('uuid'),
        }

    return _builder


class FakeLLM:
    """Fake LLM client exposing the model-override hook and recording calls."""

    def __init__(self, model: str = 'model-A'):
        self.model = model
        self.seen: list[str | None] = []
        self._override: str | None = None

    def set_model_override(self, model: str | None) -> None:
        self.seen.append(model)
        self._override = model


class FailoverGraphiti:
    """Fake graphiti client whose add_episode fails on configured models."""

    def __init__(
        self,
        llm: FakeLLM,
        fail_on: set[str] | None = None,
        fail_error: type[Exception] = RateLimitError,
    ):
        self.llm_client = llm
        self.fail_on = fail_on or set()
        self.fail_error = fail_error
        self.calls: list[tuple[str | None, dict]] = []

    async def add_episode(self, **kwargs):
        model = self.llm_client._override or self.llm_client.model
        self.calls.append((model, kwargs))
        if model in self.fail_on:
            raise self.fail_error(f'transient failure on {model}')
        return None


def attempted_for(journal: QueueJournal, uuid: str) -> list[str]:
    """Read the attempted_models JSON column of a row, given its uuid."""
    conn = sqlite3.connect(journal.db_path)
    try:
        row = conn.execute(
            'SELECT attempted_models FROM episode_queue WHERE uuid=?', (uuid,)
        ).fetchone()
    finally:
        conn.close()
    if row is None or not row[0]:
        return []
    return json.loads(row[0])


async def wait_done(service: QueueService, count: int = 1, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if await service.journal.count_done() >= count:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f'Timed out waiting for {count} done rows')


async def enqueue(service: QueueService, uuid: str) -> None:
    await service.add_episode(
        group_id='g1',
        name=f'ep-{uuid}',
        content='body',
        source_description='d',
        episode_type='text',
        entity_types=None,
        uuid=uuid,
    )


class TestFailoverFallsBack:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rate_limit_falls_back_to_next_model(self, tmp_path):
        """Primary (model-A) 429s -> the episode is retried on model-B -> done."""
        llm = FakeLLM('model-A')
        client = FailoverGraphiti(llm, fail_on={'model-A'})
        service = QueueService(
            make_config(tmp_path / 'f.db', model_fallbacks=['model-B'])
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            await enqueue(service, 'u-fb')
            await wait_done(service)
            # Override pinned to primary first, then the fallback, then reset.
            assert llm.seen[0] == 'model-A'
            assert 'model-B' in llm.seen
            assert llm.seen[-1] is None
            # The successful call ran on the fallback.
            model, _ = client.calls[-1]
            assert model == 'model-B'
            # attempted_models recorded the failed primary model only.
            assert attempted_for(service.journal, 'u-fb') == ['model-A']
            # One episode, one row, no duplicate claims left behind.
            assert await service.journal.claim_next('g1', 'w') is None
            assert await service.journal.count_done() == 1
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_response_falls_back(self, tmp_path):
        """EmptyResponseError on primary is transient and also fails over."""
        llm = FakeLLM('model-A')
        client = FailoverGraphiti(llm, fail_on={'model-A'}, fail_error=EmptyResponseError)
        service = QueueService(
            make_config(tmp_path / 'empty.db', model_fallbacks=['model-B'])
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            await enqueue(service, 'u-empty')
            await wait_done(service)
            assert client.calls[-1][0] == 'model-B'
            assert attempted_for(service.journal, 'u-empty') == ['model-A']
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_breaker_sees_only_final_outcome(self, tmp_path):
        """No record_failure between models: breaker stays closed on a fallback win."""
        llm = FakeLLM('model-A')
        client = FailoverGraphiti(llm, fail_on={'model-A'})
        service = QueueService(
            make_config(tmp_path / 'bk.db', model_fallbacks=['model-B'])
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            await enqueue(service, 'u-bk')
            await wait_done(service)
            snap = await service.circuit_breaker.get_snapshot()
            assert snap['state'] == 'closed'
            assert snap['failure_count'] == 0
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_fallbacks_changes_nothing(self, tmp_path):
        """Default (empty model_fallbacks): a single primary attempt, done."""
        llm = FakeLLM('model-A')
        client = FailoverGraphiti(llm, fail_on=set())
        service = QueueService(make_config(tmp_path / 'nf.db'))
        await service.initialize(client, episode_builder=make_builder())
        try:
            await enqueue(service, 'u-nf')
            await wait_done(service)
            assert len(client.calls) == 1
            assert client.calls[0][0] == 'model-A'
            assert llm.seen == ['model-A', None]
        finally:
            await service.close()


class TestExhaustionPath:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_models_exhausted_uses_standard_retry(self, tmp_path):
        """Every candidate (incl. no-override retry) fails -> bump_retry + breaker.

        The inline fallback attempts do NOT increment ``attempt``; only the final
        exhausted failure bumps it. RCA C6: attempted_models is NOT reset by
        bump_retry anymore - the next claim continues from the next
        not-yet-attempted channel (or, when every channel was tried, the
        standard non-override path via the single ``None`` candidate).
        """
        llm = FakeLLM('model-A')
        client = FailoverGraphiti(llm, fail_on={'model-A', 'model-B'})
        service = QueueService(
            make_config(
                tmp_path / 'ex.db',
                model_fallbacks=['model-B'],
                spool_backoff_base_seconds=3600.0,  # keep the row far in the future
            )
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            await enqueue(service, 'u-ex')
            # Wait until the row is backed off (pending with a future retry) -
            # i.e. the exhaustion path ran. poll until attempt >= 1.
            deadline = asyncio.get_running_loop().time() + 5.0
            attempt = 0
            while asyncio.get_running_loop().time() < deadline:
                conn = sqlite3.connect(service.journal.db_path)
                attempt = conn.execute(
                    'SELECT attempt FROM episode_queue WHERE uuid=?', ('u-ex',)
                ).fetchone()[0]
                conn.close()
                if attempt >= 1:
                    break
                await asyncio.sleep(0.02)
            assert attempt == 1
            assert await service.journal.count_done() == 0
            assert await service.journal.count_pending() == 1
            # Two add_episode runs: model-A (primary) then model-B (fallback);
            # the chain is exhausted after both, so the standard retry runs.
            models = [m for m, _ in client.calls]
            assert models == ['model-A', 'model-B']
            # Backed off, failed models RECORDED (RCA C6: never reset) so the
            # next claim skips the already-degraded channels.
            assert attempted_for(service.journal, 'u-ex') == ['model-A', 'model-B']
            # Breaker saw exactly one failure (the final exhausted outcome).
            snap = await service.circuit_breaker.get_snapshot()
            assert snap['failure_count'] == 1
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_failover_candidates_skip_attempted(self, tmp_path):
        """attempted models are never re-probed within the same chain."""
        llm = FakeLLM('model-A')
        config = make_config(tmp_path / 'cand.db', model_fallbacks=['model-A', 'model-B'])
        service = QueueService(config)
        try:
            # model-A already attempted -> skip it, go straight to model-B.
            assert service._failover_candidates(['model-A'], llm) == ['model-B']
            # Nothing attempted -> active first, then fallbacks (deduped).
            assert service._failover_candidates([], llm) == ['model-A', 'model-B']
            # RCA C6 "all attempted -> start over": with every model attempted the
            # FULL chain restarts (never a deadlock) instead of [None].
            assert service._failover_candidates(['model-A', 'model-B'], llm) == (
                ['model-A', 'model-B']
            )
            # No llm client -> no active model -> standard None candidate.
            assert service._failover_candidates([], None) == [None]
        finally:
            await service.close()


class TestMigration:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_attempted_models_migrated_from_legacy_schema(self, tmp_path):
        """A phase-1 journal (no requires_serial / attempted_models) migrates."""
        db = tmp_path / 'legacy.db'
        conn = sqlite3.connect(db)
        conn.executescript(
            'CREATE TABLE episode_queue ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT, dedup_key TEXT NOT NULL, uuid TEXT, '
            'group_id TEXT NOT NULL, name TEXT NOT NULL, episode_body TEXT NOT NULL, '
            "plan_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending' "
            "CHECK (status IN ('pending', 'processing', 'done', 'failed')), "
            'attempt INTEGER NOT NULL DEFAULT 0, worker_id TEXT, lease_until TEXT, '
            'first_failure_ts TEXT, last_attempt_ts TEXT, next_retry_at TEXT, '
            'error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)'
        )
        conn.commit()
        conn.close()

        j = QueueJournal(db, lease_seconds=300.0)
        try:
            cols = {
                r[1]
                for r in sqlite3.connect(db).execute('PRAGMA table_info(episode_queue)')
            }
            assert 'attempted_models' in cols
            assert 'requires_serial' in cols
            # Reopening an already-migrated DB is idempotent.
            j2 = QueueJournal(db, lease_seconds=300.0)
            assert 'attempted_models' in {
                r[1] for r in sqlite3.connect(db).execute('PRAGMA table_info(episode_queue)')
            }
            await j2.close()
        finally:
            await j.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_attempted_models_roundtrip(self, tmp_path):
        j = QueueJournal(tmp_path / 'rt.db', lease_seconds=300.0)
        try:
            row_id, _ = await j.enqueue(make_plan(uuid='rt-1'))
            assert await j.get_attempted_models(row_id) == []
            await j.set_attempted_models(row_id, ['model-A', 'model-B'])
            assert await j.get_attempted_models(row_id) == ['model-A', 'model-B']
        finally:
            await j.close()


class TestHealthAndSerial:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_exposes_fallbacks(self, tmp_path):
        llm = FakeLLM('model-A')
        client = FailoverGraphiti(llm, fail_on=set())
        service = QueueService(
            make_config(tmp_path / 'h.db', model_fallbacks=['model-B', 'model-C'])
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            await enqueue(service, 'u-h')
            await wait_done(service)
            snap = await service.get_resilience_snapshot()
            assert snap['journal']['fallbacks'] == ['model-B', 'model-C']
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_seriality_preserved_across_failover(self, tmp_path):
        """Fallback happens inside one claim: FIFO order and no duplicates.

        Phase 4 note: the FIRST episode tries the primary (429) then the
        fallback; once reputation learns the primary is storming, the SECOND
        episode is routed straight to the live fallback. The per-group FIFO
        order and exactly-once semantics are unchanged.
        """
        llm = FakeLLM('model-A')
        client = FailoverGraphiti(llm, fail_on={'model-A'})
        service = QueueService(
            make_config(tmp_path / 'ser.db', model_fallbacks=['model-B'])
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            await enqueue(service, 'u-s1')
            await enqueue(service, 'u-s2')
            await wait_done(service, count=2)
            uuids = [kwargs['uuid'] for _, kwargs in client.calls]
            # u-s1: primary 429 then fallback success; u-s2: fallback only
            # (reputation prefers the now-known-live channel).
            assert uuids[0] == 'u-s1'  # primary attempt
            assert uuids[1] == 'u-s1'  # fallback attempt
            assert uuids[2] == 'u-s2'  # routed straight to the live fallback
            assert not any(u == 'u-s2' for u in uuids[:2])
            assert uuids.index('u-s2') > uuids.index('u-s1')
            assert await service.journal.count_done() == 2
            assert await service.journal.claim_next('g1', 'w') is None
        finally:
            await service.close()


class ReputationClock:
    """Monotonic clock shared with the simple fake below."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_tracker(clock, window=60.0, per_min=5.0, cooldown=60.0):
    return ModelReputationTracker(
        window_seconds=window,
        per_min_threshold=per_min,
        cooldown_seconds=cooldown,
        time_fn=clock,
    )


class TestReputationRouting:
    """Phase 4: _failover_candidates reorders by per-model health."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_healthy_fallback_moves_ahead_of_429_storming_primary(self, tmp_path):
        clock = ReputationClock()
        tracker = make_tracker(clock)
        for _ in range(5):  # model-A 429 rate at/above threshold -> unhealthy
            tracker.record_429('model-A')
        llm = FakeLLM('model-A')
        service = QueueService(
            make_config(tmp_path / 'rep.db', model_fallbacks=['model-B', 'model-C'])
        )
        try:
            # No attempted models, reputation says primary is storming.
            assert service._failover_candidates([], llm, tracker) == [
                'model-B',
                'model-C',
                'model-A',
            ]
            # Reputation is ignored when not supplied (configured order kept).
            assert service._failover_candidates([], llm) == [
                'model-A',
                'model-B',
                'model-C',
            ]
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reputation_keeps_healthy_primary_first(self, tmp_path):
        tracker = make_tracker(ReputationClock())
        llm = FakeLLM('model-A')
        service = QueueService(
            make_config(tmp_path / 'rep2.db', model_fallbacks=['model-B'])
        )
        try:
            assert service._failover_candidates([], llm, tracker) == ['model-A', 'model-B']
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reputation_preserves_anti_loop_skips_attempted(self, tmp_path):
        clock = ReputationClock()
        tracker = make_tracker(clock)
        for _ in range(5):
            tracker.record_429('model-A')
        llm = FakeLLM('model-A')
        service = QueueService(
            make_config(tmp_path / 'rep3.db', model_fallbacks=['model-A', 'model-B'])
        )
        try:
            # model-A already attempted this chain: never re-probed, even though
            # reputation would otherwise drop it behind the fallback.
            assert service._failover_candidates(['model-A'], llm, tracker) == ['model-B']
            # RCA C6 "all attempted -> start over": the full (healthy) chain is
            # restored; the fresh tracker keeps the healthy active model first.
            assert service._failover_candidates(['model-A', 'model-B'], llm, tracker) == (
                ['model-A', 'model-B']
            )
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_failover_records_per_model_reputation(self, tmp_path):
        """A 429 on primary + success on fallback updates reputation + snapshot."""
        llm = FakeLLM('model-A')
        client = FailoverGraphiti(llm, fail_on={'model-A'})
        service = QueueService(
            make_config(tmp_path / 'rep-e2e.db', model_fallbacks=['model-B'])
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            await enqueue(service, 'u-rep')
            await wait_done(service)
            stats = service._reputation.per_model_stats()
            # Primary recorded a 429 (and is thus unhealthy); fallback succeeded.
            assert stats['model-A']['429'] == 1
            assert stats['model-A']['healthy'] is False
            assert stats['model-B']['successes'] == 1
            assert stats['model-B']['healthy'] is True
            # The next claim would prefer the healthy fallback first.
            assert service._failover_candidates([], llm, service._reputation) == [
                'model-B',
                'model-A',
            ]
            # /health snapshot surfaces the per-model block read-only.
            snap = await service.get_resilience_snapshot()
            assert snap['journal']['per_model']['model-A']['429'] == 1
            assert snap['journal']['per_model']['model-B']['healthy'] is True
        finally:
            await service.close()


class TestJSONDecodeTransient:
    """Phase S: malformed JSON bodies are transient (fallback instead of raise)."""

    @pytest.mark.unit
    async def test_json_decode_error_is_transient_and_fails_over(self, tmp_path):
        """A JSONDecodeError on primary falls back to the next model, row done."""
        llm = FakeLLM(model='primary')
        calls = {'n': 0}

        class BrokenJSONFirst:
            llm_client = llm

            async def add_episode(self, **kwargs):
                calls['n'] += 1
                if calls['n'] == 1:
                    raise json.JSONDecodeError(
                        'Unterminated string starting at', '{"a":', 5
                    )
                return None

        service = QueueService(
            make_config(tmp_path / 'jsonfall.db', model_fallbacks=['fallback'])
        )
        await service.initialize(BrokenJSONFirst(), episode_builder=make_builder())
        try:
            await enqueue(service, 'json-1')
            await wait_done(service, 1)
            # primary transiently failed -> fallback tried and succeeded
            # (set_model_override fires setter(model) then setter(None) per
            # attempt, so seen is ['primary', None, 'fallback', None]).
            assert 'primary' in llm.seen
            assert 'fallback' in llm.seen
            assert await service.journal.count_failed() == 0
            assert await service.journal.count_done() == 1
        finally:
            await service.close()

    @pytest.mark.unit
    def test_is_transient_error_classifies_json_decode(self):
        from services.circuit_breaker import is_transient_error

        err = json.JSONDecodeError('Unterminated string', '{"a":', 5)
        assert is_transient_error(err) is True

    @pytest.mark.unit
    def test_reputation_kind_and_failover_eligible(self):
        from services.queue_service import QueueService

        err = json.JSONDecodeError('Unterminated string', '{"a":', 5)
        assert QueueService._reputation_kind(err) == 'transient'

        def setter(_):
            pass

        assert QueueService._failover_eligible(setter, 'model-A', err) is True

    @pytest.mark.unit
    def test_breaker_records_json_decode_failure(self):
        from services.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=1)

        async def _run():
            await breaker.record_failure(
                json.JSONDecodeError('Unterminated string', '{"a":', 5)
            )
            snap = await breaker.get_snapshot()
            return snap

        import asyncio as _asyncio

        snap = _asyncio.run(_run())
        assert snap['state'] == 'open'


class StormPacingClock:
    """Deterministic fake monotonic clock for the storm throttle."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class PacingGraphiti:
    """Fake graphiti that 429s the first N calls, then succeeds."""

    def __init__(self, llm, fail_for: int):
        self.llm_client = llm
        self.fail_remaining = fail_for
        self.calls: list[tuple[str | None, dict, float]] = []

    async def add_episode(self, **kwargs):
        model = self.llm_client._override or self.llm_client.model
        self.calls.append((model, kwargs, time.monotonic()))
        if self.fail_remaining > 0:
            self.fail_remaining -= 1
            raise RateLimitError('pacing 429')
        return None


class IsolationGraphiti:
    """model-A always 429s; model-C (fallback) always succeeds."""

    def __init__(self, llm):
        self.llm_client = llm
        self.calls: list[tuple[str | None, dict, float]] = []

    async def add_episode(self, **kwargs):
        model = self.llm_client._override or self.llm_client.model
        self.calls.append((model, kwargs, time.monotonic()))
        if model == 'model-A':
            raise RateLimitError('storm on A')
        return None


class TestStormPacingE2E:
    """Feature A: a 429 storm on one model slows ONLY that model, nothing is lost."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_storm_pacing_slows_pool_and_keeps_pending(self, tmp_path):
        clock = StormPacingClock()
        llm = FakeLLM('model-A')
        client = PacingGraphiti(llm, fail_for=6)
        service = QueueService(
            make_config(
                tmp_path / 'pace.db',
                model_fallbacks=[],
                spool_backoff_base_seconds=0.01,
                retryer_interval_seconds=0.05,
                storm_detection_429_threshold=6,
                storm_detection_429_window_seconds=60.0,
                storm_min_request_interval_seconds=0.05,
                storm_concurrency_ceiling=1,
            ),
            storm_time_fn=clock,
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            await enqueue(service, 'u-pace')
            await wait_done(service, count=1, timeout=10.0)
            # 6 failures then one success, all on model-A (the only channel).
            assert len(client.calls) == 7
            assert all(m == 'model-A' for m, _, _ in client.calls)
            # Storm entered after the 6th 429.
            assert service._storm.in_storm('model-A') is True
            # The storming-model call rate dropped: the success call is spaced
            # at >= the configured min interval from the last 429 call.
            ts = [ts for _, _, ts in client.calls]
            assert ts[6] - ts[5] >= 0.049
            # /health storm block exposes the storming model.
            snap = await service.get_resilience_snapshot()
            assert snap['storm']['per_model']['model-A']['in_storm'] is True
            # Pending intact: nothing lost while throttling.
            assert await service.journal.count_done() == 1
            assert await service.journal.count_pending() == 0
            assert await service.journal.count_failed() == 0
            # Quiet window + a quiet success clears the storm (probe-like exit).
            clock.advance(200.0)
            service._storm.record_success('model-A')
            assert service._storm.in_storm('model-A') is False
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_storm_only_slows_the_storming_model(self, tmp_path):
        """model-A storming; the healthy fallback is never paced (per-model)."""
        clock = StormPacingClock()
        llm = FakeLLM('model-A')
        client = IsolationGraphiti(llm)
        service = QueueService(
            make_config(
                tmp_path / 'iso.db',
                model_fallbacks=['model-C'],  # healthy fallback channel
                spool_backoff_base_seconds=0.01,
                retryer_interval_seconds=0.05,
                storm_detection_429_threshold=6,
                storm_min_request_interval_seconds=0.2,
                storm_concurrency_ceiling=1,
            ),
            storm_time_fn=clock,
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            # Drive model-A into storm directly (as its 429 events would).
            for _ in range(6):
                service._storm.record_failure('model-A')
            assert service._storm.in_storm('model-A') is True
            # Enqueue an episode: model-A (storming) is attempted first, paced,
            # then the healthy fallback finishes immediately.
            start = time.monotonic()
            await enqueue(service, 'u-iso')
            await wait_done(service, count=1, timeout=10.0)
            elapsed = time.monotonic() - start
            # Only the storming model paid the 0.2s pacing once; the healthy
            # channel was not slowed (a naive global throttle would have been
            # ordered of magnitude slower per row).
            assert elapsed < 1.0
            assert 'model-A' in service._storm_last_call
            assert 'model-C' not in service._storm_last_call
            assert service._storm.in_storm('model-C') is False
            models = [m for m, _, _ in client.calls]
            assert models[0] == 'model-A'  # storming primary attempted (paced)
            assert models[-1] == 'model-C'  # healthy fallback completed the row
            assert await service.journal.count_done() == 1
            assert await service.journal.count_pending() == 0
            assert await service.journal.count_failed() == 0
        finally:
            await service.close()


class TestClientAntiStorm:
    """Client-side anti-storm (Feature A): SDK retries off, Retry-After honoured."""

    @pytest.mark.unit
    def test_openai_client_builds_with_max_retries_zero(self, monkeypatch):
        """L1 killed: the SDK must never re-drive a call under the tenacity layer."""
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        captured: dict[str, dict] = {}

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                captured['kwargs'] = kwargs

        monkeypatch.setattr(
            'graphiti_core.llm_client.openai_generic_client.AsyncOpenAI',
            FakeAsyncOpenAI,
        )
        cfg = LLMConfig(api_key='k', model='m', base_url='http://x/v1')
        OpenAIGenericClient(config=cfg)
        assert captured['kwargs']['max_retries'] == 0
        assert captured['kwargs']['api_key'] == 'k'
        assert captured['kwargs']['base_url'] == 'http://x/v1'

    @pytest.mark.unit
    def test_wait_for_retry_respects_retry_after_capped(self):
        """Retry-After is honoured; a silly huge header cannot stall the queue."""
        from types import SimpleNamespace

        from graphiti_core.llm_client.client import _wait_for_retry

        rs = SimpleNamespace(
            outcome=SimpleNamespace(exception=lambda: RateLimitError(retry_after=5)),
            attempt_number=1,
        )
        assert _wait_for_retry(rs) == 5.0
        rs2 = SimpleNamespace(
            outcome=SimpleNamespace(exception=lambda: RateLimitError(retry_after=200)),
            attempt_number=2,
        )
        assert _wait_for_retry(rs2) == 60.0

    @pytest.mark.unit
    def test_wait_for_retry_exponential_fallback(self):
        from types import SimpleNamespace

        from graphiti_core.llm_client.client import _wait_for_retry

        rs = SimpleNamespace(
            outcome=SimpleNamespace(exception=lambda: RateLimitError()),
            attempt_number=1,
        )
        delay = _wait_for_retry(rs)
        # 2^1 * uniform(0.5, 1.5), capped at 60.
        assert 0.5 <= delay <= 60.0

    @pytest.mark.unit
    def test_retry_after_hint_walks_cause_chain(self):
        import httpx
        from graphiti_core.llm_client.client import _retry_after_hint

        assert _retry_after_hint(RateLimitError(retry_after=7)) == 7.0
        assert _retry_after_hint(RateLimitError()) is None
        req = httpx.Request('POST', 'http://x/chat/completions')
        resp = httpx.Response(429, request=req, headers={'Retry-After': '12'})
        http_err = httpx.HTTPStatusError('429', request=req, response=resp)
        wrapper = RuntimeError('wrapper')
        wrapper.__cause__ = http_err
        assert _retry_after_hint(wrapper) == 12.0

    @pytest.mark.unit
    def test_breaker_reads_retry_after_from_rate_limit_error(self):
        from services.circuit_breaker import _retry_after_from_exc

        assert _retry_after_from_exc(RateLimitError(retry_after=12)) == 12.0
        assert _retry_after_from_exc(RateLimitError()) is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_openai_rate_limit_surfaces_retry_after(self):
        """Retry-After from the raw openai error reaches our RateLimitError."""
        import httpx
        import openai
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.prompts.models import Message

        class _FakeCompletions:
            async def create(self, **kwargs):
                req = httpx.Request('POST', 'http://x/chat/completions')
                resp = httpx.Response(429, request=req, headers={'Retry-After': '9'})
                raise openai.RateLimitError('rate limited', response=resp, body=None)

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeOpenAIClient:
            chat = _FakeChat()

        cfg = LLMConfig(api_key='k', model='m', base_url='http://x/v1')
        client = OpenAIGenericClient(config=cfg, client=_FakeOpenAIClient())
        with pytest.raises(RateLimitError) as excinfo:
            await client._generate_response([Message(role='user', content='hi')])
        assert excinfo.value.retry_after == 9.0


class TestBackoffCap:
    """2026-10-22 backoff-cap: journal retry backoff never exceeds the cap.

    A 30s base grows past 1920s (~32 min) by attempt 7, freezing done for
    hours. ``retry_backoff_cap_seconds`` (default 480s) clamps the computed
    backoff, including the storm multiplier, so episodes return to work in
    minutes.
    """

    async def _enqueue_row(self, service, uuid: str):
        row_id, _ = await service.journal.enqueue(
            {
                'group_id': 'g1',
                'name': 'ep',
                'episode_body': 'body',
                'source_description': 'd',
                'source': 'text',
                'uuid': uuid,
            }
        )
        return row_id

    async def _backoff_after_failure(self, service, row_id: int, attempt: int, exc):
        """Run _handle_journal_failure on an owned row and return (backoff_s, attempt)."""
        claimed = await service.journal.claim_next('g1', service._worker_id)
        assert claimed is not None, 'row must be claimable to own the bump'
        claimed['attempt'] = attempt
        await service._handle_journal_failure('g1', claimed, exc)
        conn = sqlite3.connect(service.journal.db_path)
        try:
            stored = conn.execute(
                'SELECT attempt, next_retry_at FROM episode_queue WHERE id=?',
                (row_id,),
            ).fetchone()
        finally:
            conn.close()
        from datetime import datetime, timezone

        next_ts = datetime.fromisoformat(stored[1])
        backoff = (next_ts - datetime.now(timezone.utc)).total_seconds()
        return backoff, stored[0]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_attempt7_backoff_capped_at_480(self, tmp_path):
        """30*2^6 = 1920s raw is clamped to the 480s cap at attempt 7."""
        service = QueueService(
            make_config(
                tmp_path / 'cap.db',
                spool_backoff_base_seconds=30.0,
                retry_backoff_cap_seconds=480.0,
                max_spool_attempts=10,
            )
        )
        try:
            row_id = await self._enqueue_row(service, 'u-cap')
            backoff, attempt = await self._backoff_after_failure(
                service, row_id, attempt=6, exc=RateLimitError('x')
            )
            assert attempt == 7
            assert 0.0 <= backoff <= 480.5
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cap_applies_after_storm_multiplier(self, tmp_path):
        """A storming model (x4 or x2) still never sleeps past the cap."""
        service = QueueService(
            make_config(
                tmp_path / 'capstorm.db',
                spool_backoff_base_seconds=30.0,
                retry_backoff_cap_seconds=480.0,
                max_spool_attempts=10,
            )
        )
        try:
            # Enter storm state for the failed model (reputation_unhealthy entry).
            service._storm.record_failure('model-A', reputation_unhealthy=True)
            assert service._storm.in_storm('model-A') is True
            row_id = await self._enqueue_row(service, 'u-capstorm')
            exc = RateLimitError('storm')
            exc._storm_model = 'model-A'
            backoff, _ = await self._backoff_after_failure(
                service, row_id, attempt=6, exc=exc
            )
            # raw 30*2^6*2 = 3840s is clamped back to the 480s cap.
            assert 0.0 <= backoff <= 480.5
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_small_backoff_below_cap_untouched(self, tmp_path):
        """Early attempts stay below the cap (legacy exponential shape)."""
        service = QueueService(
            make_config(
                tmp_path / 'capsmall.db',
                spool_backoff_base_seconds=30.0,
                retry_backoff_cap_seconds=480.0,
                max_spool_attempts=10,
            )
        )
        try:
            row_id = await self._enqueue_row(service, 'u-cap-small')
            backoff, attempt = await self._backoff_after_failure(
                service, row_id, attempt=2, exc=RateLimitError('x')
            )
            assert attempt == 3
            # 30 * 2^2 = 120s, well below the 480s cap.
            assert 110.0 <= backoff <= 121.0
        finally:
            await service.close()
