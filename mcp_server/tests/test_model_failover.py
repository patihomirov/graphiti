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

import pytest
from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError

from config.schema import ResilienceConfig
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
        exhausted failure bumps it. attempted_models is reset by bump_retry so the
        next claim restarts from the active model.
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
            # Backed off and attempted_models reset for the next claim.
            assert attempted_for(service.journal, 'u-ex') == []
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
            # Both model-A (active) and model-B (fallback) already attempted:
            # nothing usable remains -> single None candidate (standard path).
            assert service._failover_candidates(['model-A', 'model-B'], llm) == [None]
            # model-A already attempted -> skip it, go straight to model-B.
            assert service._failover_candidates(['model-A'], llm) == ['model-B']
            # Nothing attempted -> active first, then fallbacks (deduped).
            assert service._failover_candidates([], llm) == ['model-A', 'model-B']
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
        """Fallback happens inside one claim: FIFO order and no duplicates."""
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
            assert uuids.count('u-s1') == 2  # first attempt + fallback
            assert uuids.count('u-s2') == 2
            assert uuids.index('u-s2') > uuids.index('u-s1')
            assert await service.journal.count_done() == 2
            assert await service.journal.claim_next('g1', 'w') is None
        finally:
            await service.close()
