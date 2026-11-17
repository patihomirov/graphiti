"""Regression tests: wall-clock deadline vs real attempts, and /health truth.

Diagnosed on 2026-11-17 after a night-long llm-gateway outage:

* The claim-time wall-clock deadline (RCA C3) surrendered 94 backlog episodes
  whose single recorded failure was a transport error during the outage: one
  ``Connection error`` bump made ``attempt`` 1, and the very next claim killed
  the row as ``deadline exceeded 1 attempts`` - without a single genuine
  processing attempt. The deadline now requires
  ``resilience.episode_wall_deadline_min_attempts`` recorded failures (default 2)
  before it may surrender a row.
* /health reported ``queue_depth`` next to the legacy in-memory
  ``max_queue_depth`` (20) while the journal ceiling is ``journal_max_pending``,
  so the pair read as a permanent overflow.
* /health kept reporting ``circuit_breaker: open`` after the backend recovered:
  the journal pool drains rows without calling ``allow_request()`` and
  ``record_success`` only closed from ``half_open``, so an open breaker could
  never close again.

Everything runs against tmp_path SQLite journals with fake graphiti clients:
no network, no graphiti-core data.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from config.schema import ResilienceConfig
from graphiti_mcp_server import _effective_queue_max_depth
from services.circuit_breaker import CircuitBreaker
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
        journal_lease_seconds=30.0,
        spool_enabled=False,
        spool_backoff_base_seconds=0.01,
        # Keep the retryer alarm out of the way: these tests drive the claim path
        # directly, so a fast retryer would wake the pool into claiming the row
        # first (race on ``processed``).
        retryer_interval_seconds=3600.0,
    )
    cfg.update(over)
    return ResilienceConfig(**cfg)


def make_builder():
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
    def __init__(self, model: str = 'model-A'):
        self.model = model
        self._override: str | None = None

    def set_model_override(self, model: str | None) -> None:
        self._override = model


class FakeGraphiti:
    """add_episode always succeeds: the row must be processed, not book-kept."""

    def __init__(self, llm: FakeLLM):
        self.llm_client = llm
        self.calls: list[str] = []

    async def add_episode(self, **kwargs):
        self.calls.append(self.llm_client._override or self.llm_client.model)
        return None


def fetch_row(db_path: str, row_id: int) -> sqlite3.Row:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute('SELECT * FROM episode_queue WHERE id=?', (row_id,)).fetchone()
    finally:
        conn.close()


def hours_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


async def seed_row(service: QueueService, uuid: str, *, attempt: int, age_hours: float):
    """Enqueue a row and stamp its failure history as if it had already run."""
    row_id, _ = await service.journal.enqueue(make_plan(uuid=uuid))
    await service.journal._execute(
        'UPDATE episode_queue SET created_at=?, attempt=?, error=? WHERE id=?',
        (hours_ago(age_hours), attempt, 'Connection error.', row_id),
    )
    return row_id


# --------------------------------------------------------------------------
# Bug 1: the wall-clock deadline must not spend a row it never really tried
# --------------------------------------------------------------------------

class TestWallDeadlineAttemptGate:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_network_failure_is_not_surrendered(self, tmp_path):
        """7h-old row with ONE transport failure keeps its place in the queue.

        This is the 2026-11-17 incident shape: created during the backlog, one
        ``Connection error`` recorded while llm-gateway was down, then the next
        claim killed it as ``deadline exceeded 1 attempts``.
        """
        service = QueueService(
            make_config(tmp_path / 'one-fail.db', episode_wall_deadline_seconds=21600.0)
        )
        client = FakeGraphiti(FakeLLM('model-A'))
        await service.initialize(client, episode_builder=make_builder())
        try:
            row_id = await seed_row(service, 'u-one-fail', attempt=1, age_hours=7.0)
            processed = await service._process_journal_once(
                'g1', worker_id=service._worker_id
            )
            row = fetch_row(service.journal.db_path, row_id)
            assert processed is True
            assert row['status'] == 'done'
            assert 'deadline' not in (row['error'] or '')
            assert row['attempt'] == 1
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_repeated_failures_are_still_surrendered(self, tmp_path):
        """A row that really failed twice and blew the budget is surrendered."""
        service = QueueService(
            make_config(tmp_path / 'n-fails.db', episode_wall_deadline_seconds=21600.0)
        )
        client = FakeGraphiti(FakeLLM('model-A'))
        await service.initialize(client, episode_builder=make_builder())
        try:
            row_id = await seed_row(service, 'u-n-fail', attempt=2, age_hours=7.0)
            processed = await service._process_journal_once(
                'g1', worker_id=service._worker_id
            )
            row = fetch_row(service.journal.db_path, row_id)
            assert processed == 'deadline'
            assert row['status'] == 'failed'
            assert 'deadline exceeded' in (row['error'] or '')
            assert client.calls == []
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_min_attempts_one_restores_legacy_behaviour(self, tmp_path):
        """``episode_wall_deadline_min_attempts=1`` keeps the old attempt >= 1 gate."""
        service = QueueService(
            make_config(
                tmp_path / 'legacy.db',
                episode_wall_deadline_seconds=21600.0,
                episode_wall_deadline_min_attempts=1,
            )
        )
        client = FakeGraphiti(FakeLLM('model-A'))
        await service.initialize(client, episode_builder=make_builder())
        try:
            row_id = await seed_row(service, 'u-legacy', attempt=1, age_hours=7.0)
            processed = await service._process_journal_once(
                'g1', worker_id=service._worker_id
            )
            row = fetch_row(service.journal.db_path, row_id)
            assert processed == 'deadline'
            assert row['status'] == 'failed'
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_gate_matrix(self, tmp_path):
        """The gate is shared by the claim path and the watchdog."""
        service = QueueService(make_config(tmp_path / 'gate.db'))
        try:
            old = hours_ago(7.0)
            assert service._wall_deadline_exceeded(0, old) is False
            assert service._wall_deadline_exceeded(1, old) is False
            assert service._wall_deadline_exceeded(2, old) is True
            # A fresh row is never overdue, however many attempts it burned.
            assert service._wall_deadline_exceeded(5, hours_ago(0.5)) is False
            # Unparseable / missing creation stamps are never surrendered.
            assert service._wall_deadline_exceeded(5, None) is False
            assert service._wall_deadline_exceeded(5, 'not-a-date') is False
        finally:
            await service.close()

    @pytest.mark.unit
    def test_default_min_attempts(self):
        assert ResilienceConfig().episode_wall_deadline_min_attempts == 2


# --------------------------------------------------------------------------
# Bug 2a: /health must publish the ceiling of the ACTIVE intake path
# --------------------------------------------------------------------------

class TestHealthQueueCeiling:
    @pytest.mark.unit
    def test_journal_ceiling_wins_when_journal_enabled(self):
        snapshot = {
            'queue_depth': 76,
            'max_queue_depth': 20,
            'journal': {'enabled': True, 'max_pending': 300},
        }
        assert _effective_queue_max_depth(snapshot) == 300

    @pytest.mark.unit
    def test_legacy_ceiling_without_journal(self):
        snapshot = {
            'queue_depth': 5,
            'max_queue_depth': 20,
            'journal': {'enabled': False},
        }
        assert _effective_queue_max_depth(snapshot) == 20

    @pytest.mark.unit
    def test_missing_journal_ceiling_falls_back_to_legacy(self):
        snapshot = {'max_queue_depth': 20, 'journal': {'enabled': True}}
        assert _effective_queue_max_depth(snapshot) == 20

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_snapshot_publishes_journal_ceiling(self, tmp_path):
        service = QueueService(make_config(tmp_path / 'ceiling.db', journal_max_pending=300))
        try:
            snapshot = await service.get_resilience_snapshot()
            assert snapshot['journal']['enabled'] is True
            assert snapshot['max_queue_depth'] == service.resilience.max_queue_depth
            assert _effective_queue_max_depth(snapshot) == 300
        finally:
            await service.close()


# --------------------------------------------------------------------------
# Bug 2b: an open breaker must close on a real success (journal path)
# --------------------------------------------------------------------------

class TestBreakerClosesOnJournalSuccess:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_record_success_closes_open_breaker(self):
        breaker = CircuitBreaker(failure_threshold=2, open_timeout_seconds=3600.0)
        for _ in range(2):
            await breaker.record_failure(RuntimeError('connection reset'))
        assert (await breaker.get_snapshot())['state'] == 'open'

        await breaker.record_success()

        snapshot = await breaker.get_snapshot()
        assert snapshot['state'] == 'closed'
        assert snapshot['failure_count'] == 0
        assert snapshot['retry_after_seconds'] == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_half_open_probe_success_still_closes(self):
        breaker = CircuitBreaker(failure_threshold=1, open_timeout_seconds=3600.0)
        await breaker.record_failure(RuntimeError('trip'))
        # Journal pool never probes: the success itself must be enough.
        await breaker.record_success()
        assert breaker.state == 'closed'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_journal_success_heals_tripped_breaker_end_to_end(self, tmp_path):
        """The 10:13 MSK symptom: successes flow, /health still says open."""
        service = QueueService(make_config(tmp_path / 'breaker.db'))
        client = FakeGraphiti(FakeLLM('model-A'))
        await service.initialize(client, episode_builder=make_builder())
        try:
            for _ in range(3):
                await service.circuit_breaker.record_failure(
                    RuntimeError('connection reset by peer')
                )
            assert (await service.get_resilience_snapshot())['state'] == 'open'

            await seed_row(service, 'u-heal', attempt=0, age_hours=0.1)
            processed = await service._process_journal_once(
                'g1', worker_id=service._worker_id
            )
            assert processed is True

            snapshot = await service.get_resilience_snapshot()
            assert snapshot['state'] == 'closed'
            assert snapshot['failure_count'] == 0
        finally:
            await service.close()
