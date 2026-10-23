"""RCA Tier-0/Tier-1 regression tests (2026-10-23).

Covers every fix from the audited "bad episodes" RCA:
- C7: SDK 5xx / APIStatusError transient classification (+ zai proxy loss);
- C1: permanent -> fast-fail (is_permanent_error matrix, <= permanent_fail_fast_attempts);
- C2: poison-detection (repeat_signature / repeat_count -> quarantined);
- C3: per-episode wall-clock deadline (claim + watchdog);
- C6: bump_retry no longer resets attempted_models;
- C10: per-worker worker_id + steward steal-guard for a whole owner set;
- B1: add_memory validation gate (rejects malformed input before enqueue);
- schema migration for the new columns.

Everything runs against tmp_path SQLite journals with fake graphiti clients:
no network, no graphiti-core data.
"""

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone

import httpx
import openai
import pytest
from graphiti_core.llm_client.errors import RateLimitError

from config.schema import ResilienceConfig
from graphiti_mcp_server import (
    ADD_MEMORY_BODY_MAX_BYTES,
    _validate_add_memory_input,
)
from services.circuit_breaker import (
    is_permanent_error,
    is_transient_error,
)
from services.queue_journal import QueueJournal
from services.queue_service import QueueService, _error_signature


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
        retryer_interval_seconds=0.05,
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
    """add_episode succeeds unless the current override/model is in fail_on."""

    def __init__(self, llm: FakeLLM, fail_on: set[str] | None = None, fail_error=None):
        self.llm_client = llm
        self.fail_on = fail_on or set()
        self.fail_error = fail_error or RateLimitError('cfg failure')
        self.calls: list[str] = []

    async def add_episode(self, **kwargs):
        model = self.llm_client._override or self.llm_client.model
        self.calls.append(model)
        if model in self.fail_on:
            raise self.fail_error
        return None


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


def fetch_row(db_path: str, row_id: int) -> sqlite3.Row:
    cur = sqlite3.connect(db_path)
    cur.row_factory = sqlite3.Row
    try:
        return cur.execute(
            'SELECT * FROM episode_queue WHERE id=?', (row_id,)
        ).fetchone()
    finally:
        cur.close()


async def wait_status(service, uuid: str, status: str, timeout: float = 8.0) -> sqlite3.Row:
    deadline = asyncio.get_running_loop().time() + timeout
    row_id = _row_id_by_uuid(service, uuid)
    while asyncio.get_running_loop().time() < deadline:
        row = fetch_row(service.journal.db_path, row_id)
        if row is not None and row['status'] == status:
            return row
        await asyncio.sleep(0.02)
    raise AssertionError(f'Timed out waiting for row {uuid} to reach {status}')


def _row_id_by_uuid(service, uuid: str) -> int:
    cur = sqlite3.connect(service.journal.db_path)
    try:
        row = cur.execute(
            'SELECT id FROM episode_queue WHERE uuid=?', (uuid,)
        ).fetchone()
    finally:
        cur.close()
    assert row is not None, f'row {uuid} not found'
    return int(row[0])


# --------------------------------------------------------------------------
# C7: SDK status errors and transient classification
# --------------------------------------------------------------------------

class TestSDKStatusTransientC7:
    def _req(self):
        return httpx.Request('POST', 'http://x/v1')

    @pytest.mark.unit
    def test_openai_internal_server_error_transient(self):
        err = openai.InternalServerError(
            'bad gateway', response=httpx.Response(502, request=self._req()), body=None
        )
        assert is_transient_error(err) is True
        assert is_permanent_error(err) is False

    @pytest.mark.unit
    def test_openai_api_status_500_transient(self):
        err = openai.APIStatusError(
            'oops', response=httpx.Response(500, request=self._req()), body=None
        )
        assert is_transient_error(err) is True

    @pytest.mark.unit
    def test_openai_rate_limit_429_transient_not_permanent(self):
        err = openai.RateLimitError(
            'ratelimited', response=httpx.Response(429, request=self._req()), body=None
        )
        assert is_transient_error(err) is True
        assert is_permanent_error(err) is False

    @pytest.mark.unit
    def test_openai_4xx_not_429_is_permanent(self):
        err = openai.APIStatusError(
            'not found', response=httpx.Response(404, request=self._req()), body=None
        )
        assert is_transient_error(err) is False
        assert is_permanent_error(err) is True

    @pytest.mark.unit
    def test_openai_bad_request_422_permanent(self):
        err = openai.BadRequestError(
            'bad schema', response=httpx.Response(422, request=self._req()), body=None
        )
        assert is_permanent_error(err) is True

    @pytest.mark.unit
    def test_sdk_5xx_marks_zai_proxy_loss(self):
        from services.queue_service import QueueService

        err = openai.InternalServerError(
            'bad gateway', response=httpx.Response(502, request=self._req()), body=None
        )
        assert QueueService._is_zai_proxy_loss(err) is True

    @pytest.mark.unit
    def test_sdk_429_not_zai_proxy_loss(self):
        from services.queue_service import QueueService

        err = openai.RateLimitError(
            'ratelimited', response=httpx.Response(429, request=self._req()), body=None
        )
        assert QueueService._is_zai_proxy_loss(err) is False


# --------------------------------------------------------------------------
# C1: is_permanent_error matrix + fast-fail
# --------------------------------------------------------------------------

class TestIsPermanentError:
    @pytest.mark.unit
    def test_pydantic_validation_error_is_permanent(self):
        import pydantic

        class M(pydantic.BaseModel):
            title: str

        with pytest.raises(pydantic.ValidationError) as ei:
            M()
        assert is_permanent_error(ei.value) is True

    @pytest.mark.unit
    def test_neo4j_service_unavailable_is_transient(self):
        from neo4j.exceptions import ServiceUnavailable

        assert is_permanent_error(ServiceUnavailable('db down')) is False
        assert is_transient_error(ServiceUnavailable('db down')) is True

    @pytest.mark.unit
    def test_neo4j_client_error_is_permanent(self):
        from neo4j.exceptions import ClientError, CypherTypeError

        assert is_permanent_error(ClientError('boom')) is True
        assert is_permanent_error(CypherTypeError('wrong type')) is True

    @pytest.mark.unit
    def test_neo4j_session_expired_is_transient(self):
        from neo4j.exceptions import SessionExpired

        assert is_permanent_error(SessionExpired('session')) is False

    @pytest.mark.unit
    def test_httpx_status_matrix(self):
        req = httpx.Request('POST', 'http://x/v1')
        assert is_permanent_error(
            httpx.HTTPStatusError('404', request=req, response=httpx.Response(404, request=req))
        ) is True
        assert is_permanent_error(
            httpx.HTTPStatusError('500', request=req, response=httpx.Response(500, request=req))
        ) is False
        assert is_permanent_error(
            httpx.HTTPStatusError('429', request=req, response=httpx.Response(429, request=req))
        ) is False

    @pytest.mark.unit
    def test_transient_classes_are_not_permanent(self):
        import json as _json

        assert is_permanent_error(RateLimitError()) is False
        assert is_permanent_error(_json.JSONDecodeError('x', '{"a":', 2)) is False
        assert is_permanent_error(TimeoutError('t')) is False
        assert is_permanent_error(httpx.ConnectError('conn')) is False

    @pytest.mark.unit
    def test_non_transient_fallback_is_permanent(self):
        # A generic unmapped exception is treated as permanent (via !transient).
        assert is_permanent_error(ValueError('boom')) is True


class TestPermanentFastFail:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_permanent_validation_error_fails_fast_at_2(self, tmp_path):
        """A pydantic ValidationError surrenders fast (<= permanent_fail_fast_attempts)."""
        import pydantic

        class M(pydantic.BaseModel):
            title: str

        with pytest.raises(pydantic.ValidationError) as ei:
            M()

        class AlwaysValidationError:
            llm_client = FakeLLM('model-A')

            async def add_episode(self, **kwargs):
                raise ei.value

        service = QueueService(
            make_config(tmp_path / 'ff.db', max_spool_attempts=10)
        )
        await service.initialize(AlwaysValidationError(), episode_builder=make_builder())
        try:
            await enqueue(service, 'u-ff')
            row = await wait_status(service, 'u-ff', 'failed')
            # Fast-fail after permanent_fail_fast_attempts (default 2), NOT 10.
            assert str(row['error']).startswith('permanent:')
            assert row['attempt'] < 10
        finally:
            await service.close()


# --------------------------------------------------------------------------
# C2: poison-detection
# --------------------------------------------------------------------------

class TestPoisonDetection:
    @pytest.mark.unit
    def test_error_signature_normalizes_whitespace_and_caps_length(self):
        sig = _error_signature('  Request   timed  out. ')
        assert sig == 'Request timed out.'
        long = _error_signature('x' * 500)
        assert len(long) == 120

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_repeated_timeout_signature_quarantines_at_3(self, tmp_path):
        """Three identical stream-timeout failures -> 'poison' failed, not 10."""

        class AlwaysTimeout:
            llm_client = FakeLLM('model-A')

            async def add_episode(self, **kwargs):
                raise httpx.ReadTimeout('stream timeout')

        service = QueueService(
            make_config(
                tmp_path / 'poison.db',
                max_spool_attempts=10,
                journal_poison_repeat_threshold=3,
            )
        )
        await service.initialize(AlwaysTimeout(), episode_builder=make_builder())
        try:
            await enqueue(service, 'u-poison')
            row = await wait_status(service, 'u-poison', 'failed')
            # mark_failed does not rewrite attempt/repeat_count (only the error),
            # so assert on the quarantine reason itself.
            assert 'poison: repeated signature x3' in str(row['error'])
            assert row['attempt'] < 10
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_repeated_rate_limit_never_quarantines(self, tmp_path):
        """A storm of identical 429s rides the backoff out - never poison."""
        service = QueueService(
            make_config(
                tmp_path / 'rl.db',
                max_spool_attempts=10,
                journal_poison_repeat_threshold=3,
            )
        )
        try:
            j = service.journal
            row_id, _ = await j.enqueue(make_plan(uuid='u-rl'))
            for _ in range(4):
                claimed = await j.claim_next('g1', service._worker_id)
                await service._handle_journal_failure(
                    'g1', claimed, RateLimitError('Rate limit exceeded.')
                )
                # Make the backed-off row claimable again for the next failure.
                await j._execute(
                    'UPDATE episode_queue SET next_retry_at=NULL WHERE id=?',
                    (row_id,),
                )
            row = fetch_row(service.journal.db_path, row_id)
            # Same 429 signature four times -> still pending, never quarantined.
            assert row['status'] == 'pending'
            assert 'poison' not in str(row['error'])
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_signature_change_resets_counter(self, tmp_path):
        """A different error signature resets repeat_count to 1 (poison-state)."""
        service = QueueService(
            make_config(
                tmp_path / 'reset.db',
                max_spool_attempts=10,
                journal_poison_repeat_threshold=3,
            )
        )
        try:
            row = {
                'id': 1,
                'repeat_signature': None,
                'repeat_count': 0,
            }
            sig1, count1 = service._poison_state(row, 'Request timed out.')
            assert sig1 == _error_signature('Request timed out.')
            assert count1 == 1
            # Same signature again -> counter advances.
            row2 = {'repeat_signature': sig1, 'repeat_count': count1}
            sig2, count2 = service._poison_state(row2, 'Request timed out.')
            assert sig2 == sig1 and count2 == 2
            # Signature change -> counter resets to 1.
            row3 = {'repeat_signature': sig2, 'repeat_count': count2}
            sig3, count3 = service._poison_state(row3, 'Connection error.')
            assert sig3 == _error_signature('Connection error.')
            assert count3 == 1
        finally:
            await service.close()


# --------------------------------------------------------------------------
# C3: wall-clock deadline
# --------------------------------------------------------------------------

class TestEpisodeDeadline:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_overdue_attempted_row_fails_at_claim(self, tmp_path):
        """A created-6h-ago row that already attempted is surrendered at claim."""
        service = QueueService(
            make_config(
                tmp_path / 'ddl.db',
                episode_wall_deadline_seconds=21600.0,
                max_spool_attempts=10,
            )
        )
        try:
            j = service.journal
            row_id, _ = await j.enqueue(make_plan(uuid='u-ddl'))
            old_created = (
                datetime.now(timezone.utc) - timedelta(hours=7)
            ).isoformat()
            await j._execute(
                'UPDATE episode_queue SET created_at=?, attempt=1 WHERE id=?',
                (old_created, row_id),
            )
            processed = await service._process_journal_once('g1', worker_id=service._worker_id)
            # Sentinel 'deadline': the pool must not apply the inter-episode
            # pause for a stale-row surrender (RCA-C3 regression fix).
            assert processed == 'deadline'
            row = fetch_row(service.journal.db_path, row_id)
            assert row['status'] == 'failed'
            assert 'deadline exceeded' in str(row['error'])
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_never_attempted_old_row_is_not_killed(self, tmp_path):
        """A 7-day-old row that never attempted must still be claimable (backlog)."""
        service = QueueService(
            make_config(
                tmp_path / 'backlog.db',
                episode_wall_deadline_seconds=21600.0,
                max_spool_attempts=10,
            )
        )
        try:
            j = service.journal
            row_id, _ = await j.enqueue(make_plan(uuid='u-backlog'))
            old_created = (
                datetime.now(timezone.utc) - timedelta(days=7)
            ).isoformat()
            await j._execute(
                'UPDATE episode_queue SET created_at=?, attempt=0 WHERE id=?',
                (old_created, row_id),
            )
            client = FakeGraphiti(FakeLLM('model-A'), fail_on=set())
            await service.initialize(client, episode_builder=make_builder())
            processed = await service._process_journal_once('g1', worker_id=service._worker_id)
            # attempt=0 exempt: NOT killed; the row gets a fresh claim and is
            # processed normally (the fake client succeeds -> done).
            assert processed is True
            row = fetch_row(service.journal.db_path, row_id)
            assert row['status'] == 'done'
            assert 'deadline' not in (row['error'] or '')
        finally:
            await service.close()

    @pytest.mark.unit
    def test_deadline_disabled_when_zero(self):
        cfg = ResilienceConfig(episode_wall_deadline_seconds=0.0)
        assert cfg.episode_wall_deadline_seconds == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_deadline_stale_head_does_not_block_live_claim(self, tmp_path):
        """A pool worker surrendering stale rows must still reach live ones fast.

        Regression (RCA-C3): before the 'deadline' sentinel, the pool treated a
        stale surrender as a normal episode outcome and applied the inter-episode
        pause (24s) after each one, so hundreds of stale rows stalled the queue
        and live attempt=0 rows were never claimed. Now the pool continues
        without the pause, and the live row behind the stale head gets processed.
        """
        service = QueueService(
            make_config(
                tmp_path / 'stale-dead.db',
                episode_wall_deadline_seconds=21600.0,
                max_spool_attempts=10,
            )
        )
        try:
            j = service.journal
            # Stale head row: created 7 days ago, already attempted -> deadline.
            stale_id, _ = await j.enqueue(make_plan(uuid='u-stale-head'))
            old_created = (
                datetime.now(timezone.utc) - timedelta(days=7)
            ).isoformat()
            await j._execute(
                'UPDATE episode_queue SET created_at=?, attempt=1 WHERE id=?',
                (old_created, stale_id),
            )
            # Live row behind it: fresh, never attempted -> must be processed.
            await j.enqueue(make_plan(uuid='u-live-lag'))
            client = FakeGraphiti(FakeLLM('model-A'), fail_on=set())
            await service.initialize(client, episode_builder=make_builder())
            # A pool worker surrenders the stale head (sentinel, no pause), then
            # another (or the same) worker reaches the live row and completes
            # it. Waiting for the pool (background workers are started by
            # initialize) - the live row must end up done, never blocked behind
            # the stale head.
            assert (
                await service._process_journal_once('g1', worker_id=service._worker_id)
                == 'deadline'
            )
            live_row = await wait_status(service, 'u-live-lag', 'done')
            assert live_row['status'] == 'done'
        finally:
            await service.close()


# --------------------------------------------------------------------------
# C6: attempted_models preserved across bump_retry
# --------------------------------------------------------------------------

class TestAttemptedModelsPreserved:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bump_retry_keeps_attempted_models(self, tmp_path):
        j = QueueJournal(tmp_path / 'am.db', lease_seconds=300.0)
        try:
            row_id, _ = await j.enqueue(make_plan(uuid='u-am'))
            await j.set_attempted_models(row_id, ['model-A', 'model-B'])
            await j.bump_retry(
                row_id,
                attempt=2,
                error='e',
                next_retry_at='2999-01-01T00:00:00+00:00',
                repeat_signature='sig',
                repeat_count=2,
            )
            assert await j.get_attempted_models(row_id) == ['model-A', 'model-B']
            conn = sqlite3.connect(j.db_path)
            try:
                row = conn.execute(
                    'SELECT repeat_signature, repeat_count FROM episode_queue WHERE id=?',
                    (row_id,),
                ).fetchone()
            finally:
                conn.close()
            assert row[0] == 'sig'
            assert row[1] == 2
        finally:
            await j.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bump_retry_without_repeat_leaves_state_untouched(self, tmp_path):
        j = QueueJournal(tmp_path / 'am2.db', lease_seconds=300.0)
        try:
            row_id, _ = await j.enqueue(make_plan(uuid='u-am2'))
            await j.set_attempted_models(row_id, ['model-A'])
            await j.bump_retry(
                row_id,
                attempt=1,
                error='e',
                next_retry_at='2999-01-01T00:00:00+00:00',
            )
            assert await j.get_attempted_models(row_id) == ['model-A']
        finally:
            await j.close()


# --------------------------------------------------------------------------
# C10: per-worker worker_id
# --------------------------------------------------------------------------

class TestWorkerIds:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pool_workers_get_unique_ids(self, tmp_path):
        service = QueueService(
            make_config(tmp_path / 'pool.db', journal_workers=3)
        )
        try:
            service._ensure_pool()
            ids = list(service._worker_scope)
            assert len(ids) >= 3
            assert len(set(ids)) == len(ids)
            assert service._worker_base in ids
        finally:
            await service.close()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_steward_respects_whole_owner_set(self, tmp_path):
        """requeue_stale_except skips EVERY id in the owner set, not just one."""
        j = QueueJournal(tmp_path / 'steal.db', lease_seconds=300.0)
        try:
            for wid in ('p1-1', 'p1-2', 'p2'):
                rid, _ = await j.enqueue(make_plan(uuid=f'st-{wid}'))
                await j.claim_next('g1', wid)
                await j._execute(
                    "UPDATE episode_queue SET lease_until='2000-01-01T00:00:00+00:00' WHERE id=?",
                    (rid,),
                )
            # The process owns p1-1 and p1-2; p2 is foreign.
            stolen = await j.requeue_stale_except({'p1-1', 'p1-2'}, grace_seconds=1)
            assert stolen == 1
            conn = sqlite3.connect(j.db_path)
            conn.row_factory = sqlite3.Row
            try:
                rows = [dict(r) for r in conn.execute(
                    'SELECT uuid, status FROM episode_queue ORDER BY id'
                )]
            finally:
                conn.close()
            by_uuid = {r['uuid']: r['status'] for r in rows}
            assert by_uuid['st-p1-1'] == 'processing'  # never stolen
            assert by_uuid['st-p1-2'] == 'processing'  # never stolen
            assert by_uuid['st-p2'] == 'pending'  # foreign, reclaimed
        finally:
            await j.close()


# --------------------------------------------------------------------------
# B1: add_memory validation gate
# --------------------------------------------------------------------------

class TestAddMemoryGateB1:
    @pytest.mark.unit
    def test_empty_name_rejected(self):
        err = _validate_add_memory_input(
            name='   ', episode_body='body', group_id=None
        )
        assert err is not None and 'name' in err
        # None name is also rejected.
        err2 = _validate_add_memory_input(
            name=None, episode_body='body', group_id=None
        )
        assert err2 is not None

    @pytest.mark.unit
    def test_empty_body_rejected(self):
        err = _validate_add_memory_input(
            name='n', episode_body='   ', group_id=None
        )
        assert err is not None and 'episode_body' in err

    @pytest.mark.unit
    def test_oversized_body_rejected(self):
        err = _validate_add_memory_input(
            name='n', episode_body='x' * (ADD_MEMORY_BODY_MAX_BYTES + 1), group_id=None
        )
        assert err is not None
        assert 'max' in err or 'too' in err

    @pytest.mark.unit
    def test_non_string_group_id_rejected(self):
        err = _validate_add_memory_input(
            name='n', episode_body='body', group_id=12345
        )
        assert err is not None and 'group_id' in err
        # None and strings pass.
        assert _validate_add_memory_input(name='n', episode_body='body', group_id=None) is None
        assert _validate_add_memory_input(name='n', episode_body='body', group_id='g1') is None

    @pytest.mark.unit
    def test_valid_input_passes(self):
        assert _validate_add_memory_input(
            name='n', episode_body='b' * 1000, group_id='g1'
        ) is None

    @pytest.mark.unit
    def test_non_string_body_rejected(self):
        err = _validate_add_memory_input(name='n', episode_body=['x'], group_id=None)
        assert err is not None and 'episode_body' in err


# --------------------------------------------------------------------------
# schema migration for the new poison columns
# --------------------------------------------------------------------------

class TestPoisonMigration:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_repeat_columns_migrated_idempotently(self, tmp_path):
        db = tmp_path / 'mig.db'
        conn = sqlite3.connect(db)
        conn.executescript(
            'CREATE TABLE episode_queue ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT, dedup_key TEXT NOT NULL, uuid TEXT, '
            'group_id TEXT NOT NULL, name TEXT NOT NULL, episode_body TEXT NOT NULL, '
            "plan_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending' "
            "CHECK (status IN ('pending', 'processing', 'done', 'failed')), "
            'attempt INTEGER NOT NULL DEFAULT 0, worker_id TEXT, lease_until TEXT, '
            'first_failure_ts TEXT, last_attempt_ts TEXT, next_retry_at TEXT, '
            'error TEXT, requires_serial INTEGER NOT NULL DEFAULT 0, '
            'attempted_models TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)'
        )
        conn.commit()
        conn.close()

        j = QueueJournal(db, lease_seconds=300.0)
        try:
            cols = {r[1] for r in sqlite3.connect(db).execute('PRAGMA table_info(episode_queue)')}
            assert 'repeat_signature' in cols
            assert 'repeat_count' in cols
            # Reopening an already-migrated DB is idempotent.
            j2 = QueueJournal(db, lease_seconds=300.0)
            assert 'repeat_count' in {
                r[1] for r in sqlite3.connect(db).execute('PRAGMA table_info(episode_queue)')
            }
            await j2.close()
            # Fresh rows default to the neutral poison state.
            row_id, inserted = await j.enqueue(make_plan(uuid='u-mig'))
            assert inserted is True
            conn = sqlite3.connect(db)
            try:
                row = conn.execute(
                    'SELECT repeat_signature, repeat_count FROM episode_queue WHERE id=?',
                    (row_id,),
                ).fetchone()
            finally:
                conn.close()
            assert row[0] is None
            assert row[1] == 0
        finally:
            await j.close()


# --------------------------------------------------------------------------
# C7 E2E: SDK 5xx fails over to the fallback channel
# --------------------------------------------------------------------------

class TestSDK5xxFailoverE2E:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sdk_502_mid_flips_to_fallback(self, tmp_path):
        """An openai InternalServerError on the primary fails over to fallback."""
        req = httpx.Request('POST', 'http://x/v1')
        err = openai.InternalServerError(
            'bad gateway', response=httpx.Response(502, request=req), body=None
        )
        llm = FakeLLM('model-A')
        client = FakeGraphiti(llm, fail_on={'model-A'}, fail_error=err)
        service = QueueService(
            make_config(tmp_path / 'sdk.db', model_fallbacks=['model-B'])
        )
        await service.initialize(client, episode_builder=make_builder())
        try:
            await enqueue(service, 'u-sdk')
            deadline = asyncio.get_running_loop().time() + 8.0
            while asyncio.get_running_loop().time() < deadline:
                if await service.journal.count_done() >= 1:
                    break
                await asyncio.sleep(0.02)
            assert await service.journal.count_done() == 1
            assert client.calls[-1] == 'model-B'
        finally:
            await service.close()
