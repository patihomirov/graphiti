"""Unit tests for the queue-validation MCP tools.

Covers the ``mark_verified`` / ``mark_questioned`` / ``get_raw_episode`` MCP
tools (in-queue fact-check/verify protocol): success paths, unknown-row and
journal-disabled guards, ``by`` validation, full-body reads, and the reason
persistence. The tools are exercised directly with the module's ``queue_service``
global pointed at a fake exposing a real tmp_path journal (no live server).
"""

from types import SimpleNamespace

import pytest

import graphiti_mcp_server as gms
from services.queue_journal import QueueJournal


def make_plan(**over):
    plan = {
        'group_id': 'g1',
        'name': 'ep',
        'episode_body': 'some body',
        'source_description': 'desc',
        'source': 'text',
        'uuid': 'q-u-1',
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


@pytest.fixture
async def journal(tmp_path):
    j = QueueJournal(tmp_path / 'journal.db', lease_seconds=300.0)
    yield j
    await j.close()


@pytest.fixture
def patched_service(journal):
    """Point the MCP module queue_service global at a fake exposing journal."""
    fake = SimpleNamespace(journal=journal)
    prev = gms.queue_service
    gms.queue_service = fake
    yield fake
    gms.queue_service = prev


def test_new_tools_registered():
    """The three tools must be discoverable on the MCP server's tool registry."""
    tools = gms.mcp._tool_manager.list_tools()
    names = {t.name for t in tools}
    assert 'mark_verified' in names
    assert 'mark_questioned' in names
    assert 'get_raw_episode' in names


class TestMarkVerifiedTool:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_success(self, journal, patched_service):
        row_id, _ = await journal.enqueue(
            make_plan(uuid='tv-1', name='v tool', episode_body='b')
        )
        res = await gms.mark_verified(
            journal_id=row_id, by='validator-hard (GLM) 21.10 test'
        )
        assert res['ok'] is True
        assert res['journal_id'] == row_id
        assert res['status'] == 'pending'
        assert res['materialized'] is False
        row = await journal.get_raw(row_id)
        assert row['verified_by'] == 'validator-hard (GLM) 21.10 test'
        assert row['verified_at'] is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_at_passthrough(self, journal, patched_service):
        row_id, _ = await journal.enqueue(
            make_plan(uuid='tv-2', name='v tool at', episode_body='b')
        )
        res = await gms.mark_verified(
            journal_id=row_id,
            by='validator',
            at='2026-10-09T10:00:00+00:00',
        )
        assert res['ok'] is True
        row = await journal.get_raw(row_id)
        assert row['verified_at'] == '2026-10-09T10:00:00+00:00'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unknown_row(self, journal, patched_service):
        res = await gms.mark_verified(journal_id=999_999, by='x')
        assert res['ok'] is False
        assert res['reason'] == 'no such episode'
        assert res['journal_id'] == 999_999

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_journal_disabled(self):
        fake = SimpleNamespace(journal=None)
        prev = gms.queue_service
        gms.queue_service = fake
        try:
            res = await gms.mark_verified(journal_id=1, by='x')
        finally:
            gms.queue_service = prev
        assert res['ok'] is False
        assert res['reason'] == 'journal disabled'
        assert res['type'] == 'mark_verified'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_by_validation(self, journal, patched_service):
        res = await gms.mark_verified(journal_id=1, by='   ')
        assert res['ok'] is False and res['reason'] == 'by is required'
        res = await gms.mark_verified(journal_id=1, by='x' * 129)
        assert res['ok'] is False
        assert res['reason'] == 'by must be 128 characters or fewer'


class TestMarkQuestionedTool:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_success_with_reason(self, journal, patched_service):
        row_id, _ = await journal.enqueue(
            make_plan(uuid='tq-1', name='q tool', episode_body='b')
        )
        res = await gms.mark_questioned(
            journal_id=row_id,
            by='validator-hard (GLM) 21.10 test',
            reason='contradicts source X',
        )
        assert res['ok'] is True
        assert res['status'] == 'pending'
        assert res['materialized'] is False
        row = await journal.get_raw(row_id)
        assert row['questioned_by'] == 'validator-hard (GLM) 21.10 test'
        assert row['questioned_at'] is not None
        assert row['error'] == 'contradicts source X'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_success_without_reason(self, journal, patched_service):
        row_id, _ = await journal.enqueue(
            make_plan(uuid='tq-2', name='q tool noreason', episode_body='b')
        )
        res = await gms.mark_questioned(journal_id=row_id, by='validator')
        assert res['ok'] is True
        row = await journal.get_raw(row_id)
        assert row['questioned_by'] == 'validator'
        assert row['error'] is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unknown_row(self, journal, patched_service):
        res = await gms.mark_questioned(journal_id=999_999, by='x')
        assert res['ok'] is False and res['reason'] == 'no such episode'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_by_validation(self, journal, patched_service):
        res = await gms.mark_questioned(journal_id=1, by='   ')
        assert res['ok'] is False and res['reason'] == 'by is required'
        res = await gms.mark_questioned(journal_id=1, by='y' * 129)
        assert res['ok'] is False
        assert res['reason'] == 'by must be 128 characters or fewer'


class TestGetRawEpisodeTool:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_body_no_snippet_truncation(self, journal, patched_service):
        long_body = 'y' * 450
        row_id, _ = await journal.enqueue(
            make_plan(uuid='tr-1', name='full body tool', episode_body=long_body)
        )
        res = await gms.get_raw_episode(journal_id=row_id)
        # Success returns the full row dict directly (not a snippet).
        assert 'reason' not in res or res.get('ok') is not False
        assert res['episode_body'] == long_body
        assert len(res['episode_body']) == 450
        assert res['name'] == 'full body tool'
        assert res['status'] == 'pending'
        assert res['group_id'] == 'g1'
        assert res['materialized'] is False
        assert res['verified_by'] is None
        assert res['questioned_by'] is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_markers_included(self, journal, patched_service):
        row_id, _ = await journal.enqueue(
            make_plan(uuid='tr-2', name='markers tool', episode_body='b')
        )
        await journal.mark_verified(row_id, by='validator-a')
        await journal.mark_questioned(row_id, by='validator-b', reason='why')
        res = await gms.get_raw_episode(journal_id=row_id)
        assert res['verified_by'] == 'validator-a'
        assert res['questioned_by'] == 'validator-b'
        assert res['error'] == 'why'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unknown_row(self, journal, patched_service):
        res = await gms.get_raw_episode(journal_id=999_999)
        assert res['ok'] is False and res['reason'] == 'no such episode'

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_journal_disabled(self):
        fake = SimpleNamespace(journal=None)
        prev = gms.queue_service
        gms.queue_service = fake
        try:
            res = await gms.get_raw_episode(journal_id=1)
        finally:
            gms.queue_service = prev
        assert res['ok'] is False and res['reason'] == 'journal disabled'


class TestAddMemoryJournalId:
    """add_memory surfaces status/journal_id/materialized in its response."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_success_response_shape(self, journal, tmp_path):
        fake_graphiti = SimpleNamespace(
            entity_types=[], edge_types=[], edge_type_map={}
        )

        async def fake_add_episode(**kwargs):
            row_id, _ = await journal.enqueue(make_plan(uuid='am-jid-1'))
            return row_id

        fake_qs = SimpleNamespace(
            journal=journal,
            add_episode=fake_add_episode,
            resilience=SimpleNamespace(auto_previous_episode_uuid=False),
        )
        prev_g, prev_q = gms.graphiti_service, gms.queue_service
        gms.graphiti_service = fake_graphiti
        gms.queue_service = fake_qs
        try:
            resp = await gms.add_memory(
                name='ep',
                episode_body='body',
                group_id='g1',
                uuid='am-jid-1',
            )
        finally:
            gms.graphiti_service, gms.queue_service = prev_g, prev_q
        assert resp['status'] == 'queued'
        assert isinstance(resp['journal_id'], int) and resp['journal_id'] > 0
        assert resp['materialized'] is False
        assert 'message' in resp and 'queued' in resp['message']

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_message_field_preserved_for_backcompat(self):
        """Existing consumers read only 'message'; the new keys are additive."""
        fake_graphiti = SimpleNamespace(
            entity_types=[], edge_types=[], edge_type_map={}
        )

        async def fake_add_episode(**kwargs):
            return 7

        fake_qs = SimpleNamespace(
            journal=None,
            add_episode=fake_add_episode,
            resilience=SimpleNamespace(auto_previous_episode_uuid=False),
        )
        prev_g, prev_q = gms.graphiti_service, gms.queue_service
        gms.graphiti_service = fake_graphiti
        gms.queue_service = fake_qs
        try:
            resp = await gms.add_memory(name='ep', episode_body='body', group_id='g1')
        finally:
            gms.graphiti_service, gms.queue_service = prev_g, prev_q
        assert isinstance(resp, dict) and 'message' in resp
        assert resp['journal_id'] == 7
