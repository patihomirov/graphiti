"""Fixture process for the worker-level multi-model failover e2e test.

Serves a real QueueService with the durable SQLite journal enabled and, via
``resilience.model_fallbacks``, a fallback LLM model. The fake graphiti client
fails every episode on the active model (``model-A``) with an empty response
and succeeds on the fallback (``model-B``). The journal worker must retry each
episode inline on the fallback (inside the same claim), record the failed model
in ``attempted_models``, and finish all rows with zero duplicates.

No LLM and no graph DB are touched: the graphiti client is faked out entirely.

Usage: python3 scripts/model_failover_journal_fixture.py \
    --journal-path <path> --result-file <path>
"""

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from graphiti_core.llm_client.errors import EmptyResponseError  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402

from config.schema import ResilienceConfig  # noqa: E402
from services.queue_service import QueueService  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_failover_fixture')

FAKE_EPISODES = [
    ('41111111-1111-1111-1111-111111111111', 'FAILOVER-TEST episode one'),
    ('42222222-2222-2222-2222-222222222222', 'FAILOVER-TEST episode two'),
    ('43333333-3333-3333-3333-333333333333', 'FAILOVER-TEST episode three'),
]

ACTIVE_MODEL = 'model-A'
FALLBACK_MODEL = 'model-B'


class FakeLLM:
    """Fake LLM client exposing the model-override hook (model-A is active)."""

    def __init__(self):
        self.model = ACTIVE_MODEL
        self._override: str | None = None

    def set_model_override(self, model: str | None) -> None:
        self._override = model


class FailoverGraphiti:
    """Fake graphiti client: model-A empties, model-B succeeds."""

    def __init__(self, result_file: Path):
        self.llm_client = FakeLLM()
        self.result_file = result_file
        self.calls = 0

    async def add_episode(self, **kwargs):
        self.calls += 1
        model = self.llm_client._override or self.llm_client.model
        if model == ACTIVE_MODEL:
            raise EmptyResponseError('fake primary returned an empty body')
        with self.result_file.open('a', encoding='utf-8') as fh:
            fh.write(f"{kwargs.get('uuid')} {model}\n")
        return None


def make_resilience(journal_path: str) -> ResilienceConfig:
    return ResilienceConfig(
        enabled=True,
        journal_enabled=True,
        journal_path=journal_path,
        journal_lease_seconds=1.0,
        spool_enabled=False,
        max_queue_depth=20,
        failure_threshold=3,
        open_timeout_seconds=5.0,
        retryer_interval_seconds=0.5,
        spool_backoff_base_seconds=0.1,
        max_spool_attempts=5,
        model_fallbacks=[FALLBACK_MODEL],
    )


def make_builder() -> callable:
    def episode_builder(plan: dict) -> dict:
        from datetime import datetime, timezone

        return {
            'name': plan['name'],
            'episode_body': plan['episode_body'],
            'source_description': plan['source_description'],
            'source': EpisodeType[plan['source']],
            'group_id': plan['group_id'],
            'reference_time': datetime.now(timezone.utc),
            'uuid': plan.get('uuid'),
        }

    return episode_builder


async def run_fixture(args: argparse.Namespace) -> int:
    queue_service = QueueService(make_resilience(args.journal_path))
    client = FailoverGraphiti(Path(args.result_file))
    await queue_service.initialize(client, episode_builder=make_builder())
    logger.info(
        'Fixture: journal enabled=%s fallbacks=%s',
        queue_service.journal is not None,
        queue_service.resilience.model_fallbacks,
    )

    for ep_uuid, name in FAKE_EPISODES:
        await queue_service.add_episode(
            group_id='test-failover',
            name=name,
            content=f'Fake episode body for {name}',
            source_description='model failover fixture',
            episode_type=EpisodeType.text,
            entity_types=None,
            uuid=ep_uuid,
        )

    deadline = asyncio.get_running_loop().time() + 20.0
    while (
        await queue_service.journal.count_unfinished() > 0
        and asyncio.get_running_loop().time() < deadline
    ):
        await asyncio.sleep(0.2)

    done = await queue_service.journal.count_done()
    unfinished = await queue_service.journal.count_unfinished()

    conn = sqlite3.connect(args.journal_path)
    attempted = {}
    try:
        for ep_uuid, _ in FAKE_EPISODES:
            row = conn.execute(
                'SELECT attempted_models FROM episode_queue WHERE uuid=?', (ep_uuid,)
            ).fetchone()
            attempted[ep_uuid] = json.loads(row[0]) if row and row[0] else []
    finally:
        conn.close()

    logger.info('Fixture finished: done=%d unfinished=%d calls=%d', done, unfinished, client.calls)
    print(f'FAILOVER_DONE={done}', flush=True)
    print(f'FAILOVER_UNFINISHED={unfinished}', flush=True)
    print(f'FAILOVER_CALLS={client.calls}', flush=True)
    print(f'FAILOVER_ATTEMPTED={json.dumps(attempted, sort_keys=True)}', flush=True)

    await queue_service.close()
    assert done == len(FAKE_EPISODES), 'all episodes must finish on the fallback model'
    assert unfinished == 0, 'no rows may remain unfinished'
    assert all(v == [ACTIVE_MODEL] for v in attempted.values()), (
        f'attempted_models must only record the failed active model: {attempted}'
    )
    assert client.calls == len(FAKE_EPISODES) * 2, (
        f'exactly 2 runs per episode (primary fail + fallback success): calls={client.calls}'
    )
    return 0 if unfinished == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal-path', required=True)
    parser.add_argument('--result-file', required=True)
    args = parser.parse_args()
    return asyncio.run(run_fixture(args))


if __name__ == '__main__':
    sys.exit(main())
