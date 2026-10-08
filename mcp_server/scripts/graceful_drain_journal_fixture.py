"""Fixture process for the durable SQLite journal kill -9 test.

Run modes:
  serve   — start a real uvicorn server + QueueService with the durable SQLite
            journal enabled and a fake (very slow) graphiti client, enqueue 3
            fake episodes, drop a ready-file and idle until the test SIGKILLs
            the process. The first episode is in flight (stuck in a 30s fake
            LLM call), the other two are pending in the journal table.
  replay  — start QueueService on the SAME journal path with a fast fake client
            and an episode_builder; initialize() requeues the abandoned
            processing row, the journal worker processes all rows exactly once,
            recording uuids to a result file.

No LLM and no Neo4j are touched: the graphiti client is faked out entirely.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from config.schema import ResilienceConfig  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402
from services.queue_service import QueueService  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('graceful_drain_journal_fixture')

FAKE_EPISODES = [
    ('11111111-1111-1111-1111-111111111111', 'JOURNAL-TEST episode one'),
    ('22222222-2222-2222-2222-222222222222', 'JOURNAL-TEST episode two'),
    ('33333333-3333-3333-3333-333333333333', 'JOURNAL-TEST episode three'),
]


class SlowClient:
    """Fake graphiti client: every add_episode takes ~30s (a 'stuck LLM call')."""

    async def add_episode(self, **kwargs):
        logger.info('SlowClient: start processing %s (will hang)', kwargs.get('uuid'))
        await asyncio.sleep(30)
        logger.info('SlowClient: finished %s', kwargs.get('uuid'))


class FastClient:
    """Fake graphiti client for replay: records uuids and returns instantly."""

    def __init__(self, result_file: Path):
        self.result_file = result_file

    async def add_episode(self, **kwargs):
        with self.result_file.open('a', encoding='utf-8') as fh:
            fh.write(f"{kwargs.get('uuid')}\n")
        logger.info('FastClient: replayed %s', kwargs.get('uuid'))
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


async def run_serve(args: argparse.Namespace) -> int:
    queue_service = QueueService(make_resilience(args.journal_path))
    await queue_service.initialize(SlowClient(), episode_builder=make_builder())
    logger.info('Fixture: journal enabled=%s path=%s', queue_service.journal is not None, args.journal_path)

    import uvicorn

    async def dummy_app(scope, receive, send):
        if scope['type'] == 'http':
            await send(
                {
                    'type': 'http.response.start',
                    'status': 200,
                    'headers': [(b'content-type', b'text/plain')],
                }
            )
            await send({'type': 'http.response.body', 'body': b'journal-kill9 fixture'})

    uvicorn_config = uvicorn.Config(dummy_app, host='127.0.0.1', port=0, log_level='warning')
    server = uvicorn.Server(uvicorn_config)
    serve_task = asyncio.create_task(server.serve(), name='fixture-http-server')

    # Enqueue 3 fake episodes: the worker picks the first one up immediately
    # (SlowClient hangs on it), episodes two and three stay pending.
    for ep_uuid, name in FAKE_EPISODES:
        await queue_service.add_episode(
            group_id='test-drain',
            name=name,
            content=f'Fake episode body for {name}',
            source_description='journal kill9 fixture',
            episode_type=EpisodeType.text,
            entity_types=None,
            uuid=ep_uuid,
        )
    pending = await queue_service.journal.count_unfinished()
    logger.info('Fixture: enqueued %d episodes, journal unfinished=%d', len(FAKE_EPISODES), pending)

    Path(args.ready_file).write_text('ready\n', encoding='utf-8')

    # Idle forever; the test SIGKILLs this process (no graceful drain path).
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    await queue_service.close()
    return 0


async def run_replay(args: argparse.Namespace) -> int:
    queue_service = QueueService(make_resilience(args.journal_path))
    await queue_service.initialize(FastClient(Path(args.result_file)), episode_builder=make_builder())

    deadline = asyncio.get_running_loop().time() + 20.0
    while (
        await queue_service.journal.count_unfinished() > 0
        and asyncio.get_running_loop().time() < deadline
    ):
        await asyncio.sleep(0.2)

    unfinished = await queue_service.journal.count_unfinished()
    logger.info('Replay finished: unfinished=%d', unfinished)
    print(f'REPLAY_UNFINISHED={unfinished}', flush=True)
    assert await queue_service.journal.count_done() == 3, 'exactly 3 rows must be done after replay'
    await queue_service.close()
    return 0 if unfinished == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['serve', 'replay'], required=True)
    parser.add_argument('--journal-path', required=True)
    parser.add_argument('--ready-file')
    parser.add_argument('--result-file')
    args = parser.parse_args()

    if args.mode == 'serve':
        return asyncio.run(run_serve(args))
    return asyncio.run(run_replay(args))


if __name__ == '__main__':
    sys.exit(main())
