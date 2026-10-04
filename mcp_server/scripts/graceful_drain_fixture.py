"""Fixture process for the graceful-drain test.

Run modes:
  serve   — start a real uvicorn server + QueueService with a fake (very slow)
            graphiti client, enqueue 3 fake episodes, drop a ready-file and
            wait for SIGTERM. On SIGTERM the GracefulShutdownCoordinator drains
            the queue to the spool exactly like the production server does.
  replay  — point an EpisodeRetryer at the same spool with a fast fake client
            and replay every pending episode, recording uuids to a result file.

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
from services.episode_spool import EpisodeRetryer  # noqa: E402
from services.queue_service import QueueService  # noqa: E402
from services.shutdown import GracefulShutdownCoordinator  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('graceful_drain_fixture')

FAKE_EPISODES = [
    ('11111111-1111-1111-1111-111111111111', 'DRAIN-TEST episode one'),
    ('22222222-2222-2222-2222-222222222222', 'DRAIN-TEST episode two'),
    ('33333333-3333-3333-3333-333333333333', 'DRAIN-TEST episode three'),
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


def make_resilience(spool_dir: str) -> ResilienceConfig:
    return ResilienceConfig(
        enabled=True,
        spool_enabled=True,
        spool_dir=spool_dir,
        max_queue_depth=20,
        failure_threshold=3,
        open_timeout_seconds=5.0,
        retryer_interval_seconds=0.5,
        spool_backoff_base_seconds=0.1,
        max_spool_attempts=5,
    )


async def run_serve(args: argparse.Namespace) -> int:
    queue_service = QueueService(make_resilience(args.spool_dir))
    await queue_service.initialize(SlowClient())

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
            await send({'type': 'http.response.body', 'body': b'graceful-drain fixture'})

    uvicorn_config = uvicorn.Config(dummy_app, host='127.0.0.1', port=0, log_level='warning')
    uvicorn_config.timeout_graceful_shutdown = 5.0
    server = uvicorn.Server(uvicorn_config)

    coordinator = GracefulShutdownCoordinator(
        queue_service=queue_service,
        retryer=None,
        server=server,
        drain_timeout_seconds=args.drain_timeout,
    )

    serve_task = asyncio.create_task(server.serve(), name='fixture-http-server')
    await coordinator.install_after(serve_task)

    # Enqueue 3 fake episodes: the worker picks the first one up immediately
    # (SlowClient hangs on it), episodes two and three stay pending.
    for ep_uuid, name in FAKE_EPISODES:
        await queue_service.add_episode(
            group_id='test-drain',
            name=name,
            content=f'Fake episode body for {name}',
            source_description='graceful drain fixture',
            episode_type=EpisodeType.text,
            entity_types=None,
            uuid=ep_uuid,
        )
    logger.info('Fixture: enqueued %d episodes, queue_depth=%d', len(FAKE_EPISODES), queue_service._queue_depth)

    Path(args.ready_file).write_text('ready\n', encoding='utf-8')

    await serve_task
    drain_result = await coordinator.finalize()
    logger.info('Fixture: graceful shutdown complete, drain=%s', drain_result)
    return 0


async def run_replay(args: argparse.Namespace) -> int:
    queue_service = QueueService(make_resilience(args.spool_dir))
    await queue_service.initialize(FastClient(Path(args.result_file)))

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

    retryer = EpisodeRetryer(
        spool=queue_service.spool,
        breaker=queue_service.circuit_breaker,
        episode_builder=episode_builder,
        graphiti_client=FastClient(Path(args.result_file)),
        interval_seconds=0.5,
        max_attempts=5,
        backoff_base_seconds=0.1,
    )
    retryer.start()

    deadline = asyncio.get_running_loop().time() + 20.0
    while queue_service.spool.count_pending() > 0 and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.2)
    await retryer.stop()

    pending = queue_service.spool.count_pending()
    logger.info('Replay finished: pending=%d', pending)
    print(f'REPLAY_PENDING={pending}', flush=True)
    return 0 if pending == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['serve', 'replay'], required=True)
    parser.add_argument('--spool-dir', required=True)
    parser.add_argument('--ready-file')
    parser.add_argument('--result-file')
    parser.add_argument('--drain-timeout', type=float, default=2.0)
    args = parser.parse_args()

    if args.mode == 'serve':
        return asyncio.run(run_serve(args))
    return asyncio.run(run_replay(args))


if __name__ == '__main__':
    sys.exit(main())
