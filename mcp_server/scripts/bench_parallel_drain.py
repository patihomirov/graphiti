"""Benchmark of the journal worker pool draining a parallel load.

Spins up a real :class:`QueueService` on a tmp SQLite journal with a fake
graphiti client whose ``add_episode`` sleeps for ``--sleep`` seconds (a stand-in
for the combined LLM extraction + Neo4j write), enqueues ``--episodes`` rows
across ``--groups`` groups, and measures wall-clock time until the journal
drains (``count_unfinished() == 0``).

By varying ``--workers`` you compare pool sizes on an identical workload:

- On ``--groups >= --workers`` independent (parallel-safe) groups the worker
  pool must show the inter-group speedup: throughput grows ~linearly with the
  pool size (N=2 >= 1.6x, N=4 >= 3.2x in the design contract).
- With ``--serial`` the episodes form a single serial zone (auto-previous /
  saga without an explicit previous saga episode), so the per-group processing
  lock keeps them strictly FIFO: N=2 must be within +/-10% of N=1.

No network and no graphiti-core data are touched; the client is faked out.

Usage::

    python3 scripts/bench_parallel_drain.py --workers 4 --episodes 60 \
        --sleep 1.0 --groups 4 --repeats 3
"""

import argparse
import asyncio
import logging
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from config.schema import ResilienceConfig  # noqa: E402
from services.queue_service import QueueService  # noqa: E402


class SlowClient:
    """Fake graphiti client: each add_episode 'processes' for --sleep seconds."""

    def __init__(self, sleep: float):
        self.sleep = float(sleep)
        self.processed: list[str] = []

    async def add_episode(self, **kwargs):
        await asyncio.sleep(self.sleep)
        self.processed.append(str(kwargs.get('uuid')))


def make_builder() -> callable:
    def episode_builder(plan: dict) -> dict:
        return {
            'name': plan['name'],
            'episode_body': plan['episode_body'],
            'source_description': plan['source_description'],
            'source': 'text',
            'group_id': plan['group_id'],
            'reference_time': datetime.now(timezone.utc),
            'uuid': plan.get('uuid'),
        }

    return episode_builder


def make_resilience(journal_path: str, workers: int) -> ResilienceConfig:
    return ResilienceConfig(
        enabled=True,
        journal_enabled=True,
        journal_path=journal_path,
        journal_lease_seconds=3600.0,
        journal_workers=workers,
        spool_enabled=False,
        max_queue_depth=1_000_000,
        failure_threshold=100,
        open_timeout_seconds=30.0,
        retryer_interval_seconds=0.2,
        spool_backoff_base_seconds=0.1,
        max_spool_attempts=5,
    )


async def run_once(service: QueueService, episodes: int, groups: int, serial: bool) -> float:
    """Enqueue ``episodes`` rows and return the wall-clock seconds to drain."""
    for i in range(episodes):
        group = f'bench-g{(i % groups) + 1}'
        plan_overrides = {'group_id': group, 'name': f'episode-{i}', 'uuid': f'bench-{i:06d}'}
        if serial:
            # A single serial zone: saga chain without an explicit previous saga
            # episode -> requires_serial, processed strictly FIFO by one worker.
            plan_overrides.update({'saga': 'bench-saga', 'saga_previous_episode_uuid': None})
        await service.add_episode(
            group_id=group,
            name=plan_overrides['name'],
            content=f'Bench episode body {i}',
            source_description='parallel drain bench',
            episode_type='text',
            entity_types=None,
            uuid=plan_overrides['uuid'],
            saga=plan_overrides.get('saga'),
            saga_previous_episode_uuid=plan_overrides.get('saga_previous_episode_uuid'),
        )

    start = asyncio.get_running_loop().time()
    # The first row may already be in flight by the time we measure; 0.05s of
    # settle lets the pool pick up a row before the clock starts.
    await asyncio.sleep(0.05)
    while await service.journal.count_unfinished() > 0:
        await asyncio.sleep(0.02)
    return asyncio.get_running_loop().time() - start


async def bench(args: argparse.Namespace) -> int:
    wall_times: list[float] = []
    for _ in range(args.repeats):
        tmp_root = Path('/tmp/opencode')
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='bench-drain-', dir=tmp_root) as td:
            service = QueueService(make_resilience(str(Path(td) / 'bench.db'), args.workers))
            try:
                await service.initialize(SlowClient(args.sleep), episode_builder=make_builder())
                wall = await run_once(service, args.episodes, args.groups, args.serial)
                wall_times.append(wall)
            finally:
                await service.close()

    med = statistics.median(wall_times)
    throughput = args.episodes / med
    mode = 'serial' if args.serial else 'parallel-safe'
    print(
        f'RESULT mode={mode} workers={args.workers} groups={args.groups} '
        f'episodes={args.episodes} sleep={args.sleep} repeats={args.repeats} '
        f'wall(median)={med:.2f}s throughput={throughput:.2f} ep/s '
        f'runs={[round(w, 2) for w in wall_times]}',
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4, help='Global journal worker pool size')
    parser.add_argument('--episodes', type=int, default=60, help='Total episodes to enqueue')
    parser.add_argument('--sleep', type=float, default=1.0, help='Fake add_episode duration (s)')
    parser.add_argument('--groups', type=int, default=4, help='Number of groups (round-robin)')
    parser.add_argument('--repeats', type=int, default=1, help='Runs per configuration (median reported)')
    parser.add_argument('--serial', action='store_true', help='Force a single serial zone workload')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    return asyncio.run(bench(args))


if __name__ == '__main__':
    sys.exit(main())
