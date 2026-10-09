"""Fixture process for the Phase-4 per-model routing + breakthrough e2e test.

Serves a real QueueService with the durable SQLite journal enabled, a
fallback LLM model and enqueue_breakthrough_max_pending>0. The fake graphiti
client always returns 429 on the active model (``model-A``) and succeeds on
the fallback (``model-B``). The scenario:

1. The circuit breaker is tripped OPEN (failure_threshold 429s).
2. While OPEN, 8 new canaries are enqueued - bounded breakthrough lets ALL of
   them in (no CircuitOpenError / graphiti_backpressure).
3. The journal pool drains every row on the live model-B channel: per-model
   reputation learns model-A is storming, so after the first row every later
   row is routed straight to model-B.
4. After the drain, sleeping past open_timeout and enqueuing one more canary
   takes the breaker to half-open; the successful probe on model-B closes it.
5. per_model stats in the resilience snapshot mirror all of the above.

No LLM and no graph DB are touched: the graphiti client is faked out entirely.

Usage: python3 scripts/model_routing_journal_fixture.py \
    --journal-path <path> --result-file <path>
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from graphiti_core.llm_client.errors import RateLimitError  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402

from config.schema import ResilienceConfig  # noqa: E402
from services.circuit_breaker import CircuitOpenError  # noqa: E402
from services.queue_service import QueueService  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_routing_fixture')

ACTIVE_MODEL = 'model-A'
FALLBACK_MODEL = 'model-B'
MAX_QUEUE_DEPTH = 20
FIRST_WAVE = 8  # accepted during the OPEN window via breakthrough
PROBE_UUIDS = ['59999999-9999-9999-9999-999999999999']


def canary_uuids() -> list[str]:
    return [f'50000000-0000-0000-0000-{i:012d}' for i in range(1, FIRST_WAVE + 1)]


class FakeLLM:
    """Fake LLM client exposing the model-override hook (model-A is active)."""

    def __init__(self):
        self.model = ACTIVE_MODEL
        self._override: str | None = None

    def set_model_override(self, model: str | None) -> None:
        self._override = model


class RoutingGraphiti:
    """Fake graphiti client: model-A always 429s, model-B succeeds."""

    def __init__(self, result_file: Path):
        self.llm_client = FakeLLM()
        self.result_file = result_file
        self.calls = 0

    async def add_episode(self, **kwargs):
        self.calls += 1
        model = self.llm_client._override or self.llm_client.model
        if model == ACTIVE_MODEL:
            raise RateLimitError('fake primary 429 storm')
        with self.result_file.open('a', encoding='utf-8') as fh:
            fh.write(f"{kwargs.get('uuid')} {model}\n")
        return None


def make_resilience(journal_path: str) -> ResilienceConfig:
    return ResilienceConfig(
        enabled=True,
        journal_enabled=True,
        journal_path=journal_path,
        journal_lease_seconds=1.0,
        journal_workers=2,
        spool_enabled=False,
        max_queue_depth=MAX_QUEUE_DEPTH,
        enqueue_breakthrough_max_pending=FIRST_WAVE + 2,
        failure_threshold=3,
        open_timeout_seconds=4.0,
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
    client = RoutingGraphiti(Path(args.result_file))
    await queue_service.initialize(client, episode_builder=make_builder())

    # 1. Trip the breaker open (3 transient 429 failures).
    for _ in range(3):
        await queue_service._breaker.record_failure(RateLimitError())
    assert queue_service._breaker.state == 'open', 'breaker must be open before intake'
    logger.info('Fixture: breaker open, enqueue_breakthrough_max_pending=%d',
                queue_service.resilience.enqueue_breakthrough_max_pending)

    # 2. First wave: while OPEN, every canary must be ACCEPTED (no
    #    CircuitOpenError / graphiti_backpressure) up to the bounded capacity.
    accepted = 0
    for u in canary_uuids():  # first wave only
        try:
            await queue_service.add_episode(
                group_id='test-routing',
                name=f'routing-{u[:8]}',
                content=f'Fake routing episode {u}',
                source_description='model routing fixture',
                episode_type=EpisodeType.text,
                entity_types=None,
                uuid=u,
            )
            accepted += 1
        except CircuitOpenError as e:
            logger.error('Unexpected backpressure during OPEN intake: %s', e)
            break
        except Exception as e:  # noqa: BLE001 - any intake failure fails the test
            logger.error('Unexpected intake error: %s', e)
            break

    # 3. Drain the first wave on the live fallback channel.
    deadline = asyncio.get_running_loop().time() + 30.0
    while (
        await queue_service.journal.count_unfinished() > 0
        and asyncio.get_running_loop().time() < deadline
    ):
        await asyncio.sleep(0.2)

    # 4. Past open_timeout, enqueue one probe canary: it flips the breaker to
    #    half-open and its success on model-B closes it.
    await asyncio.sleep(queue_service.resilience.open_timeout_seconds + 1.0)
    probe = PROBE_UUIDS[0]
    probe_accepted = False
    try:
        await queue_service.add_episode(
            group_id='test-routing',
            name='routing-probe',
            content='Fake routing probe',
            source_description='model routing fixture',
            episode_type=EpisodeType.text,
            entity_types=None,
            uuid=probe,
        )
        probe_accepted = True
    except CircuitOpenError as e:
        logger.error('Probe canary rejected: %s', e)

    deadline = asyncio.get_running_loop().time() + 30.0
    while (
        await queue_service.journal.count_unfinished() > 0
        and asyncio.get_running_loop().time() < deadline
    ):
        await asyncio.sleep(0.2)
    final_state = await queue_service._breaker.get_snapshot()

    # 5. Per-model stats from the snapshot.
    snap = await queue_service.get_resilience_snapshot()
    per_model = snap['journal'].get('per_model', {})

    done = await queue_service.journal.count_done()
    unfinished = await queue_service.journal.count_unfinished()

    logger.info('Fixture finished: done=%d unfinished=%d calls=%d', done, unfinished, client.calls)
    print(f'ROUTING_ACCEPTED={accepted}', flush=True)
    print(f'ROUTING_PROBE_ACCEPTED={probe_accepted}', flush=True)
    print(f'ROUTING_DONE={done}', flush=True)
    print(f'ROUTING_UNFINISHED={unfinished}', flush=True)
    print(f'ROUTING_BREAKER_STATE={final_state["state"]}', flush=True)
    print(f'ROUTING_PRIMARY_429={per_model.get(ACTIVE_MODEL, {}).get("429", 0)}', flush=True)
    print(f'ROUTING_PRIMARY_HEALTHY={per_model.get(ACTIVE_MODEL, {}).get("healthy", None)}', flush=True)
    print(f'ROUTING_FALLBACK_SUCCESSES={per_model.get(FALLBACK_MODEL, {}).get("successes", 0)}', flush=True)
    print(f'ROUTING_FALLBACK_HEALTHY={per_model.get(FALLBACK_MODEL, {}).get("healthy", None)}', flush=True)

    await queue_service.close()

    expected_total = FIRST_WAVE + 1
    assert accepted == FIRST_WAVE, f'all first-wave canaries must be accepted: {accepted}'
    assert probe_accepted is True, 'probe canary must be accepted'
    assert done == expected_total, 'every canary must finish'
    assert unfinished == 0, 'no rows may remain unfinished'
    assert final_state['state'] == 'closed', (
        f'successful probe on the live channel must close the breaker: {final_state["state"]}'
    )
    assert per_model.get(ACTIVE_MODEL, {}).get('healthy') is False, (
        '429-storming primary must be unhealthy in per_model stats'
    )
    assert per_model.get(FALLBACK_MODEL, {}).get('successes', 0) >= expected_total, (
        'fallback must carry all successful runs'
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
