"""End-to-end test of the Phase-4 per-model routing + bounded breakthrough.

Scenario:

1. Spawn the fixture subprocess: a real QueueService with the durable SQLite
   journal enabled, ``resilience.model_fallbacks = ['model-B']`` and
   ``enqueue_breakthrough_max_pending`` > 0. The fake graphiti client 429s on
   the active model (``model-A``) and succeeds on the fallback (``model-B``).
2. The breaker is tripped OPEN; while open, 8 canaries are enqueued and must
   ALL be accepted (no CircuitOpenError / graphiti_backpressure) - the bounded
   breakthrough path.
3. The pool drains every row on ``model-B``: per-model reputation learns the
   primary is storming, so later rows are routed straight to the live channel.
4. Past ``open_timeout`` a probe canary flips the breaker to half-open; its
   success on ``model-B`` closes the breaker.
5. Assert: all 9 rows done on ``model-B``, breaker closed, per_model stats
   mark the primary unhealthy and the fallback healthy.

Usage: python3 scripts/test_model_routing_journal.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV_PY = REPO / '.venv' / 'bin' / 'python'
FIXTURE = REPO / 'scripts' / 'model_routing_journal_fixture.py'

FIRST_WAVE = 8
TOTAL = FIRST_WAVE + 1  # 8 canaries + 1 probe


def main() -> int:
    tmp_root = Path('/tmp/opencode')
    tmp_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix='journal-routing-', dir=tmp_root))
    journal_path = workdir / 'journal.db'
    result_file = workdir / 'processed.txt'

    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO / 'src')

    checks: list[tuple[str, bool, str]] = []

    print(f'== journal model routing test ==\nworkdir: {workdir}')
    run = subprocess.run(
        [
            str(VENV_PY),
            str(FIXTURE),
            '--journal-path', str(journal_path),
            '--result-file', str(result_file),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    (workdir / 'fixture.log').write_text(run.stdout + run.stderr, encoding='utf-8')

    checks.append(('fixture rc=0', run.returncode == 0, f'rc={run.returncode}'))

    def line(prefix: str) -> str:
        for ln in run.stdout.splitlines():
            if ln.startswith(prefix):
                return ln.split('=', 1)[1]
        return ''

    checks.append(
        ('all first-wave canaries accepted while breaker open',
         line('ROUTING_ACCEPTED=') == str(FIRST_WAVE),
         f"accepted={line('ROUTING_ACCEPTED=')}")
    )
    checks.append(
        ('probe canary accepted',
         line('ROUTING_PROBE_ACCEPTED=') == 'True',
         f"probe={line('ROUTING_PROBE_ACCEPTED=')}")
    )
    checks.append(
        ('all rows done', line('ROUTING_DONE=') == str(TOTAL), f"done={line('ROUTING_DONE=')}")
    )
    checks.append(
        ('no unfinished rows', line('ROUTING_UNFINISHED=') == '0',
         f"unfinished={line('ROUTING_UNFINISHED=')}")
    )
    checks.append(
        ('breaker closed after success on live channel',
         line('ROUTING_BREAKER_STATE=') == 'closed',
         f"state={line('ROUTING_BREAKER_STATE=')}")
    )
    checks.append(
        ('primary marked unhealthy in per_model stats',
         line('ROUTING_PRIMARY_HEALTHY=') == 'False',
         f"primary_healthy={line('ROUTING_PRIMARY_HEALTHY=')}")
    )
    checks.append(
        ('primary recorded a 429',
         int(line('ROUTING_PRIMARY_429=') or 0) >= 1,
         f"primary_429={line('ROUTING_PRIMARY_429=')}")
    )
    checks.append(
        ('fallback marked healthy',
         line('ROUTING_FALLBACK_HEALTHY=') == 'True',
         f"fallback_healthy={line('ROUTING_FALLBACK_HEALTHY=')}")
    )
    checks.append(
        ('fallback carried every successful run',
         int(line('ROUTING_FALLBACK_SUCCESSES=') or 0) >= TOTAL,
         f"fallback_successes={line('ROUTING_FALLBACK_SUCCESSES=')}")
    )

    # Every uuid processed exactly once, always on the fallback model.
    processed = result_file.read_text(encoding='utf-8').splitlines() if result_file.exists() else []
    lines = [(u, m) for u, m in (ln.split() for ln in processed)]
    uuids = [u for u, _ in lines]
    checks.append(
        ('every uuid processed exactly once (no duplicates)',
         len(uuids) == TOTAL and len(set(uuids)) == TOTAL,
         f'uuids={len(uuids)} unique={len(set(uuids))}')
    )
    checks.append(
        ('all successes on the fallback model',
         all(m == 'model-B' for _, m in lines),
         f'models={sorted(set(m for _, m in lines))}')
    )

    print('\n== checks ==')
    failed = 0
    for name, ok, detail in checks:
        status = 'PASS' if ok else 'FAIL'
        failed += 0 if ok else 1
        print(f'  [{status}] {name} ({detail})')

    print('\n== fixture log tail ==')
    for ln in (workdir / 'fixture.log').read_text(encoding='utf-8').splitlines()[-12:]:
        print('  ' + ln)

    print(f'\nRESULT: {"PASS" if failed == 0 else f"FAIL ({failed} checks)"}')
    print(f'artifacts: {workdir}')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
