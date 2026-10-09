"""End-to-end test of the worker-level multi-model failover (Phase 3, variant 'b').

Scenario:

1. Spawn the fixture subprocess: a real QueueService with the durable SQLite
   journal enabled and ``resilience.model_fallbacks = ['model-B']``. The fake
   graphiti client empties on the active model (``model-A``) and succeeds on the
   fallback (``model-B``). Three fake episodes are enqueued.
2. The journal worker must, for each episode, fail over inline (inside the same
   claim) from ``model-A`` to ``model-B`` on the empty response, record the
   failed model in the ``attempted_models`` journal column, and mark the row
   done - with no duplicates.
3. Assert: all 3 episodes done exactly once (each uuid recorded once), the
   successful call ran on ``model-B``, ``attempted_models`` holds only
   ``['model-A']`` per row, and the process rc=0.

Usage: python3 scripts/test_model_failover_journal.py
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV_PY = REPO / '.venv' / 'bin' / 'python'
FIXTURE = REPO / 'scripts' / 'model_failover_journal_fixture.py'

EXPECTED_UUIDS = [
    '41111111-1111-1111-1111-111111111111',
    '42222222-2222-2222-2222-222222222222',
    '43333333-3333-3333-3333-333333333333',
]


def main() -> int:
    tmp_root = Path('/tmp/opencode')
    tmp_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix='journal-failover-', dir=tmp_root))
    journal_path = workdir / 'journal.db'
    result_file = workdir / 'processed.txt'

    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO / 'src')

    checks: list[tuple[str, bool, str]] = []

    print(f'== journal model failover test ==\nworkdir: {workdir}')
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
        timeout=60,
    )
    (workdir / 'fixture.log').write_text(run.stdout + run.stderr, encoding='utf-8')

    checks.append(('fixture rc=0', run.returncode == 0, f'rc={run.returncode}'))

    def line(prefix: str) -> str:
        for ln in run.stdout.splitlines():
            if ln.startswith(prefix):
                return ln
        return ''

    done = line('FAILOVER_DONE=')
    checks.append(('all episodes done', done == f'FAILOVER_DONE={len(EXPECTED_UUIDS)}', done))
    checks.append(
        ('no unfinished rows', line('FAILOVER_UNFINISHED=') == 'FAILOVER_UNFINISHED=0',
         line('FAILOVER_UNFINISHED='))
    )
    checks.append(
        ('2 runs per episode (fail + fallback)',
         line('FAILOVER_CALLS=') == f'FAILOVER_CALLS={len(EXPECTED_UUIDS) * 2}',
         line('FAILOVER_CALLS='))
    )

    # attempted_models must record only the failed active model for every row.
    attempted_line = line('FAILOVER_ATTEMPTED=')
    expected_attempted = {u: ['model-A'] for u in EXPECTED_UUIDS}
    if attempted_line:
        try:
            attempted = json.loads(attempted_line.split('=', 1)[1])
            checks.append(
                ('attempted_models = [model-A] per row',
                 attempted == expected_attempted,
                 str(attempted))
            )
        except ValueError:
            checks.append(('attempted_models parseable', False, attempted_line))

    # Each uuid processed exactly once (no duplicates), always on the fallback.
    processed = result_file.read_text(encoding='utf-8').splitlines() if result_file.exists() else []
    lines = [(u, m) for u, m in (ln.split() for ln in processed)]
    uuids = [u for u, _ in lines]
    checks.append(
        ('every uuid processed exactly once',
         len(uuids) == len(EXPECTED_UUIDS) and sorted(uuids) == sorted(EXPECTED_UUIDS),
         f'uuids={sorted(uuids)}')
    )
    checks.append(
        ('all successes ran on the fallback model',
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
    for ln in (workdir / 'fixture.log').read_text(encoding='utf-8').splitlines()[-10:]:
        print('  ' + ln)

    print(f'\nRESULT: {"PASS" if failed == 0 else f"FAIL ({failed} checks)"}')
    print(f'artifacts: {workdir}')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
