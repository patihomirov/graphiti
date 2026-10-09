"""End-to-end test of the durable SQLite journal surviving a hard kill (-9).

The in-memory queue (queue_service.py) and the graceful-drain-on-SIGTERM
path both lose episodes when the process is SIGKILLed. The durable journal is
the write-path source of truth: add_episode persists the full plan to the
episode_queue table BEFORE processing, so a kill -9 loses nothing.

Scenario:

1. Spawn the fixture subprocess (real uvicorn server + QueueService with the
   journal enabled and a hanging fake graphiti client). It enqueues 3 fake
   episodes: the first is in flight (stuck in a 30s fake LLM call), two are
   pending in the journal table.
2. SIGKILL the subprocess (no graceful drain, no spool — the rows are durable).
3. Spawn a second fixture subprocess on the SAME journal path with a fast fake
   client: initialize() requeues the abandoned processing row, the journal
   worker processes all 3 rows exactly once.
4. Assert: no duplicates (each uuid appears exactly once), rc=0.

Usage: python3 scripts/test_graceful_drain_journal.py
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV_PY = REPO / '.venv' / 'bin' / 'python'
FIXTURE = REPO / 'scripts' / 'graceful_drain_journal_fixture.py'

EXPECTED_UUIDS = [
    '11111111-1111-1111-1111-111111111111',
    '22222222-2222-2222-2222-222222222222',
    '33333333-3333-3333-3333-333333333333',
]


def wait_for_file(path: Path, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text(encoding='utf-8').strip():
            return True
        time.sleep(0.1)
    return False


def main() -> int:
    tmp_root = Path('/tmp/opencode')
    tmp_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix='journal-kill9-', dir=tmp_root))
    journal_path = workdir / 'journal.db'
    ready_file = workdir / 'ready.flag'
    result_file = workdir / 'replayed.txt'
    serve_log = workdir / 'serve.log'

    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO / 'src')

    checks: list[tuple[str, bool, str]] = []

    # ---------------------------------------------------------------- serve
    print(f'== journal kill -9 test ==\nworkdir: {workdir}')
    serve = subprocess.Popen(
        [
            str(VENV_PY),
            str(FIXTURE),
            '--mode', 'serve',
            '--journal-path', str(journal_path),
            '--ready-file', str(ready_file),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if not wait_for_file(ready_file):
        serve.kill()
        out = serve.stdout.read() if serve.stdout else ''
        print(f'FAIL: fixture never became ready\n{out}')
        return 1
    print(f'[ok] fixture ready (pid={serve.pid}), 1 in flight + 2 pending in journal')

    # Give the worker a moment to pick up episode #1 (hang in SlowClient),
    # then SIGKILL — no graceful drain, no spool. Wait for the 1s lease on the
    # in-flight row to expire so the replay's requeue_stale can reclaim it.
    time.sleep(0.5)
    serve.kill()
    print('[..] SIGKILL sent')
    out, _ = serve.communicate(timeout=30)
    serve_log.write_text(out, encoding='utf-8')
    time.sleep(1.5)

    checks.append(('fixture was killed by -9', serve.returncode == -signal.SIGKILL, f'rc={serve.returncode}'))
    checks.append(('journal db exists on disk', journal_path.exists(), str(journal_path)))

    # --------------------------------------------------------------- replay
    replay = subprocess.run(
        [
            str(VENV_PY),
            str(FIXTURE),
            '--mode', 'replay',
            '--journal-path', str(journal_path),
            '--result-file', str(result_file),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    (workdir / 'replay.log').write_text(replay.stdout + replay.stderr, encoding='utf-8')

    replayed = result_file.read_text(encoding='utf-8').split() if result_file.exists() else []
    checks.append(('replay subprocess rc=0', replay.returncode == 0, f'rc={replay.returncode}'))
    checks.append(
        ('replay drained the journal',
         replay.stdout.count('REPLAY_UNFINISHED=0') == 1,
         replay.stdout.strip().splitlines()[-1] if replay.stdout.strip() else '')
    )
    checks.append(
        ('all 3 episodes replayed exactly once',
         sorted(replayed) == EXPECTED_UUIDS and len(replayed) == len(EXPECTED_UUIDS),
         f'replayed={sorted(replayed)}'
         )
    )

    # --------------------------------------------------------------- verdict
    print('\n== checks ==')
    failed = 0
    for name, ok, detail in checks:
        status = 'PASS' if ok else 'FAIL'
        failed += 0 if ok else 1
        print(f'  [{status}] {name} ({detail})')

    print('\n== serve log tail ==')
    for line in serve_log.read_text(encoding='utf-8').splitlines()[-10:]:
        print('  ' + line)
    print('\n== replay log tail ==')
    for line in (workdir / 'replay.log').read_text(encoding='utf-8').splitlines()[-10:]:
        print('  ' + line)

    print(f'\nRESULT: {"PASS" if failed == 0 else f"FAIL ({failed} checks)"}')
    print(f'artifacts: {workdir}')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
