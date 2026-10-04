"""End-to-end test of the graceful drain on SIGTERM (no LLM, no Neo4j).

Scenario (per the 2026-10-04 lesson: systemctl restart silently killed the
in-memory episode queue):

1. Spawn the fixture subprocess (real uvicorn server + QueueService with a
   hanging fake graphiti client). It enqueues 3 fake episodes: the first is
   in flight (stuck in a 30s fake LLM call), two are pending.
2. Send SIGTERM to the subprocess.
3. Assert: the process exits cleanly and all 3 episodes are on disk in the
   spool (2 pending spilled + 1 in-flight cancelled and spilled).
4. Spawn the replay subprocess: the EpisodeRetryer must ingest all 3 spooled
   episodes through a fast fake client, emptying the spool.

Usage: python3 scripts/test_graceful_drain.py
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
FIXTURE = REPO / 'scripts' / 'graceful_drain_fixture.py'

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
    workdir = Path(tempfile.mkdtemp(prefix='graceful-drain-', dir=tmp_root))
    spool_dir = workdir / 'spool'
    ready_file = workdir / 'ready.flag'
    result_file = workdir / 'replayed.txt'
    serve_log = workdir / 'serve.log'

    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO / 'src')

    checks: list[tuple[str, bool, str]] = []

    # ---------------------------------------------------------------- serve
    print(f'== graceful drain test ==\nworkdir: {workdir}')
    serve = subprocess.Popen(
        [
            str(VENV_PY),
            str(FIXTURE),
            '--mode', 'serve',
            '--spool-dir', str(spool_dir),
            '--ready-file', str(ready_file),
            '--drain-timeout', '2.0',
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
    print(f'[ok] fixture ready (pid={serve.pid}), queue holds 3 fake episodes')

    # Give the worker a moment to pick up episode #1, then SIGTERM.
    time.sleep(0.5)
    serve.send_signal(signal.SIGTERM)
    print('[..] SIGTERM sent')

    try:
        out, _ = serve.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        serve.kill()
        out, _ = serve.communicate()
        print('FAIL: fixture did not exit within 30s after SIGTERM')
        return 1
    serve_log.write_text(out, encoding='utf-8')

    checks.append(('fixture exited cleanly (rc=0)', serve.returncode == 0, f'rc={serve.returncode}'))
    checks.append(
        ('drain logged', 'Graceful drain: spooled' in out, 'Graceful drain log line'),
    )

    spooled = sorted(p.stem for p in spool_dir.glob('*.json')) if spool_dir.exists() else []
    checks.append(
        ('all 3 episodes in spool', spooled == EXPECTED_UUIDS, f'spooled={spooled}')
    )
    checks.append(
        ('in-flight episode spooled with drain reason', 'cancelled during graceful drain' in out, 'cancel+spool path')
    )

    # --------------------------------------------------------------- replay
    replay = subprocess.run(
        [
            str(VENV_PY),
            str(FIXTURE),
            '--mode', 'replay',
            '--spool-dir', str(spool_dir),
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
    checks.append(('replay emptied the spool', replay.stdout.count('REPLAY_PENDING=0') == 1, replay.stdout.strip().splitlines()[-1] if replay.stdout else ''))
    checks.append(
        ('all 3 episodes replayed', sorted(replayed) == EXPECTED_UUIDS, f'replayed={sorted(replayed)}')
    )

    # --------------------------------------------------------------- verdict
    print('\n== checks ==')
    failed = 0
    for name, ok, detail in checks:
        status = 'PASS' if ok else 'FAIL'
        failed += 0 if ok else 1
        print(f'  [{status}] {name} ({detail})')

    print(f'\n== drain log tail ==')
    for line in serve_log.read_text(encoding='utf-8').splitlines():
        if 'Graceful' in line or 'drain' in line.lower() or 'spool' in line.lower():
            print('  ' + line)

    print(f'\nRESULT: {"PASS" if failed == 0 else f"FAIL ({failed} checks)"}')
    print(f'artifacts: {workdir}')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
