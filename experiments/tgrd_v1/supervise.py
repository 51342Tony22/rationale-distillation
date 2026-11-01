"""Durable sequential job queue with a hard wall-clock run window."""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from runtime import ROOT, OUT, now, read, save, deadline, append


def main():
    with (OUT / 'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(OUT / 'queue_state.json') if (OUT / 'queue_state.json').exists() else {'completed': [], 'failed': []}
        caffeinate = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
        try:
            while time.time() < deadline():
                queue = read(OUT / 'queue.json')
                pending = [j for j in queue if j['id'] not in state['completed'] + state['failed']]
                if not pending:
                    state.update(status='waiting_for_next_reviewed_stage', updated_utc=now(), pid=os.getpid())
                    save(OUT / 'queue_state.json', state)
                    time.sleep(15)
                    continue
                job = pending[0]
                if job.get('requires') and not all((ROOT / p).exists() for p in job['requires']):
                    state.update(status='waiting_for_review_or_prerequisite', waiting_job=job['id'], updated_utc=now())
                    save(OUT / 'queue_state.json', state)
                    time.sleep(15)
                    continue
                path = OUT / 'logs' / f"{job['id']}.log"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a') as log:
                    child = subprocess.Popen(job['argv'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state.update(status='running', active_job=job['id'], child_pid=child.pid, pid=os.getpid(), updated_utc=now())
                    save(OUT / 'queue_state.json', state)
                    append(OUT / 'events.jsonl', {'event':'job_started','job':job,'pid':child.pid,'at_utc':now()})
                    remaining = max(1, deadline() - time.time())
                    try:
                        code = child.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            code = child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            code = child.wait()
                        state.update(status='deadline_checkpointed', active_job=job['id'], updated_utc=now())
                        save(OUT / 'queue_state.json', state)
                        return
                state['completed' if code == 0 else 'failed'].append(job['id'])
                state.update(status='job_completed' if code == 0 else 'job_failed', exit_code=code, updated_utc=now())
                save(OUT / 'queue_state.json', state)
                append(OUT / 'events.jsonl', {'event':'job_finished','job':job['id'],'exit_code':code,'at_utc':now()})
                if code != 0:
                    # Do not cascade into training/evaluation after a failed prerequisite.
                    return
            state.update(status='deadline_checkpointed', updated_utc=now())
            save(OUT / 'queue_state.json', state)
        finally:
            if caffeinate.poll() is None:
                caffeinate.terminate()


if __name__ == '__main__':
    main()
