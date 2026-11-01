"""Boundary-safe handoff to a full-data queue with no obsolete20-hour cutoff."""
from __future__ import annotations

import fcntl
import os
import subprocess
import time
from runtime_full import *


def main():
    with (OUT/'full_supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status('full_supervisor_waiting_boundary',full_train_target=7473)
        while True:
            check_time()
            old=read(OUT/'queue_state.json')
            if old.get('failed'):
                raise RuntimeError(f'Legacy prerequisite failed: {old["failed"]}')
            if {'teacher_cal40','teacher_train500'} <= set(old['completed']):break
            time.sleep(10)
        state=read(OUT/'full_queue_state.json') if (OUT/'full_queue_state.json').exists() else {'completed':[],'failed':[]}
        keeper=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
        try:
            while True:
                check_time()
                queue=read(OUT/'full_queue.json')
                pending=[j for j in queue if j['id'] not in state['completed']+state['failed']]
                if not pending:
                    state.update(status='waiting_for_next_reviewed_stage',updated_utc=now(),pid=os.getpid())
                    save(OUT/'full_queue_state.json',state)
                    time.sleep(15)
                    continue
                job=pending[0]
                if not all((ROOT/p).exists() for p in job.get('requires',[])):
                    state.update(status='waiting_for_review_or_prerequisite',waiting_job=job['id'],updated_utc=now())
                    save(OUT/'full_queue_state.json',state)
                    time.sleep(15)
                    continue
                logpath=OUT/'logs'/f'{job["id"]}.log'
                logpath.parent.mkdir(parents=True,exist_ok=True)
                with logpath.open('a') as log:
                    child=subprocess.Popen(job['argv'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',active_job=job['id'],child_pid=child.pid,pid=os.getpid(),updated_utc=now())
                    save(OUT/'full_queue_state.json',state)
                    append(OUT/'events.jsonl',{'event':'full_job_started','job':job,'at_utc':now(),'pid':child.pid})
                    while child.poll() is None:
                        try:check_time()
                        except TimeoutError:
                            # Let token/step-level worker checks checkpoint before terminating.
                            child.terminate()
                            child.wait(timeout=30)
                            raise
                        time.sleep(10)
                code=child.returncode
                state['completed' if code==0 else 'failed'].append(job['id'])
                state.update(status='job_completed' if code==0 else 'job_failed',exit_code=code,updated_utc=now())
                save(OUT/'full_queue_state.json',state)
                append(OUT/'events.jsonl',{'event':'full_job_finished','job':job['id'],'exit_code':code,'at_utc':now()})
                if code!=0:return
        finally:
            if keeper.poll() is None:keeper.terminate()


if __name__=='__main__':
    main()
