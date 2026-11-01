"""Explicit full-GSM8K authorization; keeps original signed runtime.py intact."""
from __future__ import annotations

import time
from runtime import *
import runtime as legacy_runtime


def deadline():
    value = read(OUT / 'full_run_authorization.json')
    assert value['supersedes_original_20_hour_cutoff']
    if value['hard_deadline_utc'] is None:
        return float('inf')
    import datetime
    return datetime.datetime.fromisoformat(value['hard_deadline_utc']).timestamp()


def check_time(reserve=0):
    if (OUT/'STOP_REQUESTED').exists():
        raise TimeoutError('Explicit local stop marker exists; retained completed records')
    if time.time()+reserve >= deadline():
        raise TimeoutError('Full-run configured deadline reached')


def generate(*args, **kwargs):
    # Original implementation consults its module global deadline on every token.
    # Override in THIS new process only, preserving the old on-disk source/signature.
    legacy_runtime.deadline = deadline
    return legacy_runtime.generate(*args, **kwargs)
