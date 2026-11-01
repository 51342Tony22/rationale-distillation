"""Versioned TGRD execution helpers; never overwrite the earlier pilots."""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/math_distill_v1'))
import baseline

OUT = ROOT / 'output/tgrd_v1'
DATA = ROOT / 'data/tgrd_v1'
BASE = json.loads((ROOT / 'experiments/math_distill_v1/config.json').read_text())


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def save(path, value):
    baseline.save(Path(path), value)


def freeze(path, value):
    baseline.freeze(Path(path), value)


def append(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    value = dict(value)
    value['record_sha256'] = digest(json.dumps(value, sort_keys=True))
    with p.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return value


def checked_rows(path):
    result = rows(path)
    for row in result:
        value = dict(row)
        signature = value.pop('record_sha256')
        assert digest(json.dumps(value, sort_keys=True)) == signature, f'Corrupt row: {path}'
    return result


def deadline():
    c = read(OUT / 'run_window.json')
    return dt.datetime.fromisoformat(c['deadline_utc']).timestamp()


def check_time(reserve=0):
    if time.time() + reserve >= deadline():
        raise TimeoutError('Authorized 20-hour run window reached; durable outputs retained')


def status(stage, **values):
    save(OUT / 'status.json', {'updated_utc': now(), 'pid': os.getpid(), 'stage': stage, **values})


@contextlib.contextmanager
def gpu():
    p = ROOT / 'output/local_math_gpu.lock'
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def signature(stage, paths, extra=None):
    value = {'inputs': {str(Path(p).relative_to(ROOT)): baseline.sha(Path(p)) for p in paths},
             'extra': extra or {}}
    freeze(OUT / 'signatures' / f'{stage}.json', value)


def load_model(kind='student'):
    import mlx.core as mx
    from mlx_lm import load
    mx.set_memory_limit(14 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    model, tokenizer = load(str(ROOT / BASE[f'{kind}_dir']),
                            tokenizer_config={'local_files_only': True, 'trust_remote_code': False})
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, question, *, teacher=False, limit=1024, user_text=None):
    import mlx.core as mx
    import re
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    if teacher:
        text = user_text if user_text is not None else question
        prompt = tokenizer.apply_chat_template([
            {'role': 'system', 'content': BASE['teacher_system']},
            {'role': 'user', 'content': text}], tokenize=False, add_generation_prompt=True)
    else:
        prompt = BASE['student_prompt'].format(question=question)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    assert len(ids) + limit <= 4096, 'Context overflow; no truncation permitted'
    start = time.perf_counter()
    output, last, boundary = '', None, None
    model.eval()
    for part in stream_generate(model, tokenizer, ids, max_tokens=limit, sampler=make_sampler(temp=0.0)):
        last = part
        output += part.text
        if not teacher:
            boundary = re.search(r'(?:^|\n)\s*Question:', output)
            if boundary:
                break
        if time.time() >= deadline():
            raise TimeoutError('Generation interrupted at run-window deadline')
    assert last is not None
    return {'prompt_sha256': digest(prompt), 'prompt_tokens': len(ids), 'output': output,
            'output_sha256': digest(output), 'scoring_text': output[:boundary.start()] if boundary else output,
            'generation_tokens': last.generation_tokens,
            'finish_reason': 'next_question' if boundary else last.finish_reason,
            'seconds': time.perf_counter() - start, 'peak_mlx_memory_bytes': mx.get_peak_memory()}
