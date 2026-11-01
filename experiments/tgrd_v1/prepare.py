"""Freeze the 20-hour execution window and data/source inventory."""
from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import random
import re
from pathlib import Path

from runtime import ROOT, HERE, DATA, OUT, BASE, baseline, freeze, read, rows, save, now, status


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / 'run_window.json').exists():
        start = dt.datetime.now(dt.timezone.utc)
        freeze(OUT / 'run_window.json', {
            'started_utc': start.isoformat(), 'deadline_utc': (start + dt.timedelta(hours=20)).isoformat(),
            'authorized_hours': 20, 'authorization': 'User: 可以跑完整实验，20小时，开干吧',
            'policy': 'Sequential GPU work; checkpoint and stop at deadline; no unreported scope reduction',
            'full_plan': 'FYP_TGRD_MASTER_EXPERIMENT_PLAN_ZH.md'})
    train = rows(ROOT / 'data/math_distill_v1/train.jsonl')
    dev = read(ROOT / 'data/math_distill_v1/pilot100/dev.json')
    assert len(train) == 500 and len(dev) == 250
    assert not {r['source_id'] for r in train} & {r['source_id'] for r in dev}
    exposure = set()
    scanned = []
    # Inventory old source manifests/data only; avoid scanning model weights.
    for p in sorted((ROOT / 'data').rglob('*')):
        if not p.is_file() or DATA in p.parents or p.suffix not in {'.json', '.jsonl'} or p.stat().st_size > 25_000_000:
            continue
        text = p.read_text(errors='replace')
        if 'gsm8k:train:' in text:
            found = set(re.findall(r'gsm8k:train:\d+', text))
            exposure.update(found)
            scanned.append({'path': str(p.relative_to(ROOT)), 'sha256': baseline.sha(p), 'ids': len(found)})
    exposure.update(r['source_id'] for r in train + dev)
    original = rows(ROOT / 'data/math_distill_v1/source/train.jsonl')
    remaining = [i for i in range(len(original)) if f'gsm8k:train:{i}' not in exposure]
    random.Random(20261026).shuffle(remaining)
    def source(i):
        q = original[i]
        rationale, gold = q['answer'].rsplit('####', 1)
        return {'source_id': f'gsm8k:train:{i}', 'source_index': i, 'question': q['question'],
                'gold': gold.strip(), 'source_answer': q['answer'],
                'reference_rationale': re.sub(r'<<[^>]*>>', '', rationale).strip(),
                'question_sha256': baseline.sha_text(q['question']) if hasattr(baseline, 'sha_text') else __import__('hashlib').sha256(q['question'].encode()).hexdigest()}
    freeze(DATA / 'train500.json', train)
    freeze(DATA / 'dev250.json', dev)
    freeze(DATA / 'cal40.json', dev[:40])
    freeze(DATA / 'probe60_candidates.json', [source(i) for i in remaining[:60]])
    freeze(DATA / 'holdout200_candidates.json', [source(i) for i in remaining[60:260]])
    freeze(DATA / 'historical_exposure.json', {'source_ids': sorted(exposure), 'scanned': scanned,
            'limits': 'Current workspace manifests only; older external workspaces and pretraining exposure not fully recoverable. New holdouts remain provisional until near-duplicate audit.'})
    assets = [ROOT / 'experiments/math_distill_v1/config.json', ROOT / 'data/math_distill_v1/source_manifest.json',
              ROOT / 'data/math_distill_v1/pilot100/manifest.json', ROOT / 'FYP_TGRD_MASTER_EXPERIMENT_PLAN_ZH.md']
    freeze(DATA / 'source_manifest.json', {'base': BASE, 'source_hashes': {str(p.relative_to(ROOT)): baseline.sha(p) for p in assets},
              'sets': {p.name: baseline.sha(p) for p in sorted(DATA.glob('*.json')) if p.name != 'source_manifest.json'},
              'versions': {n: importlib.metadata.version(n) for n in ['mlx','mlx-lm','transformers','numpy']},
              'official_test_model_evaluation_started': False})
    status('prepared', train_candidates=500, existing_dev=250, provisional_probe=60, provisional_holdout=200,
           historical_manifest_ids=len(exposure), gpu_model_loaded=False)
    print(json.dumps(read(OUT / 'status.json'), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
