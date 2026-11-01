"""Versioned public evaluator using genuine adopted-quality-v3 gates only."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from runtime_full import (ROOT, HERE, OUT, BASE, read, save, append, checked_rows,
    baseline, digest, signature, generate, gpu, check_time)
from budgeted_materials import PROTOCOL, SELECTOR_CONDITIONS
from evaluation_selector_quality_v3 import validate_request
from evaluate import test_items
import score_sensitivity_v2 as sensitivity


def validate_saved(saved, items, condition, seed, signature_sha):
    if len(saved) > len(items):
        raise ValueError('Prediction file exceeds the complete test inventory')
    for row, item in zip(saved, items):
        boundary = re.search(r'(?:^|\n)\s*Question:', row['output'])
        text = row['output'][:boundary.start()] if boundary else row['output']
        v2, _ = sensitivity.parse(text)
        score = baseline.score(text, item['gold'])
        if (any(row.get(k) != v for k, v in item.items())
                or row.get('protocol') != PROTOCOL or row.get('condition') != condition
                or row.get('seed') != seed
                or row.get('evaluation_signature_file_sha256') != signature_sha
                or digest(row['output']) != row.get('output_sha256')
                or row.get('prompt_sha256') != digest(BASE['student_prompt'].format(question=item['question']))
                or row.get('scoring_text') != text
                or any(row.get(k) != v for k, v in score.items())
                or row.get('v2_correct') != (v2 is not None and v2 == baseline.number(item['gold']))):
            raise ValueError('Saved prediction differs from its frozen input/model or recomputed score')


def write_summary(folder, saved, condition, seed, verified, signature_sha):
    complete = len(saved) == 1319
    save(folder / 'summary.json', {'protocol': PROTOCOL, 'condition': condition, 'seed': seed,
        'completed': len(saved), 'total': 1319, 'complete': complete,
        'correct': sum(x['correct'] for x in saved),
        'v2_correct': sum(x['v2_correct'] for x in saved),
        'accuracy': sum(x['correct'] for x in saved) / len(saved) if saved else None,
        'generation_seconds': sum(x['seconds'] for x in saved),
        'training_artifact': verified['training_artifact'],
        'evaluation_signature_file_sha256': signature_sha,
        'partial_score_is_not_full_test_result': not complete,
        'single_seed_does_not_establish_stability': True})


def main(condition, seed=42, adapter=None):
    # Gate, actual final adapters, transfer sample/configuration/rules and
    # historical exposure disclosures are checked BEFORE the public loader.
    verified = validate_request(condition, seed, adapter)
    items = test_items()
    if len(items) != 1319 or len({r['source_id'] for r in items}) != 1319:
        raise ValueError('Complete official GSM8K test inventory required')
    folder = OUT / 'evaluation' / PROTOCOL / condition / f'seed{seed}'
    stage = f'{PROTOCOL}_eval_GSM8K_{condition}_s{seed}'
    paths = [Path(__file__), HERE / 'evaluation_selector_quality_v3.py', HERE / 'evaluate.py',
        HERE / 'runtime.py', HERE / 'runtime_full.py', Path(sensitivity.__file__), Path(baseline.__file__),
        ROOT / 'experiments/math_distill_v1/config.json',
        ROOT / 'data/math_distill_v1/source/test.jsonl', *verified['signature_paths']]
    signature(stage, paths, {'protocol': PROTOCOL, 'condition': condition, 'seed': seed,
        'ordered_ids': [r['source_id'] for r in items], 'prompt': BASE['student_prompt'],
        'greedy': True, 'max_new_tokens': 1024, 'dataset_count': 1319,
        'training_artifact': verified['training_artifact']})
    signature_sha = baseline.sha(OUT / 'signatures' / (stage + '.json'))
    saved = checked_rows(folder / 'predictions.jsonl')
    validate_saved(saved, items, condition, seed, signature_sha)
    write_summary(folder, saved, condition, seed, verified, signature_sha)
    if len(saved) == 1319:
        return  # cached completion repairs an interrupted summary without loading a model
    with gpu():
        check_time(reserve=60)
        import mlx.core as mx
        from mlx_lm import load
        mx.set_memory_limit(14 * 1024**3); mx.set_cache_limit(512 * 1024**2)
        mx.random.seed(seed); baseline.verify_student(BASE)
        model, tokenizer = load(str(ROOT / BASE['student_dir']),
            adapter_path=str(verified['adapter_directory']),
            tokenizer_config={'local_files_only': True, 'trust_remote_code': False})
        for item in items[len(saved):]:
            check_time(reserve=30)
            output = generate(model, tokenizer, item['question'], limit=1024)
            v2, _ = sensitivity.parse(output['scoring_text'])
            value = {**item, **output, **baseline.score(output['scoring_text'], item['gold']),
                'v2_correct': v2 is not None and v2 == baseline.number(item['gold']),
                'protocol': PROTOCOL, 'condition': condition, 'seed': seed,
                'evaluation_signature_file_sha256': signature_sha}
            committed = append(folder / 'predictions.jsonl', value); saved.append(committed)
            write_summary(folder, saved, condition, seed, verified, signature_sha)
            if len(saved) % 10 == 0:
                print(f'{condition}/seed{seed}: {len(saved)}/1319', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest='command', required=True)
    pretest = sub.add_parser('freeze-pretest')
    for name in ('roster', 'rules', 'disclosures'):
        pretest.add_argument('--' + name, type=Path, required=True)
    sub.add_parser('freeze-gate')
    runner = sub.add_parser('run')
    runner.add_argument('--condition', choices=SELECTOR_CONDITIONS, required=True)
    runner.add_argument('--seed', type=int, choices=(42,), default=42)
    runner.add_argument('--adapter')
    args = parser.parse_args()
    if args.command == 'freeze-pretest':
        from evaluation_selector_quality_v3 import freeze_pretest
        freeze_pretest(args.roster, args.rules, args.disclosures)
    elif args.command == 'freeze-gate':
        from evaluation_selector_quality_v3 import freeze_gate
        freeze_gate()
    else:
        main(args.condition, args.seed, args.adapter)
