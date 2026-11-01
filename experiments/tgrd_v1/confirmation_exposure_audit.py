"""CPU-only provisional P60/H200 exposure audit; never loads public test data."""
from __future__ import annotations

import collections
import hashlib
import json
import re
import unicodedata

from runtime_full import ROOT, DATA, OUT, baseline, checked_rows, freeze, read


RULE = {
    'version': 1,
    'scope': 'Known workspace train-root exposure and lexical duplicate flags only',
    'normalization': 'NFKC casefold; Unicode word tokens; punctuation ignored',
    'literal_shingle_words': 5,
    'number_abstracted_shingle_words': 4,
    'jaccard_threshold_numerator': 3,
    'jaccard_threshold_denominator': 4,
    'decision': 'Flags require full-text review; zero flags is not semantic independence proof',
    'replacement_policy': 'No automatic removal, replacement or selection by model performance',
    'public_test_access': False,
}


def tokens(question, abstract_numbers=False):
    text = unicodedata.normalize('NFKC', question).casefold()
    # Keep comma-grouped/decimal numbers together, including in number abstraction.
    values = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?|[^\W\d_]+', text, flags=re.UNICODE)
    return tuple('<number>' if abstract_numbers and v[0].isdigit() else v for v in values)


def shingles(question, width, abstract_numbers=False):
    words = tokens(question, abstract_numbers)
    # A nonempty short question is represented by its entire token tuple.
    if len(words) < width:
        return {words} if words else set()
    return {words[i:i + width] for i in range(len(words) - width + 1)}


def validate_rosters(full_rows, historical_ids, rosters):
    sources = {}
    for row in full_rows:
        sid = row['source_id']
        if sid in sources or not sid.startswith('gsm8k:train:'):
            raise ValueError('Duplicate or non-training root in source roster')
        if hashlib.sha256(row['question'].encode()).hexdigest() != row['question_sha256']:
            raise ValueError('Question hash mismatch: ' + sid)
        sources[sid] = row
    if not set(historical_ids) <= sources.keys():
        raise ValueError('Historical IDs absent from current source roster')
    owners = {}
    for name, rows in rosters.items():
        for row in rows:
            sid = row['source_id']
            if sid in owners or sid in historical_ids:
                raise ValueError('Candidate overlaps historical/candidate roster: ' + sid)
            if sid not in sources or row != sources[sid]:
                raise ValueError('Candidate differs from signed training root: ' + sid)
            owners[sid] = name
    return sources, owners


def duplicate_flags(sources, historical_ids, owners):
    """Exhaustive positive-overlap pairs through an inverted shingle index.

    Evaluate candidate/historical and all distinct candidate/candidate pairs;
    never compare a root to itself. Integer thresholding avoids rounding ties.
    """
    ids = sorted(set(historical_ids) | set(owners))
    flags = {}
    for metric, width, abstract in [('literal', 5, False), ('number_abstracted', 4, True)]:
        bags = {sid: shingles(sources[sid]['question'], width, abstract) for sid in ids}
        inverted = collections.defaultdict(set)
        for sid, bag in bags.items():
            for part in bag:
                inverted[part].add(sid)
        for sid in sorted(owners):
            others = set().union(*(inverted[part] for part in bags[sid])) if bags[sid] else set()
            for other in sorted(others - {sid}):
                # Candidate pairs are assessed once, regardless of P60/H200 ownership.
                if other in owners and other < sid:
                    continue
                intersection = len(bags[sid] & bags[other])
                union = len(bags[sid] | bags[other])
                if not union or intersection * 4 < union * 3:
                    continue
                key = tuple(sorted((sid, other)))
                item = flags.setdefault(key, {
                    'root_ids': list(key),
                    'cohorts': [owners.get(x, 'known_historical') for x in key],
                    'metrics': {},
                    'review_state': 'pending_full_text_duplicate_review',
                })
                item['metrics'][metric] = {'intersection': intersection, 'union': union,
                                           'jaccard': intersection / union}
    return [flags[key] for key in sorted(flags)]


def main():
    destination = OUT / 'confirmation_exposure_v1'
    # Frozen before inspecting any duplicate results. No current test/prediction input.
    freeze(destination / 'rule.json', RULE)
    paths = [DATA / name for name in ['train7473.json', 'historical_exposure.json',
                                     'probe60_candidates.json', 'holdout200_candidates.json']]
    historical = read(paths[1])
    if len(historical['source_ids']) != len(set(historical['source_ids'])):
        raise ValueError('Duplicate historical IDs')
    # The historical inventory is a dated observation, not a live rescan of all train.
    for artifact in historical['scanned']:
        if baseline.sha(ROOT / artifact['path']) != artifact['sha256']:
            raise ValueError('Historical inventory artifact changed: ' + artifact['path'])
    rosters = {'P60': read(paths[2]), 'H200': read(paths[3])}
    if [len(rosters[k]) for k in ['P60', 'H200']] != [60, 200]:
        raise ValueError('Unexpected provisional confirmation roster sizes')
    sources, owners = validate_rosters(read(paths[0]), set(historical['source_ids']), rosters)
    if len(sources) != 7473:
        raise ValueError('Expected complete training roster')
    flags = duplicate_flags(sources, set(historical['source_ids']), owners)
    review_ids = set()
    additional = OUT / 'material_audits/teacher_additional_reviews_v1.jsonl'
    if additional.exists():
        review_ids.update(r['source_id'] for r in checked_rows(additional))
    calibration = OUT / 'material_audits/cal40_teacher_full_text_reviews_v1.json'
    if calibration.exists():
        review_ids.update(r['source_id'] for r in read(calibration))
    # Snapshot growing generation without editing/locking its output stream.
    teacher = OUT / 'materials/teacher_targets_train7473.jsonl'
    raw = teacher.read_bytes()
    raw = raw[:raw.rfind(b'\n') + 1]
    generated = []
    for line in raw.splitlines():
        row = json.loads(line)
        signature = row.pop('record_sha256')
        if hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() != signature:
            raise ValueError('Teacher record signature mismatch')
        if (hashlib.sha256(row['output'].encode()).hexdigest() != row['output_sha256'] or
                row['question_sha256'] != sources[row['source_id']]['question_sha256']):
            raise ValueError('Teacher output/question mismatch')
        generated.append(row['source_id'])
    if generated != [r['source_id'] for r in read(paths[0])[:len(generated)]]:
        raise ValueError('Teacher prefix order differs from frozen full roster')
    report = {
        'rule': RULE,
        'input_hashes': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths},
        'assistant_review_snapshot_hashes': {
            str(p.relative_to(ROOT)): baseline.sha(p)
            for p in [additional, calibration] if p.exists()},
        'known_historical_root_count': len(historical['source_ids']),
        'candidate_counts': {k: len(v) for k, v in rosters.items()},
        'exact_source_id_disjointness_verified': True,
        'flags': flags,
        'flagged_pair_count': len(flags),
        'candidate_full_text_review_exposure_ids': sorted(set(owners) & review_ids),
        'teacher_generated_prefix': {'complete_records': len(generated),
                                    'sha256': hashlib.sha256(raw).hexdigest(),
                                    'candidate_counts': {
                                        name: sum(sid in set(generated) for sid in
                                                  [r['source_id'] for r in rows])
                                        for name, rows in rosters.items()}},
        'source_rosters_modified': False,
        'confirmation_gate_granted': False,
        'formal_training_gate_granted': False,
        'public_test_loaded': False,
        'limits': [historical['limits'],
                   'Lexical thresholds do not detect every paraphrase, semantic family, or external exposure.',
                   'Teacher generation is distinct from student training and selection exposure.',
                   'Full-text root reviews are preparation exposure; no candidate performance was consulted.',
                   'Final full-train reuse ends independence for P60/H200 after that training.',
                   'No candidate labels or teacher quality were used to replace any roster member.'],
    }
    # Preserve the first snapshot. Subsequent invocations must use a new version directory.
    freeze(destination / 'audit.json', report)
    print(json.dumps({k: report[k] for k in ['candidate_counts', 'flagged_pair_count',
                     'candidate_full_text_review_exposure_ids', 'confirmation_gate_granted']}))


if __name__ == '__main__':
    main()
