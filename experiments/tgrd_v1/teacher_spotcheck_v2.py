"""Finite descriptive semantic spotcheck; never a hidden full7473 review gate."""
from pathlib import Path
from runtime_full import ROOT, OUT, read, checked_rows, freeze, save, digest, baseline, now
from automatic_teacher_quality import FOLDER, GATE, PROTOCOL

REPORT = FOLDER / 'spotcheck_report.json'


def validate_reviews(sources, teachers, roster, reviews):
    ids = roster['source_ids']
    if (roster.get('protocol') != PROTOCOL or roster.get('seed') != 42
            or roster.get('root_count') != 128 or len(ids) != 128 or len(set(ids)) != 128
            or roster.get('replacement_on_failure') is not False
            or roster.get('transfer128_roster') is not False
            or roster.get('public_test_used') is not False
            or [r['source_id'] for r in reviews] != ids):
        raise ValueError('Need the complete fixed128 quality sample, including negative decisions')
    for row in reviews:
        sid = row['source_id']; source, teacher = sources[sid], teachers[sid]
        if (row.get('review_kind') != 'assistant_full_text_mathematical_review'
                or row.get('all_text_read') is not True or type(row.get('accepted')) is not bool
                or not row.get('reason') or row.get('question_sha256') != source['question_sha256']
                or row.get('output_sha256') != teacher['output_sha256']
                or digest(source['question']) != source['question_sha256']
                or digest(teacher['output']) != teacher['output_sha256']):
            raise ValueError('Quality sample requires genuine output-bound semantic decisions')
    bad = sum(not r['accepted'] for r in reviews)
    return {'sample_count': 128, 'semantic_accepted': 128 - bad,
        'semantic_flagged': bad, 'flagged_fraction': bad / 128,
        'estimate_role': 'descriptive_strict_review_flags_not_independent_confirmation',
        'changes_automatic_training_roster': False, 'public_test_used': False}


def prepare():
    quality = read(GATE); roster = read(FOLDER / 'spotcheck_roster.json')
    if not set(roster['source_ids']) <= set(quality['accepted_root_ids']):
        raise ValueError('Sample not drawn from the automatic common root set')
    from full_selector import SOURCE, TEACHERS
    sources = {r['source_id']: r for r in read(SOURCE)}
    teachers = {r['source_id']: r for r in checked_rows(TEACHERS)}
    rows = read(OUT / 'material_audits/cal40_teacher_full_text_reviews_v1.json')
    rows += checked_rows(OUT / 'material_audits/teacher_additional_reviews_v1.jsonl')
    if len({r['source_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate historical/current decisions')
    byid = {r['source_id']: r for r in rows}
    missing = [sid for sid in roster['source_ids'] if sid not in byid]
    if missing:
        value = {'at_utc': now(), 'completed': 128 - len(missing), 'total': 128,
            'missing_ids': missing, 'complete': False, 'public_test_used': False}
        save(FOLDER / 'spotcheck_progress.json', value)
        return value
    selected = [byid[sid] for sid in roster['source_ids']]
    counts = validate_reviews(sources, teachers, roster, selected)
    decisions = FOLDER / 'spotcheck_actual_reviews.json'
    freeze(decisions, selected)
    paths = [Path(__file__), GATE, FOLDER / 'spotcheck_roster.json', decisions, SOURCE, TEACHERS]
    value = {'schema': 'tgrd_teacher_quality_spotcheck_report_v2', 'protocol': PROTOCOL,
        'complete': True, **counts,
        'historical_reviews_may_overlap_not_all_new_or_blind': True,
        'full7473_semantic_review_claimed': False, 'formal_training_enabled': False,
        'inputs': {str(p.relative_to(ROOT)): baseline.sha(p) for p in paths}}
    freeze(REPORT, value)
    save(FOLDER / 'spotcheck_progress.json', {'complete': True, 'completed': 128, 'total': 128})
    return value


def verify():
    report = read(REPORT)
    if (report.get('schema') != 'tgrd_teacher_quality_spotcheck_report_v2'
            or report.get('protocol') != PROTOCOL or report.get('complete') is not True
            or report.get('full7473_semantic_review_claimed') is not False
            or report.get('public_test_used') is not False or not report.get('inputs')):
        raise ValueError('Fixed descriptive128 sample is not complete')
    for name, checksum in report['inputs'].items():
        if baseline.sha(ROOT / name) != checksum:
            raise ValueError('Quality sample evidence changed: ' + name)
    from full_selector import SOURCE, TEACHERS
    counts = validate_reviews({r['source_id']: r for r in read(SOURCE)},
        {r['source_id']: r for r in checked_rows(TEACHERS)},
        read(FOLDER / 'spotcheck_roster.json'), read(FOLDER / 'spotcheck_actual_reviews.json'))
    if any(report.get(k) != v for k, v in counts.items()):
        raise ValueError('Quality sample descriptive counts changed')
    return [REPORT, *[ROOT / name for name in report['inputs']]]


if __name__ == '__main__':
    value = prepare()
    print({k: v for k, v in value.items() if k not in ('missing_ids', 'inputs')}, flush=True)
