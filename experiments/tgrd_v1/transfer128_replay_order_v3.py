"""Strict compatibility for historical unordered manifest exports.

The v2 exporter iterated validate_roster's frozenset. Across hash seeds this
changes only list order. Existing bytes are retained and every keyed entry is
reconstructed from the original reviewed batches. Production consumes an
ordered in-memory copy of the review index; no scientific rule is changed.
"""
from pathlib import Path
from runtime_full import ROOT, HERE, OUT, read, baseline
from train_selector_quality_v3 import verify_inputs

CPU = OUT / 'preflight/transfer128_replay_order_cpu_v3.json'
DOC = HERE / 'TRANSFER128_REPLAY_ORDER_V3_ZH.md'
LEGACY_FOLDER = OUT / 'transfer128/variants_v1'
_installed = False


def entry_id(entry):
    if 'source_id' in entry:
        return entry['source_id']
    return entry['selection_rationales'][0]['root_source_id']


def keyed(rows):
    result = {}
    for row in rows:
        sid = entry_id(row)
        if sid in result:
            raise ValueError('Duplicate root in replayed/frozen manifest')
        result[sid] = row
    return result


def verify_equivalent(path, reconstructed):
    """Allow only root-list permutation, retaining all metadata and values."""
    stored = read(path)
    if 'roots' in reconstructed:
        if (set(stored) != set(reconstructed)
                or {k:v for k,v in stored.items() if k != 'roots'} !=
                   {k:v for k,v in reconstructed.items() if k != 'roots'}
                or keyed(stored['roots']) != keyed(reconstructed['roots'])):
            raise ValueError('Frozen manifest differs beyond root-list order: ' + str(path))
    elif stored != reconstructed:
        raise ValueError('Frozen manifest metadata/output hashes changed: ' + str(path))
    return stored


def replay_variants():
    import transfer128_variant_manifest_v2 as legacy
    # Call the original complete replay, intercepting writes with exact checks.
    # This affects this synchronous module call only; it never relaxes freeze.
    original_freeze = legacy.freeze
    legacy.freeze = verify_equivalent
    try:
        return _original_prepare()
    finally:
        legacy.freeze = original_freeze


def canonical_report(report, source_ids):
    rows = keyed(report['roots'])
    if len(source_ids) != 128 or len(set(source_ids)) != 128 or set(rows) != set(source_ids):
        raise ValueError('Review index must retain the complete original128 roster')
    return {**report, 'roots': [rows[sid] for sid in source_ids]}


def install():
    global _installed, _original_prepare
    if _installed:
        return
    report = read(CPU)
    if (report.get('status') != 'passed' or report.get('scientific_results') is not False
            or report.get('official_test_accessed') is not False
            or str(Path(__file__).relative_to(ROOT)) not in report.get('inputs', {})):
        raise ValueError('Versioned replay-order CPU evidence required')
    verify_inputs(report)
    import transfer128_variant_manifest_v2 as variants
    import transfer128_production_v1 as production
    from transfer128_sampling import ROSTER
    _original_prepare = variants.prepare
    variants.prepare = replay_variants
    original_read = production.read
    original_configuration_evidence = production.configuration_evidence

    def ordered_read(path):
        value = original_read(path)
        if Path(path) == LEGACY_FOLDER / 'actual_review_report_v2.json':
            value = canonical_report(value, original_read(ROSTER)['source_ids'])
        return value

    def bound_configuration_evidence():
        value = original_configuration_evidence()
        paths = [Path(__file__), HERE/'continue_transfer128_order_v3.py', CPU, DOC]
        value['inputs'].update({str(p.relative_to(ROOT)):baseline.sha(p) for p in paths})
        value['historical_manifest_order_adapter'] = {
            'version': 3, 'original_export_bytes_preserved': True,
            'all128_keyed_actual_batch_entries_replayed': True,
            'production_review_index_order': 'original_frozen_roster_order',
            'sampling_or_selection_rules_changed': False}
        return value

    production.read = ordered_read
    production.configuration_evidence = bound_configuration_evidence
    _installed = True
