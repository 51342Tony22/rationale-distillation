"""Explicit development cohorts with separate immutable output namespaces."""
from dataclasses import dataclass
from pathlib import Path
from runtime_full import DATA, OUT

COHORTS = ('bootstrap10','cal40_reviewed34_v2')


@dataclass(frozen=True)
class Cohort:
    name: str
    relations: Path
    variants: Path
    reviews: Path
    cases: Path
    diagnostics: Path
    tokenization_report: Path
    pool_folder: Path
    trials_folder: Path
    expected_roots: int
    signature_prefix: str


def cohort_paths(name='bootstrap10'):
    if name == 'bootstrap10':
        return Cohort(name, DATA/'bootstrap_calibration_relations_v2.json',
            DATA/'bootstrap_calibration_variants_v2.json',
            OUT/'material_audits/calibration_bootstrap_teacher_reviews.json',
            OUT/'calibration/positions_v2_cases.json',
            OUT/'calibration/positions_v2_diagnostics.jsonl',
            OUT/'preflight/positions_v2_tokenization.json',
            OUT/'calibration/shared_pool_v1', OUT/'calibration/trials_v1',10,'bootstrap')
    if name == 'cal40_reviewed34_v2':
        folder = OUT/'calibration'/name
        return Cohort(name, DATA/'cal40_reviewed_relations_v2.json',
            DATA/'cal40_reviewed_variants_v2.json',
            OUT/'material_audits/cal40_teacher_full_text_reviews_v1.json',
            folder/'positions_v2_cases.json', folder/'positions_v2_diagnostics.jsonl',
            OUT/'preflight/cal40_reviewed34_v2_tokenization.json',
            folder/'shared_pool_v1', folder/'trials_v1',34,name)
    raise ValueError('Unknown calibration cohort: '+name)
