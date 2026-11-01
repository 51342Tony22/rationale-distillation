"""Read-only queue admission check for the adopted explanation v3 stage."""
from pathlib import Path
from runtime_full import OUT, read
from train_selector_quality_v3 import validate_gate
from train_selector_stage import verify_inputs
from explanation_quality_v3 import authority, REPORT, EXPORT

def main():
    authority()
    cpu=read(OUT/'preflight/explanation_quality_v3_cpu.json');verify_inputs(cpu)
    if cpu.get('status')!='passed' or cpu.get('total_CPU_tests_passed',0)<157:
        raise ValueError('Current integrated v3 CPU evidence required')
    for path in (REPORT,EXPORT):verify_inputs(read(path))
    for name in ('teacher_full7473','full_fixed_explanations_general7b_v2'):
        verify_inputs(read(OUT/'signatures'/f'{name}.json'))
    verify_inputs(read(OUT/'preflight/full_fixed_explanation_generation_complete_v1.json'))
    bundle,paths=validate_gate('M1',42)
    print({'validated_roots':len(bundle['rows']),'evidence_files':len(paths),
        'actual_sample128_complete':True,'public_evaluation_enabled':False},flush=True)

if __name__=='__main__':main()
