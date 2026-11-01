"""Synthetic provenance regressions; no outputs become scientific evidence."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import transfer128_production_v1 as p
from runtime_full import read,freeze,append,digest,baseline,BASE
import test_trial_engine as fixtures
from fixed_trial_engine_v1 import run_fixed_root


class ConfigurationTests(unittest.TestCase):
    def summary(self):
        return {'protocol':p.PROTOCOL,'cohort':'cal34_general7b_v2','recipe_id':'local_explanation_general7b_v2',
            'actual_model_results_analyzed':True,'all24_seed_configuration_cells_complete':True,
            'public_test_used':False,'status':'positive_development_signal_requires_confirmation',
            'selected_configuration':{'updates':1,'threshold':.03,'trial_seed':701},'root_ids':['dev']}
    def test_negative_development_has_no_selected_config_and_cannot_expand(self):
        s=self.summary();s.update(status='no_positive_development_support',selected_configuration=None)
        r=p.configuration_fields(s);self.assertFalse(r['scientific_trial_enabled'])
        self.assertEqual(r['automatic_expansion'],'unsupported_no_expansion')
        s['selected_configuration']={'updates':1,'threshold':.03,'trial_seed':701}
        with self.assertRaises(ValueError):p.configuration_fields(s)
    def test_wrong_recipe_partial_grid_and_boolean_hyperparameters_rejected(self):
        original=self.summary()
        for change in ({'recipe_id':'old_math7b'},{'all24_seed_configuration_cells_complete':False},
                {'actual_model_results_analyzed':False},{'selected_configuration':{'updates':True,'threshold':.03,'trial_seed':701}},
                {'selected_configuration':{'updates':1,'threshold':True,'trial_seed':701}},
                {'selected_configuration':{'updates':1,'threshold':.03,'trial_seed':1701}}):
            with self.assertRaises(ValueError):p.configuration_fields({**original,**change})
    def test_actual_grid_missing_blocks_before_GPU_or_material_qualification(self):
        with patch('calibration_summary.main',side_effect=ValueError('Missing actual grid')) as actual, \
                patch.object(p,'material_evidence') as material,patch.object(p,'gpu') as gpu:
            with self.assertRaisesRegex(ValueError,'Missing actual grid'):p.configuration_evidence()
            actual.assert_called_once();material.assert_not_called();gpu.assert_not_called()
    def test_unsupported_run_creates_no_roots_and_is_not_false_fallback_completion(self):
        with tempfile.TemporaryDirectory() as d,patch.object(p,'FOLDER',Path(d)), \
                patch.object(p,'CONFIGURATION',Path(d)/'config.json'), \
                patch.object(p,'load_configuration',return_value={'scientific_trial_enabled':False}), \
                patch.object(p,'material_evidence') as material,patch.object(p,'gpu') as gpu:
            freeze(p.CONFIGURATION,{'synthetic_test_only':True})
            r=p.run();self.assertEqual(r['trial_roots_executed'],0)
            self.assertFalse(r['fixed128_trials_complete']);self.assertFalse((Path(d)/'roots').exists())
            material.assert_not_called();gpu.assert_not_called()


class ActualFileReplayTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.TrialEngineTests();f.setUp();self.addCleanup(f.doCleanups);self.f=f
        for split,rows in f.probes.items():
            for i,row in enumerate(rows):
                q=f'Synthetic {split} probe {i}: add three and four.'
                row.update(source_id=f.source['source_id'],question=q,question_sha256=digest(q),gold='7')
        self.checksum=digest('test_config_not_science');self.params={'updates':1,'threshold':.03,'trial_seed':701}
        def evaluate(learner,probes,path):
            if not path.exists():
                for index,probe in enumerate(probes):
                    correct=(learner.state=='explanation' and probe['split']=='select') or (learner.state=='ordinary' and probe['split']=='audit')
                    output='The answer is \\boxed{'+('7' if correct else '9')+'}.'
                    nll=.5 if learner.state=='explanation' and probe['split']=='select' else 1.
                    append(path,{**probe,'output':output,'output_sha256':digest(output),
                        'scoring_text':output,'generation_tokens':12,'finish_reason':'stop',
                        'prompt_sha256':digest(BASE['student_prompt'].format(question=probe['question'])),
                        'answer_nll':nll,**baseline.score(output,probe['gold'])})
            return p.prediction_metrics(path,probes)
        self.result=run_fixed_root(f.learner,f.source,f.pool,f.probes,f.folder,seed=701,updates=1,
            initial_hash=digest('S0'),weight_hash=fixtures.weight_hash,restore=fixtures.restore,
            evaluate=evaluate,threshold=.03,configuration_sha256=self.checksum)
    def replay(self):
        return p.replay_root(self.f.source,self.f.pool,self.f.probes,self.f.folder,
            self.checksum,self.params,digest('S0'))
    def test_raw_branches_and_predictions_replay_winner_despite_negative_audit(self):
        result,paths=self.replay();self.assertEqual(result,self.result)
        self.assertEqual(result['decision']['selected_id'],'explain')
        self.assertEqual(result['Vaudit_accuracy_gain'],-1.)
        self.assertTrue(paths);self.assertFalse(result['audit_can_change_selection'])
    def test_changed_trained_adapter_and_changed_selected_result_are_fatal(self):
        path=self.f.folder/'branch1/adapter.safetensors';original=path.read_bytes();path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'trained branch'):self.replay()
        path.write_bytes(original)
        path=self.f.folder/'trial_result.json';result=read(path);result['decision']['selected_id']='e0'
        path.write_text(__import__('json').dumps(result))
        with self.assertRaises(ValueError):self.replay()
    def test_validly_resigned_prediction_still_cannot_lie_about_score_or_probe(self):
        path=self.f.folder/'branch1/Vsel.jsonl';rows=p.checked_rows(path)
        path.unlink();rows[0]['correct']=False
        for row in rows:
            row.pop('record_sha256');append(path,row)
        with self.assertRaisesRegex(ValueError,'score'):self.replay()
    def test_changed_frozen_configuration_or_extra_audit_prevents_replay(self):
        self.params['threshold']=.1
        with self.assertRaisesRegex(ValueError,'input differs'):self.replay()
        self.params['threshold']=.03
        # The selected branch and e0 have audits; a fabricated third candidate
        # would fail complete training and Vsel checks before any audit use.
        self.f.pool['candidates'].append({'candidate_id':'unselected','completion':'x',
            'completion_sha256':digest('x'),'student_tokens':40})
        with self.assertRaises(ValueError):self.replay()


if __name__=='__main__':unittest.main()
