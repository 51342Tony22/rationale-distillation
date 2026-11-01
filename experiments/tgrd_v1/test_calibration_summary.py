"""Synthetic configuration effects are software checks, not experiment data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import calibration_summary as summary
from calibration_summary import aggregate,SEEDS,UPDATES,THRESHOLDS,RULE
from runtime_full import ROOT,OUT,freeze,read,append,digest,baseline
from trial_engine import run_root
from test_trial_engine import FakeLearner,restore,weight_hash


class CalibrationSummaryTests(unittest.TestCase):
    def cells(self,gain=None):
        gain=gain or (lambda s,u,t:0)
        return [{'seed':seed,'updates':updates,'threshold':threshold,'roots':[
            {'source_id':'synthetic:1','baseline_correct':3,'selected_correct':3+gain(seed,updates,threshold),
             'selected_id':'explanation','Vaudit_nll_gain':-.1,'untrained_correct':6},
            {'source_id':'synthetic:2','baseline_correct':1,'selected_correct':1,
             'selected_id':'e0','Vaudit_nll_gain':0.,'untrained_correct':0}]}
            for seed in SEEDS for updates in UPDATES for threshold in THRESHOLDS]

    def test_exact_count_ties_use_fewer_updates_then_higher_threshold(self):
        result=aggregate(self.cells(lambda s,u,t:1),['synthetic:1','synthetic:2'])
        self.assertEqual(result['selected_configuration'],{'updates':1,'threshold':.10,'trial_seed':701})
        self.assertAlmostEqual(result['best_primary_development_mean_gain'],1/12)
        self.assertEqual(len(result['configuration_summaries']),24)
        self.assertEqual(result['configuration_summaries'][0]['selected_nonempty_roots'],1)
        self.assertEqual(result['configuration_summaries'][0]['untrained_Vsel_ceiling_roots'],1)
        self.assertFalse(result['independent_confirmation_passed'])
        self.assertFalse(result['formal_training_enabled'])

    def test_repeat_seed_cannot_reselect_primary_configuration(self):
        def gain(seed,updates,threshold):
            if seed==701:return 1 if updates==3 and threshold==.03 else 0
            return -1 if updates==3 and threshold==.03 else 2
        result=aggregate(self.cells(gain),['synthetic:1','synthetic:2'])
        self.assertEqual(result['selected_configuration'],{'updates':3,'threshold':.03,'trial_seed':701})
        self.assertFalse(result['stability']['repeat_gain_positive'])
        self.assertLess(result['stability']['repeat_mean_gain'],0)

    def test_all_nonpositive_keeps_negative_cells_and_does_not_scale(self):
        result=aggregate(self.cells(lambda s,u,t:-1),['synthetic:1','synthetic:2'])
        self.assertEqual(result['status'],'no_positive_development_support')
        self.assertIsNone(result['selected_configuration'])
        self.assertEqual(len(result['configuration_summaries']),24)
        self.assertTrue(all(c['audit_negative_roots']==1 for c in result['configuration_summaries']))
        self.assertFalse(result['public_evaluation_enabled'])
        result=aggregate(self.cells(),['synthetic:1','synthetic:2'])
        self.assertIsNone(result['selected_configuration'])

    def test_missing_duplicate_grid_or_dropped_reordered_root_is_rejected(self):
        cells=self.cells()
        for changed in (cells[:-1],cells+[cells[0]],
                        [dict(cells[0],roots=cells[0]['roots'][:1])]+cells[1:],
                        [dict(cells[0],roots=list(reversed(cells[0]['roots'])))]+cells[1:]):
            with self.assertRaises(ValueError):aggregate(changed,['synthetic:1','synthetic:2'])

    def test_e0_audit_cannot_diverge_from_baseline_and_counts_cannot_be_fractional(self):
        for key,value in [('selected_correct',2),('Vaudit_nll_gain',.01)]:
            changed=copy.deepcopy(self.cells());changed[0]['roots'][1][key]=value
            with self.assertRaises(ValueError):aggregate(changed,['synthetic:1','synthetic:2'])
        for key,value in [('baseline_correct',3.5),('selected_correct',7),('Vaudit_nll_gain',float('nan'))]:
            changed=copy.deepcopy(self.cells());changed[0]['roots'][0][key]=value
            with self.assertRaises(ValueError):aggregate(changed,['synthetic:1','synthetic:2'])


class CalibrationSummaryIntegrationTests(unittest.TestCase):
    """Temporary synthetic state-machine files; no MLX or real GPU gates."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='synthetic_summary_integration_',dir=OUT/'preflight')
        self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name)
        self.out=self.folder/'output';self.data=self.folder/'data'
        self.paths=SimpleNamespace(trials_folder=self.out/'trials',pool_folder=self.out/'pool',
            cases=self.out/'cases.json',reviews=self.out/'reviews.json',variants=self.data/'variants.json',
            expected_roots=1,signature_prefix='synthetic_summary')
        q='Synthetic engineering question?';sid='synthetic:1'
        self.source={'source_id':sid,'question':q,'question_sha256':digest(q),'gold':'1'}
        self.pool={'source_id':sid,'question_sha256':digest(q),'teacher_output_sha256':digest('ordinary'),
            'candidates':[{'candidate_id':cid,'completion':text,'completion_sha256':digest(text),'student_tokens':tokens}
                          for cid,text,tokens in [('e0','ordinary',0),('explain','explanation',40)]]}
        self.probes={split:[{'id':f'{sid}:{split}:{i}','source_id':sid,'split':split,
            'question':f'Synthetic {split} question {i}?','question_sha256':digest(f'Synthetic {split} question {i}?'),
            'gold':'1'} for i in range(6)] for split in ('select','audit')}
        freeze(self.data/'cal40.json',[self.source])
        freeze(self.paths.pool_folder/'reviewed_pools.json',[self.pool])
        teacher={'source_id':sid,'output':'ordinary','output_sha256':digest('ordinary')}
        append(self.out/'materials/teacher_targets_cal40.jsonl',teacher)
        freeze(self.paths.reviews,[{'source_id':sid,'accepted':True,'all_text_read':True,
            'question_sha256':digest(q),'output_sha256':digest('ordinary')}])
        freeze(self.paths.cases,[{'source_id':sid,'candidates':[]}])
        freeze(self.paths.variants,[{'source_id':sid,'variants':self.probes}])
        freeze(self.paths.trials_folder/'summary_rule_v1.json',
            {**RULE,'cohort':'synthetic','expected_roots':1,'predeclared_before_any_actual_grid_result':True})
        # Deliberately fake reports are confined to this disposable test folder.
        signature=self.out/'signatures/trial_driver_gpu_v1.json'
        freeze(signature,{'inputs':{},'synthetic_engineering_fixture_only':True})
        freeze(self.out/'preflight/trial_driver_gpu_v1.json',
            {'status':'passed','signature_file_sha256':baseline.sha(signature),'synthetic_engineering_fixture_only':True})
        def evaluate(learner,probes,path):
            split=probes[0]['split']
            if split=='select':count,nll={'S0':(0,2.),'ordinary':(1,1.),'explanation':(3,.5)}[learner.state]
            else:count,nll={'ordinary':(3,.8),'explanation':(0,1.5)}[learner.state]
            for i,probe in enumerate(probes):
                output='The answer is '+('1' if i<count else '2')
                append(path,{**probe,'output':output,'output_sha256':digest(output),'scoring_text':output,
                    'correct':i<count,'answer_nll':nll,'synthetic_engineering_fixture_only':True})
            return summary.summarize(read_rows(path),[p['id'] for p in probes])
        for seed in SEEDS:
            for updates in UPDATES:
                folder=self.paths.trials_folder/f'seed{seed}_updates{updates}'
                run_root(FakeLearner(),self.source,self.pool,self.probes,folder/'synthetic_1',
                    seed=seed,updates=updates,initial_hash=digest('S0'),weight_hash=weight_hash,
                    restore=restore,evaluate=evaluate)
                freeze(folder/'status.json',{'complete_for_declared_cohort':True,'roots':1,'seed':seed,
                    'updates':updates,'cohort':'synthetic','public_test_used':False})
                freeze(self.out/f'signatures/trial_synthetic_summary_s{seed}_u{updates}_v1.json',{'inputs':{}})

    def analyze(self):
        with patch.multiple(summary,OUT=self.out,DATA=self.data,cohort_paths=lambda name:self.paths,
                            validate_probes=lambda bundle,source,rebuild:bundle['variants']), \
             patch('explanation_pool.assemble'), \
             patch('material_builder.validate_pool',side_effect=lambda source,teacher,pool,steps:pool['candidates']):
            summary.main('synthetic')

    def test_actual_trial_state_machine_files_recompute_complete_grid_and_keep_negative_audits(self):
        self.analyze()
        result=read(self.paths.trials_folder/'development_grid_summary_v1.json')
        self.assertEqual(result['status'],'no_positive_development_support')
        self.assertIsNone(result['selected_configuration'])
        self.assertEqual(len(result['configuration_summaries']),24)
        self.assertTrue(all(c['paired_correct_count_gain']==-3 for c in result['configuration_summaries']))
        self.assertFalse(result['formal_training_enabled'])

    def test_changed_raw_score_or_audit_based_reselection_is_rejected(self):
        folder=self.paths.trials_folder/'seed701_updates1/synthetic_1'
        target=folder/'branch1/Vsel.jsonl';original=target.read_bytes();rows=read_rows(target)
        target.unlink();rows[0]['correct']=False
        for row in rows:row.pop('record_sha256');append(target,row)
        with self.assertRaises(ValueError):self.analyze()
        target.write_bytes(original)
        result_path=folder/'development_results.json';result=read(result_path)
        result['configurations'][0]['selected_id']='e0'
        result_path.write_text(json.dumps(result))
        with self.assertRaises(ValueError):self.analyze()


def read_rows(path):
    from runtime_full import checked_rows
    return checked_rows(path)


if __name__=='__main__':unittest.main()
