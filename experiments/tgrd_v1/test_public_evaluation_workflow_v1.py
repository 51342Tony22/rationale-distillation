"""Temporary/synthetic workflow tests, never a public dataset or scientific gate."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import public_evaluation_workflow_v1 as m
from runtime_full import save,digest
import evaluation_selector_quality_v3 as selector
import evaluate_selector_quality_v3 as selector_eval
import evaluate_transfer128_v1 as transfer_eval
import transfer128_production_v1 as production


class PublicWorkflowTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name).resolve();self.out=self.root/'output/tgrd_v1'
        self.negative=dict(scientific_trial_enabled=False,development_status='no_positive_development_support',selected_configuration=None)
        self.positive=dict(scientific_trial_enabled=True,development_status='positive_development_signal_requires_confirmation',
            selected_configuration=dict(updates=3,threshold=.03,trial_seed=701))

    def exposure(self):
        return m.exposure_contents('历史项目曾查看部分官方测试子集，不能宣称项目从未看过官方test。',
            dict(limits='Synthetic history not complete',source_ids=['gsm8k:train:1']))

    def test_historical_public_exposure_is_not_zero_or_an_invented_clean_subset(self):
        value=self.exposure()
        self.assertTrue(value['known_partial_official_test_exposure'])
        self.assertIsNone(value['exposed_official_test_ids'])
        self.assertIsNone(value['recoverable_official_test_exposure_count'])
        self.assertFalse(value['current_test_is_first_ever_blind_evaluation'])
        self.assertFalse(value['exposed_unexposed_sensitivity_possible_from_recovered_IDs'])
        with self.assertRaises(ValueError):m.exposure_contents('No recoverable statement',dict(limits='x',source_ids=['gsm8k:train:1']))
        with self.assertRaises(ValueError):m.exposure_contents('历史项目曾查看部分官方测试子集',dict(limits='x',source_ids=['gsm8k:test:1']))

    def test_scope_uses_complete_development_outcome_not_public_scores_or_loose_flags(self):
        self.assertEqual(m.choose_scope(self.negative),['M1','NLL','M3'])
        self.assertEqual(m.choose_scope(self.positive),['M1','NLL','M3','M7','M6'])
        for invalid in ({'scientific_trial_enabled':True},
            {**self.negative,'selected_configuration':dict(updates=3)},
            {**self.positive,'selected_configuration':{}},
            {**self.positive,'development_status':'partial_positive'}):
            with self.assertRaises(ValueError):m.choose_scope(invalid)

    def observations(self):
        ids=[f'gsm8k:train:{i}' for i in range(7179)]
        excluded=['gsm8k:train:0','gsm8k:train:1','gsm8k:train:2','gsm8k:train:3']
        roster=dict(source_ids=[f'gsm8k:train:{i}' for i in range(100,228)],excluded_development_ids=excluded)
        reviews=[dict(source_id=f'gsm8k:train:{i}',all_text_read=True) for i in range(1433)]
        args=(ids,roster,dict(source_ids=['gsm8k:train:1']),[dict(source_id='gsm8k:train:3')],reviews)
        return args

    def test_development_recycling_and_1433_reads_are_counted_from_actual_ledgers(self):
        args=self.observations();observed=m.source_observations(*args)
        self.assertEqual(observed['teacher_unread'],6040)
        self.assertEqual(observed['historical_train_roots_recycled_into_formal'],['gsm8k:train:1'])
        args[-1][0]['all_text_read']=False
        with self.assertRaises(ValueError):m.source_observations(*args)

    def test_transfer_roster_overlap_cannot_be_hidden_by_good_probe_or_public_results(self):
        args=self.observations();args[1]['source_ids'][0]='gsm8k:train:1'
        with self.assertRaises(ValueError):m.source_observations(*args)

    def disclosure_args(self):
        export=dict(complete=True,actual_reviewed=2821,actual_rejected=444,mechanically_usable_unread=8192,
            all_text_read=False,semantic_correctness_verified=False)
        report=dict(accepted=109,rejected=19,existing_exact_reused=31,new_actual_reads=97)
        extension=dict(complete=True,requests=412,new_outputs=142,exact_output_reuses=270,
            automatically_eligible=200,excluded=212,new_automatically_eligible=60,
            all_text_read=False,semantic_correctness_verified=False,new142_covered_by_original_descriptive_sample=False)
        return [self.exposure(),m.source_observations(*self.observations()),
            dict(complete=True,sample_count=128,semantic_accepted=115,semantic_flagged=13),
            export,report,extension,dict(M1=100,NLL=110,M3=109),['M1','NLL','M3']]

    def test_concrete_noise_length_and_new142_scope_disclosures_are_not_semantic_clearance(self):
        args=self.disclosure_args();value=m.disclosure_contents(*args)
        self.assertIn('91',value['automatic_teacher_quality_noise'])
        self.assertIn('8192',value['automatic_explanation_quality_noise'])
        self.assertIn('6040',value['automatic_teacher_quality_noise'])
        self.assertIn('未执行128科学试学',value['finite128_confirmation_scope'])
        self.assertFalse(value['new142_covered_by_original_descriptive_sample'])
        self.assertEqual(value['actual_supervised_tokens_including_EOS']['NLL'],110)
        args[5]['semantic_correctness_verified']=True
        with self.assertRaises(ValueError):m.disclosure_contents(*args)

    def test_false_quality_counts_and_missing_actual_token_budgets_are_rejected(self):
        args=self.disclosure_args();args[3]['mechanically_usable_unread']=0
        with self.assertRaises(ValueError):m.disclosure_contents(*args)
        args=self.disclosure_args();args[6].pop('M3')
        with self.assertRaises(ValueError):m.disclosure_contents(*args)
        args=self.disclosure_args();args[5]['automatically_eligible']=400
        with self.assertRaises(ValueError):m.disclosure_contents(*args)

    def test_missing_actual_grid_blocks_before_disclosures_pretest_or_public_loader(self):
        with patch.object(production,'load_configuration',side_effect=ValueError('six actual grid jobs incomplete')), \
             patch.object(m,'materialize_disclosures') as disclosures,patch.object(selector,'freeze_pretest') as pretest, \
             patch.object(selector_eval,'test_items') as data,patch.object(transfer_eval,'test_items') as transfer_data:
            with self.assertRaises(ValueError):m.run('M1')
            disclosures.assert_not_called();pretest.assert_not_called();data.assert_not_called();transfer_data.assert_not_called()

    def test_scope_cannot_first_freeze_after_current_predictions(self):
        save(self.out/'evaluation'/m.PROTOCOL/'M1/seed42/predictions.jsonl',{'synthetic_presence_only':True})
        cpu=self.out/'cpu.json';save(cpu,dict(status='passed',device='explicit_MLX_CPU',scientific_results=False,inputs={'synthetic':'unused'}))
        with patch.object(m,'OUT',self.out),patch.object(m,'SCOPE',self.out/'scope.json'),patch.object(m,'CPU',cpu), \
             patch.object(m,'verify_inputs',return_value=[]),patch.object(production,'load_configuration',return_value=self.negative), \
             patch.object(selector,'collect_training') as collect:
            with self.assertRaises(ValueError):m.prepare()
            collect.assert_not_called()

    def test_negative_scope_does_not_call_G4_G5_evaluators_or_create_predictions(self):
        workflow=dict(conditions=['M1','NLL','M3'],inputs={'synthetic':'temporary only'})
        with patch.object(m,'OUT',self.out),patch.object(m,'prepare',return_value=workflow),patch.object(m,'verify_inputs',return_value=[]), \
             patch.object(selector_eval,'main') as old,patch.object(transfer_eval,'main') as new:
            result=m.run('M7')
        old.assert_not_called();new.assert_not_called()
        self.assertEqual(result['status'],'unsupported_not_run');self.assertFalse(result['official_predictions_generated'])
        self.assertFalse((self.out/'evaluation').exists())

    def test_final_aggregation_admission_precedes_official_question_loader(self):
        workflow=dict(conditions=['M1','NLL','M3'],inputs={'synthetic':'temporary only'})
        with patch.object(m,'prepare',return_value=workflow),patch.object(m,'verify_inputs',return_value=[]), \
             patch.object(selector,'validate_request',side_effect=ValueError('actual public gate missing')), \
             patch('evaluate.test_items') as data:
            with self.assertRaises(ValueError):m.finalize()
            data.assert_not_called()

    def synthetic_results(self):
        scope=['M1','NLL','M3'];rows={};metadata={}
        for group in scope:
            rows[group]=[dict(source_id=f'gsm8k:test:{i}',question_sha256=digest(f'Synthetic item{i}'),gold=str(i),
                correct=i%2==0,seconds=.1) for i in range(1319)]
            metadata[group]=dict(supervised_tokens_including_eos=100)
        rows['NLL'][1]['correct']=True;rows['NLL'][2]['correct']=False
        return scope,rows,metadata

    def test_full1319_paired_aggregation_keeps_negative_and_unsupported_groups_distinct(self):
        value=m.complete_results(*self.synthetic_results())
        self.assertEqual(value['paired_vs_M1']['NLL']['wrong_to_right'],1)
        self.assertEqual(value['paired_vs_M1']['NLL']['right_to_wrong'],1)
        self.assertEqual(value['paired_vs_M1']['NLL']['net_correct_difference'],0)
        self.assertEqual(value['unsupported_groups'],['M7','M6'])
        self.assertFalse(value['multi_seed_effectiveness_established'])

    def test_partial_duplicate_or_changed_gold_results_cannot_be_final(self):
        scope,rows,metadata=self.synthetic_results();rows['M3'].pop()
        with self.assertRaises(ValueError):m.complete_results(scope,rows,metadata)
        scope,rows,metadata=self.synthetic_results();rows['NLL'][0]['source_id']='gsm8k:test:1'
        with self.assertRaises(ValueError):m.complete_results(scope,rows,metadata)
        scope,rows,metadata=self.synthetic_results();rows['M3'][0]['gold']='wrong'
        with self.assertRaises(ValueError):m.complete_results(scope,rows,metadata)


if __name__=='__main__':unittest.main()
