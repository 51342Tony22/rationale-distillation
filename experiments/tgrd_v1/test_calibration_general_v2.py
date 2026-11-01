"""Versioned pool and full-grid routing checks; no scientific trial outputs."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import calibration_general_v2 as general
import calibration_summary as summary
import test_calibration_summary as fixtures
from runtime_full import OUT, freeze, read, baseline


class GeneralCalibrationTests(unittest.TestCase):
    def test_namespaces_cannot_resume_old_math_grid_or_use_nll_pool(self):
        context = general.GeneralContext(); original = general.cohort_paths('cal40_reviewed34_v2')
        self.assertNotEqual(context.paths.trials_folder, original.trials_folder)
        self.assertNotEqual(context.paths.pool_folder, original.pool_folder)
        self.assertNotEqual(context.paths.signature_prefix, original.signature_prefix)
        self.assertEqual(context.paths.variants, original.variants)
        self.assertEqual(context.paths.cases, original.cases)
        self.assertEqual(context.paths.expected_roots, 34)

    def pool_fixture(self):
        ids = [f'synthetic:{i}' for i in range(34)]
        sources = {sid: {'question_sha256': 'q'} for sid in ids}
        teachers = {sid: {'output_sha256': 't'} for sid in ids}
        reviews = {sid: {'accepted': True, 'all_text_read': True,
            'question_sha256': 'q', 'output_sha256': 't'} for sid in ids}
        cases = [{'source_id': sid, 'candidates': []} for sid in ids]
        pools = [{'source_id': sid, 'recipe_id': general.RECIPE_ID, 'selector': 'joint',
            'candidates': [{'candidate_id': 'e0'}, {'candidate_id': 'explain',
                'recipe_id': general.RECIPE_ID}]} for sid in ids]
        return pools, cases, sources, teachers, reviews, {sid: {} for sid in ids}

    def test_complete_ordered34_including_fallbacks_and_correct_recipe_required(self):
        args = self.pool_fixture()
        with patch.object(general, 'validate_pool'), patch.object(general, 'validate_probes'):
            self.assertEqual(len(general.validate_pools(*args)), 34)
            for mutate in (
                    lambda a: a[0].pop(),
                    lambda a: a[0].reverse(),
                    lambda a: a[0][0].update(selector='nll_only'),
                    lambda a: a[0][0].update(recipe_id='old_math'),
                    lambda a: a[0][0]['candidates'][1].update(recipe_id='old_math'),
                    lambda a: a[4]['synthetic:0'].update(all_text_read=False)):
                bad = copy.deepcopy(args); mutate(bad)
                with self.assertRaises(ValueError): general.validate_pools(*bad)

    def cost_fixture(self):
        manifest = {'root_ids': [f'synthetic:{i}' for i in range(34)]}
        plan = {'schema': 'tgrd_transfer_cost_plan_v2', 'protocol': general.PROTOCOL,
            'cohort': general.COHORT, 'recipe_id': general.RECIPE_ID,
            'materials_manifest_sha256': 'm', 'grid_seeds': [701,1701],
            'grid_updates': [1,3,5], 'thresholds': list(general.THRESHOLDS),
            'root_ids': manifest['root_ids'], 'frozen_before_any_actual_grid_result': True,
            'public_test_used': False, 'actual_cost_measurement': {
                'actual_student_GPU': True, 'generation_limit': 1024,
                'generations': 6, 'elapsed_seconds': 60}, 'inputs': {'synthetic_only': 'x'}}
        return plan, manifest

    def test_cpu_estimate_or_wrong_materials_cannot_unlock_grid(self):
        plan, manifest = self.cost_fixture(); general.validate_cost_plan(plan, manifest, 'm')
        for changes in ({'recipe_id':'old_math'}, {'materials_manifest_sha256':'changed'},
                        {'grid_seeds':[42]}, {'grid_updates':[1]}, {'inputs':{}},
                        {'root_ids':[]}, {'frozen_before_any_actual_grid_result':False}):
            with self.assertRaises(ValueError):
                general.validate_cost_plan({**plan, **changes}, manifest, 'm')
        for changes in ({'actual_student_GPU':False}, {'generation_limit':32},
                        {'generations':0}, {'generations':5}, {'elapsed_seconds':True},
                        {'elapsed_seconds':float('nan')}):
            bad = copy.deepcopy(plan); bad['actual_cost_measurement'].update(changes)
            with self.assertRaises(ValueError): general.validate_cost_plan(bad, manifest, 'm')

    def test_result_recipe_and_required_job_signature_cannot_be_relabelled(self):
        context = general.GeneralContext(); identity = context.report_identity()
        required = [general.HERE/'calibration_general_v2.py', general.FOLDER/'materials_manifest.json',
            general.COST_PLAN, general.CPU_REPORT, context.paths.pool_folder/'reviewed_pools.json']
        signed = {'extra': {**identity, 'seed':701, 'updates':3,
            'thresholds':list(general.THRESHOLDS), 'variant_labels_in_gradients':False},
            'inputs':{str(p.relative_to(general.ROOT)):'synthetic' for p in required}}
        context.verify_job(identity, signed, 701, 3)
        for key, value in [('recipe_id','old_math'), ('cohort','cal40_reviewed34_v2'),
                           ('variant_labels_in_gradients',True)]:
            bad = copy.deepcopy(signed); bad['extra'][key] = value
            with self.assertRaises(ValueError): context.verify_job(identity, bad, 701, 3)
        bad = copy.deepcopy(signed); bad['inputs'].pop(str(general.COST_PLAN.relative_to(general.ROOT)))
        with self.assertRaises(ValueError): context.verify_job(identity, bad, 701, 3)

    def test_old_cost_or_cpu_gates_cannot_start_real_worker(self):
        # Reject before preparing materials, loading a learner or taking GPU.
        with patch.object(general, 'GeneralContext') as context, patch.object(general, 'gpu') as gpu:
            for updates, seed in ((3,42), (2,701), (None,701), (1,None)):
                with self.assertRaises(ValueError): general.run(updates,seed)
            context.assert_not_called(); gpu.assert_not_called()


class GeneralSummaryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.CalibrationSummaryIntegrationTests()
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.marker = self.f.out/'adopted_recipe_marker.json'
        freeze(self.marker, {'synthetic_fixture_only':True})
        self.context = SimpleNamespace(paths=self.f.paths,
            prepare=lambda: None, evidence_paths=lambda: [self.marker],
            report_identity=lambda: {'recipe_id':general.RECIPE_ID}, verify_job=lambda *args: None)

    def analyze(self):
        with patch.multiple(summary, OUT=self.f.out, DATA=self.f.data,
                validate_probes=lambda bundle,source,rebuild:bundle['variants']), \
             patch('explanation_pool.assemble', side_effect=AssertionError('Must not rebuild old Math7B pools')), \
             patch('material_builder.validate_pool', side_effect=lambda source,teacher,pool,steps:pool['candidates']):
            summary.main('synthetic', context=self.context)

    def test_context_recomputes_all24_cells_keeps_negative_and_signs_new_evidence(self):
        self.analyze()
        report = read(self.f.paths.trials_folder/'development_grid_summary_v1.json')
        self.assertEqual(report['recipe_id'],general.RECIPE_ID)
        self.assertEqual(len(report['configuration_summaries']),24)
        self.assertIsNone(report['selected_configuration'])
        self.assertTrue(all(c['paired_correct_count_gain']==-3 for c in report['configuration_summaries']))
        self.assertEqual(report['inputs'][str(self.marker.relative_to(general.ROOT))],baseline.sha(self.marker))

    def test_context_summary_rejects_changed_selection_before_audit(self):
        path=self.f.paths.trials_folder/'seed701_updates1/synthetic_1/decisions_before_audit.json'
        row=read(path);row['decisions'][0]['selected_id']='e0'
        path.write_text(__import__('json').dumps(row))
        with self.assertRaises(ValueError):self.analyze()


if __name__ == '__main__':
    unittest.main()
