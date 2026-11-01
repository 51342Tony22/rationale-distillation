"""Final finite ledger cannot hide missing roots, repaired sources or bad probes."""
import copy
import unittest
from runtime_full import OUT,read
from transfer128_variant_manifest_v2 import FOLDER,replay_batches


class CompleteVariantLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports=[read(p) for p in sorted((FOLDER/'batches').glob('*.json'))]
        cls.sources={r['source_id']:r for r in read(OUT/'transfer128/preparation_v1/root_preparation_v1.json')}
        cls.ids=list(cls.sources)

    def test_complete_actual_ledger_replays_all_sources_both_splits_and_rationales(self):
        entries,results=replay_batches(self.reports,self.sources,self.ids)
        self.assertEqual(len(entries),128)
        self.assertEqual(sum(r['qualified'] for r in results),124)
        self.assertEqual({e['source_id'] for e in entries if not e['accepted']},
            {f'gsm8k:train:{i}' for i in [509,3822,6327,6831]})

    def test_missing_or_duplicate_root_cannot_be_called_complete(self):
        with self.assertRaisesRegex(ValueError,'Complete original128'):
            replay_batches(self.reports[:-1],self.sources,self.ids)
        with self.assertRaisesRegex(ValueError,'exactly one'):
            replay_batches(self.reports+[self.reports[0]],self.sources,self.ids)

    def test_changed_frozen_probe_or_observed_rationale_cannot_export(self):
        reports=copy.deepcopy(self.reports)
        r=next(r for report in reports for r in report['validated_results'] if r['qualified'])
        r['bundle']['variants']['audit'][0]['gold']='999999'
        with self.assertRaisesRegex(ValueError,'changed'):
            replay_batches(reports,self.sources,self.ids)
        reports=copy.deepcopy(self.reports)
        r=next(r for report in reports for r in report['validated_results'] if r['qualified'])
        r['Vsel_rationales']['selection_rationales'][0]['completion']='hidden answer'
        with self.assertRaisesRegex(ValueError,'changed'):
            replay_batches(reports,self.sources,self.ids)

    def test_rejected_source_cannot_be_silently_repaired_or_given_a_bundle(self):
        sources=copy.deepcopy(self.sources)
        sources['gsm8k:train:6831']['question']=sources['gsm8k:train:6831']['question'].replace('less than the third','less than the second')
        with self.assertRaisesRegex(ValueError,'exact-root'):
            replay_batches(self.reports,sources,self.ids)
        reports=copy.deepcopy(self.reports)
        rejected=next(r for report in reports for r in report['validated_results'] if not r['qualified'])
        rejected['bundle']={'manufactured':True}
        with self.assertRaisesRegex(ValueError,'changed'):
            replay_batches(reports,self.sources,self.ids)


if __name__=='__main__':unittest.main()
