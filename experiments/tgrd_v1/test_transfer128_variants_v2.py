"""Grouped numeric references retain raw provenance and all semantic guards."""
import copy
import unittest
from runtime_full import OUT,read,digest
from transfer128_variants_v2 import normalized_source,validate_entry


class NumericReferenceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.sources={r['source_id']:r for r in read(OUT/'transfer128/preparation_v1/root_preparation_v1.json')}
        self.entries=read(OUT/'transfer128/variants_v1/authored_batches/batch02_actual12.json')
        self.entry=next(e for e in self.entries if e['source_id']=='gsm8k:train:1504')
        self.source=self.sources[self.entry['source_id']]

    def test_grouped_reference_replays_all_probes_and_observed_only_rationales(self):
        original=copy.deepcopy(self.source)
        result=validate_entry(self.entry,self.source)
        self.assertEqual(self.source,original)
        binding=result['numeric_reference_binding']
        self.assertEqual(binding['raw_gold'],'28,800')
        self.assertEqual(binding['raw_gold_sha256'],digest('28,800'))
        self.assertEqual(binding['comparison_gold'],'28800')
        self.assertEqual(len(result['bundle']['variants']['select']),6)
        self.assertEqual(len(result['bundle']['variants']['audit']),6)
        self.assertEqual(len(result['Vsel_rationales']['selection_rationales']),6)

    def test_malformed_grouping_and_reference_disagreement_cannot_be_repaired(self):
        for raw in ['28,80','2,88,000','28,800,','28,800 dollars',' 28,800']:
            with self.assertRaisesRegex(ValueError,'Malformed'):
                normalized_source({**self.source,'gold':raw})
        with self.assertRaisesRegex(ValueError,'reference answer'):
            validate_entry(self.entry,{**self.source,'gold':'28,801'})

    def test_normalization_does_not_supply_mathematical_acceptance_or_hidden_givens(self):
        for change in ({'original_question_text_read':False},{'all_six_templates_text_read':False}):
            with self.assertRaises(ValueError):validate_entry({**self.entry,**change},self.source)
        entry=copy.deepcopy(self.entry);entry['spec']['review']['accepted']=False
        with self.assertRaisesRegex(ValueError,'semantic review'):validate_entry(entry,self.source)
        entry=copy.deepcopy(self.entry);entry['spec']['templates']['structural'][0]['answer']='c'
        with self.assertRaisesRegex(ValueError,'hidden generation parameter'):
            validate_entry(entry,self.source)

    def test_plain_labels_and_historical_batch_remain_exactly_replayable(self):
        old=read(OUT/'transfer128/variants_v1/authored_batches/batch01_actual12.json')
        for entry in old:
            result=validate_entry(entry,self.sources[entry['source_id']])
            self.assertEqual(result['qualified'],entry['accepted'])
        for entry in self.entries:
            result=validate_entry(entry,self.sources[entry['source_id']])
            self.assertTrue(result['qualified'])
            normalized,binding=normalized_source(self.sources[entry['source_id']])
            self.assertEqual(binding['question_sha256'],digest(normalized['question']))


if __name__=='__main__':unittest.main()
