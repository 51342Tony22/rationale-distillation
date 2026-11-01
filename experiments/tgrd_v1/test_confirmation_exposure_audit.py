import copy
import hashlib
import unittest

from confirmation_exposure_audit import duplicate_flags, tokens, validate_rosters


def row(sid, question):
    return {'source_id': 'gsm8k:train:' + str(sid), 'question': question,
            'question_sha256': hashlib.sha256(question.encode()).hexdigest(), 'gold': '3'}


class ExposureAuditTests(unittest.TestCase):
    def test_number_abstraction_keeps_decimal_and_grouped_numbers_whole(self):
        self.assertEqual(tokens('Ａda pays $1,200.50 for 6 pens.', True),
                         ('ada', 'pays', '<number>', 'for', '<number>', 'pens'))

    def test_historical_and_within_cross_cohort_duplicates_are_flagged(self):
        full = [row(i, 'Ada buys 12 green pens and gives 4 pens away today.') for i in range(4)]
        sources, owners = validate_rosters(full, {'gsm8k:train:0'},
                                          {'P60': full[1:3], 'H200': full[3:]})
        flags = duplicate_flags(sources, {'gsm8k:train:0'}, owners)
        self.assertEqual(len(flags), 6)  # 3 historical + 3 candidate pairs; no self-pair.
        self.assertTrue(all(len(set(f['root_ids'])) == 2 for f in flags))
        self.assertTrue(all(f['metrics']['literal']['jaccard'] == 1 for f in flags))

    def test_numeric_template_duplicate_and_unrelated_question(self):
        full = [row(0, 'Ada buys 12 green pens and gives 4 pens away today.'),
                row(1, 'Ada buys 30 green pens and gives 9 pens away today.'),
                row(2, 'A rocket travels between distant planets during the winter months.')]
        sources, owners = validate_rosters(full, {'gsm8k:train:0'}, {'P60': full[1:]})
        flags = duplicate_flags(sources, {'gsm8k:train:0'}, owners)
        self.assertEqual([f['root_ids'] for f in flags], [['gsm8k:train:0', 'gsm8k:train:1']])
        self.assertEqual(flags[0]['metrics']['number_abstracted']['jaccard'], 1)
        self.assertNotIn('literal', flags[0]['metrics'])

    def test_disjointness_and_exact_source_provenance_fail_closed(self):
        full = [row(0, 'Ada has three pens.'), row(1, 'Ben has five books.')]
        with self.assertRaises(ValueError):
            validate_rosters(full, {'gsm8k:train:0'}, {'P60': full[:1]})
        with self.assertRaises(ValueError):
            validate_rosters(full, set(), {'P60': full[:1], 'H200': full[:1]})
        bad = copy.deepcopy(full[1]); bad['gold'] = '777'
        with self.assertRaises(ValueError):
            validate_rosters(full, set(), {'P60': [bad]})
        bad = copy.deepcopy(full); bad[0]['question'] += ' Changed.'
        with self.assertRaises(ValueError):
            validate_rosters(bad, set(), {})


if __name__ == '__main__':
    unittest.main()
