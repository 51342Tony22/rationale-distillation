import unittest
from trial_selection import choose, summarize


class TrialSelectionTests(unittest.TestCase):
    def test_both_gains_must_be_strictly_positive_and_abstention_is_allowed(self):
        candidates = [{'candidate_id':'e0','student_tokens':0}, {'candidate_id':'a','student_tokens':40}]
        base = {'accuracy':.5,'answer_nll':1.}
        for score, threshold in [({'accuracy':.5,'answer_nll':.7},0),
                                 ({'accuracy':.7,'answer_nll':1.},0),
                                 ({'accuracy':.7,'answer_nll':.9375},.1),
                                 ({'accuracy':.7,'answer_nll':.75},.1)]:
            decision = choose({'e0':base,'a':score},candidates,threshold)
            self.assertEqual(decision['selected_id'],'a' if score['answer_nll']==.75 else 'e0')
        self.assertEqual(choose({'e0':base},candidates[:1],0)['selected_id'],'e0')

    def test_accuracy_precedes_nll_then_length_then_fixed_id(self):
        candidates = [{'candidate_id':'e0','student_tokens':0}, *[
            {'candidate_id':cid,'student_tokens':length} for cid,length in [('a',40),('b',32),('c',32),('d',32)]]]
        scores = {'e0':{'accuracy':0,'answer_nll':2.},
                  'a':{'accuracy':.5,'answer_nll':1.}, 'b':{'accuracy':.5,'answer_nll':1.},
                  'c':{'accuracy':.5,'answer_nll':1.}, 'd':{'accuracy':1/6,'answer_nll':.1}}
        self.assertEqual(choose(scores,candidates,0)['selected_id'],'b')

    def test_probe_order_completeness_and_finite_nll_are_required(self):
        ids = [str(i) for i in range(6)]
        rows = [{'id':str(i),'correct':i<3,'answer_nll':1.} for i in range(6)]
        self.assertEqual(summarize(rows,ids),{'accuracy':.5,'answer_nll':1.})
        for bad in [rows[:5],list(reversed(rows)),[{**rows[0],'answer_nll':float('nan')},*rows[1:]]]:
            with self.assertRaises(ValueError):
                summarize(bad,ids)
        with self.assertRaises(ValueError):
            choose({'e0':{'accuracy':0,'answer_nll':1.}},[{'candidate_id':'e0'}],.02)


if __name__ == '__main__':
    unittest.main()
