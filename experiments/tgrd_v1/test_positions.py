import unittest
from positions_v2 import genuine_transition,usable_evidence,propose


class PositionsTests(unittest.TestCase):
    def test_assignments_and_headers_do_not_become_transitions(self):
        self.assertFalse(genuine_transition('- Oranges = 70'))
        self.assertFalse(genuine_transition('Step1: find the total'))
        self.assertTrue(genuine_transition(r'\[4 \times20=80\]'))
        self.assertTrue(genuine_transition(r'x-\frac14x=24'))
        self.assertFalse(usable_evidence('Step 2:','prior_reasoning'))
        self.assertFalse(usable_evidence('Find the number of buckets:','prior_reasoning'))
        self.assertTrue(usable_evidence('The remaining money is24.','prior_reasoning'))

    def test_proposal_keeps_signed_excess_and_reports_invalid(self):
        rows=[{'step_index':0,'mean_nll':2,'mask_excess':-1,'status':'diagnostic_complete'},
              {'step_index':1,'mean_nll':1,'mask_excess':2,'status':'diagnostic_complete'},
              {'step_index':2,'mean_nll':9,'mask_excess':None,'status':'insufficient_matched_controls'}]
        result=propose(rows)
        self.assertEqual(result['positions'],[0,1])
        self.assertEqual(result['excluded_for_missing_controls'],[2])
        self.assertEqual(propose([])['positions'],[])


if __name__=='__main__':unittest.main()
