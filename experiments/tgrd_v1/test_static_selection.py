import unittest
from static_selection import rsr, lalp, paragraph_spans, scoring_inputs, select_static


class CharacterTokenizer:
    def encode(self,text,add_special_tokens=False):
        return list(range(1,len(text)+1))
    def __call__(self,text,**kwargs):
        return {'input_ids':self.encode(text), 'offset_mapping':[(i,i+1) for i in range(len(text))]}


class StaticSelectionTests(unittest.TestCase):
    def test_rsr_is_ratio_of_sums_with_rank_clipping_not_mean_token_ratio(self):
        self.assertEqual(rsr([1,1000],[1.,3.]),101/4)
        self.assertNotEqual(rsr([1,3],[1.,9.]),(1/1+3/9)/2)
        with self.assertRaises(ValueError):rsr([0],[1.])

    def test_lalp_gives_equal_step_weight_instead_of_equal_token_weight(self):
        score=lalp([[1.]*9,[3.]])
        self.assertEqual(score,-2.)
        self.assertNotEqual(score,-1.2)

    def test_local_window_keeps_prompt_and_recent_steps_but_no_earlier_steps(self):
        prompt='Q:'; completion='first\n\nsecond\n\nthird'
        prepared=scoring_inputs(CharacterTokenizer(),prompt,completion,window=1)
        self.assertEqual(''.join(completion[a:b] for a,b in paragraph_spans(completion)),completion)
        self.assertEqual(len(prepared['steps']),3)
        self.assertEqual([token for row in prepared['steps'] for token in row['target_ids']],prepared['target_ids'])
        last=prepared['steps'][-1]
        self.assertEqual(last['context_ids'],prepared['context_ids']+prepared['steps'][1]['target_ids'])
        self.assertEqual(last['preceding_step_span'],[1,2])
        no_history=scoring_inputs(CharacterTokenizer(),prompt,completion,window=0)
        self.assertTrue(all(row['context_ids']==prepared['context_ids'] for row in no_history['steps']))

    def test_scoring_direction_and_empty_branch_policy_match_the_main_groups(self):
        rows=[{'candidate_id':'e0','student_tokens':0,'RSR':.01,'LALP':0.},
              {'candidate_id':'a','student_tokens':40,'RSR':2.,'LALP':-1.},
              {'candidate_id':'b','student_tokens':40,'RSR':1.,'LALP':-2.}]
        self.assertEqual(select_static(rows,'RSR'),'b')
        self.assertEqual(select_static(rows,'LALP'),'a')
        self.assertEqual(select_static(rows[:1],'RSR'),'e0')

    def test_empty_and_oversize_sequences_fail_without_truncation(self):
        with self.assertRaises(ValueError):scoring_inputs(CharacterTokenizer(),'Q:','')
        with self.assertRaises(ValueError):scoring_inputs(CharacterTokenizer(),'Q:','x'*4096)


if __name__=='__main__':unittest.main()
