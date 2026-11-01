import copy
import unittest

from runtime_full import BASE
from whole_span_masking import aligned_positions, whole_units, matched_controls, diagnose_whole_step


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids':[ord(c) for c in text],
                'offset_mapping':[(i,i+1) for i in range(len(text))]}


class IndependentBackend:
    def __init__(self):
        self.calls = []

    def score(self, ids, targets):
        self.calls.append((list(ids),list(targets)))
        # Independent context effects: attended span +3; controls +1,+2,+3.
        value = 2 + sum(delta for token,delta in [(2,3),(6,1),(10,2),(14,3)] if token not in ids)
        return [value] * len(targets)

    def observe(self, ids, targets, queries):
        weights = [0.] * len(ids)
        weights[2] = weights[3] = .35
        for index in (6,7,10,11,14,15):weights[index] = .05
        return self.score(ids,targets), [weights[:] for _ in queries]


class WholeSpanTests(unittest.TestCase):
    def fixture(self):
        ids = list(range(40))
        step = {'step_index':3,'start_position':28,'targets':[30,31]}
        units = [{'kind':'prior_reasoning','span_type':'solution_paragraph',
                  'text':str(i), 'positions':[i,i+1]} for i in (2,6,10,14)]
        return ids,step,units

    def test_boundaries_reject_cut_tokens(self):
        self.assertEqual(aligned_positions([(0,2),(2,4)],1,4),[])
        self.assertEqual(aligned_positions([(0,2),(2,4)],0,3),[])
        self.assertEqual(aligned_positions([(0,2),(2,4)],0,4,10),[10,11])
        self.assertEqual(aligned_positions([(0,1),(0,0),(1,2)],0,2),[])

    def test_sentences_and_paragraphs_include_formatting_and_punctuation(self):
        question = 'Oil costs $1.50. Two wheels need oil. How much?'
        completion = 'First part.\n\nThen calculate:\n\\[2 * 3 = 6\\]\n\nThe answer is \\boxed{6}.'
        prompt = BASE['student_prompt'].format(question=question)
        prepared = whole_units(prompt,question,completion,CharacterTokenizer())
        questions = [u for u in prepared['units'] if u['kind']=='question']
        paragraphs = [u for u in prepared['units'] if u['kind']=='prior_reasoning']
        self.assertEqual(len(questions),3)
        self.assertIn('$1.50.',questions[0]['text'])
        self.assertEqual(len(paragraphs),2)
        self.assertIn('\\[2 * 3 = 6\\]',paragraphs[1]['text'])
        for unit in prepared['units']:
            self.assertEqual(''.join(chr(prepared['input_ids'][i]) for i in unit['positions']),unit['text'])
        self.assertNotIn('boxed',''.join(u['text'] for u in prepared['units']))
        with self.assertRaises(ValueError):whole_units('changed',question,completion,CharacterTokenizer())

    def test_controls_match_whole_span_kind_shape_and_exact_count(self):
        _,_,units = self.fixture();top=units[0]
        extras = [dict(units[1],kind='question'),dict(units[2],span_type='line'),
                  dict(units[3],positions=[20,21,22]),dict(top)]
        a,pool = matched_controls(top,[top,units[1],*extras],seed=7,key='a')
        self.assertEqual(pool,1);self.assertEqual(a,[units[1]])
        self.assertEqual((a,pool),matched_controls(top,[*extras,units[1],top],seed=7,key='a'))
        self.assertEqual(top['positions'],[2,3])

    def test_signed_effect_and_unchanged_original_targets(self):
        ids,step,units=self.fixture();backend=IndependentBackend()
        before=copy.deepcopy((ids,step,units))
        result=diagnose_whole_step(ids,step,units,backend,seed=7,key='a')
        self.assertEqual(result['status'],'whole_span_diagnostic_complete')
        self.assertEqual(result['attended_mask']['delta_nll'],3)
        self.assertEqual(result['random_delta_mean'],2)
        self.assertEqual(result['mask_excess'],1)
        self.assertEqual(sorted(mask['delta_nll'] for mask in result['whole_control_masks']),[1,2,3])
        for forwarded,targets in backend.calls:
            self.assertEqual([forwarded[i] for i in targets],[30,31])
            self.assertLessEqual(max(forwarded),31)
        self.assertEqual((ids,step,units),before)
        self.assertFalse(result['semantic_relevance_matched'])
        self.assertFalse(result['position_distance_matched'])

    def test_missing_controls_never_substitute_token_fragments_or_change_selector(self):
        ids,step,units=self.fixture();backend=IndependentBackend()
        result=diagnose_whole_step(ids,step,units[:2],backend,seed=7,key='a')
        self.assertEqual(result['status'],'insufficient_exact_whole_span_controls')
        self.assertIsNone(result['mask_excess'])
        self.assertEqual(len(result['whole_control_masks']),1)
        self.assertFalse(result['selector_positions_changed'])
        self.assertFalse(result['learning_gain_measured'])
        # A current/future paragraph cannot become attended evidence or control.
        future={'kind':'prior_reasoning','span_type':'solution_paragraph','text':'future','positions':[28,29]}
        result2=diagnose_whole_step(ids,step,units[:2]+[future],IndependentBackend(),seed=7,key='a')
        self.assertEqual(result2['evidence'],result['evidence'])

    def test_attention_instrumentation_mismatch_is_an_error(self):
        ids,step,units=self.fixture();backend=IndependentBackend();original=backend.observe
        def changed(*args):
            nll,weights=original(*args)
            return [x+.01 for x in nll],weights
        backend.observe=changed
        with self.assertRaises(ValueError):diagnose_whole_step(ids,step,units,backend,seed=7,key='a')


if __name__=='__main__':unittest.main()
