"""Actual-model tests for training updates, exact branch reset and NLL alignment."""
from __future__ import annotations

from pathlib import Path
from runtime_full import *
from learning import Learner, tokenize, loss


def main():
    import mlx.core as mx
    import mlx.nn as nn
    with gpu():
        signature('learning_preflight',[Path(__file__),HERE/'learning.py',HERE/'runtime.py',HERE/'runtime_full.py',
                  OUT/'full_run_authorization.json'],{'purpose':'Engineering only; synthetic math; no public test'})
        baseline.verify_student(BASE)
        learner=Learner(701)
        question='A box contains 3 red pencils and 4 blue pencils. How many pencils are in the box?'
        completion='There are 3 + 4 = 7 pencils in total.\nThe answer is 7.'
        sample=tokenize(learner.tokenizer,question,completion)
        full=sample['prompt_ids']+sample['target_ids']
        p=len(sample['prompt_ids'])
        initial={k:mx.array(v) for k,v in learner.weights().items()}
        mx.eval(initial)
        def evaluate():
            learner.model.eval()
            value=loss(learner.model,mx.array([full[:-1]]),mx.array([sample['target_ids']]),p)
            mx.eval(value)
            return float(value.item())
        before=evaluate()
        # Independently index the full-sequence CE to verify the prompt boundary.
        learner.model.eval()
        hidden=learner.model.model(mx.array([full[:-1]]))
        logits=learner.model.model.embed_tokens.as_linear(hidden).astype(mx.float32)
        all_ce=nn.losses.cross_entropy(logits,mx.array([full[1:]]),reduction='none')
        independent=float(mx.mean(all_ce[:,p-1:]).item())
        assert abs(before-independent)<1e-5,(before,independent)
        first=learner.update_batch([sample])
        after=evaluate()
        changed=max(float(mx.max(mx.abs(initial[k]-v)).item()) for k,v in learner.weights().items())
        assert changed>0 and learner.updates==1
        learner.reset(701)
        reset_diff=max(float(mx.max(mx.abs(initial[k]-v)).item()) for k,v in learner.weights().items())
        assert reset_diff==0 and learner.updates==0
        restored=evaluate()
        assert abs(restored-before)<1e-6
        repeated=learner.update_batch([sample])
        repeat_after=evaluate()
        assert abs(first['loss']-repeated['loss'])<1e-6
        assert abs(after-repeat_after)<1e-6
        save(OUT/'preflight/learning.json',{'status':'passed','at_utc':now(),
            'trainable_parameters':learner.parameter_count,'completion_loss':before,
            'independent_full_CE_masked_loss':independent,'after_one_update_loss':after,
            'changed_weight_max_abs':changed,'reset_weight_max_abs':reset_diff,
            'restored_loss':restored,'same_seed_repeated_loss':repeat_after,
            'optimizer_reset_verified':True,'official_test_accessed':False,
            'checks':['completion_alignment','actual_parameter_update','exact_branch_reset',
                      'new_optimizer_after_reset','same_seed_repeatability','evaluation_dropout_disabled']})
        print(json.dumps(read(OUT/'preflight/learning.json'),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
