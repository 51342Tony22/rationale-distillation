"""Fresh LoRA initialization, auditable branch reset and completion-only updates."""
from __future__ import annotations

import math
import random

from runtime_full import BASE, load_model, check_time

LORA = {'rank': 16, 'scale': 2.0, 'dropout': 0.05,
        'keys': ['self_attn.q_proj', 'self_attn.v_proj']}


def enable_grad_checkpoint(layer):
    # mlx_lm patches the CLASS, including future model instances. Wrapping the
    # same class again nests recomputation and changes dropout's PRNG sequence.
    # Cold-reload and trial learners must use the same single wrapper.
    from mlx_lm.tuner.trainer import grad_checkpoint
    cls = type(layer)
    if not getattr(cls.__call__, '_tgrd_gradient_checkpoint', False):
        grad_checkpoint(layer)
        cls.__call__._tgrd_gradient_checkpoint = True


def tokenize(tokenizer, question, completion):
    prompt = BASE['student_prompt'].format(question=question)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(target_ids) > 4096:
        raise ValueError('Training would truncate; material must be audited before use')
    return {'prompt_ids': prompt_ids, 'target_ids': target_ids}


def loss(model, inputs, targets, prompt_tokens):
    import mlx.core as mx
    import mlx.nn as nn
    hidden = model.model(inputs)[:, prompt_tokens - 1:]
    logits = model.model.embed_tokens.as_linear(hidden) if model.args.tie_word_embeddings else model.lm_head(hidden)
    assert logits.shape[1] == targets.shape[1]
    return nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction='mean')


class Learner:
    def __init__(self, seed):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from mlx_lm.tuner.utils import linear_to_lora_layers
        self.model, self.tokenizer = load_model('student')
        self.model.freeze()
        mx.random.seed(seed)
        linear_to_lora_layers(self.model, 16, LORA)
        enable_grad_checkpoint(self.model.layers[0])
        self.initial = {name: mx.array(value) for name, value in tree_flatten(self.model.trainable_parameters())}
        mx.eval(self.initial)
        self.parameter_count = sum(x.size for x in self.initial.values())
        assert self.parameter_count == 1245184, self.parameter_count
        self.value_grad = nn.value_and_grad(self.model, loss)
        self.reset(seed)

    def weights(self):
        from mlx.utils import tree_flatten
        return dict(tree_flatten(self.model.trainable_parameters()))

    def reset(self, seed):
        import mlx.core as mx
        import mlx.optimizers as optim
        self.model.load_weights(list(self.initial.items()), strict=False)
        mx.eval(self.model.parameters())
        mx.random.seed(seed)
        self.optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
        self.model.eval()
        self.updates = 0

    def update_batch(self, samples):
        import mlx.core as mx
        from mlx.utils import tree_map
        self.model.train()
        accumulated, values, tokens = None, [], 0
        for sample in samples:
            check_time(reserve=15)
            full = sample['prompt_ids'] + sample['target_ids']
            objective, grads = self.value_grad(self.model, mx.array([full[:-1]]),
                                              mx.array([sample['target_ids']]), len(sample['prompt_ids']))
            grads = tree_map(lambda x: x.astype(mx.float32), grads)
            accumulated = grads if accumulated is None else tree_map(lambda a,b: a+b, accumulated, grads)
            mx.eval(objective, accumulated)
            value = float(objective.item())
            assert math.isfinite(value)
            values.append(value)
            tokens += len(sample['target_ids'])
        grads = tree_map(lambda x: x/len(samples), accumulated)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        self.updates += 1
        self.model.eval()
        mx.clear_cache()
        return {'loss': sum(values)/len(values), 'tokens': tokens, 'updates': self.updates}

    def trial(self, question, completion, updates=3):
        sample = tokenize(self.tokenizer, question, completion)
        return [self.update_batch([sample]) for _ in range(updates)]

    def answer_nll(self, question, gold):
        import mlx.core as mx
        sample = tokenize(self.tokenizer, question, str(gold))
        # The auxiliary score includes the final numeric answer only, not EOS.
        targets = sample['target_ids'][:-1]
        ids = sample['prompt_ids'] + targets
        self.model.eval()
        result = loss(self.model, mx.array([ids[:-1]]), mx.array([targets]), len(sample['prompt_ids']))
        mx.eval(result)
        return float(result.item())

    def save(self, path):
        import mlx.core as mx
        mx.save_safetensors(str(path), self.weights())


def order_for_epoch(count, seed, epoch):
    rng = random.Random(seed)
    result = list(range(count))
    for _ in range(epoch + 1):
        rng.shuffle(result)
    return result
