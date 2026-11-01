"""Regression for repeated MLX model construction with dropout recomputation."""
import unittest


class RepeatedModelCheckpointTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        device = mx.default_device()
        mx.set_default_device(mx.cpu)
        self.addCleanup(lambda: mx.set_default_device(device))

    def test_repeated_model_construction_preserves_dropout_gradient_and_random_state(self):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx.utils import tree_flatten
        from learning import enable_grad_checkpoint

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)
                self.dropout = nn.Dropout(.25)

            def __call__(self, x):
                return self.linear(self.dropout(x))

        def result():
            mx.random.seed(42)
            model = Block()
            enable_grad_checkpoint(model)
            mx.eval(model.parameters())
            model.train()
            value, grads = nn.value_and_grad(model, lambda m: mx.sum(
                m(mx.array([[1., 2., 3.], [3., 2., 1.]]))))(model)
            mx.eval(value, grads, list(mx.random.state))
            return (float(value.item()),
                    {k: np.asarray(v).copy() for k, v in tree_flatten(grads)},
                    [np.asarray(v).copy() for v in mx.random.state])

        first = result()
        for _ in range(3):
            later = result()
            self.assertEqual(first[0], later[0])
            self.assertEqual(set(first[1]), set(later[1]))
            for k in first[1]:
                np.testing.assert_array_equal(first[1][k], later[1][k])
            for a, b in zip(first[2], later[2]):
                np.testing.assert_array_equal(a, b)

