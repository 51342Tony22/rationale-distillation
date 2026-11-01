"""CPU-only loss/alignment and LoRA save/load checks on a tiny random model."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import baseline
from pilot_common import C, OUT


def main():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen2 import Model, ModelArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters
    from pilot_train import completion_loss
    mx.set_default_device(mx.cpu)
    args = ModelArgs(model_type="qwen2", hidden_size=32, num_hidden_layers=1,
                     intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                     rms_norm_eps=1e-6, vocab_size=32, max_position_embeddings=128,
                     tie_word_embeddings=True)
    mx.random.seed(7)
    model = Model(args)
    base = Model(args)
    base.update(model.parameters())
    model.freeze()
    cfg = {"fine_tune_type": "lora", "num_layers": 1,
           "lora_parameters": {"rank": 2, "scale": 2.0, "dropout": 0.0, "keys": C["lora_keys"]}}
    linear_to_lora_layers(model, 1, cfg["lora_parameters"])
    model.eval()
    inputs, targets = mx.array([[1, 2, 3, 4, 5]]), mx.array([[4, 5, 6]])
    actual = completion_loss(model, inputs, targets, 3)
    dense = nn.losses.cross_entropy(model(inputs)[:, 2:].astype(mx.float32), targets, reduction="mean")
    error = abs(float(actual.item()) - float(dense.item()))
    assert error < 1e-6
    loss_and_grad = nn.value_and_grad(model, completion_loss)
    value, grad = loss_and_grad(model, inputs, targets, 3)
    assert all("lora_" in key for key, _ in tree_flatten(grad))
    assert any(float(mx.max(mx.abs(g)).item()) > 0 for _, g in tree_flatten(grad))
    optimizer = optim.AdamW(learning_rate=1e-3)
    optimizer.update(model, grad)
    mx.eval(model.parameters(), optimizer.state)
    before_save = model(inputs)
    with tempfile.TemporaryDirectory(prefix="fyp-pilot-cpu-adapter-") as temp:
        folder = Path(temp)
        baseline.save(folder / "adapter_config.json", cfg)
        mx.save_safetensors(str(folder / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
        base = load_adapters(base, temp)
        base.eval()
        delta = float(mx.max(mx.abs(base(inputs) - before_save)).item())
        assert delta == 0
    report = {"status": "passed", "device": str(mx.default_device()), "model": "tiny random CPU Qwen2, not research weights",
              "completion_loss_vs_dense_error": error, "predictor_alignment": "input index P-1 predicts first completion target",
              "targets": 3, "padding_targets": 0, "prompt_targets": 0, "last_target_is_included": True,
              "gradients": "LoRA only; nonzero; optimizer update passed", "adapter_reload_max_logits_difference": delta,
              "training_runner_sha256": baseline.sha(Path(__file__).with_name("pilot_train.py")),
              "preflight_sha256": baseline.sha(Path(__file__))}
    baseline.save(OUT / "cpu_training_preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
