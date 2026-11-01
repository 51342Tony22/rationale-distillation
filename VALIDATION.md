# Snapshot validation

Date: 2026-11-01.

Result: 59 packaged tests passed; 7 later variant tests passed; existing invalid-escape warning in math_normalize.py.

```bash
PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src:experiments/tgrd_v1:experiments/math_distill_v1 python -m unittest discover -s experiments/tgrd_v1 -p test_variants.py -q
```

Checks used the existing local project dependency environment against this upload copy. Dependencies were not freshly installed from scratch for this packaging check.

All Python source files in the new repository snapshots passed an AST syntax check. The selected upload files, including DOCX XML, were scanned for common credential-token and private-key patterns; no matching credentials were found. Pattern checks are limited and are not a comprehensive security audit. No model training, paid teacher requests, or full benchmark reruns were performed during packaging. Saved metrics describe historical experiments.

The remaining artifact-dependent, model-dependent, and GPU/MLX tests in `experiments/` were not run. Their input datasets, model weights, and historical execution state are deliberately excluded.
