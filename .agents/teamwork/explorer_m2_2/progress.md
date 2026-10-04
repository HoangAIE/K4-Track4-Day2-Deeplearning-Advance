# Progress — explorer_m2_2

- Last visited: 2026-10-03T15:58:30Z
- Status: Task Complete — Hard Handoff Ready
- Completed Tasks:
  1. Detailed formulation of single-factor ablations on `convnext_tiny`: T05 (label smoothing), T06 (focal loss), T07 (balanced sampler), T08 (EMA decay).
  2. In-depth quantitative analysis of class imbalance: 5,463 Negatives (52.02%) vs rare weeds (5.76%-6.43%), baseline B02 per-class diagnostics (Chinee Apple recall 89.78%, Negatives recall 98.74%).
  3. Mathematical derivation of how each mechanism targets imbalance.
  4. Verified CLI parser in `code/train.py`, discovering critical requirement: `--set label_smoothing=0.1` must be passed explicitly or `LabelSmoothingCE` degenerates to vanilla CrossEntropy.
  5. Tested dry-run configuration parsing and object creation.
  6. Confirmed checkpoint paths (`runs/T0x/seed0/`) and curve naming conventions (`curves/T0x_convnext_tiny.png` + aliases `curves/T0x_<description>.png`).
  7. Estimated runtime per experiment (~16 min on RTX 5060 Ti).
  8. Authored `analysis.md` and 5-component `handoff.md`.
