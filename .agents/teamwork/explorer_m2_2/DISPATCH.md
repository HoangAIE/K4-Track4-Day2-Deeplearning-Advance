# Milestone 2 Task: M2 Ablation Design for Axis C (Loss) and Axis D/F (Sampling & Regularization)

Read:
- `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\ORIGINAL_REQUEST.md`
- `d:\K4-Track4-Day2-Deeplearning-Advance\PROJECT.md`
- `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\worker_m1\handoff.md`

Your tasks:
1. Formulation of experiment configurations on `convnext_tiny`:
   - Axis C (Loss function):
     - T05: `loss="ls"` with `label_smoothing=0.1` (`--set backbone=convnext_tiny loss=ls label_smoothing=0.1`)
     - T06: `loss="focal"` with $\gamma=2.0$ (`--set backbone=convnext_tiny loss=focal`)
   - Axis D (Sampling):
     - T07: `sampler="balanced"` (`--set backbone=convnext_tiny sampler=balanced`)
   - Axis F (Regularization):
     - T08: `ema_decay=0.999` (`--set backbone=convnext_tiny ema_decay=0.999`)
2. Verify exact CLI flags and parameters in `code/train.py`.
3. Check how these losses specifically target the dominant Negative class and rare weed classes.
4. Document detailed execution plan in `analysis.md` and complete with `handoff.md`.
Ensure you update your `progress.md`. Do NOT edit source code.

## 2026-10-03T15:52:43Z
You are Explorer 2 for Milestone 2 (M2 Ablation Design for Axis C and Axis D/F) in the DeepWeeds Deep Learning Day 2 Lab project.
Your working directory is: `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\explorer_m2_2\`
Authoritative request: `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\ORIGINAL_REQUEST.md`
Project blueprint: `d:\K4-Track4-Day2-Deeplearning-Advance\PROJECT.md`
Read `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\worker_m1\handoff.md`.

Your tasks:
1. Examine single-factor ablations on `convnext_tiny`:
   - Axis C (Loss function):
     - `loss="ls"` with `label_smoothing=0.1` (CRITICAL: must explicitly set `label_smoothing=0.1`)
     - `loss="focal"` with $\gamma=2.0$
   - Axis D (Sampling):
     - `sampler="balanced"`
   - Axis F (Regularization):
     - `ema_decay=0.999`
2. Analyze how each mechanism targets class imbalance (Negative class dominance vs rare weeds).
3. Confirm CLI arguments in `code/train.py`, runtime expectations, checkpoint paths (`runs/T0x/seed0/`), and curve names (`curves/T0x_<description>.png`).
4. Write your detailed strategy to `.agents/teamwork/explorer_m2_2/analysis.md` and complete with `handoff.md`.
5. Send a message to parent when done.
Update `progress.md` during execution. Do NOT modify source code. Maintain zero test set leakage.
