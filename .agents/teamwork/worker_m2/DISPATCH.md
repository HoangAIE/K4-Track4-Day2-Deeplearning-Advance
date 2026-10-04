## 2026-10-03T16:00:04Z
You are Worker 2 for Milestone 2 (Step 2 — Training Recipe Ablation T01..T09) in the DeepWeeds Deep Learning Day 2 Lab project.
Your working directory is: `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\worker_m2\`
Authoritative request: `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\ORIGINAL_REQUEST.md`
Project blueprint: `d:\K4-Track4-Day2-Deeplearning-Advance\PROJECT.md`
Read the Explorer handoff reports before starting:
- `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\explorer_m2_1\handoff.md`
- `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\explorer_m2_2\handoff.md`
- `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\explorer_m2_3\handoff.md`

Your tasks:
1. Ensure Windows environment compatibility: `$env:PYTHONUTF8=1` and `num_workers=0`.
2. Baseline T00 is already established in B02 on `convnext_tiny` (Val Macro-F1 = 96.04%, Val Top-1 = 96.97%).
3. Execute the single-factor ablation experiments on `convnext_tiny`:
   - Axis A (Initialization):
     - T01: `init="scratch"`: `python code/train.py --exp_id T01 --seed 0 --fold 0 --set backbone=convnext_tiny init=scratch num_workers=0`
     - T02: `init="frozen"`: `python code/train.py --exp_id T02 --seed 0 --fold 0 --set backbone=convnext_tiny init=frozen num_workers=0`
   - Axis B (Augmentation):
     - T03: `aug="trivial"`: `python code/train.py --exp_id T03 --seed 0 --fold 0 --set backbone=convnext_tiny aug=trivial num_workers=0`
     - T04: `mix="cutmix"`: `python code/train.py --exp_id T04 --seed 0 --fold 0 --set backbone=convnext_tiny mix=cutmix num_workers=0`
   - Axis C (Loss function):
     - T05: `loss="ls"` with `label_smoothing=0.1`: `python code/train.py --exp_id T05 --seed 0 --fold 0 --set backbone=convnext_tiny loss=ls label_smoothing=0.1 num_workers=0`
     - T06: `loss="focal"` with $\gamma=2.0$: `python code/train.py --exp_id T06 --seed 0 --fold 0 --set backbone=convnext_tiny loss=focal focal_gamma=2.0 num_workers=0`
   - Axis D / F (Sampling / EMA):
     - T07: `sampler="balanced"`: `python code/train.py --exp_id T07 --seed 0 --fold 0 --set backbone=convnext_tiny sampler=balanced num_workers=0`
     - T08: `ema_decay=0.999`: `python code/train.py --exp_id T08 --seed 0 --fold 0 --set backbone=convnext_tiny ema_decay=0.999 num_workers=0`
4. Execute Combined Recipe experiment (T09):
   - Combine the best compatible factors based on empirical results (e.g., `loss=focal focal_gamma=2.0 aug=color ema_decay=0.999` or as justified in Explorer 3's handoff).
5. Verify run artifacts in `runs/T0x/seed0/` (`best_checkpoint.pt`, `config.json`, `summary.json`, `history.csv`, `val_logits.npy`) and dual-axis training curves in `curves/`. Ensure curves have filenames matching `curves/T0x_*.png`.
6. Update `results.xlsx` sheet `Training`:
   - 9 columns: `exp_id`, `backbone`, `trục thay đổi (A–G)`, `khác T00 ở điểm nào`, `seed`, `macro-F1 val`, `top-1 val`, `Δ so với T00`, `ghi chú`.
   - Include baseline T00 row at the top with $\Delta = 0.0000$.
   - Calculate $\Delta = \text{Macro-F1}_{\text{exp}} - \text{Macro-F1}_{\text{T00}}$ for all runs.
   - Format cleanly using `openpyxl`.
7. Conduct scientific analysis comparing each $\Delta$ to estimated seed noise (~0.1-0.3 points) and answer core questions:
   - Does scratch init work? Is frozen backbone good enough?
   - Does augmentation / CutMix help generalization or hurt convergence?
   - Which loss handles the dominant Negative class best & improves rare weed F1?
   - Do combined effects compound linearly or exhibit diminishing/destructive interference?
   - Identify the Best Recipe.
8. Document all commands, runtime logs, exact metrics, and analysis in `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\worker_m2\handoff.md`.
9. Send message to parent when done.
