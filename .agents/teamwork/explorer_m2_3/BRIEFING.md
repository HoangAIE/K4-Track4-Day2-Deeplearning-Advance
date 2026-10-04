# BRIEFING — 2026-10-03T15:59:00Z

## Mission
Design combined recipe experiment (T_combo) across axes A, B, C, D, F avoiding negative interference, and formulate schema and update logic for results.xlsx sheet Training.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigation, recipe synthesis, excel schema & delta update design
- Working directory: d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\explorer_m2_3
- Original parent: c4718f41-3030-43ce-8fcc-465340c45738
- Milestone: Milestone 2 (M2 Combined Recipe Design & Excel Training Sheet Schema)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement or modify source code
- Zero test set leakage (only validation set metrics for M2 decisions)
- Exact Excel columns required: exp_id, backbone, trục thay đổi (A–G), khác T00 ở điểm nào, seed, macro-F1 val, top-1 val, Δ so với T00, ghi chú
- Baseline T00 at top (convnext_tiny, Δ = 0.0000)
- Delta formula: Δ = Macro-F1_exp - Macro-F1_T00

## Current Parent
- Conversation ID: c4718f41-3030-43ce-8fcc-465340c45738
- Updated: not yet

## Investigation State
- **Explored paths**: `code/train.py`, `code/model.py`, `code/dataset.py`, `code/losses.py`, `code/update_excel.py`, `results.xlsx`, `worker_m1/handoff.md`, `runs/B02/seed0/summary.json`, `ORIGINAL_REQUEST.md`, `PROJECT.md`, `GUIDE.md`, `RUBRIC.md`.
- **Key findings**:
  1. Identified interference risks: CutMix + Label Smoothing causes compound over-smoothing / target dilution on fine-grained weeds; Balanced Sampler + Focal Loss causes double-penalty starvation on Negative background diversity while over-fitting minority weed samples.
  2. Designed primary recipe `T09` (`convnext_tiny` + `finetune` + `aug="color"` + `loss="focal"` + `ema_decay=0.999`) avoiding negative interference and providing orthogonal improvements.
  3. Designed alternative recipe `T10` (`mix="cutmix"` + `loss="focal"` + `ema_decay=0.999`).
  4. Identified bug in `code/train.py`: `ema.restore(model)` is called before saving the best checkpoint, saving standard non-shadow weights.
  5. Established exact schema and openpyxl implementation for `Training` sheet in `results.xlsx` with T00 baseline in row 2, format `+0.0000;-0.0000;0.0000`, and statistical significance test against $\sigma_{\text{seed}} \approx 0.0030$.
- **Unexplored areas**: None within M2 scope.

## Key Decisions Made
- Anchored T00 to row 2 of `Training` sheet inheriting directly from B02 (`convnext_tiny`, Macro-F1 0.9604, Top-1 0.9697, Delta 0.0000).
- Formulated `T09` (`aug=color`, `loss=focal`, `ema_decay=0.999`) as the primary candidate combination recipe for Milestone 2.
- Provided drop-in Python update code for `code/update_excel.py` including `upsert_training_row` and `populate_training_from_runs`.

## Artifact Index
- analysis.md — Comprehensive analysis of T_combo design & results.xlsx Training sheet schema
- handoff.md — 5-component hard handoff report
- progress.md — Liveness heartbeat
- DISPATCH.md — Initial dispatch log
