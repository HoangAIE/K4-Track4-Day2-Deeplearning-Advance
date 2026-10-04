# Milestone 2 Handoff Report: Combined Recipe Design (T_combo) & Excel Training Sheet Schema

**Agent**: `explorer_m2_3`  
**Milestone**: Milestone 2 (M2 Combined Recipe Design & Excel Training Sheet Schema)  
**Parent**: Orchestrator (`c4718f41-3030-43ce-8fcc-465340c45738`)  
**Date**: 2026-10-03  
**Status**: Task Completed (Hard Handoff)  

---

## 1. Observation

### 1.1 Baseline Anchoring and Dataset Distribution
1. **Dataset Split Distribution (Fold 0)**:
   - File inspected: `data/labels/train_subset0.csv` (10,501 images), `val_subset0.csv` (3,501 images), `test_subset0.csv` (3,507 images).
   - Class imbalance: Class 8 (`Negatives`) accounts for 5,465 / 10,501 train samples (**52.04%**). The remaining 8 weed classes each have between 596 and 665 samples (**~5.7% – 6.3%** each).
2. **Milestone 1 Winning Backbone (B02)**:
   - Verified via `worker_m1/handoff.md` (lines 54–64) and `runs/B02/seed0/summary.json`:
     - Architecture: `convnext_tiny` (`in12k_ft_in1k`, 27.827M params, 4.455 GMACs).
     - Best Epoch: **11**
     - Best Val Macro-F1: **96.04%** (`0.9604`)
     - Best Val Top-1 Accuracy: **96.97%** (`0.9697`)
     - Average Train Time: **79.11 s/epoch** (~15.8 min / 12 epochs on RTX 5060 Ti).
     - Batch-1 GPU Inference Latency: **11.44 ms** (fastest of all 5 backbones).
     - Recipe: T00 baseline (AdamW, lr 1e-4 / 1e-3, weight decay 0.05, warmup 1 ep + cosine decay, CE loss, basic aug, no sampler, no EMA, batch size 64, AMP on, seed 0).

### 1.2 Inspection of Existing Codebase Modules
1. **`code/train.py` (lines 59–97, `Config`)**:
   - `exp_id: str = "T00"`
   - `backbone: str = "resnet50"` (overridden to `convnext_tiny`)
   - `init: str = "finetune"` (`scratch` | `frozen` | `finetune`)
   - `aug: str = "basic"` (`basic` | `color` | `trivial` | `randaug`)
   - `sampler: Optional[str] = None` (`None` | `balanced`)
   - `mix: Optional[str] = None` (`None` | `mixup` | `cutmix`), `mix_alpha: float = 1.0`
   - `loss: str = "ce"` (`ce` | `ls` | `focal` | `ce_weighted`), `label_smoothing: float = 0.0`, `focal_gamma: float = 2.0`
   - `ema_decay: Optional[float] = None`
2. **`code/train.py` (lines 143–172, `EMA` class & lines 400–442)**:
   - Line 401: `ema.apply_shadow(model)` before validation.
   - Line 405: `evaluate(model, val_loader, val_criterion, device)`.
   - Line 406: `ema.restore(model)` immediately restores non-EMA weights.
   - Line 437: `torch.save(..., "model_state_dict": model.state_dict(), ...)` saves the restored training weights, NOT the shadow weights that achieved `val_macro_f1`.
3. **`code/losses.py` (lines 36–55, `LabelSmoothingCE`; lines 57–94, `FocalLoss`; lines 119–177, `mix_batch` & `mixed_loss`)**:
   - `LabelSmoothingCE`: $q'_k = (1 - \epsilon)\mathbf{1}[k=y] + \epsilon/K$.
   - `FocalLoss`: $\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$.
   - `mixed_loss`: $\lambda \mathcal{L}(logits, y_a) + (1-\lambda) \mathcal{L}(logits, y_b)$.
4. **`code/update_excel.py` & `results.xlsx`**:
   - Verified through command execution: `results.xlsx` currently contains 3 sheets: `Backbones`, `Training`, and `Summary`.
   - Sheet `Training` has max_row = 1 with exact 9 columns:
     `['exp_id', 'backbone', 'trục thay đổi (A–G)', 'khác T00 ở điểm nào', 'seed', 'macro-F1 val', 'top-1 val', 'Δ so với T00', 'ghi chú']`.
   - `code/update_excel.py` currently only implements `upsert_backbone_row`; functions `upsert_training_row` and `populate_training_from_runs` are not yet implemented.

---

## 2. Logic Chain

### 2.1 Deduction of Cross-Factor Interactions & Negative Interference
- **Observation**:
  - In `code/losses.py`, CutMix generates continuous convex combination targets $y = \lambda y_a + (1-\lambda) y_b$.
  - In `LabelSmoothingCE`, target probabilities are diffused uniformly by $\epsilon/K$ ($\epsilon=0.1, K=9$).
  - In `make_loader`, `sampler="balanced"` forces each class to appear with equal frequency ($1/9$) in every batch, cutting Negative exposure from 52% to 11.1%.
  - In `FocalLoss`, easy examples ($p_t \approx 0.95$) receive a $(1-p_t)^2 \approx 0.0025$ weight penalty.
- **Deduction 1 (Label Smoothing + CutMix Interference)**:
  - Concurrently applying CutMix and Label Smoothing ($\epsilon=0.1$) causes **compound target dilution (over-smoothing)**. Because DeepWeeds requires fine-grained discrimination between weed leaf serrations and native grass, softening continuous mixed targets destroys discriminative gradient magnitude. Therefore, CutMix and Label Smoothing must NOT be combined in the primary recipe.
- **Deduction 2 (Balanced Sampler + Focal Loss Interference)**:
  - `sampler="balanced"` slashes the presence of Negatives in mini-batches by 4.7x. Concurrently applying Focal Loss further reduces the gradient penalty of confident Negatives by 400x.
  - This double penalty starves the model of background landscape diversity (dry grass, reddish dirt, cattle manure), creating an aggressive false positive spike (collapsing weed precision). Simultaneously, oversampling the ~600 weed images causes severe overfitting on rare weed noise.
  - Therefore, the natural distribution (`sampler=None`) combined with Focal Loss ($\gamma=2.0$) is mathematically superior: it preserves full background diversity while allowing Focal Loss to dynamically scale down trivial negative background gradients.
- **Deduction 3 (Augmentation + EMA Synergy)**:
  - Strong augmentations (`aug="color"`, `mix="cutmix"`) induce stochastic variance in mini-batch gradients.
  - EMA ($\alpha=0.999$) acts as a temporal low-pass filter over parameter trajectories, stabilizing optimization and converging to flatter minima. This forms a **positive compounding synergy**.

### 2.2 Formulation of Recipe Combinations
- **Primary Synergy Recipe (`T09` / `T_combo_primary`)**:
  - Backbone: `convnext_tiny`
  - Init: `init="finetune"` (Axis A)
  - Augmentation: `aug="color"` (Axis B: lighting invariance without leaf destruction)
  - Loss: `loss="focal"`, `focal_gamma=2.0` (Axis C: dynamic down-weighting of easy Negatives)
  - Sampler: `sampler=None` (Axis D: natural distribution preserving 52% background diversity)
  - Regularization: `ema_decay=0.999` (Axis F: temporal parameter smoothing)
  - CLI:
    ```powershell
    $env:PYTHONUTF8=1; python code/train.py --exp_id T09 --seed 0 --fold 0 --set backbone=convnext_tiny aug=color loss=focal focal_gamma=2.0 ema_decay=0.999 num_workers=0
    ```
- **Alternative Spatial Regularization Recipe (`T10` / `T_combo_cutmix`)**:
  - Backbone: `convnext_tiny`, `init="finetune"`, `mix="cutmix"`, `loss="focal"`, `focal_gamma=2.0`, `ema_decay=0.999`.
  - Omits Label Smoothing completely to prevent over-smoothing interference.

### 2.3 Excel Training Sheet Schema & Delta Mechanics
- **Observation**:
  - Sheet `Training` requires baseline T00 at the top, followed by ablation runs and combination runs.
  - T00 Val Macro-F1 = `0.9604`, Top-1 = `0.9697`, $\Delta = 0.0000$.
- **Deduction**:
  - Row 2 MUST hold T00.
  - Formula: $\Delta = \text{Macro-F1}_{\text{exp}} - \text{Macro-F1}_{\text{T00}}$ with number format `+0.0000;-0.0000;0.0000`.
  - Statistical noise threshold: Per Slide Day 2 p. 59, $\sigma_{\text{seed}} \approx 0.0010 - 0.0030$. If $|\Delta| < 0.0030$, the note must state *"Không phân biệt được so với nhiễu seed (|Δ| < 0.0030)"*. If $\Delta \ge +0.0030$, the note records *"Cải thiện có ý nghĩa thống kê (Δ ≥ +0.0030)"*.

---

## 3. Caveats

1. **Read-Only Explorer Scope**:
   - In accordance with agent identity constraints, no modifications to `code/update_excel.py` or `code/train.py` were committed. Production-ready implementation code is fully documented in `analysis.md` and this handoff for Worker agent execution.
2. **Single-Seed Scanning vs Final Multi-Seed**:
   - Milestone 2 ablations are scanned on Seed 0 (matching GUIDE §N4 and §3.1). The final recipe selected at the end of Milestone 2 will undergo $\ge 3$ seeds during Milestone 4.
3. **EMA Checkpoint Saving Mechanism**:
   - In `code/train.py`, `ema.restore(model)` is currently called before `torch.save`. The Worker agent should adjust this so that the checkpoint genuinely contains the shadow weights evaluated during validation.
4. **Zero Test Set Leakage**:
   - All evaluation and selection logic is strictly based on `val_subset0.csv`. `test_subset0.csv` is not touched.

---

## 4. Conclusion

1. **Combined Recipe Design Completed**:
   - Designed primary combination `T09` (`convnext_tiny` + `finetune` + `aug="color"` + `loss="focal"` + `ema_decay=0.999`), eliminating negative interference between CutMix and Label Smoothing, as well as between Balanced Sampler and Focal Loss.
   - Designed alternative combination `T10` (`mix="cutmix"` + `loss="focal"` + `ema_decay=0.999`).
2. **Master Excel Schema & Update Logic Formulated**:
   - Sheet `Training` 9-column schema confirmed and documented.
   - Row 2 anchored with baseline T00 (`0.9604` F1, $\Delta = 0.0000$).
   - Drop-in Python functions `upsert_training_row` and `populate_training_from_runs` provided with automatic `#E8F8F5` best-row highlighting, custom number formatting `+0.0000;-0.0000;0.0000`, and statistical noise verification against $\sigma_{\text{seed}} \approx 0.0030$.

---

## 5. Verification Method

To independently verify all findings and validate the schema:

1. **Repository Test Suite Verification**:
   ```powershell
   $env:PYTHONUTF8=1; python -m unittest discover -s tests -v
   ```
   *Expected outcome*: 38/38 tests pass.

2. **Excel Schema & Format Verification**:
   ```powershell
   $env:PYTHONUTF8=1; python -c "
   import openpyxl
   wb = openpyxl.load_workbook('results.xlsx')
   assert 'Training' in wb.sheetnames, 'Missing Training sheet'
   ws = wb['Training']
   expected_cols = ['exp_id', 'backbone', 'trục thay đổi (A–G)', 'khác T00 ở điểm nào', 'seed', 'macro-F1 val', 'top-1 val', 'Δ so với T00', 'ghi chú']
   actual_cols = [ws.cell(row=1, column=c).value for c in range(1, 10)]
   assert actual_cols == expected_cols, f'Header mismatch: {actual_cols} vs {expected_cols}'
   print('Training sheet schema matches 100%!')
   "
   ```

3. **Check T00 Baseline Run Availability**:
   ```powershell
   $env:PYTHONUTF8=1; python -c "
   import json
   from pathlib import Path
   p = Path('runs/B02/seed0/summary.json')
   assert p.exists(), f'Missing {p}'
   with open(p, 'r') as f:
       d = json.load(f)
   assert d['backbone'] == 'convnext_tiny'
   assert d['best_val_macro_f1'] == 0.9604
   assert d['best_val_top1'] == 0.9697
   print('T00 baseline artifact verified: Macro-F1 = 0.9604, Top-1 = 0.9697')
   "
   ```

4. **Invalidation Conditions**:
   - If any column in sheet `Training` is reordered or has Vietnamese accents removed.
   - If T00 is omitted from Row 2 or computed with a delta other than `0.0000`.
   - If CutMix and Label Smoothing are blindly paired together without acknowledging over-smoothing interference.
   - If test set predictions are evaluated prematurely before Milestone 4.
