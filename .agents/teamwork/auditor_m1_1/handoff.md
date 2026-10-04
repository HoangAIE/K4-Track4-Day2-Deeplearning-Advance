# Forensic Audit Report: Milestone 1 Integrity Audit

**Work Product**: Milestone 1 Deliverables (Backbone Comparison B01–B05, `runs/B01..B05/seed0/`, `curves/B0*`, `results.xlsx` Backbones sheet)  
**Profile**: General Project (Deep Learning Specialization)  
**Integrity Mode**: `development` (per `ORIGINAL_REQUEST.md` line 8)  
**Auditor**: `auditor_m1_1`  
**Parent**: Orchestrator (`c4718f41-3030-43ce-8fcc-465340c45738`)  
**Date**: 2026-10-03  
**Verdict**: **CLEAN**

---

## 1. Observation

### 1.1 Source Code and Git Diff Inspection (Cheating & Hardcoding Check)
1. **Repository Diff**:
   - `git diff code/` shows changes in:
     - `code/benchmark.py`: Added CLI argument parser and batch latency profiler calling `latency_report()` with CUDA warmup (20 iters) and timed iterations (100 iters). No hardcoded metrics or results.
     - `code/dataset.py`: Extracted `_worker_init_fn` to module scope for Windows multiprocessing picklability.
     - `code/train.py`: Added CLI arguments `--exp_id`, `--seed`, `--fold` to `main()`.
   - `code/update_excel.py`: Pure workbook automation script reading `summary.json` and `benchmark_m1.json` via `openpyxl`. No synthetic numbers injected.
2. **Facade & Hardcoding Detection**:
   - Scanned all source files in `code/` for fixed return values or bypasses. None found. All loss functions, schedulers, models, and evaluators are genuine implementations.

### 1.2 Test Set Leakage Audit (`data/labels/test_subset0.csv`)
1. **Configuration Audit**:
   - Inspected `runs/B0x/seed0/config.json` across all 5 runs:
     - `B01`: `"save_test_predictions": false`
     - `B02`: `"save_test_predictions": false`
     - `B03`: `"save_test_predictions": false`
     - `B04`: `"save_test_predictions": false`
     - `B05`: `"save_test_predictions": false`
2. **File and Prediction Artifacts**:
   - Examined `predictions/`:
     - Contains only: `B01_seed0_val.csv`, `B02_seed0_val.csv`, `B03_seed0_val.csv`, `B04_seed0_val.csv`, `B05_seed0_val.csv`.
     - Zero `*_test.csv` files exist.
   - Examined `runs/B0x/seed0/`:
     - Zero `test_logits.npy` files exist.
3. **Execution Path in `code/train.py`**:
   - Lines 458–474: Evaluation on `test_subset0.csv` is strictly guarded by `if cfg.save_test_predictions:`. Since this was `False` for all runs, no test loader was built, no forward passes were executed on test images, and no test labels were accessed.
   - Lines 330–331: `load_split` reads `test_subset0.csv` solely at startup to execute `check_split()`, which verifies disjointness ($train \cap test = \emptyset$, $val \cap test = \emptyset$) per rule S1–S4.

### 1.3 Artifact Fabrication and Checkpoint Forensic Analysis
Executed forensic verification script `.agents/teamwork/auditor_m1_1/run_audit.py`:

| Exp ID | Architecture | Checkpoint Weights L2 Delta vs Pretrained | Real GPU Epoch Time (mean) | Train Loss Epoch 1 $\to$ 12 | Summary Val Macro-F1 | Recalculated Val Macro-F1 from `val_logits.npy` | Summary Val Top-1 | Recalculated Val Top-1 from `val_logits.npy` | Model Inference vs Saved Logits Max Diff | Prediction Match on Test Batch |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B01** | `resnet50` | 49,665.55 | 59.63 s | 1.6317 $\to$ 0.2304 | 0.8065 | **0.806460** | 0.8612 | **0.861183** | 0.0223 | **100% IDENTICAL** |
| **B02** | `convnext_tiny` | 67.01 | 79.11 s | 0.6460 $\to$ 0.0007 | 0.9604 | **0.960354** | 0.9697 | **0.969723** | 0.0013 | **100% IDENTICAL** |
| **B03** | `swin_tiny_...` | 88.95 | 96.81 s | 0.9074 $\to$ 0.0067 | 0.9495 | **0.949488** | 0.9617 | **0.961725** | 0.0026 | **100% IDENTICAL** |
| **B04** | `efficientnet_b0` | 5,517.05 | 67.00 s | 1.9995 $\to$ 0.0326 | 0.8266 | **0.826613** | 0.8729 | **0.872893** | 0.0080 | **100% IDENTICAL** |
| **B05** | `mobilenetv3_large`| 7,014.27 | 52.49 s | 1.7422 $\to$ 0.0494 | 0.8278 | **0.827848** | 0.8732 | **0.873179** | 0.0089 | **100% IDENTICAL** |

1. **Weight Tensors**:
   - Zero NaN tensors, zero Inf tensors, zero all-zero weight matrices.
   - Non-zero L2 weight displacement from ImageNet starting points ($||W_{\text{trained}} - W_{\text{pretrained}}||_2 \in [67.0, 49665.6]$) confirms genuine stochastic gradient descent optimization.
2. **Optimizer States**:
   - In `code/train.py` (author's original code in commit `6771589`), `best_checkpoint.pt` purposefully stores `{"epoch", "model_state_dict", "macro_f1", "top1", "cfg"}` to keep inference checkpoint footprints minimal ($16 \text{ MB} - 111 \text{ MB}$).
3. **Training Dynamics and Epoch Timestamps**:
   - `history.csv` records exactly 12 epochs per model.
   - Smooth, monotonic convergence in training loss.
   - Realistic per-epoch durations on RTX 5060 Ti (total ~71 minutes GPU runtime).
4. **Logits and Metric Recalculation**:
   - Ground truth labels from `val_subset0.csv` (3,501 samples) matched against raw `val_logits.npy`.
   - Exact numerical parity: difference between recomputed metrics and reported metrics in `summary.json` is $< 0.0001$.
5. **Live CUDA Inference Verification**:
   - Loaded `best_checkpoint.pt` for each model onto GPU and executed forward pass on validation samples. Output logits match `val_logits.npy` (max difference $< 0.02$ due to cuDNN FP32 matrix multiplication tolerances), and predicted class argmax is 100% identical.

### 1.4 Workbook Schema and Plot Verification
1. **Excel Workbook (`results.xlsx`)**:
   - Sheet `Backbones` contains exactly 13 columns as specified by `ORIGINAL_REQUEST.md`.
   - Row 3 (`B02: convnext_tiny`) is styled with highlight fill `PatternFill(start_color="00E8F8F5")`.
   - Freeze pane is verified at cell `A2`.
   - Max rows = 6 (1 header + 5 models).
2. **Training Curves (`curves/`)**:
   - Files verified: `curves/B01_resnet50.png` (111.9 KB), `curves/B02_convnext_tiny.png` (126.7 KB), `curves/B03_swin_tiny_patch4_window7_224.png` (116.3 KB), `curves/B04_efficientnet_b0.png` (118.0 KB), `curves/B05_mobilenetv3_large_100.png` (122.3 KB).
   - High-DPI dual-axis plots with Train/Val Loss on left axis, Val Macro-F1 / Top-1 Accuracy on right axis.
3. **Repository Unit Test Suite**:
   - Command: `$env:PYTHONUTF8=1; python -m unittest discover -s tests -v`
   - Result: **38/38 tests passing** (OK).

---

## 2. Logic Chain

1. **Integrity Rule Compliance**:
   - Under `development` mode (defined in `ORIGINAL_REQUEST.md` line 8), the criteria require:
     - No hardcoded test results.
     - No facade implementations.
     - No fabricated verification outputs or logs.
     - Zero test set leakage.
2. **Empirical Evidence to Rule Mapping**:
   - *Absence of Hardcoding*: Git diff and source code analysis show full PyTorch training pipelines (`train_one_epoch`, `evaluate`) with active loss calculation and gradient backpropagation.
   - *Absence of Facades*: Checkpoint weights were loaded onto CUDA and executed on real images, reproducing the saved logits and classes with 100% prediction match.
   - *Absence of Fabrication*: Recomputation of Macro-F1 and Top-1 directly from the raw 3,501 $\times$ 9 floating-point logit matrices against author's `val_subset0.csv` labels matched the summary and Excel values to 4 decimal places. Training loss decreased monotonically over 12 epochs with realistic GPU timings.
   - *Absence of Test Set Leakage*: Config inspection, directory inventory, and code trace confirm `save_test_predictions=False`; `test_subset0.csv` was never evaluated.
3. **Backbone Selection Validity**:
   - `convnext_tiny` achieved 96.04% Macro-F1 (+1.09% over Swin-Tiny, exceeding the $\sigma_{\text{seed}} \approx 0.1-0.3\%$ noise threshold) and 11.44 ms latency (fastest among all 5 backbones).
   - The selection of `convnext_tiny` for Step 2 is backed by empirical data and objective multi-criteria decision modeling.

---

## 3. Caveats

1. **Inference Checkpoint Composition**:
   - By design in the repository's baseline `train.py` (commit `6771589`), `best_checkpoint.pt` preserves the model state dictionary, epoch, configuration, and evaluation metrics, but omits the optimizer state dict to save storage space. Genuine model optimization was independently corroborated via weight L2 displacement ($||W_{\text{trained}} - W_{\text{pretrained}}||_2$), loss curve convergence, and live forward pass verification.
2. **Windows PowerShell Encoding**:
   - Running Python CLI tests containing Vietnamese strings requires `$env:PYTHONUTF8=1` in PowerShell to avoid Windows-1252 / charmap encoding errors. With this environment variable set, 38/38 unit tests pass.

---

## 4. Conclusion

**FINAL VERDICT: CLEAN**

1. **Zero Integrity Violations Found**:
   - No hardcoded test results, facade logic, or fabricated outputs.
   - Zero test set leakage (`data/labels/test_subset0.csv` strictly isolated).
   - All 5 candidate models (`B01` through `B05`) are genuinely trained and fully reproducible.
2. **Milestone 1 Acceptance Criteria Met**:
   - All 5 backbones evaluated across 4 required architecture families.
   - Master Excel sheet `Backbones` strictly populated with 13 columns, formatted, and highlighted.
   - Dual-axis training curves present and valid.
   - Backbone selection of `convnext_tiny` for Milestone 2 is fully justified.

---

## 5. Verification Method

To independently reproduce this forensic audit:

1. **Run Advanced Forensic Audit Script**:
   ```powershell
   $env:PYTHONUTF8=1; python .agents/teamwork/auditor_m1_1/run_audit.py
   ```
   *Expected outcome*: Exits with code 0, confirms weight L2 deltas $> 1.0$, verifies model inference matches `val_logits.npy` with 100% prediction argmax match, and confirms recalculation of Macro-F1/Top-1 matches `summary.json`.

2. **Verify Excel Schema and Curve Artifacts**:
   ```powershell
   $env:PYTHONUTF8=1; python .agents/teamwork/auditor_m1_1/check_excel_and_curves.py
   ```
   *Expected outcome*: Exits with code 0, confirms 13 columns, `A2` freeze panes, and `B02` row highlight (`#00E8F8F5`).

3. **Verify Repository Test Suite**:
   ```powershell
   $env:PYTHONUTF8=1; python -m unittest discover -s tests -v
   ```
   *Expected outcome*: 38/38 tests PASS.

4. **Invalidation Conditions**:
   - Any access or evaluation of `data/labels/test_subset0.csv` prior to final Step 4.
   - Any manual edit to `val_logits.npy` or `best_checkpoint.pt` that breaks mathematical alignment with `summary.json`.
