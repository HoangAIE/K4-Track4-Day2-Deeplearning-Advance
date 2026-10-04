# Handoff Report: Recipe Ablations, Excel Logging & Curve Plotting

**Agent**: Explorer 3 (Ablation & Excel Explorer)  
**Parent Conversation ID**: `c4718f41-3030-43ce-8fcc-465340c45738`  
**Date**: 2026-10-03  
**Working Directory**: `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\explorer_survey_3\`  
**Detailed Report**: `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\explorer_survey_3\analysis.md`

---

## 1. Observation

1. **`code/train.py` Configuration**:
   - `Config` is defined as a dataclass at lines 59–97 of `code/train.py`.
   - Default parameters match baseline recipe T00:
     - `exp_id = "T00"`, `seed = 0`, `fold = 0`, `backbone = "resnet50"`, `init = "finetune"`.
     - `epochs = 12`, `batch_size = 64`, `lr_backbone = 1e-4`, `lr_head = 1e-3`, `weight_decay = 0.05`, `warmup_epochs = 1.0`, `amp = True`.
     - `aug = "basic"`, `sampler = None`, `mix = None`, `loss = "ce"`, `ema_decay = None`.
   - `build_optimizer()` (line 121) sets AdamW across 3 parameter groups defined in `code/model.py` (`backbone_decay` with wd=0.05, `backbone_no_decay` with wd=0.0, and `head` with wd=0.05).
   - `build_scheduler()` (line 127) implements linear warmup for 1 epoch followed by cosine decay down to 1e-4, updated per batch at line 219 (`scheduler.step()`).

2. **Ablation Axes Implementation**:
   - **Axis A (Init)**: In `code/model.py` (lines 35–77) and `code/train.py` (lines 178–184):
     - `init="scratch"` sets `pretrained=False`.
     - `init="frozen"` calls `freeze_backbone()`, locking backbone parameters and freezing all BatchNorm layers in `eval` mode via `frozen_train()` wrapper.
     - `init="finetune"` trains full model with differential learning rates.
   - **Axis B (Augmentation)**: In `code/dataset.py` (lines 147–189):
     - Supports `basic`, `color` (ColorJitter), `trivial` (TrivialAugmentWide), `randaug`.
     - In `code/losses.py` (lines 119–177), `mix_batch(..., mode="cutmix")` clips coordinates and adjusts $\lambda$ to actual bounding box area. `mixed_loss()` computes composite loss.
   - **Axis C (Loss)**: In `code/losses.py` (lines 17–95) and `code/train.py` (lines 364–374):
     - Supports `ce`, `focal` (FocalLoss with default gamma=2.0), `ls` (LabelSmoothingCE).
     - Verbatim code at `code/train.py` line 77: `label_smoothing: float = 0.0` and line 368: `criterion = build_criterion("ls", smoothing=cfg.label_smoothing)`.
     - Verbatim code at `code/losses.py` line 44: `if self.smoothing <= 0.0: return F.cross_entropy(logits, target)`.
     - **Observed**: Invoking `--set loss=ls` without `--set label_smoothing=0.1` will pass `smoothing=0.0`, silently defaulting to standard Cross-Entropy.
   - **Axis D (Sampler)**: In `code/dataset.py` (lines 241–248):
     - `sampler="balanced"` instantiates `WeightedRandomSampler` with weights proportional to inverse class counts and disables shuffle.
   - **Axis F (EMA)**: In `code/train.py` (lines 143–172, 381, 400–406):
     - `EMA` tracks shadow weights with decay `0.999`.
     - `ema.apply_shadow(model)` is used during validation evaluation, but `ema.restore(model)` is invoked before checkpoint saving at line 437.

3. **`results.xlsx` Status & Schema**:
   - `results.xlsx` does not exist in the repository root or subfolders.
   - `openpyxl` (3.1.5) and `pandas` (2.2.0) are installed in the Python environment.
   - Required sheets per `GUIDE.md` section 6.1:
     - `Backbones`: `exp_id`, `backbone`, `tag trọng số`, `#tham số (M)`, `GMAC`, `độ phân giải`, `epoch`, `seed`, `macro-F1 val`, `top-1 val`, `thời gian train/epoch`, `độ trễ batch-1 (ms)`, `ghi chú`.
     - `Training`: `exp_id`, `backbone`, `trục thay đổi (A–G)`, `khác T00 ở điểm nào`, `seed`, `macro-F1 val`, `top-1 val`, `Δ so với T00`, `F1 các lớp hiếm (nếu có)`, `ghi chú`.
     - `Inference`, `Final`, `PerClass`, `Latency`, `Summary`.

4. **Curves and Checkpoint Output**:
   - `code/train.py` (lines 99–106, 316–326, 435–454, 477–482) writes:
     - Checkpoint to `runs/<exp_id>/seed<seed>/best_checkpoint.pt`.
     - History to `runs/<exp_id>/seed<seed>/history.csv`.
     - Summary to `runs/<exp_id>/seed<seed>/summary.json`.
     - Predictions to `predictions/<exp_id>_seed<seed>_val.csv`.
     - Curve plots to `curves/<exp_id>_<backbone>.png` via `plot_curves()` (lines 273–311), featuring dual y-axes: Train/Val Loss (left axis) and Val Macro-F1 / Top-1 Acc (right axis).

5. **Test and Sanity Execution**:
   - `python -m unittest discover -s tests -v` returned: `Ran 38 tests in 0.896s ... OK`.
   - `$env:PYTHONUTF8=1; python code/run_step0_checks.py` passed all checks on CUDA (NVIDIA GeForce RTX 5060 Ti).

---

## 2. Logic Chain

1. **Baseline Recipe Readiness**:
   - Observation 1 demonstrates that all hyperparameters requested for T00 (AdamW, 1e-4/1e-3 LRs, wd=0.05 without norm/bias, 1 epoch warmup + cosine, batch size 64, 12 epochs, seed 0) are already hard-coded as the defaults of `Config`, `build_optimizer`, and `build_scheduler`.
   - Therefore, running `python code/train.py` with no overrides runs the baseline recipe T00 immediately without code changes.

2. **Ablation Axes Completeness & Critical Constraints**:
   - Observation 2 demonstrates that all five requested ablation axes (A: Init, B: Aug, C: Loss, D: Sampler, F: EMA) are implemented in the codebase.
   - However, for Axis C, since `label_smoothing` defaults to `0.0` in `Config` and `LabelSmoothingCE` reverts to Cross-Entropy when smoothing is $\le 0.0$, running `--set loss=ls` alone will fail to apply label smoothing.
   - Therefore, the ablation run for label smoothing must explicitly include `--set loss=ls label_smoothing=0.1`.
   - For Axis F, since shadow weights are reverted prior to checkpoint saving, validation scores in `history.csv` reflect EMA performance, but `best_checkpoint.pt` preserves the raw weights.

3. **Logging & Artifacts Pipeline**:
   - Observation 3 shows that `results.xlsx` does not exist, but necessary libraries (`openpyxl`, `pandas`) are installed.
   - Observation 4 confirms that `code/train.py` produces `runs/<exp_id>/seed0/summary.json`, `predictions/`, and `curves/<exp_id>_<backbone>.png`.
   - Therefore, generating `results.xlsx` requires aggregating `summary.json` outputs from all completed runs into the designated Excel sheets and columns.

---

## 3. Caveats

1. **Hardware / OS Specifics**: The test machine runs Windows PowerShell with Python 3.12. Running scripts printing Vietnamese UTF-8 characters directly to console requires `$env:PYTHONUTF8=1` or `PYTHONIOENCODING=utf-8` to prevent `charmap` encoding errors.
2. **Inference EMA weights**: Checkpoint saving does not preserve EMA shadow weights directly inside `best_checkpoint.pt`. If Step 3 inference requires testing the EMA model, the inference script should either recompute EMA or the checkpoint saving should be adjusted during implementation.
3. **Multi-seed Final Test Gate**: In `code/train.py`, `save_test_predictions` is `False` by default (enforcing rule S4). Test subset evaluation is only activated when explicitly requested for Step 4 final runs.

---

## 4. Conclusion

The training engine `code/train.py` and supporting modules are fully implemented, verified, and ready to execute Step 1 (Backbone comparison) and Step 2 (Recipe ablations).
- All requested ablation axes (A, B, C, D, F) are supported.
- For Axis C, `label_smoothing=0.1` must be explicitly passed along with `loss=ls`.
- `results.xlsx` must be constructed using `pandas.ExcelWriter(engine="openpyxl")` with the exact column schemas documented above.
- Curves and checkpoints are correctly directed to `curves/` and `runs/<exp_id>/seed0/`.

---

## 5. Verification Method

1. **Unit Test Suite**:
   ```powershell
   $env:PYTHONUTF8=1; python -m unittest discover -s tests -v
   ```
   *Expected outcome*: 38 tests pass with `OK`.

2. **Step 0 Pipeline Verification**:
   ```powershell
   $env:PYTHONUTF8=1; python code/run_step0_checks.py
   ```
   *Expected outcome*: All 5 stages complete with `>>> TẤT CẢ KIỂM TRA BƯỚC 0 ĐÃ VƯỢT QUA XUẤT SẮC! <<<`.

3. **Dry-Run Parameter Verification**:
   ```powershell
   $env:PYTHONUTF8=1; python -c "from code.train import Config, parse_overrides; cfg = Config(**parse_overrides(['loss=ls', 'label_smoothing=0.1', 'sampler=balanced', 'ema_decay=0.999'])); print(cfg)"
   ```
   *Expected outcome*: `loss='ls'`, `label_smoothing=0.1`, `sampler='balanced'`, `ema_decay=0.999`.
