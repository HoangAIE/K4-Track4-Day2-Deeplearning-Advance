# Project: DeepWeeds Deep Learning Day 2 Lab

## Architecture
- **Problem**: 9-class weed classification on DeepWeeds dataset with severe class imbalance (Negative class ~52%, 8 weed classes ~6% each).
- **Core Engine**: PyTorch with timm backbones, mixed precision (AMP), AdamW optimizer with differential learning rates (1e-4 backbone, 1e-3 classifier head) and weight decay 0.05.
- **Dataset Splitting**: Fixed Fold 0 partition (10,501 train, 3,501 val, 3,507 test) with strict zero test leakage. All validation, model selection, and hyperparameter tuning strictly use `val_subset0.csv`.
- **Target Metric**: Validation Macro-F1 across 9 classes (primary objective) and Top-1 Accuracy.
- **Profiling**: Batch-1 inference latency (ms) with warmup and CUDA synchronization, GMACs, and parameter counts.
- **Reporting**: Excel workbook `results.xlsx` (`Backbones` and `Training` sheets), training curves in `curves/`, checkpoints and run configs in `runs/<exp_id>/seed0/`.

## Feature Inventory
| # | Feature | Description | Milestone | Source | Status |
|---|---------|-------------|-----------|--------|--------|
| F1 | B01 ResNet-50 | Train B01 (`resnet50`) with T00 recipe, benchmark latency/GMACs, log curves & metrics | M1 | ORIGINAL_REQUEST §R1 | DONE |
| F2 | B02 ConvNeXt-Tiny | Train B02 (`convnext_tiny`) with T00 recipe, benchmark latency/GMACs, log curves & metrics | M1 | ORIGINAL_REQUEST §R1 | DONE |
| F3 | B03 Swin/ViT Transformer | Train B03 (`swin_tiny_patch4_window7_224`) with T00 recipe, benchmark latency/GMACs, log curves & metrics | M1 | ORIGINAL_REQUEST §R1 | DONE |
| F4 | B04 EfficientNet-B0 | Train B04 (`efficientnet_b0`) with T00 recipe, benchmark latency/GMACs, log curves & metrics | M1 | ORIGINAL_REQUEST §R1 | DONE |
| F5 | B05 MobileNetV3-Large | Train B05 (`mobilenetv3_large_100`) with T00 recipe, benchmark latency/GMACs, log curves & metrics | M1 | ORIGINAL_REQUEST §R1 | DONE |
| F6 | Backbone Trade-off Selection | Quantitative trade-off analysis (Macro-F1 vs Latency vs Params) -> `convnext_tiny` selected | M1 | ORIGINAL_REQUEST §R1 | DONE |
| F7 | Axis A Ablation (Init) | `scratch` vs `frozen` vs `finetune` (T00) single-factor ablations on `convnext_tiny` | M2 | ORIGINAL_REQUEST §R2 | PLANNED |
| F8 | Axis B Ablation (Augmentation) | `basic` vs `color` / `trivial` or `cutmix` single-factor ablations | M2 | ORIGINAL_REQUEST §R2 | PLANNED |
| F9 | Axis C Ablation (Loss Function) | `ce` vs `ls` (label_smoothing=0.1) vs `focal` ($\gamma=2.0$) single-factor ablations | M2 | ORIGINAL_REQUEST §R2 | PLANNED |
| F10 | Axis D / F Ablation (Sampling / EMA) | `sampler="balanced"` and/or `ema_decay=0.999` single-factor ablations | M2 | ORIGINAL_REQUEST §R2 | PLANNED |
| F11 | Combined Recipe Exploration | Combination of best-performing individual factors to evaluate compounding vs cancelling effects | M2 | ORIGINAL_REQUEST §R2 | PLANNED |
| F12 | Excel Workbook Logging | Populate `results.xlsx` (`Backbones` and `Training` sheets) with all required columns and formats | M1, M2, M3 | ORIGINAL_REQUEST §R1, §R2, §Acceptance | IN_PROGRESS (Backbones DONE) |
| F13 | Curve Plotting & Checkpoint Persistence | Generate and save dual-axis curves to `curves/` and run artifacts to `runs/<exp_id>/seed0/` | M1, M2 | ORIGINAL_REQUEST §R1, §R2, §R4 | IN_PROGRESS (B01..B05 DONE) |
| F14 | Scientific Analysis & Noise Comparison | Statistical comparison of $\Delta$ against seed noise ($\approx 0.1-0.3$ pt), answering core scientific questions, identifying Best Recipe | M3 | ORIGINAL_REQUEST §R3 | PLANNED |
| F15 | Zero-Leakage & Data Integrity | Ensure strict zero test set leakage (`test_subset0.csv` untouched) and development integrity | M1, M2, M3 | ORIGINAL_REQUEST §R4 | IN_PROGRESS |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Step 1 — Backbone Comparison (B01..B05) | Train 5 backbones (B01..B05), profile latency & GMACs, generate curves, populate `Backbones` sheet in `results.xlsx`, and select optimal backbone for Step 2 (`convnext_tiny`) | Phase 0 Survey | DONE |
| M2 | Step 2 — Training Recipe Ablation (T01..T0x) | On `convnext_tiny`, run single-variable ablations across >= 3 axes (A, B, C, D/F) + best combination, generate curves, populate `Training` sheet in `results.xlsx` | M1 | IN_PROGRESS |
| M3 | Step 3 — Scientific Analysis & Verification | Statistical delta vs noise analysis, answer core experimental questions, final Excel and artifacts audit, deliver comprehensive lab completion report | M1, M2 | PLANNED |

## Interface Contracts
### Training Invocation Contract
- Script: `code/train.py`
- Environment: `$env:PYTHONUTF8=1` on Windows PowerShell
- Arguments: `--exp_id <EXP_ID> --seed 0 --fold 0 --set <KEY1>=<VAL1> <KEY2>=<VAL2> ...`
- Output directories:
  - Artifacts: `runs/<EXP_ID>/seed0/` containing `config.json`, `best_checkpoint.pt`, `val_logits.npy`, `history.csv`, `summary.json`
  - Plots: `curves/<EXP_ID>_<BACKBONE>.png` (and aliased if needed)

### Benchmark Invocation Contract
- Script: `code/benchmark.py`
- Invocation: `python code/benchmark.py --models <MODEL_NAME> --batch_size 1 --device cuda`
- Output: Latency percentiles (mean, p50, p95), parameter count (M), GMACs.

### Excel Logging Contract
- File: `results.xlsx`
- Sheet `Backbones`: `exp_id`, `backbone`, `tag trọng số`, `#tham số (M)`, `GMAC`, `độ phân giải`, `epoch`, `seed`, `macro-F1 val`, `top-1 val`, `thời gian train/epoch`, `độ trễ batch-1 (ms)`, `ghi chú`
- Sheet `Training`: `exp_id`, `backbone`, `trục thay đổi (A–G)`, `khác T00 ở điểm nào`, `seed`, `macro-F1 val`, `top-1 val`, `Δ so với T00`, `ghi chú`

## Code Layout
- `code/train.py`: Primary training loop, Config definition, experiment orchestration.
- `code/model.py`: Backbone architecture factories, parameter and GMAC counters, freeze/finetune logic.
- `code/dataset.py`: DeepWeeds dataset loaders, transforms, augmentation pipelines.
- `code/losses.py`: CrossEntropy, LabelSmoothing, FocalLoss, and class-weighted losses.
- `code/benchmark.py`: Inference latency and computational complexity profiling.
- `eval.py`: Standard Macro-F1 and Top-1 evaluation metrics.
- `data/images/`: 17,509 JPEG images (256x256).
- `data/labels/`: Fold 0 CSV splits (`train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`).
- `runs/`: Experiment checkpoints, histories, and JSON summaries.
- `curves/`: Training and validation metric progression plots.
- `results.xlsx`: Master results workbook.
