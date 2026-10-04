# Detailed Investigation & Analysis Report: Recipe Ablations, Excel Logging & Curve Plotting

**Author:** Explorer 3 (Ablation & Excel Explorer)  
**Date:** 2026-10-03  
**Working Directory:** `.agents/teamwork/explorer_survey_3/`  
**Scope:** Investigation of `code/train.py`, baseline recipe T00, ablation axes (A, B, C, D, F), `results.xlsx` structure, and curve/checkpoint saving mechanisms for DeepWeeds Lab Day 2.

---

## 1. Executive Summary

This investigation analyzed the training engine (`code/train.py`), supporting modules (`code/model.py`, `code/losses.py`, `code/dataset.py`, `code/benchmark.py`), ablation axes required by `ORIGINAL_REQUEST.md` (Axes A, B, C, D, F), the status and format of `results.xlsx`, and curve/checkpoint persistence.

### Key Conclusions:
1. **Pipeline Completeness**: The core training infrastructure in `code/train.py` and supporting modules is already implemented and verified. Unit tests (`38/38` passing) and Step 0 pipeline checks (`code/run_step0_checks.py` passing on NVIDIA RTX 5060 Ti GPU) confirm mathematical and functional correctness.
2. **Baseline Recipe T00 Compliance**: `Config` in `code/train.py` defaults exactly to T00 specifications: AdamW, 3 parameter groups (weight decay 0.05 on 2D weights, 0.0 on bias/norm), linear warmup 1 epoch + cosine annealing per step down to 1e-4, Cross-Entropy loss, AMP enabled, batch size 64, 12 epochs, seed 0, fold 0.
3. **Ablation Axes (A, B, C, D, F) Wiring**:
   - **Axis A (Init)**: `scratch`, `frozen`, `finetune` are fully implemented in `code/model.py` and `code/train.py` (including eval-mode BatchNorm freezing).
   - **Axis B (Augmentation)**: `basic`, `color`, `trivial`, `randaug`, and `cutmix` (with border clipping and actual area lambda adjustment) are fully implemented.
   - **Axis C (Loss)**: `ce`, `ls`, `focal`, and `ce_weighted` are implemented. **CRITICAL FINDING**: In `Config`, `label_smoothing` defaults to `0.0`. If invoking `--set loss=ls` without `--set label_smoothing=0.1`, `LabelSmoothingCE` falls back to standard Cross-Entropy! The ablation command must specify `loss=ls label_smoothing=0.1`.
   - **Axis D (Sampler)**: `sampler=None` vs `sampler="balanced"` is fully implemented via `WeightedRandomSampler` with inverse class frequencies.
   - **Axis F (EMA)**: `EMA` class with decay `0.999` is implemented and used for validation metrics. **SUBTLE CAVEAT**: `ema.restore(model)` is called before saving `best_checkpoint.pt`, meaning saved checkpoint weights are the raw training weights, not the EMA shadow weights.
4. **`results.xlsx` Status**: Does **NOT** exist yet. It must be generated using `pandas.ExcelWriter(engine="openpyxl")`. The required sheets per `GUIDE.md` section 6.1 are `Backbones`, `Training`, `Inference`, `Final`, `PerClass`, `Latency`, and `Summary`. The exact required columns have been documented below.
5. **Curve & Checkpoint Artifacts**:
   - `code/train.py` outputs to `runs/<exp_id>/seed<seed>/` (`config.json`, `best_checkpoint.pt`, `val_logits.npy`, `history.csv`, `summary.json`).
   - Predictions output to `predictions/<exp_id>_seed<seed>_val.csv` (and test CSV only when `save_test_predictions=True`).
   - Training curves are saved to `curves/<exp_id>_<backbone>.png` with dual y-axes (Train/Val Loss on left, Val Macro-F1 and Top-1 Acc on right).

---

## 2. Deep Dive: `code/train.py` & Baseline Recipe T00

### 2.1 Definition of `Config`
In `code/train.py` (lines 59–97), configuration is structured as a Python `@dataclass`:

```python
@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: Optional[str] = None     # None | balanced
    mix: Optional[str] = None         # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: Optional[float] = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: Optional[float] = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False
```

### 2.2 Verification of T00 Baseline Recipe Parameters
The parameters mandated by `ORIGINAL_REQUEST.md` R1 and `GUIDE.md` section 1.4 map to `code/train.py` and supporting modules as follows:

| Parameter | Specification | Implementation in `code/train.py` & `code/model.py` | Verification Status |
|---|---|---|---|
| **Optimizer** | AdamW | `torch.optim.AdamW(groups)` in `build_optimizer()` (line 124) | Exact match |
| **Parameter Groups** | 3 groups (slide p. 52) | `param_groups()` in `code/model.py` (lines 114–169):<br>1. `backbone_decay`: 2D weights, lr=1e-4, wd=0.05<br>2. `backbone_no_decay`: 1D norm/bias, lr=1e-4, wd=0.0<br>3. `head`: classifier parameters, lr=1e-3, wd=0.05 | Verified (53 decay tensors, 106 no-decay tensors, 2 head tensors for ResNet50) |
| **Backbone LR** | 1e-4 | `lr_backbone = 1e-4` (line 83) | Exact match |
| **Head LR** | 1e-3 | `lr_head = 1e-3` (line 84) | Exact match |
| **Weight Decay** | 0.05 (excluding norm/bias) | `weight_decay = 0.05` (line 85), skipped in group 2 | Exact match |
| **Warmup & Scheduler** | Linear 1 epoch + Cosine Annealing | `build_scheduler()` (lines 127–140): linear warmup for `warmup_epochs * steps_per_epoch`, cosine decay to 1e-4 floor, stepped per batch (line 219) | Exact match |
| **Loss** | Cross-Entropy | `loss = "ce"` -> `build_criterion("ce")` -> `nn.CrossEntropyLoss()` (lines 76, 373) | Exact match |
| **AMP** | On (FP16 autocast + GradScaler) | `amp = True`, `torch.amp.GradScaler("cuda")`, `torch.autocast(device_type=device.type)` (lines 88, 198, 202, 380) | Exact match |
| **Batch Size** | 64 | `batch_size = 64` (line 82) | Exact match |
| **Epochs** | 12 | `epochs = 12` (line 81) | Exact match |
| **Seed** | 0 | `seed = 0` (line 63), `set_seed()` sets random, numpy, torch CPU/CUDA, cudnn deterministic (lines 109–118) | Exact match |
| **Data Integrity** | Fold 0, no test leak | `fold = 0`, validation evaluated on `val_subset0.csv`, test evaluation gated by `save_test_predictions = False` (lines 64, 96, 458) | Exact match |

### 2.3 Command Line Interface (CLI)
`parse_overrides(args.set)` in lines 504–543 dynamically inspects dataclass fields of `Config` and performs type conversions:
- Handles `bool` ("true"/"false"), `int`, `float`, and `None` ("none"/"null").
- Example invocation:
  ```bash
  python code/train.py --set exp_id=B01 backbone=resnet50 seed=0
  python code/train.py --set exp_id=T01 init=scratch seed=0
  ```

---

## 3. Deep Dive: Ablation Axes Implementation Analysis

### 3.1 Axis A (Initialization): `scratch` vs `frozen` vs `finetune`
- **Location**: `code/model.py` (lines 30–113) and `code/train.py` (lines 178–184, 351–357).
- **Mechanism**:
  - `init="scratch"`: `pretrained=False`. All weights are randomly initialized. All layers are trainable (`requires_grad=True`).
  - `init="frozen"`: `pretrained=True`. Calls `freeze_backbone(model)`:
    - Backbone weights set to `param.requires_grad = False`.
    - Only `model.get_classifier()` has `requires_grad = True`.
    - **Crucial detail**: `freeze_backbone` sets all `BatchNorm2d/1d/SyncBatchNorm` layers to `m.eval()`, and wraps `model.train()` with `frozen_train()` so BatchNorm running statistics are NEVER modified during training.
    - During training in `train_one_epoch` (lines 178–182): `model.eval()`, classifier head `.train()`.
  - `init="finetune"`: `pretrained=True`. Backbone and head are both trainable with differential learning rates (1e-4 vs 1e-3).
- **Wiring Status**: **Fully implemented and operational.**

### 3.2 Axis B (Augmentation): `basic` vs `color` / `trivial` vs `cutmix`
- **Location**: `code/dataset.py` (lines 147–189) and `code/losses.py` (lines 119–177).
- **Mechanism**:
  - `aug="basic"`: `RandomResizedCrop(224, scale=(0.8, 1.0))`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize()`.
  - `aug="color"`: `RandomResizedCrop(224, scale=(0.8, 1.0))`, `RandomHorizontalFlip()`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`, `ToTensor()`, `Normalize()`.
  - `aug="trivial"`: `TrivialAugmentWide()`, `RandomResizedCrop(224, scale=(0.8, 1.0))`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize()`.
  - `aug="randaug"`: `RandAugment(num_ops=2, magnitude=9)`, `RandomResizedCrop`, `RandomHorizontalFlip`, `Normalize`.
  - `mix="cutmix"`: Handled in `mix_batch(images, labels, alpha=cfg.mix_alpha, mode="cutmix")` and `mixed_loss(criterion, logits, targets)`.
    - Samples $\lambda \sim \text{Beta}(\alpha, \alpha)$.
    - Cuts bounding box from shuffled images `images[perm]`.
    - Clips coordinates to image boundaries `[0, w]` and `[0, h]`.
    - Recomputes actual bounding box area: `lam_actual = 1.0 - (actual_area) / (w * h)`.
    - Computes mixed loss: $\lambda \mathcal{L}(logits, y_a) + (1 - \lambda)\mathcal{L}(logits, y_b)$.
- **Wiring Status**: **Fully implemented and operational.**

### 3.3 Axis C (Loss Functions): `ce` vs `ls` vs `focal`
- **Location**: `code/losses.py` (lines 17–95) and `code/train.py` (lines 364–374).
- **Mechanism**:
  - `loss="ce"`: standard `nn.CrossEntropyLoss()`.
  - `loss="focal"`: `FocalLoss(gamma=cfg.focal_gamma, alpha=None)`. In `code/train.py`, `focal_gamma` defaults to `2.0`. Tested in `code/run_step0_checks.py` (verified $\gamma=0$ perfectly matches CE).
  - `loss="ls"`: `LabelSmoothingCE(smoothing=cfg.label_smoothing)`. Formula: $(1 - \epsilon)\text{NLL} + \epsilon \cdot \text{UniformLoss}$. Tested in `code/run_step0_checks.py` (verified $\epsilon=0$ matches CE).
- **CRITICAL CAVEAT**:
  In `code/train.py`:
  ```python
  # Line 77
  label_smoothing: float = 0.0
  ...
  # Lines 368-369
  elif cfg.loss in ("ls", "label_smoothing"):
      criterion = build_criterion("ls", smoothing=cfg.label_smoothing)
  ```
  And in `code/losses.py` (line 44):
  ```python
  if self.smoothing <= 0.0:
      return F.cross_entropy(logits, target)
  ```
  If a user executes `--set exp_id=T03 loss=ls`, `cfg.label_smoothing` remains `0.0`, resulting in standard Cross-Entropy!
  **Recommendation**:
  When running the Axis C label smoothing ablation, either:
  1. Always invoke CLI with `--set loss=ls label_smoothing=0.1`, OR
  2. In `code/train.py`, update line 369 to default to 0.1 if `cfg.label_smoothing == 0.0` (e.g. `smoothing=cfg.label_smoothing if cfg.label_smoothing > 0.0 else 0.1`).
- **Wiring Status**: **Implemented, but invocation requires passing `label_smoothing=0.1` explicitly.**

### 3.4 Axis D (Sampling): `sampler=None` vs `sampler="balanced"`
- **Location**: `code/dataset.py` (lines 238–248) and `code/train.py` (lines 72, 337–341).
- **Mechanism**:
  - `sampler=None`: Standard random shuffling (`shuffle=True`, `sampler=None`).
  - `sampler="balanced"`: Computes train class frequencies $N_c$. Assigns sample weights $w_i = \frac{1}{N_{y_i}}$. Creates `torch.utils.data.WeightedRandomSampler(weights=..., num_samples=len(weights), replacement=True)` and disables dataset shuffling.
- **Wiring Status**: **Fully implemented and operational.**

### 3.5 Axis F (Regularization / EMA): `ema_decay=None` vs `ema_decay=0.999`
- **Location**: `code/train.py` (lines 143–172, 381, 400–406).
- **Mechanism**:
  - `EMA` tracks shadow parameters: $W_{shadow} \leftarrow d \cdot W_{shadow} + (1 - d) \cdot W$.
  - Updated after every batch (`ema.update(model)` at line 222).
  - Evaluated on validation set using shadow parameters via `ema.apply_shadow(model)` and restored via `ema.restore(model)` (lines 401, 406).
- **SUBTLE CAVEAT**:
  When `val_macro_f1 > best_macro_f1` (line 433), `torch.save` stores `model.state_dict()` at line 437. Because `ema.restore(model)` was executed at line 406, `model.state_dict()` contains the raw non-EMA weights!
  If this checkpoint is later reloaded for Step 3 inference, it will use the non-EMA weights.
  **Recommendation**: For proper EMA checkpoint saving, either clone the shadow weights into the saved checkpoint or document that validation metric during training reflects EMA while checkpoint holds raw parameters.
- **Wiring Status**: **Implemented for in-training validation metrics.**

---

## 4. Deep Dive: `results.xlsx` Specifications

### 4.1 Current Status
- `results.xlsx` does **not exist** in the repository.
- Both `pandas` (version 2.2.0) and `openpyxl` (version 3.1.5) are installed in the Python environment, allowing direct programmatic creation and modification via `pd.ExcelWriter("results.xlsx", engine="openpyxl")`.

### 4.2 Required Sheets & Exact Column Headers
According to `GUIDE.md` section 6.1 and `ORIGINAL_REQUEST.md`:

#### 1. Sheet `Backbones` (Step 1 Comparison)
Required for acceptance criteria R1. Must contain $\ge 5$ backbones covering all 4 mandated architectural families.
- **Columns (13 columns)**:
  1. `exp_id` (e.g. `B01`, `B02`, `B03`, `B04`, `B05`)
  2. `backbone` (e.g. `resnet50`, `convnext_tiny`, `swin_tiny_patch4_window7_224`, `efficientnet_b0`, `mobilenetv3_large_100`)
  3. `tag trọng số` (timm / torchvision pretrained weight tag, e.g. `torchvision_resnet50`, `in1k`)
  4. `#tham số (M)` (total parameters in millions, e.g. `25.56M`)
  5. `GMAC` (Giga Multiply-Accumulates for $3 \times 224 \times 224$ input)
  6. `độ phân giải` (always `224` for Step 1)
  7. `epoch` (always `12` for T00 recipe)
  8. `seed` (always `0`)
  9. `macro-F1 val` (computed on `val_subset0.csv` using `eval.py`)
  10. `top-1 val` (Top-1 Accuracy on `val_subset0.csv`)
  11. `thời gian train/epoch` (average training epoch time in seconds, e.g. `45.2s`)
  12. `độ trễ batch-1 (ms)` (GPU forward latency p50 with warmup & CUDA sync)
  13. `ghi chú` (qualitative trade-off remarks and observations)

#### 2. Sheet `Training` (Step 2 Recipe Ablation)
Required for acceptance criteria R2. Must contain baseline `T00`, single-factor ablations across $\ge 3$ axes, and $\ge 1$ combination experiment.
- **Columns (10 columns)**:
  1. `exp_id` (e.g. `T00`, `T01`, `T02`, `T03`, `T04`, `T05`, `T06`, `T07`)
  2. `backbone` (the chosen optimal backbone from Step 1)
  3. `trục thay đổi (A–G)` (e.g. `-` for T00, `A` for Init, `B` for Aug, `C` for Loss, `D` for Sampler, `F` for EMA, `Combo` for combination)
  4. `khác T00 ở điểm nào` (explicit difference, e.g. `Mốc chuẩn T00`, `init="scratch"`, `loss="focal"`, `loss="ls" (eps=0.1)`, `aug="color"`, `mix="cutmix"`, `sampler="balanced"`, `ema_decay=0.999`, etc.)
  5. `seed` (`0`)
  6. `macro-F1 val`
  7. `top-1 val`
  8. `Δ so với T00` (difference in Macro-F1 relative to T00, formatted e.g. `+0.0125` or `-0.0050`)
  9. `F1 các lớp hiếm (nếu có)` (Macro-F1 of rare weed species: Chinee Apple and Snake Weed)
  10. `ghi chú` (analysis relative to standard error: note whether $\Delta < \text{std} \approx 0.003$ is noise)

#### 3. Complete List of All Sheets in `results.xlsx` (GUIDE.md Section 6.1)
For maximum grading compliance (RUBRIC.md section E, 8 points), the Excel file should contain all 7 sheets:
1. `Backbones`: Comparative table of $\ge 5$ backbones.
2. `Training`: Ablation table on chosen backbone.
3. `Inference`: Step 3 inference techniques (TTA, ensemble, temperature scaling, BN fusion).
4. `Final`: Step 4 multi-seed test results (`F01..F03` + `T00`, test Macro-F1, Top-1, ECE, mean $\pm$ std).
5. `PerClass`: Per-class precision, recall, F1 on test set.
6. `Latency`: Benchmark table (batch 1, batch 32, FP32, AMP, FP16, p50, p95, p99, img/s).
7. `Summary`: One-page executive summary table highlighting top-10 models and cost trade-offs.

---

## 5. Deep Dive: Curve & Checkpoint Saving

### 5.1 Run Output Hierarchy
In `code/train.py`, directory structure adheres strictly to requirements:
```
runs/
└── <exp_id>/
    └── seed<seed>/              # e.g., runs/B01/seed0/, runs/T00/seed0/
        ├── config.json          # dataclass serialized configuration
        ├── best_checkpoint.pt   # state_dict, epoch, metrics, cfg
        ├── val_logits.npy       # raw validation logits (N x 9)
        ├── test_logits.npy      # raw test logits (only if save_test_predictions=True)
        ├── history.csv          # epoch-by-epoch loss, metrics, lr, time
        └── summary.json         # final metrics summary
predictions/
├── <exp_id>_seed<seed>_val.csv  # format compliant with eval.py
└── <exp_id>_seed<seed>_test.csv # only generated when save_test_predictions=True
curves/
└── <exp_id>_<backbone>.png     # e.g., curves/B01_resnet50.png, curves/T00_resnet50.png
```

### 5.2 Curve Plot Specifications
The function `plot_curves(history, path, title)` in `code/train.py` (lines 273–311) produces publication-quality charts:
- **Dimensions**: `figsize=(9, 5.5)`, `dpi=150`.
- **Dual Y-Axes**:
  - **Left Y-Axis (Loss)**:
    - Train Loss: Red dashed line (`#e74c3c`, `linestyle="--"`, marker `o`, markersize 4).
    - Val Loss: Dark red solid line (`#c0392b`, `linewidth=2`, marker `s`, markersize 4).
  - **Right Y-Axis (Metrics %)**:
    - Val Macro-F1: Solid blue line (`#2980b9`, `linewidth=2.5`, marker `^`, markersize 5).
    - Val Top-1 Accuracy: Green dash-dot line (`#27ae60`, `linestyle="-."`, marker `d`, markersize 4).
- **Styling**: Distinct colors, readable markers, formatted grid (`alpha=0.6, linestyle=":"`), unified legend in `center right`, bold title displaying experiment ID, backbone, init, loss, aug, and seed.
- **Naming Convention**:
  - For Step 1: `curves/B0x_<backbone>.png` (e.g. `B01_resnet50.png`, `B02_convnext_tiny.png`).
  - For Step 2: `curves/T0x_<backbone>.png` (e.g. `T00_resnet50.png`, `T01_resnet50.png`).
  - *Note*: If the user prefers `curves/T0x_<mota>.png` (e.g. `T01_scratch.png`), a descriptive tag can easily be appended to `exp_id` or `curve_path`.

---

## 6. Execution & Verification Findings

To ensure all claims are empirically grounded:
1. **Unit Tests**:
   - Command: `python -m unittest discover -s tests -v`
   - Result: **38 tests passed in 0.896s**.
   - Validates metric consistency between `eval.py` and `scikit-learn`, prediction CSV formatting contracts, CLI argument parsing, and starter contract compliance.
2. **Pipeline Sanity Check**:
   - Command: `$env:PYTHONUTF8=1; python code/run_step0_checks.py`
   - Result: **All Step 0 checks passed on NVIDIA GeForce RTX 5060 Ti GPU**.
   - Verified split integrity (10,501 train, 3,501 val, 3,507 test; zero overlap; all 17,509 images present).
   - Verified Focal loss ($\gamma=0$) $\equiv$ CE (error $= 0.0$).
   - Verified Label smoothing ($\epsilon=0$) $\equiv$ CE (error $= 0.0$).
   - Verified CutMix actual area lambda clipping.
   - Verified 3 parameter groups in ResNet50 (decay=53, no-decay=106, head=2).
   - Verified freeze backbone keeps BatchNorm modules in eval mode.
   - Verified small-batch overfit (loss $< 0.000012$ in 60 steps).
   - Verified end-to-end forward/backward optimization step.
3. **Hardware & Environment**:
   - Python: `3.12.10` (Windows 64-bit).
   - PyTorch: `2.12.0+cu132`.
   - Torchvision: `0.27.0+cu132`.
   - GPU: `NVIDIA GeForce RTX 5060 Ti` (CUDA available).
   - Required libraries `timm` (1.0.30) and `openpyxl` (3.1.5) are installed and verified.

---

## 7. Recommendations for Implementation Phase

1. **Excel Generation Automation**:
   - Create a lightweight utility script (e.g., `code/update_excel.py`) that reads all `runs/<exp_id>/seed0/summary.json` files and updates `results.xlsx` with formatted sheets (`Backbones`, `Training`, `Latency`, `Summary`).
2. **Ablation CLI Safety for Loss Axis**:
   - When executing `loss="ls"`, instruct agents or shell scripts to explicitly provide `--set exp_id=T0x loss=ls label_smoothing=0.1` to prevent silent fallback to standard cross-entropy.
3. **EMA Checkpoint Clarity**:
   - When running the EMA ablation (`ema_decay=0.999`), note that `summary.json` and curves record the EMA validation score, whereas `best_checkpoint.pt` holds un-shadowed weights. For Step 3 inference on EMA models, load shadow weights or evaluate directly.
4. **Execution Sequence Recommendation**:
   - **Step 1 (Backbones B01–B05)**: Run `resnet50` (B01), `convnext_tiny` (B02), `swin_tiny` or `vit_small` (B03), `efficientnet_b0` (B04), `mobilenetv3_large_100` (B05). Measure latency with `code/benchmark.py`. Select the optimal trade-off backbone.
   - **Step 2 (Training Ablations T00–T07)**:
     - T00: Baseline (Finetune, Basic Aug, CE, Standard Sampler, No EMA).
     - Axis A: T01 (`init=scratch`), T02 (`init=frozen`).
     - Axis B: T03 (`aug=color`), T04 (`aug=trivial`), T05 (`mix=cutmix`).
     - Axis C: T06 (`loss=ls label_smoothing=0.1`), T07 (`loss=focal focal_gamma=2.0`).
     - Axis D: T08 (`sampler=balanced`).
     - Axis F: T09 (`ema_decay=0.999`).
     - Best Combination: T10 (Combines top-performing orthogonal factors).
