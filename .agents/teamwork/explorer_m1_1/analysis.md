# Milestone 1: Training Plan & Execution Feasibility Analysis

**Author**: Explorer 1 (`explorer_m1_1`)  
**Target Milestone**: M1 (Backbone Comparison B01–B05 under Baseline Recipe T00)  
**Execution Environment**: Windows 11, Python 3.12, PyTorch 2.12.0+cu132, CUDA 13.2  
**Hardware Target**: NVIDIA GeForce RTX 5060 Ti (15.93 GB dedicated VRAM / 17.1 GB total addressable)  
**Date**: 2026-10-03  

---

## 1. Executive Summary

Milestone 1 evaluates 5 backbone architectures covering 4 required architectural paradigms on the DeepWeeds 9-class weed classification task using baseline recipe **T00** on Fold 0.
Through empirical GPU benchmarking and codebase inspection, we confirm:
1. **Zero OOM Risk**: The peak memory reserved at batch size 64 across all 5 models is **5.20 GB** (`swin_tiny`), which utilizes only **31.9%** of available VRAM on the RTX 5060 Ti GPU. No reduction in batch size is necessary.
2. **Total Execution Runtime**: Training all 5 models sequentially for 12 epochs each requires **~25 to 31 minutes** total (~1.5–8.4 minutes per model).
3. **Execution Commands**: Training commands are precisely mapped to `code/train.py` using `--set` key-value overrides.
4. **Critical Windows Multiprocessing Discovery**: In `code/dataset.py`, `_worker_init_fn` is defined as a local closure inside `make_loader`. On Windows (`spawn` mode), this causes `AttributeError: Can't get local object 'make_loader.<locals>._worker_init_fn'` when `num_workers > 0`. We provide both a zero-code CLI remedy (`num_workers=0`) and a 4-line patch (`proposed_dataset_worker_init.patch`).

---

## 2. Model Architecture & Pretrained Weight Verification

The 5 required backbones cover the 4 architectural families specified in `ORIGINAL_REQUEST §R1` and `GUIDE.md §2.1`:

| Exp ID | Architecture Family | Backbone Identifier (`timm`) | Pretrained Tag (`timm 1.0.30`) | Parameters (M) | GMACs (224x224) |
|---|---|---|---|---|---|
| **B01** | ResNet (Standard Baseline) | `resnet50` | `a1_in1k` | 23.526 M | 4.087 |
| **B02** | ConvNeXt / ResNeXt | `convnext_tiny` | `in12k_ft_in1k` | 27.827 M | 4.455 |
| **B03** | Vision Transformer | `swin_tiny_patch4_window7_224` | `ms_in1k` | 27.526 M | 4.350 |
| **B04** | Lightweight CNN | `efficientnet_b0` | `ra_in1k` | 4.019 M | 0.385 |
| **B05** | Lightweight Mobile | `mobilenetv3_large_100` | `ra_in1k` | 4.214 M | 0.215 |

*Notes on weights*: All 5 models are verified to load successfully with weights cached locally. Parameter counts and GMACs match the calculations in `code/model.py` (`count_params`, `count_gmacs`).

---

## 3. Precise Training Commands Formulation (Recipe T00)

### 3.1 Baseline Hyperparameters (Recipe T00)
- **Optimizer**: AdamW
- **Learning Rate**: Differential LR — Backbone = `1e-4` (0.0001), Classifier Head = `1e-3` (0.001)
- **Weight Decay**: `0.05` (with zero weight decay on biases and 1D normalization layers via `param_groups`)
- **LR Schedule**: 1 epoch linear warmup followed by Cosine Annealing (decaying toward `1e-4`)
- **Loss Function**: Standard Cross-Entropy (`ce`)
- **Mixed Precision**: Automatic Mixed Precision (`amp=True`)
- **Batch Size**: 64
- **Epochs**: 12
- **Seed**: 0
- **Fold**: 0 (`train_subset0.csv`, `val_subset0.csv`)
- **Image Size**: 224x224
- **Augmentation**: Basic (`RandomResizedCrop(224)` + `RandomHorizontalFlip` for train; `Resize(256)` + `CenterCrop(224)` for val)
- **Test Set Protection**: `save_test_predictions=False` (Strict zero leakage)

### 3.2 Individual Training Invocations

All commands must be executed in PowerShell with `$env:PYTHONUTF8=1` from repository root:

#### B01: ResNet-50
```powershell
$env:PYTHONUTF8=1; python code/train.py --set exp_id=B01 backbone=resnet50 seed=0 fold=0 epochs=12 batch_size=64 lr_backbone=1e-4 lr_head=1e-3 weight_decay=0.05 warmup_epochs=1.0 loss=ce amp=true num_workers=0
```

#### B02: ConvNeXt-Tiny
```powershell
$env:PYTHONUTF8=1; python code/train.py --set exp_id=B02 backbone=convnext_tiny seed=0 fold=0 epochs=12 batch_size=64 lr_backbone=1e-4 lr_head=1e-3 weight_decay=0.05 warmup_epochs=1.0 loss=ce amp=true num_workers=0
```

#### B03: Swin-Tiny (swin_tiny_patch4_window7_224)
```powershell
$env:PYTHONUTF8=1; python code/train.py --set exp_id=B03 backbone=swin_tiny_patch4_window7_224 seed=0 fold=0 epochs=12 batch_size=64 lr_backbone=1e-4 lr_head=1e-3 weight_decay=0.05 warmup_epochs=1.0 loss=ce amp=true num_workers=0
```

#### B04: EfficientNet-B0
```powershell
$env:PYTHONUTF8=1; python code/train.py --set exp_id=B04 backbone=efficientnet_b0 seed=0 fold=0 epochs=12 batch_size=64 lr_backbone=1e-4 lr_head=1e-3 weight_decay=0.05 warmup_epochs=1.0 loss=ce amp=true num_workers=0
```

#### B05: MobileNetV3-Large (mobilenetv3_large_100)
```powershell
$env:PYTHONUTF8=1; python code/train.py --set exp_id=B05 backbone=mobilenetv3_large_100 seed=0 fold=0 epochs=12 batch_size=64 lr_backbone=1e-4 lr_head=1e-3 weight_decay=0.05 warmup_epochs=1.0 loss=ce amp=true num_workers=0
```

### 3.3 Sequential Pipeline Script (Automated M1 Runner)
To execute all 5 models in sequence automatically without manual intervention:
```powershell
$env:PYTHONUTF8=1
$models = @(
    @{exp="B01"; bb="resnet50"},
    @{exp="B02"; bb="convnext_tiny"},
    @{exp="B03"; bb="swin_tiny_patch4_window7_224"},
    @{exp="B04"; bb="efficientnet_b0"},
    @{exp="B05"; bb="mobilenetv3_large_100"}
)

foreach ($m in $models) {
    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host "STARTING $($m.exp) ($($m.bb)) under Recipe T00" -ForegroundColor Cyan
    Write-Host "========================================================`n" -ForegroundColor Cyan
    python code/train.py --set exp_id=$($m.exp) backbone=$($m.bb) seed=0 num_workers=0
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Training failed for $($m.exp)! Halting."
        break
    }
}
```

---

## 4. Hardware Profiling: VRAM Feasibility & OOM Analysis

### 4.1 Empirical GPU Memory Measurements
Target Hardware: NVIDIA GeForce RTX 5060 Ti (Dedicated VRAM: 15,928 MB / 15.93 GB; Total Addressable: 17.1 GB).  
Batch Size: 64, Input Resolution: 224x224, Precision: PyTorch AMP (`torch.autocast('cuda')`).

Measurements taken during forward-backward iterations with gradient clipping, AdamW optimizer step, and GradScaler updates:

| Exp ID | Backbone | Peak Allocated VRAM | Peak Reserved VRAM | Peak VRAM % of 15.93 GB | Headroom Remaining | OOM Risk Assessment |
|---|---|---|---|---|---|---|
| **B01** | `resnet50` | 3,132.3 MB (~3.06 GB) | 3,606.0 MB (~3.52 GB) | 22.1% | 12.32 GB | **ZERO RISK** |
| **B02** | `convnext_tiny` | 4,296.6 MB (~4.20 GB) | 4,536.0 MB (~4.43 GB) | 27.8% | 11.39 GB | **ZERO RISK** |
| **B03** | `swin_tiny_patch4_window7_224` | 4,785.6 MB (~4.67 GB) | 5,200.0 MB (~5.08 GB) | 31.9% | 10.73 GB | **ZERO RISK** |
| **B04** | `efficientnet_b0` | 3,061.9 MB (~2.99 GB) | 3,318.0 MB (~3.24 GB) | 20.3% | 12.61 GB | **ZERO RISK** |
| **B05** | `mobilenetv3_large_100` | 1,556.5 MB (~1.52 GB) | 1,944.0 MB (~1.90 GB) | 11.9% | 13.98 GB | **ZERO RISK** |

### 4.2 OOM Risk Conclusion
- Highest memory consumer is Swin-Tiny at **5.20 GB** reserved VRAM.
- Dedicated hardware has **15.93 GB** VRAM, leaving **over 68% safety margin**.
- **No backbone is at risk of OOM at batch size 64**. Batch size 64 should be strictly maintained as specified in `ORIGINAL_REQUEST §R1`.

---

## 5. Training Time & Throughput Estimation

### 5.1 Dataset Specifications
- Training split (`train_subset0.csv`): 10,501 images = 165 batches of 64.
- Validation split (`val_subset0.csv`): 3,501 images = 55 batches of 64.
- Disk & DataLoader throughput measured on host SSD with `num_workers=0`: **527.1 images/sec** (~19.9s pure dataset loading time per epoch).

### 5.2 Estimated Epoch & Total Runtime

| Model | Train Step Time (ms/step) | 165 Train Steps (s) | 55 Val Steps (s) | Est. Total Time / Epoch | Est. Total Runtime (12 Epochs) |
|---|---|---|---|---|---|
| **B01** `resnet50` | 152.3 ms | 25.1 s | 4.5 s | **~33 - 38 s** | **~6.6 - 7.6 min** |
| **B02** `convnext_tiny` | 137.8 ms | 22.7 s | 3.8 s | **~29 - 35 s** | **~5.8 - 7.0 min** |
| **B03** `swin_tiny` | 176.2 ms | 29.1 s | 5.5 s | **~37 - 42 s** | **~7.4 - 8.4 min** |
| **B04** `efficientnet_b0` | 78.7 ms | 13.0 s | 2.5 s | **~18 - 24 s** | **~3.6 - 4.8 min** |
| **B05** `mobilenetv3_large` | 46.5 ms | 7.7 s | 1.8 s | **~12 - 16 s** | **~2.4 - 3.2 min** |
| **TOTAL (5 Backbones)** | — | — | — | — | **~25.8 - 31.0 min** |

All 5 backbones can be trained to completion sequentially within **half an hour**.

---

## 6. Inference Latency & Computational Profile

Using `code/benchmark.py` (`latency_report`) with 15 warmup cycles and 100 benchmark iterations with CUDA synchronization:

| Exp ID | Model | FP32 Latency p50 | FP32 Latency p95 | FP32 Mean Latency | FP32 Throughput | AMP Latency p50 |
|---|---|---|---|---|---|---|
| **B01** | `resnet50` | 5.34 ms | 6.26 ms | 5.41 ms | 187.3 img/s | 4.95 ms |
| **B02** | `convnext_tiny` | **3.41 ms** | **4.63 ms** | **3.63 ms** | **293.3 img/s** | 5.72 ms |
| **B03** | `swin_tiny_patch4_window7_224` | 7.13 ms | 10.85 ms | 7.44 ms | 140.3 img/s | 9.29 ms |
| **B04** | `efficientnet_b0` | 5.37 ms | 7.04 ms | 5.63 ms | 186.2 img/s | 7.70 ms |
| **B05** | `mobilenetv3_large_100` | 4.16 ms | 5.66 ms | 4.42 ms | 240.4 img/s | 5.55 ms |

### Architectural Insight:
- `convnext_tiny` achieves the lowest batch-1 latency (**3.41 ms**, 293 img/s), faster even than lightweight models `mobilenetv3` (4.16 ms) and `efficientnet_b0` (5.37 ms), due to depthwise 7x7 conv kernels and high GPU core saturation on modern Ada/Blackwell tensor architectures.
- In agreement with `GUIDE.md §4.1`, at batch=1 AMP overhead makes FP32 faster than AMP for almost all backbones (`convnext_tiny` FP32 3.41 ms vs AMP 5.72 ms).

---

## 7. Artifact Directory Structure & Output Verification

Inspection of `code/train.py` confirms exact compliance with project contracts:

### 7.1 Checkpoint Directory: `runs/<exp_id>/seed0/`
For each run `B0x`, the following files are produced and saved:
- `runs/B0x/seed0/config.json`: Complete dump of dataclass `Config`.
- `runs/B0x/seed0/best_checkpoint.pt`: Checkpoint state containing `epoch`, `model_state_dict`, `macro_f1`, `top1`, and `cfg`.
- `runs/B0x/seed0/val_logits.npy`: Saved raw logits array of shape `(3501, 9)` for downstream Step 3 inference experiments (temperature scaling, ensembling, probability averaging) without requiring GPU re-execution.
- `runs/B0x/seed0/history.csv`: Per-epoch log containing `epoch`, `train_loss`, `val_loss`, `val_macro_f1`, `val_top1`, `lr`, `epoch_time_s`.
- `runs/B0x/seed0/summary.json`: Final experiment summary dictionary.

### 7.2 Curve Plotting: `curves/<exp_id>_<backbone>.png`
- File naming dynamically generated at line 479 of `code/train.py`:
  - `curves/B01_resnet50.png`
  - `curves/B02_convnext_tiny.png`
  - `curves/B03_swin_tiny_patch4_window7_224.png`
  - `curves/B04_efficientnet_b0.png`
  - `curves/B05_mobilenetv3_large_100.png`
- Layout: Dual-axis plot with Train Loss (dashed red), Val Loss (solid red), Val Macro-F1 (solid blue, primary metric), and Val Top-1 Accuracy (dash-dot green).

### 7.3 Predictions Directory: `predictions/`
- `predictions/B0x_seed0_val.csv` saved via `eval.save_predictions` for validation evaluation.

---

## 8. Windows Multiprocessing Compatibility Finding & Proposed Patch

### 8.1 Issue Description
In `code/dataset.py` lines 249-253:
```python
def make_loader(df, images_dir, transform, batch_size, train, sampler=None, num_workers=2):
    ...
    def _worker_init_fn(worker_id: int):
        worker_seed = (torch.initial_seed() + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    ...
    return DataLoader(..., worker_init_fn=_worker_init_fn)
```
On Windows, multiprocessing uses `spawn` instead of `fork`. When passing `_worker_init_fn` to worker processes, Python's `pickle` fails because `_worker_init_fn` is defined as a closure inside `make_loader`. This produces:
`AttributeError: Can't get local object 'make_loader.<locals>._worker_init_fn'`.

### 8.2 Solutions
1. **Immediate Execution Workaround (Zero code edit)**:
   Add `num_workers=0` to `--set` in the training command. This avoids multiprocessing spawn and runs data loading synchronously in the main thread at **527 images/second** with zero error.
2. **Permanent Clean Patch**:
   Move `_worker_init_fn` to module scope in `code/dataset.py`.
   A patch file `proposed_dataset_worker_init.patch` has been written to `.agents/teamwork/explorer_m1_1/proposed_dataset_worker_init.patch`.

---

## 9. Zero Test-Set Leakage Protocol

Compliance with `README.md §2.1` (Rules S1–S6):
1. **Model Selection**: All decisions (backbone comparison in M1, recipe ablations in M2) are exclusively derived from `val_subset0.csv`.
2. **Test Isolation**: `test_subset0.csv` is never evaluated during M1. In `code/train.py`, `cfg.save_test_predictions` is strictly `False`.
3. **Partition Integrity**: Partitions are validated through `check_split`, ensuring zero overlap (`train ∩ val = 0`, `train ∩ test = 0`, `val ∩ test = 0`) and union equality to 17,509 images.
