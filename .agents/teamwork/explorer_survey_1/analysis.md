# Comprehensive Architecture, Environment & Benchmark Survey Report
**Project:** DeepWeeds Deep Learning Day 2 Lab  
**Author:** Explorer 1 (Architecture & Benchmark Explorer)  
**Date:** 2026-10-03  
**Status:** Completed  

---

## 1. Executive Summary

This investigation analyzed the execution environment, model architecture support in `code/model.py`, benchmarking harness in `code/benchmark.py`, training engine in `code/train.py`, and the evaluation protocol in `eval.py` for the DeepWeeds Day 2 Lab.

### Key Takeaways
1. **Hardware & Environment**:
   - Python: **3.12.10**, PyTorch: **2.12.0+cu132**, CUDA: **13.2 runtime**, GPU: **NVIDIA GeForce RTX 5060 Ti (17.10 GB VRAM)**.
   - Dependencies: `timm` (v1.0.30) and `openpyxl` (v3.1.5) were initially absent from the environment and have been installed.
   - **Critical OS finding**: On Windows, PowerShell subprocesses default to CP1252 encoding. Vietnamese console logs in `dataset.py`, `train.py`, and `eval.py` crash with `UnicodeEncodeError: 'charmap'` unless **`$env:PYTHONUTF8=1`** is set.
2. **Backbone Readiness (R1)**:
   - All five required backbone models across 4 architectural families are **fully supported out-of-the-box**:
     - `B01` ResNet: `resnet50` (Params: 23.526M, GMACs: 4.087, tag: `a1_in1k`)
     - `B02` ConvNeXt: `convnext_tiny` (Params: 27.827M, GMACs: 4.455, tag: `in12k_ft_in1k`)
     - `B03` Transformer: `swin_tiny_patch4_window7_224` (Params: 27.526M, GMACs: 4.350, tag: `ms_in1k`) and `vit_small_patch16_224` (Params: 21.669M, GMACs: 4.241, tag: `augreg_in21k_ft_in1k`)
     - `B04` Lightweight: `efficientnet_b0` (Params: 4.019M, GMACs: 0.385, tag: `ra_in1k`)
     - `B05` Lightweight: `mobilenetv3_large_100` (Params: 4.214M, GMACs: 0.215, tag: `ra_in1k`)
   - Pretrained weights for all 6 candidate backbones have been successfully downloaded and cached locally in `~/.cache/huggingface/hub/`.
3. **Benchmarking Accuracy (R1 / R4)**:
   - `code/benchmark.py` strictly adheres to slide rules: ≥10 warmup runs, explicit `torch.cuda.synchronize()` before and after every measurement, percentile reporting (p50, p95, p99, mean, std), batch size 1, resolution 224×224.
   - Measured batch-1 GPU latencies on RTX 5060 Ti are well under the 100 ms real-time ceiling (all p95 < 10 ms).
   - Inverted residual models (`efficientnet_b0`, `mobilenetv3_large_100`) have very low GMACs (0.2–0.4) but batch-1 GPU latencies comparable to `resnet50` (p50 4–5 ms) due to memory bandwidth and kernel launch overheads, validating slide 43 ("FLOPs is not latency").
4. **Evaluation & Data Leakage Protection (R1 / R2 / R4)**:
   - `code/train.py` integrates seamlessly with `eval.py`: checkpoint selection is driven strictly by **Val Macro-F1**, unweighted across all 9 classes.
   - `save_test_predictions` is `False` by default (Rule S4), guaranteeing zero test leakage during Steps 1 & 2.
   - Dataset fold 0 integrity verified: 10,501 train, 3,501 val, 3,507 test; mutual overlaps are empty; all 17,509 images exist on disk.

---

## 2. Environment & Dependency Audit

| Component | Detected Specification | Verification Command / Source | Status |
|---|---|---|---|
| OS | Windows (AMD64) | `platform.platform()` | Compatible |
| Python | 3.12.10 | `sys.version` | Supported |
| PyTorch | 2.12.0+cu132 | `torch.__version__` | Supported |
| CUDA Runtime | CUDA 13.2 | `torch.version.cuda` | Active |
| GPU Device | NVIDIA GeForce RTX 5060 Ti | `torch.cuda.get_device_name(0)` | Active |
| VRAM | 17.10 GB total memory | `torch.cuda.get_device_properties(0)` | Ample for BS=64 |
| `torchvision` | 0.27.0+cu132 | `torchvision.__version__` | Installed |
| `timm` | 1.0.30 | `pip list` / `import timm` | Installed & Verified |
| `openpyxl` | 3.1.5 | `pip list` / `import openpyxl` | Installed & Verified |
| `pandas` | 2.2.0 | `pandas.__version__` | Installed |
| `numpy` | 1.26.4 | `numpy.__version__` | Installed |
| `scipy` | 1.12.0 | `scipy.__version__` | Installed |
| `scikit-learn` | 1.4.2 | `sklearn.__version__` | Installed |
| `matplotlib` | 3.11.1 | `matplotlib.__version__` | Installed |
| `thop` | 0.1.1-2209072238 | `thop.__version__` | Installed |

### Critical Environment Note
- **Windows UTF-8 Encoding**: The Python test and execution commands must run with `$env:PYTHONUTF8=1` in PowerShell. Without this, standard library print calls containing Vietnamese diacritics raise `UnicodeEncodeError: 'charmap' codec can't encode character`. All 38 unittests in `tests/` pass with exit code 0 when `PYTHONUTF8=1` is active.

---

## 3. Architecture Deep Dive (`code/model.py`)

### 3.1 Model Creation & Mapping
In `code/model.py`:
- `SUGGESTED_BACKBONES` dictionary defines default architecture aliases:
  ```python
  SUGGESTED_BACKBONES = {
      "resnet50": "resnet50",
      "resnext50": "resnext50_32x4d",
      "convnext_tiny": "convnext_tiny",
      "deit_small": "deit_small_patch16_224",
      "swin_tiny": "swin_tiny_patch4_window7_224",
      "efficientnet_b0": "efficientnet_b0",
      "mobilenetv3": "mobilenetv3_large_100",
  }
  ```
- `build_model(name, pretrained=True, num_classes=9, drop_rate=0.0, init="finetune")`:
  - `model_name = SUGGESTED_BACKBONES.get(name, name)` resolves aliases or passes canonical timm model names directly.
  - Automatically modifies classifier head for 9 classes via `timm.create_model(..., num_classes=9)`.
  - Attaches `model._init_mode`, `model._backbone_name`, and `model.pretrained_tag`.
  - Fallback logic: includes torchvision fallback for `resnet50` if timm encounters an error.

### 3.2 5-Backbone Candidate Suite (B01..B05)
All 5 required backbones were instantiated, initialized with pretrained weights, and analyzed:

| Exp ID | Architecture Family | Candidate Name (`timm`) | Weight Tag | Params (M) | GMACs (224×224) | Classifier Module |
|---|---|---|---|---|---|---|
| **B01** | ResNet (Baseline) | `resnet50` | `a1_in1k` | 23.526 | 4.087 | `nn.Linear` (2048 → 9) |
| **B02** | ConvNeXt | `convnext_tiny` | `in12k_ft_in1k` | 27.827 | 4.455 | `nn.Linear` (768 → 9) |
| **B03a**| Transformer (Swin) | `swin_tiny_patch4_window7_224` | `ms_in1k` | 27.526 | 4.350 | `nn.Linear` (768 → 9) |
| **B03b**| Transformer (ViT) | `vit_small_patch16_224` | `augreg_in21k_ft_in1k` | 21.669 | 4.241 | `nn.Linear` (384 → 9) |
| **B04** | Lightweight | `efficientnet_b0` | `ra_in1k` | 4.019 | 0.385 | `nn.Linear` (1280 → 9) |
| **B05** | Lightweight | `mobilenetv3_large_100` | `ra_in1k` | 4.214 | 0.215 | `timm.layers.linear.Linear` (960 → 9) |

*Note on parameter counting in `resnet50`*: Standard ImageNet-1K ResNet-50 has 25.56M parameters with a 1000-class head. With a 9-class head (replacing $2048 \times 1000$ with $2048 \times 9$), parameters drop by $\sim 2.03\text{M}$ to **23.53M**, exactly matching `count_params(m) = 23.526M`.

### 3.3 Initialization Modes (Ablation Axis A)
- `init="scratch"`: `pretrained=False`, all layers trained from random initialization.
- `init="frozen"`: `pretrained=True`, `freeze_backbone(model)` is invoked:
  - Sets `param.requires_grad = False` for all backbone parameters.
  - Keeps classifier head `requires_grad = True`.
  - Sets all `BatchNorm` layers to `eval()`.
  - Monkey-patches `model.train()` to force BatchNorm to remain in `eval()` mode during training, preventing corrupting running mean and variance on small/frozen batches.
- `init="finetune"`: `pretrained=True`, all layers trainable with differential learning rates.

### 3.4 3-Way Parameter Groups (`model.param_groups`)
Implements slide 52 requirements:
1. `backbone_decay`: Backbone tensors with `ndim > 1` (convolutional kernels, linear weight matrices): `lr = lr_backbone` ($1\times 10^{-4}$), `weight_decay = 0.05`.
2. `backbone_no_decay`: Backbone tensors with `ndim <= 1` (biases, batch norm scales/shifts, layer norm): `lr = lr_backbone` ($1\times 10^{-4}$), `weight_decay = 0.0`.
3. `head`: Classifier head weights and biases: `lr = lr_head` ($1\times 10^{-3}$), `weight_decay = 0.05`.

---

## 4. Benchmark Harness Deep Dive (`code/benchmark.py`)

### 4.1 Methodology & Adherence to Strict Protocols
`code/benchmark.py` implements latency benchmarking via `bench()` and `latency_report()`:
- **Warmup**: Executes ≥10 forward passes before timing starts.
- **CUDA Synchronization**: Calls `torch.cuda.synchronize()` immediately prior to `time.perf_counter()` and immediately following execution before reading the stop time.
- **Sample Size**: Runs 50–100 measured iterations (`iters >= 50`).
- **Statistics**: Computes percentiles `p50`, `p95`, `p99`, `mean`, `std`, and throughput `images_per_s`.
- **Inference Mode**: Uses `torch.inference_mode()` with `model.eval()`.
- **Precision Modes**: Supports `"fp32"`, `"amp"`, and `"fp16"`.

### 4.2 Measured Latencies on RTX 5060 Ti (Batch Size = 1, Resolution = 224×224)

| Model Name | Exp ID | FP32 p50 (ms) | FP32 p95 (ms) | AMP p50 (ms) | AMP p95 (ms) | Real-time Budget (<100 ms) |
|---|---|---|---|---|---|---|
| `resnet50` | B01 | 4.05 | 4.87 | 4.96 | 5.77 | PASS (p95 = 4.87 ms) |
| `convnext_tiny` | B02 | 3.55 | 4.31 | 5.53 | 6.22 | PASS (p95 = 4.31 ms) |
| `swin_tiny_patch4_window7_224` | B03 | 6.87 | 9.61 | 8.62 | 9.40 | PASS (p95 = 9.61 ms) |
| `vit_small_patch16_224` | B03 alt | 3.26 | 3.86 | 4.36 | 5.23 | PASS (p95 = 3.86 ms) |
| `efficientnet_b0` | B04 | 5.19 | 6.16 | 7.11 | 9.57 | PASS (p95 = 6.16 ms) |
| `mobilenetv3_large_100` | B05 | 4.12 | 4.93 | 5.61 | 7.41 | PASS (p95 = 4.93 ms) |

### 4.3 Critical Observations on Latency vs GMACs
1. **The FLOPs Paradox**: Although `mobilenetv3_large_100` has only 0.215 GMACs (vs 4.087 GMACs for `resnet50` — a $19\times$ reduction), its batch-1 GPU latency is 4.12 ms vs 4.05 ms for ResNet-50. This confirms slide 43: depthwise separable convolutions are memory-bound rather than compute-bound on modern high-throughput GPUs.
2. **ConvNeXt vs ResNet**: `convnext_tiny` is slightly faster (p50: 3.55 ms) than `resnet50` (p50: 4.05 ms) despite having more parameters (27.8M vs 23.5M) and slightly higher GMACs (4.45 vs 4.09), thanks to $7\times 7$ depthwise convolutions with fewer total stages.
3. **Transformers**: `vit_small_patch16_224` achieves outstanding batch-1 latency (3.26 ms) because of straightforward matrix multiplications, whereas `swin_tiny` experiences overhead from shifted window partitioning (6.87 ms).

---

## 5. Training Engine & Evaluation Pipeline (`code/train.py`, `eval.py`)

### 5.1 Metric Computation Formulas
- **Top-1 Accuracy**:
  $$\text{Top-1} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(y_i = \hat{y}_i)$$
  In `eval.py:268`: `(y_true == y_pred).mean()`. Unweighted across classes; biased towards class 8 (`Negatives` = 52% of dataset).
- **Per-Class Precision, Recall, and F1**:
  $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
  In `eval.py:234-242`: Handled with exact zero-division safeguarding.
- **Macro-F1**:
  $$\text{Macro-F1} = \frac{1}{9} \sum_{c=0}^8 F1_c$$
  In `eval.py:268`: `pc["f1"].mean()`. Crucial for DeepWeeds due to severe class imbalance. Rare classes like Chinee Apple and Snake Weed carry equal weight to Negatives.

### 5.2 Model Selection and Data Integrity Protocol
- **Selection Criterion**: `train.py:432` evaluates `val_macro_f1 > best_macro_f1`. Checkpoints (`best_checkpoint.pt`) and validation predictions are saved **exclusively based on Val Macro-F1**.
- **Zero Test Leakage**:
  - `Config.save_test_predictions` is initialized to `False` (`train.py:96`).
  - Lines 458-474 only evaluate and export `test_subset0.csv` predictions when explicitly requested at Step 4.
  - During Steps 1 and 2, test data is never evaluated or used for early stopping/tuning.
- **Artifacts Generated Per Run**:
  - `runs/<exp_id>/seed<k>/config.json`
  - `runs/<exp_id>/seed<k>/history.csv` (epoch-level metrics)
  - `runs/<exp_id>/seed<k>/best_checkpoint.pt`
  - `runs/<exp_id>/seed<k>/val_logits.npy`
  - `predictions/<exp_id>_seed<k>_val.csv` (formatted for `eval.py`)
  - `curves/<exp_id>_<backbone>.png` (plots train/val loss and val macro-F1 / top-1)

---

## 6. Recommendations for Steps 1 & 2 Execution

1. **Environment Flag**: Always execute Python commands with `$env:PYTHONUTF8=1` in the PowerShell terminal session to ensure clean logging and metric aggregation.
2. **Backbone Recommendation for Step 1 (B01..B05)**:
   - Run B01 (`resnet50`), B02 (`convnext_tiny`), B03 (`vit_small_patch16_224` or `swin_tiny_patch4_window7_224`), B04 (`efficientnet_b0`), and B05 (`mobilenetv3_large_100`).
   - For B03, `convnext_tiny` and `vit_small_patch16_224` are especially promising given their low batch-1 latency on the RTX 5060 Ti.
3. **Excel Reporting**:
   - `openpyxl` is installed. When recording to `results.xlsx`, populate sheet `Backbones` and sheet `Training` using `pd.ExcelWriter("results.xlsx", engine="openpyxl")`.
4. **Step 2 Ablation Strategy**:
   - Focus on:
     - **Axis A (Init)**: `init="scratch"` vs `init="frozen"` vs `init="finetune"` (T00).
     - **Axis B (Aug)**: `aug="basic"` (T00) vs `aug="color"` vs `aug="trivial"` or `mix="cutmix"`.
     - **Axis C (Loss)**: `loss="ce"` (T00) vs `loss="ls"` vs `loss="focal"`.
     - **Axis D (Sampling)**: `sampler=None` vs `sampler="balanced"`.
     - **Axis F (EMA)**: `ema_decay=None` vs `ema_decay=0.999`.

---
*Report compiled and verified by Explorer 1.*
