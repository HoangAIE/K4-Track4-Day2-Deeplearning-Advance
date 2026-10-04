# Milestone 1 Benchmark & Complexity Extraction Analysis

**Author:** Explorer 2 (`explorer_m1_2`)  
**Date:** 2026-10-03  
**Target:** Milestone 1 (M1) — Backbone Comparison (B01–B05)  
**Hardware Profile:** NVIDIA GeForce RTX 5060 Ti (17,094 MiB VRAM), PyTorch 2.12.0+cu132, timm 1.0.30, CUDA Driver 13.2  

---

## 1. Executive Summary

This report establishes the verified empirical benchmarking methodology, architectural complexity measurements, latency profiling, and quantitative selection criteria for the 5 target backbones in Milestone 1 of the DeepWeeds Deep Learning Day 2 Lab.

### Key Empirical Findings:
1. **Model Complexity & Latency Profile (Batch 1, 224x224, CUDA Sync, Warmup >= 10)**:
   - **B01 (`resnet50`)**: 23.526M params, 4.087 GMACs, Pretrained Tag: `a1_in1k`, FP32 p50: **4.39 ms** (p95: 7.08 ms, p99: 9.65 ms).
   - **B02 (`convnext_tiny`)**: 27.827M params, 4.455 GMACs, Pretrained Tag: `in12k_ft_in1k`, FP32 p50: **3.53 ms** (p95: 4.51 ms, p99: 4.63 ms).
   - **B03 (`swin_tiny_patch4_window7_224`)**: 27.526M params, 4.350 GMACs, Pretrained Tag: `ms_in1k`, FP32 p50: **7.23 ms** (p95: 11.06 ms, p99: 15.65 ms).
   - **B04 (`efficientnet_b0`)**: 4.019M params, 0.385 GMACs, Pretrained Tag: `ra_in1k`, FP32 p50: **5.46 ms** (p95: 8.94 ms, p99: 12.66 ms).
   - **B05 (`mobilenetv3_large_100`)**: 4.214M params, 0.215 GMACs, Pretrained Tag: `ra_in1k`, FP32 p50: **4.50 ms** (p95: 6.19 ms, p99: 8.31 ms).

2. **The "FLOPs Is Not Latency" Disconnect (Slide 43)**:
   Although `mobilenetv3_large_100` requires **19× fewer GMACs** than `resnet50` (0.215 vs 4.087 GMACs), its batch-1 inference latency is actually **slower** on desktop/server GPU (`mobilenetv3_large_100`: 4.50 ms vs `resnet50`: 4.39 ms vs `convnext_tiny`: 3.53 ms). Depthwise separable convolutions exhibit low arithmetic intensity, resulting in memory-bandwidth-bound and kernel-launch-bound bottlenecks on high-performance GPUs.

3. **AMP Autocast Penalty at Batch Size 1 (Slide 73)**:
   At batch size 1, PyTorch AMP autocast (`torch.autocast("cuda")`) is consistently **slower** than native FP32 across all 5 architectures due to precision casting and runtime type-checking overheads that eclipse compute savings on tiny single-image tensors. Pure FP32 is strongly recommended for batch-1 robotic inference.

4. **Quantitative Decision Rule Established**:
   A 4-tier selection framework (Statistical Noise Gate $\Delta > 0.30\%$, Real-Time Budget $L_{\text{p95}} \le 100$ ms, Multi-Criteria Utility $U_{\text{tradeoff}}$, and Pareto Frontier) is formally defined to objectively select the winning backbone for Step 2.

---

## 2. Codebase & Script Verification

### 2.1 Inspection of `code/benchmark.py`
The existing implementation in `code/benchmark.py` provides three core functions:
- `bench(fn, warmup=10, iters=100, sync=None)`: Discards `warmup` iterations, performs CUDA synchronization before and after every single measurement (`sync = torch.cuda.synchronize`), records high-resolution wall times with `time.perf_counter()`, and computes `p50`, `p95`, `p99`, `mean`, and sample `std`.
- `latency_report(model, batch_size, img_size, dtype="fp32", device="cuda", warmup=10, iters=100)`: Sets `model.eval()`, handles input tensor allocation, selects execution modes (`fp32`, `amp`, `fp16`), executes `bench()`, and returns a dictionary with GPU name, dtype, batch size, image size, percentiles, and throughput (`images_per_s`).
- `tta_latency(model, k_views=2, ...)`: Measures multi-view test-time augmentation latency.

### 2.2 Inspection of `code/model.py`
- `build_model(name, pretrained=True, num_classes=9, drop_rate=0.0, init="finetune")`: Instantiates backbones using `timm.create_model`, sets proper 9-class heads, attaches `model.pretrained_tag`, and configures freezing/finetuning modes.
- `count_params(model)`: Computes total model parameters in millions: `round(sum(p.numel() for p in model.parameters()) / 1e6, 3)`.
- `count_gmacs(model, img_size=224)`: Attaches forward hooks to `nn.Conv2d` and `nn.Linear` layers to compute exact Multiply-Accumulate operations for a $3 \times 224 \times 224$ input tensor: `total_macs / 1e9`.

### 2.3 Identified Gap & Proposed CLI Patch
- **Contract Mismatch**: `PROJECT.md` line 48 specifies an invocation contract:
  `python code/benchmark.py --models <MODEL_NAME> --batch_size 1 --device cuda`
  However, `code/benchmark.py` originally lacked an `argparse` CLI entrypoint (`if __name__ == "__main__": main()`).
- **Resolution**: We created a non-invasive Git patch file:
  `.agents/teamwork/explorer_m1_2/proposed_benchmark_cli.patch`
  This patch provides an `argparse` entrypoint supporting `--models` (single, multiple, or `all`), `--batch_size`, `--img_size`, `--dtype`, `--device`, `--warmup`, `--iters`, and `--out` (JSON export).

---

## 3. Verified Benchmark & Complexity Metrics (B01–B05)

### 3.1 Comprehensive Multi-Dtype Comparison Table
All measurements conducted on NVIDIA GeForce RTX 5060 Ti, resolution $224 \times 224$, 15 warmup iterations, 100 timed iterations with explicit `torch.cuda.synchronize()`:

| Exp ID | Backbone Name | Architecture Family | Pretrained Tag | Params (M) | GMACs | Batch 1 FP32 (ms) [p50 / p95 / p99 / mean] | Batch 1 AMP (ms) [p50 / p95 / p99 / mean] | Batch 1 FP16 (ms) [p50 / p95 / p99 / mean] | Batch 32 AMP Throughput (img/s) |
|---|---|---|---|---|---|---|---|---|---|
| **B01** | `resnet50` | ResNet | `a1_in1k` | 23.526 | 4.087 | **4.39** / 7.08 / 9.65 / 4.77 | 5.22 / 8.80 / 10.69 / 5.66 | 5.65 / 10.20 / 12.64 / 6.10 | 1,707.6 |
| **B02** | `convnext_tiny` | ConvNeXt | `in12k_ft_in1k` | 27.827 | 4.455 | **3.53** / 4.51 / 4.63 / 3.70 | 5.41 / 6.60 / 11.81 / 5.79 | 3.49 / 4.59 / 7.87 / 3.74 | 1,467.2 |
| **B03** | `swin_tiny_patch4_window7_224` | Vision Transformer | `ms_in1k` | 27.526 | 4.350 | **7.23** / 11.06 / 15.65 / 7.61 | 9.14 / 10.41 / 15.12 / 9.23 | 7.21 / 15.03 / 18.66 / 7.96 | 1,083.6 |
| **B04** | `efficientnet_b0` | Lightweight | `ra_in1k` | 4.019 | 0.385 | **5.46** / 8.94 / 12.66 / 5.88 | 7.62 / 8.99 / 16.13 / 7.80 | 7.16 / 8.22 / 17.22 / 7.38 | 3,146.5 |
| **B05** | `mobilenetv3_large_100` | Lightweight | `ra_in1k` | 4.214 | 0.215 | **4.50** / 6.19 / 8.31 / 4.83 | 6.36 / 11.34 / 15.41 / 6.84 | 7.29 / 8.30 / 8.77 / 7.12 | 5,263.2 |

### 3.2 In-Depth Analysis of Architectural Trade-offs
1. **Speed vs Compute (ConvNeXt-Tiny Dominance in Single-Frame Latency)**:
   - `convnext_tiny` is the fastest model at batch size 1 (p50 = 3.53 ms, p95 = 4.51 ms, mean = 3.70 ms, 283.3 img/s).
   - Despite having 27.8M parameters and 4.455 GMACs, its modernized 7x7 depthwise convolutions and inverted bottleneck blocks map efficiently onto modern GPU streaming multiprocessors without fragmenting CUDA thread blocks.
2. **Transformer Latency Overhead (Swin-Tiny)**:
   - `swin_tiny_patch4_window7_224` is the slowest model in single-frame inference (p50 = 7.23 ms, p95 = 11.06 ms).
   - Windowed self-attention, patch partition, and shifting window operations entail dynamic memory permutations and non-coalesced memory access patterns that introduce noticeable latency when batch size is 1.
3. **Lightweight Models (MobileNetV3 vs EfficientNet-B0)**:
   - `mobilenetv3_large_100` (4.50 ms) outperforms `efficientnet_b0` (5.46 ms) by nearly 1.0 ms at batch 1.
   - At batch 32, `mobilenetv3_large_100` delivers an impressive **5,263.2 img/s**, confirming its suitability for batched throughput workloads.
   - In parameter footprint, both lightweight models require only ~4.0M–4.2M parameters, which is 6× smaller than ResNet-50 and ConvNeXt-Tiny.

---

## 4. Benchmark Invocation & Parsing Guide

### 4.1 CLI Command Lines
Once the patch in `.agents/teamwork/explorer_m1_2/proposed_benchmark_cli.patch` is applied to `code/benchmark.py`:

```powershell
# Benchmark a single model (batch size 1, FP32, CUDA)
python code/benchmark.py --models resnet50 --batch_size 1 --device cuda

# Benchmark all 5 models and export results to JSON
python code/benchmark.py --models all --batch_size 1 --device cuda --out runs/backbone_benchmarks.json

# Benchmark lightweight models with AMP
python code/benchmark.py --models efficientnet_b0 mobilenetv3_large_100 --batch_size 1 --dtype amp --device cuda
```

### 4.2 Standalone Python Invocation (Zero Source Modification)
If running prior to applying the patch, execute the following one-liner in PowerShell:

```powershell
python -c "import sys; sys.path.insert(0, 'code'); from model import build_model, count_params, count_gmacs; from benchmark import latency_report; [print(f'{m:<28} | Params: {count_params(build_model(m, True, 9)):.3f}M | GMACs: {count_gmacs(build_model(m, True, 9)):.3f} | Latency p50: {latency_report(build_model(m, True, 9), 1, 224, \"fp32\", \"cuda\")[\"p50\"]} ms') for m in ['resnet50', 'convnext_tiny', 'swin_tiny_patch4_window7_224', 'efficientnet_b0', 'mobilenetv3_large_100']]"
```

### 4.3 Python Integration API for Worker
To populate `results.xlsx` sheet `Backbones`, the Worker script can extract the profiling dictionary directly:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("code").resolve()))
import torch
from model import build_model, count_params, count_gmacs
from benchmark import latency_report

def get_backbone_profile(model_name: str, device: str = "cuda") -> dict:
    model = build_model(model_name, pretrained=True, num_classes=9)
    model.to(device)
    n_params = count_params(model)
    gmacs = count_gmacs(model, img_size=224)
    rep = latency_report(model, batch_size=1, img_size=224, dtype="fp32", device=device, warmup=15, iters=100)
    tag = getattr(model, "pretrained_tag", "unknown")
    return {
        "backbone": model_name,
        "tag": tag,
        "params_m": n_params,
        "gmacs": gmacs,
        "latency_p50_ms": rep["p50"],
        "latency_p95_ms": rep["p95"],
        "latency_mean_ms": rep["mean"],
        "images_per_s": rep["images_per_s"],
    }
```

---

## 5. Quantitative Trade-off Selection Framework

### 5.1 Formulation of Problem & Objective
Under `ORIGINAL_REQUEST.md` §R1 and `RUBRIC.md` §1.B, the selection of the optimal backbone for Step 2 cannot simply be "pick highest F1 blindly". It must be justified by an objective quantitative trade-off balancing:
1. **Classification Quality**: Val Macro-F1 across 9 classes (primary evaluation metric under severe class imbalance).
2. **Inference Latency**: Batch-1 $L_{\text{p50}}$ (ms) for real-time agricultural robotics.
3. **Memory Footprint**: Total parameters $P$ (Millions) and GMACs for edge hardware constraints.

### 5.2 Four-Tier Quantitative Decision Criteria

#### Tier 1: Statistical Noise Gate ($\Delta F_1 > \sigma_{\text{noise}}$)
- Slide 59 and `RUBRIC.md` establish that seed noise on DeepWeeds is $\sigma_{\text{noise}} \approx 0.10\% - 0.30\%$.
- **Rule**: If two candidate models $A$ and $B$ have $|\text{Macro-F1}(A) - \text{Macro-F1}(B)| \le 0.30\%$, the performance difference is statistically indistinguishable.
- **Action**: In this scenario, the model with lower latency ($L_{\text{p50}}$) and lower parameters ($P$) is definitively preferred.

#### Tier 2: Real-Time Robotics Feasibility Gate (Hard Constraint)
- **Rule**: $L_{\text{p95}} \le 100$ ms (satisfying Rubric item I5) and $P \le 50$ M.
- **Verification**: All 5 candidate models strictly satisfy this gate ($L_{\text{p95}} \le 11.06$ ms $\ll 100$ ms).

#### Tier 3: Multi-Criteria Utility Index ($U_{\text{tradeoff}}$)
Across the 5 models $\mathcal{M} = \{B01, B02, B03, B04, B05\}$, normalize each metric into $[0, 1]$:

$$s_{\text{F1}}(m) = \frac{\text{Macro-F1}(m) - \min_{k} \text{Macro-F1}(k)}{\max_{k} \text{Macro-F1}(k) - \min_{k} \text{Macro-F1}(k) + 10^{-6}}$$

$$s_{\text{lat}}(m) = \frac{\max_{k} L_{\text{p50}}(k) - L_{\text{p50}}(m)}{\max_{k} L_{\text{p50}}(k) - \min_{k} L_{\text{p50}}(k) + 10^{-6}}$$

$$s_{\text{param}}(m) = \frac{\max_{k} P(k) - P(m)}{\max_{k} P(k) - \min_{k} P(k) + 10^{-6}}$$

The composite utility is computed as:
$$U(m) = w_{\text{F1}} \cdot s_{\text{F1}}(m) + w_{\text{lat}} \cdot s_{\text{lat}}(m) + w_{\text{param}} \cdot s_{\text{param}}(m)$$

We define two standard evaluation profiles:
- **Profile A (Academic / Lab Rubric - F1 Oriented)**:
  $$w_{\text{F1}} = 0.60, \quad w_{\text{lat}} = 0.25, \quad w_{\text{param}} = 0.15$$
- **Profile B (Field Robotic Sprayer - Latency & Footprint Oriented)**:
  $$w_{\text{F1}} = 0.40, \quad w_{\text{lat}} = 0.40, \quad w_{\text{param}} = 0.20$$

#### Tier 4: Pareto Efficiency (Non-Dominated Sorting)
A model $m$ dominates $m'$ ($m \succ m'$) if:
$$\text{Macro-F1}(m) \ge \text{Macro-F1}(m'), \quad L_{\text{p50}}(m) \le L_{\text{p50}}(m'), \quad P(m) \le P(m')$$
with at least one strict inequality.
The non-dominated models form the **Pareto Optimal Frontier**. The final backbone must be a Pareto-optimal model with the highest $U(m)$.

### 5.3 Concrete Decision Tree for Worker Post-Training
When all 5 training runs (B01..B05) are complete:
1. Extract `best_val_macro_f1`, `params_m`, and `latency_p50_ms` for each model.
2. Determine the model with highest Macro-F1 ($m^*_{\text{F1}}$).
3. If $m^*_{\text{F1}} == \text{convnext\_tiny}$:
   - Check latency: $L_{\text{p50}}(\text{convnext\_tiny}) = 3.53$ ms, which is the fastest of all 5 models!
   - Result: ConvNeXt-Tiny simultaneously achieves highest accuracy and lowest latency, dominating the competition. ConvNeXt-Tiny is unequivocally selected.
4. If $m^*_{\text{F1}} == \text{swin\_tiny}$:
   - Check $\Delta F_1 = F_1(\text{swin\_tiny}) - F_1(\text{convnext\_tiny})$.
   - If $\Delta F_1 \le 0.30\%$: Select ConvNeXt-Tiny because Swin-Tiny is 2.05× slower (7.23 ms vs 3.53 ms) and the difference is within noise.
   - If $\Delta F_1 > 0.30\%$: Calculate Profile A Utility $U(\text{swin\_tiny})$ vs $U(\text{convnext\_tiny})$. If $U(\text{swin}) > U(\text{convnext})$, select Swin-Tiny.
5. If a lightweight model (`mobilenetv3_large_100` or `efficientnet_b0`) has Macro-F1 within $0.50\%$ of ResNet-50 / ConvNeXt-Tiny:
   - Under Profile B, MobileNetV3 is highlighted as the premier edge deployment candidate (4.2M params, 0.215 GMACs).

---

## 6. Actionable Recommendations for M1 Worker

1. **Apply the Benchmark Patch**: Apply `.agents/teamwork/explorer_m1_2/proposed_benchmark_cli.patch` to enable full CLI functionality on `code/benchmark.py`.
2. **Benchmarking Execution**: Run latency profiling with `device=cuda`, `dtype=fp32`, `batch_size=1`, `img_size=224`, `warmup=15`, `iters=100`.
3. **Data Consistency**: Ensure the latency and parameter numbers in `summary.json`, `results.xlsx` (`Backbones` sheet), and the final report match exactly the values verified in this report.
4. **Log Pretrained Tags Accurately**:
   - `B01`: `a1_in1k`
   - `B02`: `in12k_ft_in1k`
   - `B03`: `ms_in1k`
   - `B04`: `ra_in1k`
   - `B05`: `ra_in1k`
5. **Selection Justification**: In the M1 milestone summary, explicitly reference the 4-tier trade-off criteria to justify the winning backbone for Milestone 2.
