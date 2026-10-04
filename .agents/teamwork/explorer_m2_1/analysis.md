# Milestone 2 Ablation Strategy & Technical Specification: Axis A & Axis B

**Document**: `analysis.md`  
**Agent**: Explorer 1 (`explorer_m2_1`)  
**Milestone**: Milestone 2 — Training Recipe Ablation (Step 2)  
**Parent Conversation ID**: `c4718f41-3030-43ce-8fcc-465340c45738`  
**Date**: 2026-10-03  
**Status**: Completed Technical Design  

---

## 1. Executive Summary & Milestone 1 Baseline Grounding

In Milestone 1, five candidate backbones spanning four architectural families were systematically evaluated under the standard baseline recipe $T00$ on DeepWeeds fold 0:
- ResNet Baseline: `resnet50` ($B01$, Val Macro-F1 = 80.65%, Latency = 12.47 ms)
- Modern ConvNet: `convnext_tiny` ($B02$, Val Macro-F1 = **96.04%**, Latency = **11.44 ms**)
- Vision Transformer: `swin_tiny_patch4_window7_224` ($B03$, Val Macro-F1 = 94.95%, Latency = 19.56 ms)
- Mobile Scaling: `efficientnet_b0` ($B04$, Val Macro-F1 = 82.66%, Latency = 15.67 ms)
- Ultra-lightweight Mobile: `mobilenetv3_large_100` ($B05$, Val Macro-F1 = 82.78%, Latency = 12.80 ms)

### 1.1 Selected Winning Backbone: `convnext_tiny`
Based on multi-criteria utility optimization combining classification accuracy ($w_{\text{F1}}=0.60$), GPU batch-1 inference latency ($w_{\text{latency}}=0.25$), and parameter complexity ($w_{\text{params}}=0.15$), **`convnext_tiny` achieved the highest composite utility score ($U=0.850$)**. 

It won on both primary operational dimensions:
1. **Highest Predictive Accuracy**: Val Macro-F1 = **96.04%** (0.9604), Val Top-1 Accuracy = **96.97%** (0.9697). It outperformed Swin-Tiny by $+1.09\%$ and ResNet-50 by $+15.39\%$.
2. **Fastest Batch-1 Inference Latency**: **11.44 ms** (p50), beating even lightweight mobile models on the NVIDIA RTX 5060 Ti GPU due to high arithmetic intensity ($FLOPs/Byte$) and GPU Tensor Core optimization.

### 1.2 Baseline Reference Recipe ($T00$) Anchor
Experiment $B02$ established the quantitative benchmark for `convnext_tiny`. In Step 2, this configuration serves as the **Anchor Baseline $T00$**:
- **Backbone**: `convnext_tiny` (Pretrained tag: `in12k_ft_in1k`, 27.827 M parameters, 4.455 GMACs).
- **Initialization**: `init="finetune"` (ImageNet-12k pretrained weights fine-tuned on ImageNet-1k; all parameters trainable).
- **Augmentation**: `aug="basic"` (`RandomResizedCrop(224, scale=(0.8, 1.0))` + `RandomHorizontalFlip()` + ImageNet normalization).
- **Mixing**: `mix=None` (no Mixup, no CutMix).
- **Loss Function**: `loss="ce"` (standard Cross-Entropy, label smoothing = 0.0).
- **Sampler**: `sampler=None` (standard uniform shuffling).
- **Optimization**: AdamW, $\text{lr}_{\text{backbone}} = 10^{-4}$, $\text{lr}_{\text{head}} = 10^{-3}$, $\text{weight\_decay} = 0.05$ (omitted on 1D normalization and bias vectors).
- **Schedule**: 1.0 epoch linear warmup followed by cosine annealing decay down to $10^{-4}$.
- **Hardware & Precision**: CUDA AMP GradScaler, batch size = 64, epochs = 12, seed = 0, fold = 0.
- **Reference Metrics**:
  - **Val Macro-F1**: **96.04%** (0.9604)
  - **Val Top-1 Accuracy**: **96.97%** (0.9697)
  - **Best Epoch**: 11 (Val Macro-F1: 83.44% at Ep 1 $\to$ 92.00% at Ep 3 $\to$ 94.25% at Ep 7 $\to$ 96.04% at Ep 11).
  - **Average Training Speed**: 79.11 s/epoch (~15.8 minutes total execution time on RTX 5060 Ti).

---

## 2. Theoretical Profile of `convnext_tiny`

To design meaningful ablations, we must examine the specific mechanics of the ConvNeXt architecture (Liu et al., *A ConvNet for the 2020s*, CVPR 2022):

1. **Macro Design**:
   - Stage compute ratio is set to $3:3:9:3$, closely mirroring Swin Transformer.
   - Non-overlapping "patchify" stem using $4 \times 4$ convolution with stride 4 replaces standard $7 \times 7$ conv + max pooling.
   - Separate downsampling layers using $2 \times 2$ convolution with stride 2 between stages.
2. **Micro Architectural Blocks**:
   - **Inverted Bottleneck**: Channels are expanded by $4\times$ ($C \to 4C$) before spatial depthwise convolution, rather than contracting channels like traditional ResNet bottlenecks.
   - **Large $7 \times 7$ Depthwise Convolution**: Moved up to the beginning of the block, effectively increasing the effective receptive field to match the self-attention window size of Vision Transformers.
   - **Activation & Normalization**: ReLU is replaced with GELU; normalizations are drastically reduced to one LayerNorm per block (no BatchNorm), eliminating cross-sample dependencies and running statistic drift.
3. **Parameter Distribution**:
   - Total parameters: **27,827,241** (~27.83 M).
   - Backbone feature extractor: **27,820,320** parameters (Stages 1–4, LayerNorms, Stem).
   - Classification head (`head.fc`): **6,921** parameters ($768 \times 9 + 9$).

---

## 3. Strict Controlled Experimentation Protocol (Principle N1 & S1–S6)

### 3.1 Single-Factor Variation Rule (Principle N1)
Per `GUIDE.md §0` (Principle N1) and `ORIGINAL_REQUEST.md §R2`:
> *"Khi so sánh hai cấu hình, chỉ khác nhau đúng một yếu tố, mọi thứ còn lại giữ nguyên (cùng backbone, cùng số epoch, cùng seed khởi đầu, cùng split). Nếu đổi hai thứ cùng lúc, bạn không kết luận được yếu tố nào có tác dụng."*

For every ablation experiment:
- **Fixed Backbone**: `convnext_tiny`
- **Fixed Dataset Split**: Fold 0 (`data/labels/train_subset0.csv`: 10,501 images; `data/labels/val_subset0.csv`: 3,501 images)
- **Fixed Optimizer**: AdamW ($\text{lr}_{\text{backbone}}=10^{-4}$, $\text{lr}_{\text{head}}=10^{-3}$, $\text{weight\_decay}=0.05$)
- **Fixed Schedule**: 1 epoch linear warmup + cosine decay
- **Fixed Batch Size**: 64
- **Fixed Epochs**: 12
- **Fixed Seed**: 0
- **Fixed Hardware Mode**: CUDA AMP (`amp=True`), `num_workers=0` (ensuring deterministic behavior and zero IPC overhead on Windows)
- **Strictly Isolated Variable**: Exactly **one** parameter modified per experiment ($T01..T05$).

### 3.2 Zero Test Set Leakage Assurance (Rules S1–S6)
- Throughout Milestone 2, `save_test_predictions` is strictly maintained as `False`.
- `test_subset0.csv` is **never** loaded or evaluated.
- All model checkpoint saving decisions, metric comparisons, and delta computations are derived exclusively from `val_subset0.csv`.

### 3.3 Noise Significance Threshold
Per Slide Day 2 p. 59, standard deviation across random seeds is $\sigma_{\text{seed}} \approx 0.10\% - 0.30\%$ ($0.0010 - 0.0030$).
- If $|\Delta \text{Macro-F1}| < 0.30\%$, the observed variation is classified as **"không phân biệt được"** (indistinguishable from seed noise).
- If $|\Delta \text{Macro-F1}| \ge 0.30\%$, the variation is classified as **statistically meaningful**.

---

## 4. Axis A: Initialization Ablation Design

Axis A investigates the impact of network weight initialization on botanical feature convergence and transferability (`ORIGINAL_REQUEST.md §R2`, `GUIDE.md §3`).

### 4.1 Scientific Context & Hypotheses
In industrial agricultural computer vision, teams often ask:
1. *Does training from scratch on domain-specific weed datasets work, or are pre-trained general vision features mandatory?*
2. *Is linear probing (freezing the backbone and training only the classifier) sufficient, or is full end-to-end backpropagation necessary?*

### 4.2 Experiment $T01$: `init="scratch"` (Random Initialization)
- **Scientific Hypothesis**:
  With only 10,501 training images in DeepWeeds fold 0, a large 28M-parameter model trained from scratch without pretraining cannot form rich hierarchical feature representations within 12 epochs using a modest learning rate ($10^{-4}$). Randomly initialized filters will suffer from severe underfitting and slow convergence.
- **Underlying Code Mechanism**:
  In `code/model.py:43`:
  ```python
  use_pretrained = (init != "scratch") and pretrained
  ```
  When `init="scratch"`, `use_pretrained` evaluates to `False`. `timm.create_model('convnext_tiny', pretrained=False, num_classes=9)` initializes all convolutional kernels and LayerNorm weights from random truncated normal distributions.
  In `code/model.py:param_groups`, all 27.83 M parameters have `requires_grad=True` and receive gradient updates.
- **Expected Convergence & Metric Behavior**:
  - Training loss will decline slowly.
  - Val Macro-F1 after 12 epochs will be drastically degraded compared to T00 (expected $\text{Val Macro-F1} \approx 30.0\% - 55.0\%$, $\Delta \ll -40.0\%$).
  - Proves conclusively that pretraining representations are vital when sample counts are in the $\sim 10^4$ regime.
- **CLI Command**:
  ```powershell
  $env:PYTHONUTF8=1; python code/train.py --exp_id T01 --seed 0 --fold 0 --set backbone=convnext_tiny init=scratch num_workers=0
  ```
- **Runtime Expectation**:
  ~79.0 s/epoch $\times$ 12 epochs $\approx$ **15.8 minutes** on NVIDIA RTX 5060 Ti.
- **Artifact Specifications**:
  - Checkpoint: `runs/T01/seed0/best_checkpoint.pt`
  - Training history: `runs/T01/seed0/history.csv`
  - Validation logits: `runs/T01/seed0/val_logits.npy` (shape: $3501 \times 9$)
  - Run metadata: `runs/T01/seed0/summary.json` and `runs/T01/seed0/config.json`
  - Training Curve: `curves/T01_convnext_tiny.png` (aliased to `curves/T01_convnext_tiny_scratch.png`)

### 4.3 Experiment $T02$: `init="frozen"` (Linear Probing / Frozen Backbone)
- **Scientific Hypothesis**:
  Pretrained representations from ImageNet-12k encode rich general semantic textures and edges. By freezing the backbone and training only the final linear projection layer, the model performs linear classification on frozen feature embeddings. This tests how linearly separable DeepWeeds classes are in the raw ImageNet-12k feature space without weed-specific adaptation.
- **Underlying Code Mechanism**:
  In `code/model.py:80`:
  ```python
  def freeze_backbone(model: nn.Module) -> None:
      for param in model.parameters():
          param.requires_grad = False
      classifier = model.get_classifier()
      if classifier is not None:
          for param in classifier.parameters():
              param.requires_grad = True
  ```
  - **Trainable Parameters**: Exactly **6,921** (Head weight: $9 \times 768 = 6912$, Head bias: 9).
  - **Frozen Parameters**: **27,820,320** ($requires\_grad=False$).
  - In `code/train.py:178`: `model.eval()` is enforced on the backbone so that LayerNorm acts deterministically, while `classifier.train()` is active.
- **Expected Convergence & Metric Behavior**:
  - Computational speedup: Backpropagation halts at the feature pooling stage. Epoch duration drops from ~79 s/epoch down to ~35–45 s/epoch (~50% reduction).
  - Accuracy: The linear head quickly reaches its capacity within 3–5 epochs, but plateaus without fine-tuning of the intermediate convolutional filters. Expected $\text{Val Macro-F1} \approx 82.0\% - 88.0\%$ ($\Delta \approx -8.0\%$ to $-14.0\%$).
  - Proves that while ImageNet features provide a solid linear baseline, domain-specific fine-tuning is required to reach state-of-the-art (>96%) discriminability.
- **CLI Command**:
  ```powershell
  $env:PYTHONUTF8=1; python code/train.py --exp_id T02 --seed 0 --fold 0 --set backbone=convnext_tiny init=frozen num_workers=0
  ```
- **Runtime Expectation**:
  ~38.0 s/epoch $\times$ 12 epochs $\approx$ **7.6 minutes** on NVIDIA RTX 5060 Ti.
- **Artifact Specifications**:
  - Checkpoint: `runs/T02/seed0/best_checkpoint.pt`
  - Training history: `runs/T02/seed0/history.csv`
  - Validation logits: `runs/T02/seed0/val_logits.npy` (shape: $3501 \times 9$)
  - Run metadata: `runs/T02/seed0/summary.json` and `runs/T02/seed0/config.json`
  - Training Curve: `curves/T02_convnext_tiny.png` (aliased to `curves/T02_convnext_tiny_frozen.png`)

---

## 5. Axis B: Augmentation & Regularization Ablation Design

Axis B investigates data diversity and regularization techniques designed to improve generalization across challenging outdoor conditions (`ORIGINAL_REQUEST.md §R2`, `GUIDE.md §3`).

### 5.1 Scientific Context & Hypotheses
Field robotics operating in pasture environments face two distinct environmental hurdles:
1. **Photometric Variability**: Fluctuating ambient sunlight, cloud cover, and camera white balance affect color representations of green foliage.
2. **Contextual Occlusion & Complex Backgrounds**: Weeds frequently interlace with non-target native grasses or soil patches.

We investigate three distinct augmentation formulations across Axis B:
- `aug="color"`: Domain-specific photometric perturbation (ColorJitter).
- `aug="trivial"`: Automated wide-range policy augmentation (TrivialAugmentWide).
- `mix="cutmix"`: Regional cut-and-paste sample mixing with soft label interpolation.

### 5.2 Experiment $T03$: `aug="color"` (Photometric Perturbation)
- **Pipeline Implementation**:
  In `code/dataset.py:164`:
  ```python
  transforms.Compose([
      transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
      transforms.ToTensor(),
      transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  ```
- **Scientific Hypothesis**:
  Mild brightness and contrast variation simulates midday vs overcast lighting conditions without destroying botanical leaf morphology. However, hue jitter ($\pm 0.1$) shifts green wavelengths, which might subtly alter chlorophyll discoloration cues for species such as *Chinee Apple* or *Lantana*.
- **Expected Convergence & Metric Behavior**:
  Convergence will track baseline closely. The model is regularized against lighting artifacts. Expected $\text{Val Macro-F1} \approx 95.8\% - 96.5\%$ ($\Delta \approx \pm 0.3\%$, near the seed noise boundary).
- **CLI Command**:
  ```powershell
  $env:PYTHONUTF8=1; python code/train.py --exp_id T03 --seed 0 --fold 0 --set backbone=convnext_tiny aug=color num_workers=0
  ```
- **Runtime Expectation**:
  ~80.5 s/epoch $\times$ 12 epochs $\approx$ **16.1 minutes** on NVIDIA RTX 5060 Ti.
- **Artifact Specifications**:
  - Checkpoint: `runs/T03/seed0/best_checkpoint.pt`
  - Training history: `runs/T03/seed0/history.csv`
  - Validation logits: `runs/T03/seed0/val_logits.npy` (shape: $3501 \times 9$)
  - Run metadata: `runs/T03/seed0/summary.json` and `runs/T03/seed0/config.json`
  - Training Curve: `curves/T03_convnext_tiny.png` (aliased to `curves/T03_convnext_tiny_color.png`)

### 5.3 Experiment $T04$: `aug="trivial"` (TrivialAugmentWide)
- **Pipeline Implementation**:
  In `code/dataset.py:172`:
  ```python
  transforms.Compose([
      transforms.TrivialAugmentWide(),
      transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  ```
- **Scientific Hypothesis**:
  `TrivialAugmentWide` (Müller & Hutter, ICCV 2021) selects a single random transformation from a wide library with uniformly sampled intensity. This introduces aggressive geometric and color transformations (posterization, solarization, shearing, translations). For fine-grained botanical classification where leaf vein patterns and thorn structures are critical, aggressive posterization or solarization can obliterate discriminative micro-textures. Furthermore, strong regularization typically requires 50–100 epochs to overcome initial underfitting; over a 12-epoch budget, it may slow early convergence.
- **Expected Convergence & Metric Behavior**:
  Training loss will decrease more slowly. Epoch 1–6 Macro-F1 will lag behind baseline T00. By Epoch 12, the model may achieve competitive generalization or fall slightly below baseline due to insufficient training length ($\text{Val Macro-F1} \approx 94.5\% - 95.8\%$, $\Delta \approx -0.5\%$ to $-1.5\%$).
- **CLI Command**:
  ```powershell
  $env:PYTHONUTF8=1; python code/train.py --exp_id T04 --seed 0 --fold 0 --set backbone=convnext_tiny aug=trivial num_workers=0
  ```
- **Runtime Expectation**:
  ~83.0 s/epoch $\times$ 12 epochs $\approx$ **16.6 minutes** on NVIDIA RTX 5060 Ti.
- **Artifact Specifications**:
  - Checkpoint: `runs/T04/seed0/best_checkpoint.pt`
  - Training history: `runs/T04/seed0/history.csv`
  - Validation logits: `runs/T04/seed0/val_logits.npy` (shape: $3501 \times 9$)
  - Run metadata: `runs/T04/seed0/summary.json` and `runs/T04/seed0/config.json`
  - Training Curve: `curves/T04_convnext_tiny.png` (aliased to `curves/T04_convnext_tiny_trivial.png`)

### 5.4 Experiment $T05$: `mix="cutmix"` (Regional CutMix Regularization)
- **Pipeline Implementation**:
  In `code/train.py:196` and `code/losses.py:119`:
  - Input: Standard `aug="basic"` transforms.
  - Slicing: For each batch, a bounding box is sampled from sample $B$ and pasted into sample $A$ with $\lambda \sim \text{Beta}(1.0, 1.0)$.
  - Area Clipping: Actual bounding box dimensions clipped to image boundaries determine exact label weight:
    $$\lambda_{\text{actual}} = 1.0 - \frac{(bbx_2 - bbx_1) \times (bby_2 - bby_1)}{W \times H}$$
  - Objective:
    $$\mathcal{L}_{\text{CutMix}} = \lambda_{\text{actual}} \mathcal{L}_{\text{CE}}(\hat{y}, y_A) + (1 - \lambda_{\text{actual}}) \mathcal{L}_{\text{CE}}(\hat{y}, y_B)$$
- **Scientific Hypothesis**:
  CutMix (Yun et al., ICCV 2019) prevents the network from over-indexing on a single localized feature (e.g., only a flower or only a large leaf), forcing the receptive fields across all 4 ConvNeXt stages to recognize distributed botanical signals. Moreover, since 52% of DeepWeeds images are class 8 (`Negatives`), CutMix frequently composites weed leaves onto pasture/dirt backgrounds, acting as a natural synthetic synthesizer of weed patches in field settings.
- **Expected Convergence & Metric Behavior**:
  Training loss will appear elevated throughout training because labels are soft composites rather than one-hot integers. However, on the validation set, CutMix will exhibit superior regularization against overfitting, particularly helping minority weed classes. Expected $\text{Val Macro-F1} \approx 95.8\% - 96.6\%$ ($\Delta \approx -0.2\%$ to $+0.6\%$).
- **CLI Command**:
  ```powershell
  $env:PYTHONUTF8=1; python code/train.py --exp_id T05 --seed 0 --fold 0 --set backbone=convnext_tiny mix=cutmix mix_alpha=1.0 num_workers=0
  ```
- **Runtime Expectation**:
  ~81.0 s/epoch $\times$ 12 epochs $\approx$ **16.2 minutes** on NVIDIA RTX 5060 Ti.
- **Artifact Specifications**:
  - Checkpoint: `runs/T05/seed0/best_checkpoint.pt`
  - Training history: `runs/T05/seed0/history.csv`
  - Validation logits: `runs/T05/seed0/val_logits.npy` (shape: $3501 \times 9$)
  - Run metadata: `runs/T05/seed0/summary.json` and `runs/T05/seed0/config.json`
  - Training Curve: `curves/T05_convnext_tiny.png` (aliased to `curves/T05_convnext_tiny_cutmix.png`)

---

## 6. Master Experiment Execution Matrix

Below is the definitive matrix detailing all single-factor ablation experiments for Milestone 2 (Axis A and Axis B), anchored against baseline $T00$:

| Exp ID | Trục (Axis) | Yếu tố thay đổi (Ablation Factor) | Cú pháp dòng lệnh thực thi (Execution Command) | Trainable Params | Est. Epoch Time | Est. Tổng thời gian | Checkpoint Path | Đường cong huấn luyện |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---|:---|
| **T00** | **Mốc (Baseline)** | Baseline T00 (`finetune`, `basic`, `ce`) | `$env:PYTHONUTF8=1; python code/train.py --exp_id T00 --seed 0 --fold 0 --set backbone=convnext_tiny num_workers=0` | 27.83 M | 79.1 s | 15.8 min | `runs/T00/seed0/best_checkpoint.pt` | `curves/T00_convnext_tiny.png` |
| **T01** | **A (Khởi tạo)** | `init="scratch"` (Random weights) | `$env:PYTHONUTF8=1; python code/train.py --exp_id T01 --seed 0 --fold 0 --set backbone=convnext_tiny init=scratch num_workers=0` | 27.83 M | 79.0 s | 15.8 min | `runs/T01/seed0/best_checkpoint.pt` | `curves/T01_convnext_tiny.png` |
| **T02** | **A (Khởi tạo)** | `init="frozen"` (Linear probe head) | `$env:PYTHONUTF8=1; python code/train.py --exp_id T02 --seed 0 --fold 0 --set backbone=convnext_tiny init=frozen num_workers=0` | **6,921** | **38.0 s** | **7.6 min** | `runs/T02/seed0/best_checkpoint.pt` | `curves/T02_convnext_tiny.png` |
| **T03** | **B (Augmentation)** | `aug="color"` (ColorJitter) | `$env:PYTHONUTF8=1; python code/train.py --exp_id T03 --seed 0 --fold 0 --set backbone=convnext_tiny aug=color num_workers=0` | 27.83 M | 80.5 s | 16.1 min | `runs/T03/seed0/best_checkpoint.pt` | `curves/T03_convnext_tiny.png` |
| **T04** | **B (Augmentation)** | `aug="trivial"` (TrivialAugmentWide) | `$env:PYTHONUTF8=1; python code/train.py --exp_id T04 --seed 0 --fold 0 --set backbone=convnext_tiny aug=trivial num_workers=0` | 27.83 M | 83.0 s | 16.6 min | `runs/T04/seed0/best_checkpoint.pt` | `curves/T04_convnext_tiny.png` |
| **T05** | **B (Augmentation)** | `mix="cutmix"` (CutMix $\alpha=1.0$) | `$env:PYTHONUTF8=1; python code/train.py --exp_id T05 --seed 0 --fold 0 --set backbone=convnext_tiny mix=cutmix mix_alpha=1.0 num_workers=0` | 27.83 M | 81.0 s | 16.2 min | `runs/T05/seed0/best_checkpoint.pt` | `curves/T05_convnext_tiny.png` |

> *Note on T00*: Because experiment $B02$ in Milestone 1 ran this exact configuration with seed 0 and fold 0, $T00$ can be materialized either by running the command above to produce `runs/T00/seed0/` and `curves/T00_convnext_tiny.png`, or by symlinking/cloning `runs/B02/seed0/` to `runs/T00/seed0/`.

---

## 7. Master Excel Logging Specification (`results.xlsx`)

Sheet `Training` in `results.xlsx` is already initialized with headers per `GUIDE.md §6.1` and `RUBRIC.md §E`. The worker must log each completed ablation experiment according to the schema below.

### 7.1 Schema Definition
`TRAINING_COLUMNS`:
1. `exp_id`: Experiment ID (`T00`, `T01`, `T02`, `T03`, `T04`, `T05`)
2. `backbone`: Architecture name (`convnext_tiny`)
3. `trục thay đổi (A–G)`: Category string (`Mốc (Baseline)`, `A (Khởi tạo)`, `B (Augmentation)`)
4. `khác T00 ở điểm nào`: Clear distinction summary
5. `seed`: Random seed integer (`0`)
6. `macro-F1 val`: Out-of-fold validation Macro-F1 across 9 classes (`float`, format `0.0000`)
7. `top-1 val`: Out-of-fold validation Top-1 Accuracy (`float`, format `0.0000`)
8. `Δ so với T00`: Difference $\text{Macro-F1}_{\text{val}}(T0x) - \text{Macro-F1}_{\text{val}}(T00)$ (`float`, format `+0.0000` or `-0.0000`)
9. `ghi chú`: Concise analytical observation

### 7.2 Pre-calculated Logging Rows for Worker Reference

| exp_id | backbone | trục thay đổi (A–G) | khác T00 ở điểm nào | seed | macro-F1 val | top-1 val | Δ so với T00 | ghi chú |
|---|---|---|---|---|---|---|---|---|
| **T00** | `convnext_tiny` | Mốc (Baseline) | Mốc chuẩn T00 (finetune, basic aug, CE loss, lr 1e-4/1e-3) | 0 | 0.9604 | 0.9697 | 0.0000 | Mốc nền từ Bước 1 (B02); đỉnh tại epoch 11 |
| **T01** | `convnext_tiny` | A (Khởi tạo) | `init="scratch"` (huấn luyện từ đầu, không pretrained) | 0 | *[val_f1]* | *[val_top1]* | *[val_f1 - 0.9604]* | Không có trọng số khởi tạo; đánh giá khả năng tự học |
| **T02** | `convnext_tiny` | A (Khởi tạo) | `init="frozen"` (đóng băng backbone, chỉ train head) | 0 | *[val_f1]* | *[val_top1]* | *[val_f1 - 0.9604]* | Linear probing (6,921 params); train nhanh ~38s/ep |
| **T03** | `convnext_tiny` | B (Augmentation) | `aug="color"` (ColorJitter b=0.2, c=0.2, s=0.2, h=0.1) | 0 | *[val_f1]* | *[val_top1]* | *[val_f1 - 0.9604]* | Biến thiên độ sáng/màu sắc; kiểm tra độ nhạy ánh sáng |
| **T04** | `convnext_tiny` | B (Augmentation) | `aug="trivial"` (TrivialAugmentWide tự động) | 0 | *[val_f1]* | *[val_top1]* | *[val_f1 - 0.9604]* | Biến dạng hình học/màu rộng; kiểm tra độ bền chi tiết lá |
| **T05** | `convnext_tiny` | B (Augmentation) | `mix="cutmix"` (CutMix alpha=1.0, nhãn mềm) | 0 | *[val_f1]* | *[val_top1]* | *[val_f1 - 0.9604]* | Trộn vùng ảnh và nhãn mềm; chống shortcut learning |

### 7.3 Automated Excel Update Helper for `worker_m2`
To eliminate manual data entry errors, the worker can execute the following standalone Python function to insert each experiment into `results.xlsx`:

```python
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

def upsert_training_record(excel_path, record):
    wb = openpyxl.load_workbook(excel_path)
    ws = wb["Training"]
    exp_id = str(record["exp_id"]).strip()
    
    # Locate or create row
    target_row = None
    for r in range(2, ws.max_row + 1):
        if str(ws.cell(row=r, column=1).value or "").strip() == exp_id:
            target_row = r
            break
    if target_row is None:
        target_row = ws.max_row + 1 if ws.cell(row=ws.max_row, column=1).value is not None else ws.max_row

    cols = [
        record["exp_id"],
        record["backbone"],
        record["trục thay đổi (A–G)"],
        record["khác T00 ở điểm nào"],
        int(record.get("seed", 0)),
        float(record.get("macro-F1 val", 0.0)),
        float(record.get("top-1 val", 0.0)),
        float(record.get("Δ so với T00", 0.0)),
        record.get("ghi chú", ""),
    ]
    
    font = Font(name="Calibri", size=10)
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"), right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"), bottom=Side(style="thin", color="D9D9D9")
    )
    
    for c_idx, val in enumerate(cols, start=1):
        cell = ws.cell(row=target_row, column=c_idx, value=val)
        cell.font = font
        cell.border = thin_border
        if c_idx in [1, 5]:
            cell.alignment = Alignment(horizontal="center", vertical="center")
        elif c_idx in [6, 7, 8]:
            cell.alignment = Alignment(horizontal="right", vertical="center")
            cell.number_format = "0.0000"
        else:
            cell.alignment = Alignment(horizontal="left", vertical="center")
            
    wb.save(excel_path)
    wb.close()
```

---

## 8. Operational Safeguards & Invalidation Conditions

1. **Windows Multiprocessing Overhead**:
   Always specify `num_workers=0` in `--set` on Windows to avoid process spawning stalls and CUDA context recreation overhead.
2. **Terminal Encoding Safeguard**:
   Always prefix PowerShell invocations with `$env:PYTHONUTF8=1;` to ensure Vietnamese character logging in terminal streams does not encounter `charmap` codec exceptions.
3. **Dual-Axis Training Curves Quality**:
   Ensure `plot_curves()` in `code/train.py` creates high-resolution PNG plots with dual Y-axes (Loss on left, Metric % on right), unified legends, and epoch markers. For any experiment, verify that `curves/<exp_id>_<backbone>.png` exists and is non-empty (>100 KB).
4. **Invalidation Conditions**:
   - If `test_subset0.csv` is accessed or evaluated before Milestone 3 / Final phase.
   - If more than one factor is modified simultaneously in an ablation experiment.
   - If random seeds differ from 0 during single-factor ablation sweeps.
   - If checkpoints in `runs/<exp_id>/seed0/` fail to save `best_checkpoint.pt` or `val_logits.npy`.

---
*Report prepared and compiled by Explorer 1 (`explorer_m2_1`) for Milestone 2 implementation.*
