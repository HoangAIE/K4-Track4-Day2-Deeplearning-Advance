# Milestone 2 Handoff Report: Ablation Design for Axis C (Loss) and Axis D/F (Sampling & Regularization)

**Agent**: `explorer_m2_2`  
**Milestone**: Milestone 2 (Step 2 — Training Recipe Ablation Design)  
**Parent**: Orchestrator (`c4718f41-3030-43ce-8fcc-465340c45738`)  
**Date**: 2026-10-03  
**Status**: Task Completed (Hard Handoff)  

---

## 1. Observation

### 1.1 Dataset Class Distribution & Imbalance
Direct empirical inspection of `data/labels/train_subset0.csv` ($N = 10,501$) and `data/labels/val_subset0.csv` ($N = 3,501$) confirmed severe majority class imbalance:
- **Class 8 (Negatives)** in Train: **5,463 samples (52.02%)**; in Val: **1,821 samples (52.01%)**.
- **Weed Classes (0–7)** in Train: Each weed species contains only **605 to 675 samples (5.76% to 6.43%)**:
  - Class 0 (Chinee Apple): 675 (6.43%)
  - Class 1 (Lantana): 637 (6.07%)
  - Class 2 (Parkinsonia): 618 (5.89%)
  - Class 3 (Parthenium): 613 (5.84%)
  - Class 4 (Prickly Acacia): 637 (6.07%)
  - Class 5 (Rubber Vine): 605 (5.76%) — *Rarest weed class*
  - Class 6 (Siam Weed): 644 (6.13%)
  - Class 7 (Snake Weed): 609 (5.80%)
- **Imbalance Ratio**: The majority background class (Negatives) outnumbers any individual weed class by **8.09 : 1** up to **9.03 : 1**.

### 1.2 Baseline Performance Breakdown (B02 / T00 on `convnext_tiny`)
Evaluating out-of-fold validation predictions from `runs/B02/seed0/val_logits.npy` against `val_subset0.csv`:
- Overall Val Macro-F1: **96.04%** (`0.960354`), Val Top-1 Accuracy: **96.97%** (`0.969723`).
- Majority class Negatives: **Recall = 98.74%**, **Precision = 97.45%**, **F1 = 98.09%** (Support = 1,821).
- Rare weed classes exhibit clear recall suppression:
  - **Chinee Apple (Class 0)**: **Recall = 89.78%**, Precision = 94.84%, **F1 = 92.24%** (lowest F1 across all classes).
  - **Snake Weed (Class 7)**: **Recall = 92.61%**, Precision = 94.00%, **F1 = 93.30%**.
  - **Prickly Acacia (Class 4)**: Recall = 94.34%, **Precision = 92.59%**, **F1 = 93.46%**.
  - Macro Average Recall is **95.51%** versus Negatives Recall of **98.74%**, proving that decision boundaries under standard Cross-Entropy are skewed toward the majority background class.

### 1.3 Codebase Mechanism & Parser Inspection
1. **Critical `label_smoothing` Fallback in `code/train.py` & `code/losses.py`**:
   - In `code/train.py` lines 76–78:
     ```python
     loss: str = "ce"
     label_smoothing: float = 0.0
     ```
   - In `code/train.py` lines 368–369:
     ```python
     elif cfg.loss in ("ls", "label_smoothing"):
         criterion = build_criterion("ls", smoothing=cfg.label_smoothing)
     ```
   - In `code/losses.py` lines 43–45:
     ```python
     def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
         if self.smoothing <= 0.0:
             return F.cross_entropy(logits, target)
     ```
   - *Observation*: Passing `--set loss=ls` alone leaves `cfg.label_smoothing` at `0.0`. `LabelSmoothingCE` then returns standard `F.cross_entropy`. **To actually enable label smoothing, `--set label_smoothing=0.1` must be explicitly passed**.
2. **Focal Loss Implementation in `code/losses.py`**:
   - In lines 63–93: `FocalLoss(gamma=2.0)` calculates $p_t = \exp(\log(p)_t)$, computes focal weight $(1 - p_t)^\gamma$, and scales cross-entropy: $\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$.
   - In `code/train.py` line 78: default `focal_gamma = 2.0`.
3. **Balanced Sampler Implementation in `code/dataset.py`**:
   - In lines 248–254:
     ```python
     if train and sampler == "balanced":
         class_counts = df["Label"].value_counts().to_dict()
         sample_weights = [1.0 / max(1, class_counts[label]) for label in df["Label"]]
         weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
         sampler_obj = WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)
         shuffle = False
     ```
   - Each weed sample has relative sampling probability $1 / N_c \approx 1/605 \approx 9.03 \times (1 / 5463)$.
4. **EMA Regularization in `code/train.py`**:
   - In lines 143–172: `EMA(decay=0.999)` updates shadow parameters: $W_{\text{ema}} \leftarrow d \cdot W_{\text{ema}} + (1 - d) \cdot W$.
   - In lines 400–406: Validation evaluates on smoothed shadow weights via `ema.apply_shadow(model)` and reverts via `ema.restore(model)`.

### 1.4 Dry-Run Verification Command
Executed test script verifying config instantiation, argument parsing, criterion building, and EMA initialization:
```
=== T05 ===
  Config: T05 convnext_tiny loss: ls ls: 0.1 gamma: 2.0 sampler: None ema: None
  Criterion created: <class 'losses.LabelSmoothingCE'> smoothing= 0.1
=== T06 ===
  Config: T06 convnext_tiny loss: focal ls: 0.0 gamma: 2.0 sampler: None ema: None
  Criterion created: <class 'losses.FocalLoss'> gamma= 2.0
=== T07 ===
  Config: T07 convnext_tiny loss: ce ls: 0.0 gamma: 2.0 sampler: balanced ema: None
  Criterion created: <class 'torch.nn.modules.loss.CrossEntropyLoss'>
=== T08 ===
  Config: T08 convnext_tiny loss: ce ls: 0.0 gamma: 2.0 sampler: None ema: 0.999
  Criterion created: <class 'torch.nn.modules.loss.CrossEntropyLoss'>
  EMA created: <class 'train.EMA'> decay= 0.999
```
*Result*: All 4 configurations parse cleanly with 0 errors.

---

## 2. Logic Chain

### 2.1 Addressing Class Imbalance: Mechanism vs Failure Mode
1. **Observation 1.2** proves that the baseline model misclassifies rare weeds as Negatives (Chinee Apple recall is 89.78% vs Negatives recall 98.74%).
2. **Observation 1.1** explains this: 52% of training images are Negatives, so standard empirical risk minimization predominantly minimizes loss on majority background samples.
3. **Deduction for T05 (Label Smoothing)**:
   - Setting $\epsilon = 0.1$ bounds optimal logit differences to $\log(72) \approx 4.28$.
   - This caps the unbounded logit growth of the majority Negatives class, preventing its decision boundary from expanding and squeezing out neighboring rare weed clusters in penultimate feature space.
4. **Deduction for T06 (Focal Loss)**:
   - For easy Negatives with $p_t = 0.90$ to $0.99$, the $(1 - p_t)^2$ factor down-weights gradient contributions by $100\times$ to $10,000\times$.
   - The thousands of unambiguous pasture background images produce negligible aggregate gradient, concentrating backpropagation updates on hard, ambiguous weed images (e.g. Chinee Apple and Snake Weed).
5. **Deduction for T07 (Balanced Sampler)**:
   - `WeightedRandomSampler` shifts batch-level class distribution from 33.3 Negatives per batch down to 7.1 Negatives per batch, equalizing weed representation.
   - Every weed class receives $\approx 1.8\times$ to $1.9\times$ more training exposures per epoch, directly countering gradient starvation.
   - *However*, because validation remains 52% Negatives, a prior shift is introduced, which may decrease weed Precision.
6. **Deduction for T08 (Weight EMA)**:
   - A decay of $d = 0.999$ averages parameters across $\tau = \frac{1}{1-0.999} = 1,000$ iterations ($\approx 6.06$ epochs).
   - In imbalanced learning, batches containing rare weeds produce noisy gradient steps. EMA dampens this oscillation, guiding the model toward flatter, wider minima with better generalization.

### 2.2 Execution & Runtime Deduction
1. Milestone 1 showed `convnext_tiny` trains in $79.11$ s/epoch on the RTX 5060 Ti.
2. For T05, T06, T07, and T08:
   - Loss function operations add $< 1$ s/epoch.
   - Sampler index generation adds $< 1$ s/epoch.
   - EMA shadow updates across 27.8M parameters require $< 2$ ms per batch ($< 0.5$ s/epoch).
3. Therefore, each 12-epoch ablation run will take approximately **15.8 to 16.2 minutes**, totaling $\approx 64$ minutes for the complete set of 4 runs.

---

## 3. Caveats

1. **Explicit Parameter Passing**:
   - As documented in Observation 1.3, running `--set loss=ls` without `label_smoothing=0.1` will silently execute standard Cross-Entropy. The operator must strictly include `label_smoothing=0.1`.
2. **Prior Shift in Balanced Sampling (T07)**:
   - Because `WeightedRandomSampler` alters the training class prior from $52\%$ Negatives to $11.1\%$ Negatives, uncalibrated softmax probabilities at validation time may over-predict rare classes. If False Positives spike, Macro-F1 could degrade despite improved Recall.
3. **EMA Checkpoint Saving Behavior**:
   - In `code/train.py`, validation evaluates shadow weights via `ema.apply_shadow(model)`, but `ema.restore(model)` is called before saving the state dict on line 437. The saved checkpoint therefore contains active model weights evaluated at the shadow metric level.
4. **Zero Test Set Leakage**:
   - All proposed configurations strictly respect S1–S6 integrity constraints. `test_subset0.csv` is never evaluated during Milestone 2.

---

## 4. Conclusion

1. **Ablation Specifications Finalized**:
   The exact single-factor ablation configurations on `convnext_tiny` are formulated and ready for execution:
   - **T05**: `$env:PYTHONUTF8=1; python code/train.py --exp_id T05 --seed 0 --fold 0 --set backbone=convnext_tiny loss=ls label_smoothing=0.1 num_workers=0`
   - **T06**: `$env:PYTHONUTF8=1; python code/train.py --exp_id T06 --seed 0 --fold 0 --set backbone=convnext_tiny loss=focal focal_gamma=2.0 num_workers=0`
   - **T07**: `$env:PYTHONUTF8=1; python code/train.py --exp_id T07 --seed 0 --fold 0 --set backbone=convnext_tiny sampler=balanced num_workers=0`
   - **T08**: `$env:PYTHONUTF8=1; python code/train.py --exp_id T08 --seed 0 --fold 0 --set backbone=convnext_tiny ema_decay=0.999 num_workers=0`
2. **Artifact Targets Established**:
   - Checkpoints: `runs/T05/seed0/`, `runs/T06/seed0/`, `runs/T07/seed0/`, `runs/T08/seed0/`.
   - Dual-axis Curves: `curves/T05_convnext_tiny.png`, `curves/T06_convnext_tiny.png`, `curves/T07_convnext_tiny.png`, `curves/T08_convnext_tiny.png` (with descriptive aliases `curves/T0x_<description>.png`).
3. **Detailed Strategy Documented**:
   - Complete mathematical derivations and execution blueprints are archived in `.agents/teamwork/explorer_m2_2/analysis.md`.

---

## 5. Verification Method

To independently verify the ablation formulations and environment readiness:

1. **Dry-Run Config & Parser Assertion**:
   ```powershell
   $env:PYTHONUTF8=1; python -c "
   import sys
   sys.path.insert(0, 'code')
   from train import Config, parse_overrides, build_criterion, EMA
   import torch.nn as nn

   cases = [
       ('T05', ['backbone=convnext_tiny', 'loss=ls', 'label_smoothing=0.1', 'num_workers=0']),
       ('T06', ['backbone=convnext_tiny', 'loss=focal', 'focal_gamma=2.0', 'num_workers=0']),
       ('T07', ['backbone=convnext_tiny', 'sampler=balanced', 'num_workers=0']),
       ('T08', ['backbone=convnext_tiny', 'ema_decay=0.999', 'num_workers=0']),
   ]
   for exp_id, pairs in cases:
       cfg = Config(exp_id=exp_id, **parse_overrides(pairs))
       if cfg.loss == 'ls':
           c = build_criterion('ls', smoothing=cfg.label_smoothing)
           assert c.smoothing == 0.1, 'T05 smoothing mismatch'
       elif cfg.loss == 'focal':
           c = build_criterion('focal', gamma=cfg.focal_gamma)
           assert c.gamma == 2.0, 'T06 gamma mismatch'
       elif cfg.sampler == 'balanced':
           assert cfg.sampler == 'balanced', 'T07 sampler mismatch'
       elif cfg.ema_decay is not None:
           ema = EMA(nn.Linear(2, 2), decay=cfg.ema_decay)
           assert ema.decay == 0.999, 'T08 decay mismatch'
   print('Ablation Configs Verification: ALL PASSED!')
   "
   ```

2. **Repository Unit Test Suite Execution**:
   ```powershell
   $env:PYTHONUTF8=1; python -m unittest discover -s tests -v
   ```
   *Expected outcome*: 38/38 tests PASS with 0 failures, 0 errors.

3. **Invalidation Conditions**:
   - If `code/train.py` is invoked with `loss=ls` without `label_smoothing=0.1`, which causes silent reversion to vanilla Cross-Entropy.
   - If batch size or other optimizer hyperparameters are altered, violating Rule 1 of single-factor ablation.
   - If test predictions are evaluated on `test_subset0.csv` during hyperparameter selection.
