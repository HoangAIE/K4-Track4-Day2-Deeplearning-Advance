# Milestone 2 Ablation Strategy: Axis C (Loss Functions) and Axis D/F (Sampling & Regularization)

**Author**: Explorer 2 (`explorer_m2_2`)  
**Scope**: Milestone 2 Single-Factor Ablations on `convnext_tiny` for Axis C (Loss), Axis D (Sampling), and Axis F (Regularization)  
**Parent**: Orchestrator (`c4718f41-3030-43ce-8fcc-465340c45738`)  
**Target File**: `.agents/teamwork/explorer_m2_2/analysis.md`  
**Date**: 2026-10-03  

---

## 1. Executive Summary & Experimental Blueprint

Milestone 1 established **`convnext_tiny`** as the winning backbone across all 5 candidates, achieving **96.04% Val Macro-F1**, **96.97% Val Top-1 Accuracy**, and **11.44 ms batch-1 GPU inference latency** on an NVIDIA GeForce RTX 5060 Ti under the baseline **T00** recipe (AdamW, $\text{lr}_{\text{backbone}}=10^{-4}$, $\text{lr}_{\text{head}}=10^{-3}$, weight decay 0.05, 1-epoch warmup + cosine decay to 12 epochs, standard Cross-Entropy, batch size 64, AMP enabled, seed 0, fold 0).

In Milestone 2, following **Rule 1** of single-factor ablation (holding all other hyperparameters strictly identical to T00), Explorer 2 formulates the exact experimental specifications for:
1. **Axis C (Loss Function)**:
   - **T05**: Label Smoothing Cross-Entropy (`loss="ls"`, $\epsilon = 0.1$)
   - **T06**: Multi-Class Focal Loss (`loss="focal"`, $\gamma = 2.0$)
2. **Axis D (Sampling Strategy)**:
   - **T07**: Class-Balanced Sampling (`sampler="balanced"`)
3. **Axis F (Weight Regularization)**:
   - **T08**: Exponential Moving Average (`ema_decay = 0.999`)

| Exp ID | Axis | Mechanism | Exact Parameter Override | Reference Baseline (T00) | Primary Target | Expected Runtime (12 ep) |
|:---:|:---:|:---|:---|:---:|:---|:---:|
| **T05** | **C** | Label Smoothing | `loss=ls label_smoothing=0.1` | `loss=ce label_smoothing=0.0` | Mitigate overconfidence on dominant Negatives; regularize feature margins | $\approx 15.8$ min (79 s/ep) |
| **T06** | **C** | Focal Loss | `loss=focal focal_gamma=2.0` | `loss=ce focal_gamma=2.0` | Dynamically suppress gradients from well-classified majority Negatives | $\approx 16.0$ min (80 s/ep) |
| **T07** | **D** | Balanced Sampler | `sampler=balanced` | `sampler=None` (uniform shuffle) | Equalize batch-level sample exposure between 8 weed classes and Negatives | $\approx 16.0$ min (80 s/ep) |
| **T08** | **F** | Weight EMA | `ema_decay=0.999` | `ema_decay=None` (active weights) | Smooth optimization trajectory over noisy weed mini-batches; find flatter minima | $\approx 16.2$ min (81 s/ep) |

---

## 2. Quantitative Grounding: Class Imbalance in DeepWeeds Fold 0

### 2.1 Empirical Split Distribution (Fold 0)
Empirical extraction from `data/labels/train_subset0.csv` ($N = 10,501$) and `data/labels/val_subset0.csv` ($N = 3,501$) reveals severe class imbalance:

| Class ID | Class Name | Category | Train Samples | Train % | Val Samples | Val % | Ratio to Majority (Train) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **0** | Chinee Apple | Weed | 675 | 6.43% | 225 | 6.43% | 1 : 8.09 |
| **1** | Lantana | Weed | 637 | 6.07% | 213 | 6.08% | 1 : 8.58 |
| **2** | Parkinsonia | Weed | 618 | 5.89% | 206 | 5.88% | 1 : 8.84 |
| **3** | Parthenium | Weed | 613 | 5.84% | 204 | 5.83% | 1 : 8.91 |
| **4** | Prickly Acacia | Weed | 637 | 6.07% | 212 | 6.06% | 1 : 8.58 |
| **5** | Rubber Vine | Weed | 605 | 5.76% | 202 | 5.77% | 1 : 9.03 (Rarest) |
| **6** | Siam Weed | Weed | 644 | 6.13% | 215 | 6.14% | 1 : 8.48 |
| **7** | Snake Weed | Weed | 609 | 5.80% | 203 | 5.80% | 1 : 8.97 |
| **8** | **Negatives** | Background | **5,463** | **52.02%** | **1,821** | **52.01%** | **1.00 : 1.00 (Majority)** |
| **Total** | *All Classes* | - | 10,501 | 100.00% | 3,501 | 100.00% | - |

**Key Imbalance Insight**:
- The non-weed **Negatives** class constitutes **52.02%** of the dataset, outnumbering any individual weed species by a factor of **8.1× to 9.0×**.
- All 8 weed classes collectively represent 5,038 training images (47.98%), each averaging ~630 images (~6.0%).
- The rarest weed is **Rubber Vine** ($N=605$, 5.76%), while the most frequent weed is **Chinee Apple** ($N=675$, 6.43%).

### 2.2 Baseline B02 Per-Class Failure Analysis
Evaluating the out-of-fold validation logits from `runs/B02/seed0/val_logits.npy` against ground truth labels yields the per-class diagnostic breakdown:

```
                precision    recall  f1-score   support

  Chinee Apple     0.9484    0.8978    0.9224       225
       Lantana     0.9539    0.9718    0.9628       213
   Parkinsonia     0.9949    0.9515    0.9727       206
    Parthenium     0.9758    0.9902    0.9830       204
Prickly Acacia     0.9259    0.9434    0.9346       212
   Rubber Vine     0.9898    0.9653    0.9774       202
     Siam Weed     0.9904    0.9628    0.9764       215
    Snake Weed     0.9400    0.9261    0.9330       203
     Negatives     0.9745    0.9874    0.9809      1821

      accuracy                         0.9697      3501
     macro avg     0.9660    0.9551    0.9604      3501
  weighted avg     0.9698    0.9697    0.9696      3501
```

**Diagnostic Observations**:
1. **Negatives Dominates Recall**: Negatives achieves **98.74% Recall** and **98.09% F1**. Because standard Cross-Entropy minimizes uniform empirical risk $\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \ell_i$, over 52% of the gradient updates reward correct classifications on background pasture images.
2. **Rare Weeds Suffer Recall Degradation**:
   - **Chinee Apple** suffers the lowest recall (**89.78%**) and lowest F1 (**92.24%**). Over 10% of true Chinee Apple plants are misclassified as either Negatives or visually similar woody weeds.
   - **Snake Weed** has recall of **92.61%** and F1 of **93.30%**.
   - **Prickly Acacia** has precision of **92.59%** and F1 of **93.46%**.
3. **Macro-Average vs Overall Accuracy Gap**:
   - Overall accuracy is high at **96.97%**, but **Macro-F1 is 96.04%** (a 0.93% drop) because rare class errors are penalized equally in unweighted macro-averaging ($\frac{1}{K} \sum_{c=0}^8 F1_c$).
   - This empirical evidence confirms the urgent need for algorithmic techniques targeting class imbalance.

---

## 3. Mathematical Mechanisms & Theoretical Analysis

### 3.1 T05: Label Smoothing Cross-Entropy (`loss="ls"`, $\epsilon = 0.1$)

#### 3.1.1 Mathematical Formulation
Under standard Cross-Entropy, target labels are Dirac delta one-hot vectors:
$$q(k) = \begin{cases} 1 & \text{if } k = y \\ 0 & \text{if } k \neq y \end{cases}$$
Label smoothing replaces hard targets with a mixture of the ground truth and a uniform distribution across all $K=9$ classes:
$$q'(k) = (1 - \epsilon) q(k) + \frac{\epsilon}{K} = (1 - \epsilon)\mathbf{1}[k=y] + \frac{\epsilon}{K}$$
For $\epsilon = 0.1$ and $K = 9$:
- Ground truth class target: $q'(y) = 1.0 - 0.1 + \frac{0.1}{9} = 0.90 + 0.01111 = \mathbf{0.91111}$
- Non-target classes: $q'(k) = \frac{0.1}{9} = \mathbf{0.01111}$

The smoothed loss decomposes into:
$$\mathcal{L}_{\text{LS}}(p, q') = (1 - \epsilon) H(q, p) + \epsilon H(u, p) = (1 - \epsilon)(-\log p_y) + \epsilon \left( -\frac{1}{K}\sum_{k=1}^K \log p_k \right)$$
where $p_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$ is the predicted softmax probability.

#### 3.1.2 Gradient and Logit Dynamics
The gradient with respect to logit $z_k$ is:
$$\frac{\partial \mathcal{L}_{\text{LS}}}{\partial z_k} = p_k - q'(k) = p_k - \left((1 - \epsilon)\mathbf{1}[k=y] + \frac{\epsilon}{K}\right)$$
Under standard CE ($\epsilon = 0$), to reach loss near 0, the model must drive $p_y \to 1.0$, requiring $z_y - z_k \to +\infty$.
Under label smoothing, zero gradient occurs when $p_k = q'(k)$, meaning:
$$z_y^* - z_k^* = \log\left(\frac{q'(y)}{q'(k)}\right) = \log\left(\frac{(K-1)(1-\epsilon)}{\epsilon}\right) = \log\left(\frac{8 \times 0.9}{0.1}\right) = \log(72) \approx \mathbf{4.277}$$

#### 3.1.3 Impact on Class Imbalance
- **Mitigating Majority Logit Explosion**: With 5,463 Negatives images continually reinforcing the background class, standard CE causes Negatives logits to grow excessively large, creating wide decision margins that encroach upon neighboring rare weed classes. Label smoothing prevents this unbounded growth by capping optimal logit differences at $\approx 4.28$.
- **Feature Space Clustering**: By enforcing equidistant boundaries to non-target classes, label smoothing encourages compact penultimate representations (as demonstrated by Müller et al., NeurIPS 2019), preventing ambiguous weed samples from being subsumed by the majority Negatives cluster.
- **Trade-off / Caveat**: Label smoothing applies a uniform prior $1/K$ across all classes. It does not disproportionately penalize Negatives over rare weeds. It regularizes confidence, but does not explicitly adjust for class prevalence.

---

### 3.2 T06: Multi-Class Focal Loss (`loss="focal"`, $\gamma = 2.0$)

#### 3.2.1 Mathematical Formulation
Focal Loss (Lin et al., ICCV 2017) introduces a dynamically modulating factor $(1 - p_t)^\gamma$ to standard Cross-Entropy:
$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$
where $p_t = \frac{\exp(z_y)}{\sum_{j=1}^K \exp(z_j)}$ is the model's estimated probability for the ground truth class $y$, and $\gamma \ge 0$ is the focusing parameter (configured to $\gamma = 2.0$).

#### 3.2.2 Suppression of Easy Background Examples
The modulating factor scales loss based on example difficulty:

| Example Classification Status | Predicted $p_t$ | Standard CE Loss ($-\log p_t$) | Modulating Factor $(1 - p_t)^2$ | Focal Loss $\text{FL}(p_t)$ | Gradient Attenuation |
|:---|:---:|:---:|:---:|:---:|:---:|
| Very Easy Negative (clear pasture) | $0.99$ | $0.0101$ | $0.0001$ | $\mathbf{0.000001}$ | **$10,000\times$ down-weighted** |
| Easy Negative (typical dry grass) | $0.90$ | $0.1054$ | $0.0100$ | $\mathbf{0.001054}$ | **$100\times$ down-weighted** |
| Moderately Easy Example | $0.75$ | $0.2877$ | $0.0625$ | $\mathbf{0.01798}$ | **$16\times$ down-weighted** |
| Uncertain / Borderline Weed | $0.50$ | $0.6931$ | $0.2500$ | $\mathbf{0.17328}$ | **$4\times$ down-weighted** |
| Hard / Ambiguous Weed (Chinee Apple) | $0.20$ | $1.6094$ | $0.6400$ | $\mathbf{1.03004}$ | **$1.56\times$ down-weighted** |
| Completely Misclassified Weed | $0.05$ | $2.9957$ | $0.9025$ | $\mathbf{2.70365}$ | **$1.11\times$ down-weighted (Preserved)** |

#### 3.2.3 Impact on Class Imbalance
- **Overcoming the "Vast Majority of Easy Negatives" Problem**:
  In DeepWeeds, out of 5,463 Negative images, an estimated 4,000+ images are clear pasture, dirt, or rock without any vegetation resemblance. Under Cross-Entropy, even when $p_t \approx 0.95$, the cumulative loss across 4,000 images is $4000 \times 0.051 \approx 204$. In contrast, 100 hard weed images with $p_t \approx 0.20$ produce $100 \times 1.61 \approx 161$. Consequently, easy negatives dominate total gradient updates!
- Under Focal Loss ($\gamma = 2.0$), the 4,000 easy negatives contribute only $4000 \times (0.05)^2 \times 0.051 \approx \mathbf{0.51}$ to total loss!
- This shifts the effective gradient vector entirely onto the hard samples—predominantly rare weed species and fine-grained boundary cases—directly addressing the low recall observed in Chinee Apple and Snake Weed.

---

### 3.3 T07: Class-Balanced Sampling (`sampler="balanced"`)

#### 3.3.1 Mathematical Formulation
In `code/dataset.py`, `WeightedRandomSampler` computes per-sample weights inversely proportional to class frequency:
$$w_i = \frac{1}{N_{y_i}}, \quad \forall i \in \{1, \dots, N_{\text{train}}\}$$
where $N_c = \sum_{i=1}^{N_{\text{train}}} \mathbf{1}[y_i = c]$.
Normalized selection probability for sample $i$ belonging to class $c$:
$$P(\text{sample } i) = \frac{w_i}{\sum_{j=1}^N w_j} = \frac{1 / N_c}{\sum_{k=0}^{K-1} N_k (1 / N_k)} = \frac{1}{K \cdot N_c} = \frac{1}{9 \cdot N_c}$$

The probability of drawing an instance from class $c$ in any given draw is:
$$P(Y = c) = \sum_{i \in \text{Class } c} P(\text{sample } i) = N_c \cdot \frac{1}{9 \cdot N_c} = \frac{\mathbf{1}}{\mathbf{9}} \approx \mathbf{11.11\%}$$

#### 3.3.2 Batch Composition Comparison

| Class ID & Name | Train Count $N_c$ | T00 (Uniform) Probability $P(c)$ | T00 Expected Samples / Batch (BS=64) | T07 (Balanced) Probability $P(c)$ | T07 Expected Samples / Batch (BS=64) | Relative Exposure Multiplier |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 0: Chinee Apple | 675 | 6.43% | 4.11 | **11.11%** | **7.11** | **+1.73×** |
| 1: Lantana | 637 | 6.07% | 3.88 | **11.11%** | **7.11** | **+1.83×** |
| 2: Parkinsonia | 618 | 5.89% | 3.77 | **11.11%** | **7.11** | **+1.89×** |
| 3: Parthenium | 613 | 5.84% | 3.74 | **11.11%** | **7.11** | **+1.90×** |
| 4: Prickly Acacia | 637 | 6.07% | 3.88 | **11.11%** | **7.11** | **+1.83×** |
| 5: Rubber Vine | 605 | 5.76% | 3.69 | **11.11%** | **7.11** | **+1.93× (Max)** |
| 6: Siam Weed | 644 | 6.13% | 3.93 | **11.11%** | **7.11** | **+1.81×** |
| 7: Snake Weed | 609 | 5.80% | 3.71 | **11.11%** | **7.11** | **+1.92×** |
| **8: Negatives** | **5,463** | **52.02%** | **33.29** | **11.11%** | **7.11** | **-4.68× (Suppressed)** |

#### 3.3.3 Impact on Class Imbalance & Critical Nuances
- **Elimination of Majority Bias in Gradient Updates**:
  In T00, over 33 of the 64 images in every batch are Negatives. With balanced sampling, Negatives is reduced to ~7 images per batch, exactly equal to each weed class.
- **The Prior Shift Hazard**:
  Because training samples are drawn from a uniform prior ($P(y=c) = 1/9$), while validation data retains the natural test distribution ($P(y=8) = 52.0\%$), the model's posterior probability:
  $$P(y=c \mid x) \propto P(x \mid y=c) P_{\text{train}}(y=c)$$
  will artificially inflate weed probabilities by a factor of $\approx 4.7\times$. This often increases False Positives on weed classes, lowering weed Precision while boosting Recall. Whether this net trade-off improves Macro-F1 is a central scientific inquiry for Milestone 2.
- **Overfitting Risk**: Because $N_{\text{samples}} = 10,501$ with `replacement=True`, a rare Rubber Vine image is sampled on average $\approx 1.93$ times per epoch. Without strong data augmentation, the backbone could memorize rare training images.

---

### 3.4 T08: Exponential Moving Average (`ema_decay = 0.999`)

#### 3.4.1 Mathematical Formulation
Exponential Moving Average maintains a shadow set of model parameters $\theta_{\text{shadow}}$ that smoothly tracks the active optimization parameters $\theta_t$:
$$\theta_{\text{shadow}}^{(t)} = d \cdot \theta_{\text{shadow}}^{(t-1)} + (1 - d) \cdot \theta_t$$
where decay $d = 0.999$.
Expanding iteratively over time:
$$\theta_{\text{shadow}}^{(T)} = (1 - d) \sum_{k=0}^{T-1} d^k \theta_{T-k}$$
The effective temporal window (number of optimization steps averaged) is:
$$\tau = \frac{1}{1 - d} = \frac{1}{1 - 0.999} = \mathbf{1,000 \text{ optimization steps}}$$
With $N_{\text{train}} = 10,501$ and $\text{batch\_size} = 64$, there are $S = \lceil 10501 / 64 \rceil = 165$ steps per epoch.
Therefore, a 1,000-step averaging window corresponds to:
$$\text{Window (epochs)} = \frac{1000}{165} \approx \mathbf{6.06 \text{ epochs}}$$

#### 3.4.2 Loss Surface Geometry and Polyak Averaging
- Stochastic gradient descent on non-convex loss surfaces oscillates around valley floors due to gradient noise.
- Under class imbalance, mini-batches with rare weeds induce high-variance, spiky gradient steps. Active weights $\theta_t$ continually bounce between steep ravines.
- EMA acts as Polyak-Ruppert averaging (Polyak & Juditsky, 1992), projecting weights toward the center of the widest loss basin.
- Wide, flat minima are proven to have superior out-of-sample generalization because feature shifts between training and validation sets do not cause steep loss spikes.

#### 3.4.3 Interaction with Evaluation in `code/train.py`
In `code/train.py`:
```python
if ema is not None:
    ema.apply_shadow(model)
val_fnames, val_true, val_logits, val_loss = evaluate(model, val_loader, val_criterion, device)
if ema is not None:
    ema.restore(model)
```
- During validation, shadow weights are evaluated.
- Because EMA eliminates stochastic weight oscillation, validation Macro-F1 curves are anticipated to show significantly smoother trajectories with fewer sudden dips across epochs 7–12.

---

## 4. Codebase Audit & CLI Invocation Verification

### 4.1 CLI Argument Parser Audit (`code/train.py`)
`code/train.py` defines `parse_overrides(pairs)` and `main()`:
```python
parser.add_argument("--exp_id", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--fold", type=int, default=None)
parser.add_argument("--set", nargs="*", default=[])
```
Field types in `Config` are parsed dynamically:
- `loss: str = "ce"` $\implies$ parsed as string.
- `label_smoothing: float = 0.0` $\implies$ parsed as float.
- `focal_gamma: float = 2.0` $\implies$ parsed as float.
- `sampler: Optional[str] = None` $\implies$ parsed as string ("balanced").
- `ema_decay: Optional[float] = None` $\implies$ parsed as float (0.999).
- `num_workers: int = 2` $\implies$ set to `0` for robust Windows execution.

### 4.2 CRITICAL DISCOVERY: Explicit `label_smoothing=0.1` Requirement
In `code/train.py`:
```python
# Line 76-78
loss: str = "ce"
label_smoothing: float = 0.0

# Line 368-369
elif cfg.loss in ("ls", "label_smoothing"):
    criterion = build_criterion("ls", smoothing=cfg.label_smoothing)
```
And in `code/losses.py`:
```python
# Line 43-45
def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if self.smoothing <= 0.0:
        return F.cross_entropy(logits, target)
```
**CRITICAL VERIFICATION**:
If an operator executes `--set loss=ls` WITHOUT explicitly passing `label_smoothing=0.1`, `cfg.label_smoothing` defaults to `0.0`. In `LabelSmoothingCE.forward()`, if `smoothing <= 0.0`, it returns standard `F.cross_entropy(logits, target)`.
**Thus, passing `loss=ls` without `label_smoothing=0.1` silently degenerates to vanilla Cross-Entropy!**
Therefore, the CLI command MUST explicitly include both flags:
`--set backbone=convnext_tiny loss=ls label_smoothing=0.1 num_workers=0`

### 4.3 Dry-Run Verification Results
Executing configuration instantiations verified that all four configurations load without error:
- **T05**: `Config(exp_id='T05', backbone='convnext_tiny', loss='ls', label_smoothing=0.1)` $\implies$ Criterion: `LabelSmoothingCE(smoothing=0.1)`
- **T06**: `Config(exp_id='T06', backbone='convnext_tiny', loss='focal', focal_gamma=2.0)` $\implies$ Criterion: `FocalLoss(gamma=2.0)`
- **T07**: `Config(exp_id='T07', backbone='convnext_tiny', sampler='balanced')` $\implies$ DataLoader: `WeightedRandomSampler(num_samples=10501, replacement=True)`
- **T08**: `Config(exp_id='T08', backbone='convnext_tiny', ema_decay=0.999)` $\implies$ Optimizer wrapper: `EMA(decay=0.999)`

---

## 5. Execution Specifications: Commands, Artifacts & Naming

### 5.1 Sequential Execution Commands
All commands MUST be run with `$env:PYTHONUTF8=1` on PowerShell:

#### Experiment T05 (Axis C — Label Smoothing)
```powershell
$env:PYTHONUTF8=1; python code/train.py --exp_id T05 --seed 0 --fold 0 --set backbone=convnext_tiny loss=ls label_smoothing=0.1 num_workers=0
```

#### Experiment T06 (Axis C — Focal Loss)
```powershell
$env:PYTHONUTF8=1; python code/train.py --exp_id T06 --seed 0 --fold 0 --set backbone=convnext_tiny loss=focal focal_gamma=2.0 num_workers=0
```

#### Experiment T07 (Axis D — Balanced Sampler)
```powershell
$env:PYTHONUTF8=1; python code/train.py --exp_id T07 --seed 0 --fold 0 --set backbone=convnext_tiny sampler=balanced num_workers=0
```

#### Experiment T08 (Axis F — Weight EMA)
```powershell
$env:PYTHONUTF8=1; python code/train.py --exp_id T08 --seed 0 --fold 0 --set backbone=convnext_tiny ema_decay=0.999 num_workers=0
```

### 5.2 Artifact Paths & Curve Naming Conventions

For each experiment `T0x`:
1. **Checkpoint & Metadata Directory**: `runs/T0x/seed0/`
   - `runs/T0x/seed0/config.json`: Complete serialized `Config` dataclass.
   - `runs/T0x/seed0/best_checkpoint.pt`: Checkpoint state dict containing `epoch`, `model_state_dict`, `macro_f1`, `top1`, and `cfg`.
   - `runs/T0x/seed0/summary.json`: Top-level metrics (`best_val_macro_f1`, `best_val_top1`, `best_epoch`, `avg_epoch_time_s`, `params_m`, `gmacs`).
   - `runs/T0x/seed0/history.csv`: Full epoch logs (columns: `epoch`, `train_loss`, `val_loss`, `val_macro_f1`, `val_top1`, `lr`, `epoch_time_s`).
   - `runs/T0x/seed0/val_logits.npy`: Raw out-of-fold validation logits (shape: $3501 \times 9$, FP32).
   - `predictions/T0x_seed0_val.csv`: Formatted validation predictions generated by `eval.save_predictions`.
2. **Curve Plots**:
   In `code/train.py`, line 479 automatically generates `curves/<exp_id>_<backbone>.png`. To ensure strict compliance with both `PROJECT.md` and `ORIGINAL_REQUEST.md` rubric requirements (`curves/T0x_<mota>.png`), curves will be generated and aliased as follows:

| Exp ID | Primary Code Curve Path | Descriptive Alias Path (`curves/T0x_<description>.png`) |
|:---:|:---|:---|
| **T05** | `curves/T05_convnext_tiny.png` | `curves/T05_label_smoothing.png` |
| **T06** | `curves/T06_convnext_tiny.png` | `curves/T06_focal_loss.png` |
| **T07** | `curves/T07_convnext_tiny.png` | `curves/T07_balanced_sampler.png` |
| **T08** | `curves/T08_convnext_tiny.png` | `curves/T08_ema_decay.png` |

---

## 6. Integration Schema for Master Workbook (`results.xlsx`)

Sheet `Training` in `results.xlsx` requires 9 columns:
`["exp_id", "backbone", "trục thay đổi (A–G)", "khác T00 ở điểm nào", "seed", "macro-F1 val", "top-1 val", "Δ so với T00", "ghi chú"]`

### 6.1 Baseline Anchor (Row 2)
The baseline anchor T00 is extracted from Milestone 1 (`B02: convnext_tiny`):
- `exp_id`: `T00`
- `backbone`: `convnext_tiny`
- `trục thay đổi (A–G)`: `Chuẩn đối chứng`
- `khác T00 ở điểm nào`: `Công thức chuẩn T00 (AdamW, CE loss, lr 1e-4/1e-3, 12 epochs)`
- `seed`: `0`
- `macro-F1 val`: `0.9604` (exact: `0.960354`)
- `top-1 val`: `0.9697` (exact: `0.969723`)
- `Δ so với T00`: `0.0000`
- `ghi chú`: `Mốc chuẩn Milestone 1 (B02); 11.44ms latency, 27.83M params`

### 6.2 Target Ablation Rows (T05–T08)

| exp_id | backbone | trục thay đổi (A–G) | khác T00 ở điểm nào | seed | macro-F1 val | top-1 val | Δ so với T00 | ghi chú |
|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **T05** | convnext_tiny | C (Hàm loss) | `loss="ls"`, `label_smoothing=0.1` | 0 | `[Pending Run]` | `[Pending Run]` | $\text{F1}_{\text{T05}} - 0.9604$ | Giảm overconfidence ở lớp Negatives, giới hạn khoảng cách logit tối ưu $\le 4.28$ |
| **T06** | convnext_tiny | C (Hàm loss) | `loss="focal"`, $\gamma=2.0$ | 0 | `[Pending Run]` | `[Pending Run]` | $\text{F1}_{\text{T06}} - 0.9604$ | Giảm tổn thất từ mẫu Negatives dễ nhận biết tới 100-10,000 lần, tập trung vào cỏ hiếm |
| **T07** | convnext_tiny | D (Cân bằng mẫu) | `sampler="balanced"` (WeightedRandomSampler) | 0 | `[Pending Run]` | `[Pending Run]` | $\text{F1}_{\text{T07}} - 0.9604$ | Cân bằng xác suất chọn mẫu giữa 9 lớp (mỗi lớp ~7 mẫu/batch), tăng tần suất cỏ hiếm ~1.9× |
| **T08** | convnext_tiny | F (Chính quy hóa) | `ema_decay=0.999` (EMA shadow weights) | 0 | `[Pending Run]` | `[Pending Run]` | $\text{F1}_{\text{T08}} - 0.9604$ | Trung bình động tham số theo cửa sổ 1.000 steps (~6 epoch), hội tụ về đáy phẳng |

### 6.3 Scientific Noise Evaluation Framework
Per Slide Day 2 p. 59 and `ORIGINAL_REQUEST §R3`:
- Standard deviation across random seeds is $\sigma_{\text{seed}} \approx 0.10\% - 0.30\%$ ($0.0010 - 0.0030$).
- **Decision Rule**:
  - If $|\Delta| < 0.0030$: The change is classified as **"không phân biệt được với nhiễu ngẫu nhiên"** (statistically indistinguishable from seed variance).
  - If $\Delta \ge +0.0030$: The change is classified as a **statistically significant improvement**.
  - If $\Delta \le -0.0030$: The change is classified as an **active performance regression**.

---

## 7. Zero Test Set Leakage Protocol
Strict data integrity (Rules S1–S6) is strictly maintained:
1. `save_test_predictions=False` is maintained across all training scripts during Milestone 2.
2. `test_subset0.csv` is NEVER loaded into memory during hyperparameter selection or ablation comparisons.
3. All model checkpoints, validation metrics, and early stopping decisions are computed strictly from `val_subset0.csv`.
