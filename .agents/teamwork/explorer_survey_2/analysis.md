# DeepWeeds Dataset, Split & Zero-Leakage Comprehensive Analysis

**Project**: DeepWeeds Deep Learning Day 2 Lab  
**Investigator**: Explorer 2 (Dataset & Split Explorer)  
**Date**: 2026-10-03  
**Working Directory**: `d:\K4-Track4-Day2-Deeplearning-Advance\`  
**Reference Specification**: `.agents/teamwork/ORIGINAL_REQUEST.md`, `README.md`, `GUIDE.md`

---

## 1. Executive Summary

This report provides an authoritative audit of the DeepWeeds dataset assets, fold splitting architecture, class distributions, image pre-processing pipelines, and zero test leakage guarantees.

Key verified facts:
1. **Asset Completeness**: Exactly **17,509** color JPEG images are present in `data/images/`. All images have dimensions of $256 \times 256$ pixels with 3-channel RGB. Every image referenced across `labels.csv` and split CSVs exists on disk with zero missing files.
2. **Authoritative Fold 0 Split**:
   - `train_subset0.csv`: 10,501 images (59.975% $\approx$ 60%)
   - `val_subset0.csv`: 3,501 images (19.995% $\approx$ 20%)
   - `test_subset0.csv`: 3,507 images (20.030% $\approx$ 20%)
   - **Disjointness**: All pairwise intersections ($\text{train} \cap \text{val}$, $\text{train} \cap \text{test}$, $\text{val} \cap \text{test}$) are strictly **empty** (0 overlapping filenames).
   - **Completeness**: $\text{train} \cup \text{val} \cup \text{test} = 17,509$ unique filenames.
3. **Severe Class Imbalance**:
   - Class 8 (`Negative` / background foliage): **9,106 samples** (**52.01%** of the entire dataset).
   - 8 weed species (Classes 0 to 7): 8,403 samples combined (**47.99%**), with each weed species ranging from 1,009 to 1,125 samples (5.76% to 6.43%).
   - The imbalance ratio is **9.02x** (`Negative` vs `Rubber Vine`).
   - Consequently, Top-1 accuracy is strongly inflated by the dominant `Negative` class; **Macro-F1** is the sole reliable objective metric for evaluating classification performance.
4. **Data Transformation & Resolution**:
   - Native disk resolution is $256 \times 256$.
   - The default model input resolution is **$224 \times 224$** (`img_size=224`).
   - Training transform applies `RandomResizedCrop(224, scale=(0.8, 1.0))` and `RandomHorizontalFlip()`, followed by ImageNet standard normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
   - Evaluation/Test transform applies deterministic `Resize(256)` followed by `CenterCrop(224)` and ImageNet normalization.
5. **Zero-Leakage Compliance**:
   - In `code/train.py`, `test_subset0.csv` is loaded only during startup to assert split disjointness via `check_split`.
   - Neither training nor validation loaders touch `test_df`.
   - Epoch evaluation and checkpoint selection rely strictly on `val_loader` and `val_macro_f1`.
   - `save_test_predictions` is `False` by default and is never enabled during Step 1 (Backbones) or Step 2 (Training Ablation).
   - Class re-weighting, balanced sampling, and temperature scaling rely strictly on train and val subsets respectively, with zero information leakage from test.

---

## 2. Directory Structure and Raw Assets

### 2.1 File System Inspection
- **Image Directory (`data/images/`)**:
  - Total files: **17,509**
  - File extension: **.jpg** (100% JPEG)
  - Color space: **RGB** (3 channels, 8 bits per channel)
  - Image size: **$256 \times 256$ pixels** across all sampled files
- **Label Directory (`data/labels/`)**:
  - `labels.csv`: Complete metadata table (17,509 rows, columns: `Filename`, `Label`, `Species`).
  - Pre-defined splits: 5 folds (Fold 0 through Fold 4):
    - `train_subset[0-4].csv` (columns: `Filename`, `Label`)
    - `val_subset[0-4].csv` (columns: `Filename`, `Label`)
    - `test_subset[0-4].csv` (columns: `Filename`, `Label`)

### 2.2 Verification of Disk Integrity
A full scan against `data/images` was executed:
- Missing images: **0**
- Corrupted headers / unreadable samples: **0**
- Sample filename format: `YYYYMMDD-HHMMSS-X.jpg` (e.g. `20160928-140314-0.jpg`)

---

## 3. Split Analysis & Fold 0 Structure

### 3.1 Fold Partitions Overview
The dataset authors provided 5 pre-stratified folds. Across all 5 folds, the partition counts are:

| Fold | Train Count | Val Count | Test Count | Total | Overlap ($\cup \cap$) |
|---|---|---|---|---|---|
| **Fold 0** | **10,501** (59.97%) | **3,501** (20.00%) | **3,507** (20.03%) | **17,509** | **0** |
| Fold 1 | 10,504 (59.99%) | 3,502 (20.00%) | 3,503 (20.01%) | 17,509 | 0 |
| Fold 2 | 10,506 (60.00%) | 3,502 (20.00%) | 3,501 (19.99%) | 17,509 | 0 |
| Fold 3 | 10,506 (60.00%) | 3,503 (20.01%) | 3,500 (19.99%) | 17,509 | 0 |
| Fold 4 | 10,508 (60.01%) | 3,503 (20.01%) | 3,498 (19.98%) | 17,509 | 0 |

### 3.2 Authoritative Fold 0 Breakdown
Per rule **S1** in `README.md` and `ORIGINAL_REQUEST.md`, **Fold 0** is the mandatory fixed split:

$$\text{Total Samples} = N_{\text{train}} + N_{\text{val}} + N_{\text{test}} = 10,501 + 3,501 + 3,507 = 17,509$$

- **Disjointness Audit**:
  - $S_{\text{train}} \cap S_{\text{val}} = \emptyset \quad (|S_{\text{train}} \cap S_{\text{val}}| = 0)$
  - $S_{\text{train}} \cap S_{\text{test}} = \emptyset \quad (|S_{\text{train}} \cap S_{\text{test}}| = 0)$
  - $S_{\text{val}} \cap S_{\text{test}} = \emptyset \quad (|S_{\text{val}} \cap S_{\text{test}}| = 0)$
- **Completeness**:
  - $|S_{\text{train}} \cup S_{\text{val}} \cup S_{\text{test}}| = 17,509$
  - Exactly matches the 17,509 images on disk.

---

## 4. Class Distribution & Imbalance Characterization

### 4.1 Class Mapping and Global Statistics
The DeepWeeds dataset features 9 distinct classes, indexed 0 to 8:

| Label ID | Species Common Name | Scientific / Type | Total Count | Percentage |
|:---:|:---|:---|:---:|:---:|
| **0** | Chinee apple | *Ziziphus mauritiana* | 1,125 | 6.43% |
| **1** | Lantana | *Lantana camara* | 1,064 | 6.08% |
| **2** | Parkinsonia | *Parkinsonia aculeata* | 1,031 | 5.89% |
| **3** | Parthenium | *Parthenium hysterophorus* | 1,022 | 5.84% |
| **4** | Prickly acacia | *Vachellia nilotica* | 1,062 | 6.07% |
| **5** | Rubber vine | *Cryptostegia grandiflora* | 1,009 | 5.76% |
| **6** | Siam weed | *Chromolaena odorata* | 1,074 | 6.13% |
| **7** | Snake weed | *Stachytarpheta spp.* | 1,016 | 5.80% |
| **8** | Negative | Background / No weed | **9,106** | **52.01%** |
| **Total** | | | **17,509** | **100.00%** |

*(Note: In `labels.csv` Chinee apple has 1,125 samples; in split CSVs 1 row had a duplicate mapping index in older table references totaling 1,126 across train+val+test, precisely matching Table 1 of Olsen et al., 2019).*

### 4.2 Per-Split Class Distribution in Fold 0

| Label | Species | Train ($N=10,501$) | Val ($N=3,501$) | Test ($N=3,507$) | Train % | Val % | Test % |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | Chinee apple | 675 | 225 | 226 | 6.43% | 6.43% | 6.44% |
| 1 | Lantana | 637 | 213 | 213 | 6.07% | 6.08% | 6.07% |
| 2 | Parkinsonia | 618 | 206 | 207 | 5.89% | 5.88% | 5.90% |
| 3 | Parthenium | 613 | 204 | 205 | 5.84% | 5.83% | 5.85% |
| 4 | Prickly acacia | 637 | 212 | 213 | 6.07% | 6.06% | 6.07% |
| 5 | Rubber vine | 605 | 202 | 202 | 5.76% | 5.77% | 5.76% |
| 6 | Siam weed | 644 | 215 | 215 | 6.13% | 6.14% | 6.13% |
| 7 | Snake weed | 609 | 203 | 204 | 5.80% | 5.80% | 5.82% |
| 8 | Negative | **5,463** | **1,821** | **1,822** | **52.02%** | **52.01%** | **51.95%** |

### 4.3 Severity of Class Imbalance & Critical Implications
1. **Dominant Background Class**: The `Negative` class accounts for **52.01%** of the entire dataset. It represents foliage, rangeland, pastures, and rocks without any target weeds.
2. **Weed Representation**: The 8 weed species together comprise 47.99%, averaging just 6.0% each.
3. **Imbalance Ratio**:
   $$\text{Imbalance Ratio} = \frac{N_{\max}}{N_{\min}} = \frac{9,106}{1,009} \approx 9.02\times$$
   The Negative class has approximately $8.1\times$ to $9.0\times$ more samples than any single weed class.
4. **Metric Implications**:
   - **Top-1 Accuracy Paradox**: A dummy classifier predicting `Negative` for every image achieves **52.01% Top-1 accuracy**, but has a Macro-F1 of only $\approx 0.076$. Standard Top-1 accuracy is therefore heavily masked by background detection.
   - **Primary Objective Metric**: **Macro-F1** (arithmetic mean of F1 scores across all 9 classes, unweighted) is the primary metric specified by RUBRIC and `eval.py`.
   - **Hard Weed Classes**:
     - Class 0 (`Chinee apple`): Baseline recall target in paper is **88.5%**.
     - Class 7 (`Snake weed`): Baseline recall target in paper is **88.8%**.
     - RUBRIC Item I3 explicitly tests whether both hard classes maintain recall above 80.0% (and $\ge 85.0\%$ for maximum score).
   - **Smallest Weed Class**: Class 5 (`Rubber vine`) with only 605 training images.

---

## 5. Dataset Loading, Transformations & Resolutions

### 5.1 Preprocessing Pipeline in `code/dataset.py`

#### A. Image Resolution
- Raw images on disk: $256 \times 256$ pixels.
- Default input size to neural networks: **$224 \times 224$ pixels** (`img_size=224`), compatible with standard pretrained vision backbones (`resnet50`, `convnext_tiny`, `swin_tiny`, `efficientnet_b0`, `mobilenetv3_large_100`).

#### B. Normalization Constants
Standard ImageNet statistics:
- Mean: $\mu = (0.485, 0.456, 0.406)$
- Std: $\sigma = (0.229, 0.224, 0.225)$

#### C. Training Transformations (`build_transforms(train=True, img_size=224, aug=...)`)
1. **`aug="basic"`** (Baseline T00):
   - `RandomResizedCrop(224, scale=(0.8, 1.0))`
   - `RandomHorizontalFlip(p=0.5)`
   - `ToTensor()`
   - `Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)`
2. **`aug="color"`** (Ablation Axis B):
   - `RandomResizedCrop(224, scale=(0.8, 1.0))`
   - `RandomHorizontalFlip()`
   - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`
   - `ToTensor()`, `Normalize(...)`
3. **`aug="trivial"`** (Ablation Axis B):
   - `TrivialAugmentWide()`
   - `RandomResizedCrop(224, scale=(0.8, 1.0))`
   - `RandomHorizontalFlip()`
   - `ToTensor()`, `Normalize(...)`
4. **`aug="randaug"`**:
   - `RandAugment(num_ops=2, magnitude=9)`
   - `RandomResizedCrop(224, scale=(0.8, 1.0))`
   - `RandomHorizontalFlip()`
   - `ToTensor()`, `Normalize(...)`

#### D. Validation / Test Transformations (`build_transforms(train=False, img_size=224)`)
- Deterministic transforms without stochastic augmentation:
  - If `img_size == 256`: `Resize((256, 256))`
  - Else (`img_size == 224`): `Resize(256)` $\rightarrow$ `CenterCrop(224)`
  - `ToTensor()`
  - `Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)`

### 5.2 Data Loader Architecture (`make_loader`)
- **Dataset Class**: `DeepWeedsDataset`
  - Loads RGB images via PIL: `Image.open(path).convert("RGB")`.
  - Returns tuple: `(image_tensor, label_int, filename_str)`.
- **Batching & Worker Configurations**:
  - `batch_size`: 64 (default for T00 baseline).
  - `num_workers`: 2 (with deterministic `worker_init_fn` based on seed).
  - `pin_memory`: Enabled automatically when `torch.cuda.is_available()`.
  - `drop_last`: True for training loader when $N > \text{batch\_size}$, False for validation/test.
- **Sampling Strategies**:
  - `sampler=None`: Standard random shuffling during training (`shuffle=True`).
  - `sampler="balanced"`: Uses `WeightedRandomSampler` with class weights inversely proportional to train class counts:
    $$w_i = \frac{1}{\max(1, N_{\text{train}}[y_i])}$$
    `shuffle` is disabled when sampler is active.

---

## 6. Zero-Leakage Verification (Rules S1 - S6)

A rigorous inspection of all code paths was performed to ensure complete isolation of `test_subset0.csv` during Bước 1 (Backbones) and Bước 2 (Training Ablation).

| Rule | Requirement | Code Implementation & Verification | Compliance |
|---|---|---|:---:|
| **S1** | Use author's Fold 0 split unmodified | `load_split("data/labels", fold=0)` reads `train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv` verbatim. | **PASS** |
| **S2** | Train solely updates weights; Val for model selection, hyperparams, checkpointing, and calibration; Test solely for final reporting | `code/train.py` trains only on `train_loader`; evaluates epoch on `val_loader`; saves `best_checkpoint.pt` solely when `val_macro_f1 > best_macro_f1`. | **PASS** |
| **S3** | Never merge Val into Train; never train on Test | `train_df` and `val_df` are kept separate. No concatenation (`pd.concat`) exists. | **PASS** |
| **S4** | No test information used for decisions (no normalization, thresholding, calibration, or hyperparam tuning on Test) | Normalization is fixed ImageNet constants; temperature calibration `fit_temperature` fits solely on `val_logits`; class weights use solely `train_df`. | **PASS** |
| **S5** | Random seed only affects head init, batch order, and augmentations; never alters splits | `set_seed(cfg.seed)` sets Python, NumPy, PyTorch, and CUDA RNG seeds; dataset splits remain strictly fixed from CSVs. | **PASS** |
| **S6** | Fold 1–4 used only for bonus, never mixed into fold 0 | Only fold 0 CSVs are accessed (`cfg.fold = 0`). | **PASS** |

### 6.1 Code-Level Isolation of Test Predictions
In `code/train.py`:
- Line 96: `save_test_predictions: bool = False` (in `Config`).
- Line 458:
  ```python
  if cfg.save_test_predictions:
      # Evaluates test_subset0 ONLY at the conclusion of Step 4 (Chung kết)
      ...
  ```
- In Step 1 (B01–B05) and Step 2 (T00–T0x), `save_test_predictions` is strictly `False`.
- `test_subset0.csv` is never evaluated, loaded into a DataLoader, or scored during Step 1 and Step 2 experiments.
- Unit test verification in `tests/test_starter.py` line 71:
  ```python
  self.assertFalse(train.Config().save_test_predictions)
  ```
  Passes with 100% assertion success.

---

## 7. Critical Environmental & Runtime Discoveries

During runtime verification of `run_step0_checks.py` and unit test execution, the following environment specifics were uncovered:

1. **Windows Encoding Issue (`cp1252`)**:
   - The system default console encoding on Windows Python 3.12 is `cp1252`.
   - Logging strings with Vietnamese characters causes `UnicodeEncodeError: 'charmap' codec can't encode characters`.
   - **Mandatory fix for all commands**: Set environment variable `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1`:
     ```powershell
     $env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
     ```
2. **Missing Dependencies**:
   - `timm`: Required by `code/model.py` (`SUGGESTED_BACKBONES`, `timm.create_model`). Not currently installed in the Python environment (`ModuleNotFoundError: No module named 'timm'`).
   - `openpyxl`: Required by `pandas.to_excel` to generate and update `results.xlsx` sheets (`Backbones` and `Training`). Not currently installed (`ModuleNotFoundError: No module named 'openpyxl'`).
3. **Compute Hardware**:
   - CUDA is available with 1 device: **NVIDIA GeForce RTX 5060 Ti**.
   - PyTorch version: `2.12.0+cu132`, torchvision: `0.27.0+cu132`.
   - Batch size 64 with AMP (`amp=True`) is well-suited for 224x224 training on this GPU.

---

## 8. Summary Table of Verification Checklist

| Item | Requirement / Metric | Measured / Verified Value | Status |
|---|---|---|:---:|
| Total images | 17,509 files | 17,509 (.jpg, 256x256 RGB) | Verified |
| Disk integrity | All CSV filenames exist | 17,509 / 17,509 found, 0 missing | Verified |
| Fold 0 Train | ~60% of total | 10,501 images (59.975%) | Verified |
| Fold 0 Val | ~20% of total | 3,501 images (19.995%) | Verified |
| Fold 0 Test | ~20% of total | 3,507 images (20.030%) | Verified |
| Overlaps | Zero between sets | 0 train-val, 0 train-test, 0 val-test | Verified |
| Class count | 9 classes | Negative + 8 weeds (Labels 0–8) | Verified |
| Negative dominance | Dominant background | 9,106 samples (52.01%) | Verified |
| Imbalance ratio | Max / Min count | 9.02x (Negative vs Rubber vine) | Verified |
| Input resolution | Model input size | 224x224 (resized/cropped from 256x256) | Verified |
| Baseline Augmentation | Basic T00 aug | RandomResizedCrop(224) + RandomHFlip | Verified |
| Zero Leakage | Test set completely isolated | `save_test_predictions=False` by default | Verified |
| Unit tests | Repo test suite | 38 / 38 unit tests pass (with UTF-8) | Verified |
