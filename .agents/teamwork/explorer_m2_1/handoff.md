# Milestone 2 Handoff Report: Single-Factor Ablation Design (Axis A & Axis B)

**Agent**: Explorer 1 (`explorer_m2_1`)  
**Milestone**: Milestone 2 — Training Recipe Ablation Design (Step 2)  
**Parent**: Orchestrator (`c4718f41-3030-43ce-8fcc-465340c45738`)  
**Date**: 2026-10-03  
**Status**: Hard Handoff (Investigation & Technical Specification Complete)  

---

## 1. Observation

### 1.1 Milestone 1 Baseline Grounding (`convnext_tiny`)
From `worker_m1/handoff.md` (lines 54–64, 116–120, 163–200) and `runs/B02/seed0/summary.json`:
- **Winning Backbone**: `convnext_tiny` was selected from Step 1 with the highest composite utility ($U=0.850$), dominating both classification accuracy and latency.
- **Pretrained Tag**: `in12k_ft_in1k` (ImageNet-12k pretraining fine-tuned on ImageNet-1k).
- **Parameters**: Total = 27.827 M (Backbone = 27,820,320 params, Classifier Head = 6,921 params).
- **Complexity**: 4.455 GMACs at $224 \times 224$ resolution.
- **Latency (batch 1, RTX 5060 Ti)**: p50 = 11.44 ms, mean = 11.69 ms, p95 = 13.83 ms.
- **Baseline Performance (T00 / B02)**:
  - Val Macro-F1: **96.04%** (`0.9604`)
  - Val Top-1 Accuracy: **96.97%** (`0.9697`)
  - Best Epoch: 11 (Plateau: 95.93% at Ep 12)
  - Train Speed: **79.11 s/epoch** (~15.8 minutes for 12 epochs)
- **Baseline Recipe T00 Setup**:
  AdamW ($\text{lr}_{\text{backbone}}=10^{-4}$, $\text{lr}_{\text{head}}=10^{-3}$, $\text{weight\_decay}=0.05$), 1 epoch linear warmup + cosine decay, Cross-Entropy loss, AMP on CUDA, batch size 64, epochs 12, seed 0, fold 0, `aug="basic"` (`RandomResizedCrop(224, (0.8, 1.0))` + `RandomHorizontalFlip()`).

### 1.2 Codebase Investigation for Axis A (Initialization)
1. **`code/model.py` (lines 30–77)**:
   ```python
   def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                   drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
       model_name = SUGGESTED_BACKBONES.get(name, name)
       use_pretrained = (init != "scratch") and pretrained
       model = timm.create_model(model_name, pretrained=use_pretrained, num_classes=num_classes, drop_rate=drop_rate)
       ...
       if init == "frozen":
           freeze_backbone(model)
       return model
   ```
   - For `init="scratch"`: `use_pretrained` evaluates to `False`. All 27.827 M parameters are randomly initialized.
   - For `init="frozen"`: `freeze_backbone(model)` sets `param.requires_grad = False` across the entire backbone, leaving only the classifier head (`model.get_classifier()`) with `requires_grad = True`.
2. **Empirical Verification of Trainable Parameters**:
   Executed command:
   ```powershell
   python -c "import sys; sys.path.insert(0, 'code'); import model; m_scratch = model.build_model('convnext_tiny', init='scratch'); print('scratch params:', model.count_params(m_scratch)); m_frozen = model.build_model('convnext_tiny', init='frozen'); trainable = sum(p.numel() for p in m_frozen.parameters() if p.requires_grad); print('frozen trainable params:', trainable)"
   ```
   Output:
   - `scratch params: 27.827`
   - `frozen trainable params: 6921`
   Exactly 6,921 trainable parameters ($768 \times 9 + 9$) in `init="frozen"`, while 27,820,320 parameters are completely frozen.
3. **`code/train.py` (lines 178–184)**:
   ```python
   if cfg.init == "frozen":
       model.eval()
       classifier = model.get_classifier()
       if classifier is not None and isinstance(classifier, nn.Module):
           classifier.train()
   else:
       model.train()
   ```
   Ensures the backbone remains in `eval()` mode throughout training while the classifier head trains.

### 1.3 Codebase Investigation for Axis B (Augmentation & CutMix)
1. **`code/dataset.py` (lines 156–189)**:
   - `aug="basic"`: `RandomResizedCrop(224, (0.8, 1.0))` + `RandomHorizontalFlip` + Normalize.
   - `aug="color"`: `RandomResizedCrop` + `RandomHorizontalFlip` + `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)` + Normalize.
   - `aug="trivial"`: `TrivialAugmentWide()` + `RandomResizedCrop` + `RandomHorizontalFlip` + Normalize.
2. **`code/losses.py` (lines 119–177) & `code/train.py` (lines 196–201)**:
   - Regional CutMix implementation: Samples $\lambda \sim \text{Beta}(1.0, 1.0)$, samples random bounding box coordinates $(cx, cy, cut\_w, cut\_h)$, clips to image boundaries, pastes cropped patch from permuted sample into target image, computes actual area fraction $\lambda_{\text{actual}}$, and applies `mixed_loss = \lambda \mathcal{L}(y_a) + (1 - \lambda) \mathcal{L}(y_b)`.
3. **Empirical Verification of Transforms & CutMix Execution**:
   Executed command:
   ```powershell
   python -c "import sys; sys.path.insert(0, 'code'); import dataset, losses, torch; tf_triv = dataset.build_transforms(train=True, aug='trivial'); tf_col = dataset.build_transforms(train=True, aug='color'); x = torch.randn(4, 3, 224, 224); y = torch.tensor([0, 1, 2, 3]); x_mix, targets = losses.mix_batch(x, y, alpha=1.0, mode='cutmix'); crit = torch.nn.CrossEntropyLoss(); logits = torch.randn(4, 9); loss = losses.mixed_loss(crit, logits, targets); print('Transform and CutMix test success, loss:', loss.item())"
   ```
   Output:
   - `Transform and CutMix test success, loss: 2.863` (passed with code 0).

### 1.4 Command-Line Parsing Compatibility in `code/train.py`
Verified command line override parser:
```powershell
python -c "import sys; sys.path.insert(0, 'code'); import train; pairs = ['backbone=convnext_tiny', 'init=scratch', 'num_workers=0']; print(train.parse_overrides(pairs)); pairs2 = ['backbone=convnext_tiny', 'mix=cutmix', 'mix_alpha=1.0', 'num_workers=0']; print(train.parse_overrides(pairs2))"
```
Output:
- `{'backbone': 'convnext_tiny', 'init': 'scratch', 'num_workers': 0}`
- `{'backbone': 'convnext_tiny', 'mix': 'cutmix', 'mix_alpha': 1.0, 'num_workers': 0}`
All parameter overrides parse seamlessly and cast into appropriate typed attributes of dataclass `Config`.

---

## 2. Logic Chain

1. **Backbone Anchoring**:
   - *Observation*: M1 selected `convnext_tiny` with 96.04% Val Macro-F1 and 11.44 ms latency.
   - *Deduction*: Per `ORIGINAL_REQUEST.md §R2`, all Step 2 ablations must anchor to `convnext_tiny` under baseline recipe T00. Experiment $B02$ represents $T00$.
2. **Controlled Single-Factor Isolation (Principle N1)**:
   - *Observation*: `GUIDE.md §0` enforces varying exactly one parameter relative to T00 per experiment.
   - *Deduction*:
     - In Axis A:
       - $T01$: modify only `init="scratch"` (all other hyperparams = T00).
       - $T02$: modify only `init="frozen"` (all other hyperparams = T00).
     - In Axis B:
       - $T03$: modify only `aug="color"`.
       - $T04$: modify only `aug="trivial"`.
       - $T05$: modify only `mix="cutmix"` (`mix_alpha=1.0`).
3. **Parameter Footprint and Runtime Dynamics**:
   - *Observation*: $T02$ (`init="frozen"`) updates only 6,921 parameters and avoids backpropagating through the 27.8M-parameter backbone.
   - *Deduction*: Epoch duration will drop to ~38 s/epoch (total runtime ~7.6 minutes), providing rapid turnaround.
   - *Observation*: $T01$, $T03$, $T04$, $T05$ all perform full model optimization (27.83 M params).
   - *Deduction*: Epoch durations are ~79–83 s/epoch, with full 12-epoch training taking ~15.8–16.6 minutes each on RTX 5060 Ti.
4. **Statistical Significance Calibration**:
   - *Observation*: Slide Day 2 p. 59 notes random seed standard deviation is $\sigma_{\text{seed}} \approx 0.10\% - 0.30\%$.
   - *Deduction*: Any $\Delta = \text{Val Macro-F1}(T0x) - \text{Val Macro-F1}(T00)$ with $|\Delta| < 0.30\%$ must be reported as "không phân biệt được" (within random noise), whereas $|\Delta| \ge 0.30\%$ represents a genuine empirical effect.

---

## 3. Caveats

1. **12-Epoch Budget on Aggressive Regularization**:
   - Regularizers such as `TrivialAugmentWide` ($T04$) or `CutMix` ($T05$) typically require longer training horizons (e.g. 50–100 epochs) to fully realize their generalization benefits. Over a 12-epoch constraint, strong data transformations may exhibit slower early convergence on training loss, which should be evaluated objectively on the validation curve without misinterpreting it as model failure.
2. **Windows Multiprocessing**:
   - In accordance with Milestone 1 findings, always invoke training commands with `num_workers=0` on Windows PowerShell to prevent inter-process pipe bottlenecks.
3. **Zero Test Set Leakage Boundary**:
   - No predictions on `test_subset0.csv` should be generated during Step 2. All comparisons and selections are confined strictly to `val_subset0.csv`.

---

## 4. Conclusion

1. **Ablation Strategy Is Fully Specified and Production-Ready**:
   - Detailed technical design and background rationale have been committed to `analysis.md`.
   - The exact command sequence and parameter dictionary are verified compatible with `code/train.py`.
2. **Experiment Execution Commands for Worker M2**:

   - **Baseline Reference ($T00$)**:
     ```powershell
     $env:PYTHONUTF8=1; python code/train.py --exp_id T00 --seed 0 --fold 0 --set backbone=convnext_tiny num_workers=0
     ```
     *(Alternatively, $T00$ artifacts can be symlinked/cloned directly from $B02$ since $B02$ is identical to $T00$)*.

   - **Axis A — Scratch Initialization ($T01$)**:
     ```powershell
     $env:PYTHONUTF8=1; python code/train.py --exp_id T01 --seed 0 --fold 0 --set backbone=convnext_tiny init=scratch num_workers=0
     ```

   - **Axis A — Frozen Backbone / Linear Probe ($T02$)**:
     ```powershell
     $env:PYTHONUTF8=1; python code/train.py --exp_id T02 --seed 0 --fold 0 --set backbone=convnext_tiny init=frozen num_workers=0
     ```

   - **Axis B — ColorJitter Augmentation ($T03$)**:
     ```powershell
     $env:PYTHONUTF8=1; python code/train.py --exp_id T03 --seed 0 --fold 0 --set backbone=convnext_tiny aug=color num_workers=0
     ```

   - **Axis B — TrivialAugmentWide ($T04$)**:
     ```powershell
     $env:PYTHONUTF8=1; python code/train.py --exp_id T04 --seed 0 --fold 0 --set backbone=convnext_tiny aug=trivial num_workers=0
     ```

   - **Axis B — Regional CutMix ($T05$)**:
     ```powershell
     $env:PYTHONUTF8=1; python code/train.py --exp_id T05 --seed 0 --fold 0 --set backbone=convnext_tiny mix=cutmix mix_alpha=1.0 num_workers=0
     ```

3. **Output Artifact Locations**:
   - Runs & Checkpoints: `runs/<exp_id>/seed0/` (`best_checkpoint.pt`, `config.json`, `summary.json`, `history.csv`, `val_logits.npy`)
   - Dual-axis Curves: `curves/<exp_id>_convnext_tiny.png` (auto-generated)
   - Master Workbook: `results.xlsx` sheet `Training`

---

## 5. Verification Method

To independently reproduce and verify the ablation design:

1. **Verify Source Integrity & Unit Test Passing**:
   ```powershell
   $env:PYTHONUTF8=1; python -m unittest discover -s tests -v
   ```
   *Expected outcome*: 38/38 tests PASS.

2. **Verify Initialization Flag Behavior in `code/model.py`**:
   ```powershell
   $env:PYTHONUTF8=1; python -c "
   import sys; sys.path.insert(0, 'code'); import model
   m_sc = model.build_model('convnext_tiny', init='scratch')
   assert sum(p.requires_grad for p in m_sc.parameters()) == len(list(m_sc.parameters()))
   m_fr = model.build_model('convnext_tiny', init='frozen')
   trainable_fr = sum(p.numel() for p in m_fr.parameters() if p.requires_grad)
   assert trainable_fr == 6921, f'Expected 6921 trainable params, got {trainable_fr}'
   print('Axis A Init Verification: PASSED!')
   "
   ```

3. **Verify Augmentation & CutMix Pipelines in `code/dataset.py` & `code/losses.py`**:
   ```powershell
   $env:PYTHONUTF8=1; python -c "
   import sys; sys.path.insert(0, 'code'); import dataset, losses, torch
   tf_col = dataset.build_transforms(train=True, aug='color')
   tf_triv = dataset.build_transforms(train=True, aug='trivial')
   x = torch.randn(4, 3, 224, 224); y = torch.tensor([0, 1, 2, 3])
   x_mix, targets = losses.mix_batch(x, y, alpha=1.0, mode='cutmix')
   assert x_mix.shape == (4, 3, 224, 224)
   print('Axis B Aug/Mix Verification: PASSED!')
   "
   ```

4. **Invalidation Conditions**:
   - If any command attempts to access or load `test_subset0.csv`.
   - If an experiment alters multiple variables simultaneously (e.g. `init=scratch` AND `aug=color` in the same run).
   - If random seeds other than 0 are used for single-factor comparative screening.
   - If curve artifacts fail to plot dual-axis metrics (Loss on left, Metric % on right).

---
*End of Milestone 2 Handoff Report.*
