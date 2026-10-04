# Survey Task: Recipe Ablations, Excel Logging & Curve Plotting

Read `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\ORIGINAL_REQUEST.md`.
Investigate `code/train.py` configuration options, how ablation axes (A, B, C, D, F) are supported in `Config`, existing loss/sampler/ema/augmentation implementations, curve saving (`curves/`), runs directory structure (`runs/<exp_id>/seed0/`), and inspect `results.xlsx` (template, sheets `Backbones` and `Training`, column headers).
Write your structured findings and recommendations to `.agents/teamwork/explorer_survey_3/analysis.md` and complete with `handoff.md`.

## 2026-10-03T13:48:43Z
You are Explorer 3 (Ablation & Excel Explorer) for the DeepWeeds Deep Learning Day 2 Lab project.
Your working directory is: `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\explorer_survey_3\`
Authoritative request is at: `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\ORIGINAL_REQUEST.md`

Your tasks:
1. Read `ORIGINAL_REQUEST.md`.
2. Inspect `code/train.py`: How is `Config` defined? How are the baseline recipe T00 parameters set (AdamW, LR backbone 1e-4, LR head 1e-3, weight decay 0.05, warmup 1 epoch, cosine annealing, cross-entropy, AMP on, batch size 64, 12 epochs, seed 0)?
3. Inspect how ablation axes are implemented:
   - Axis A (Init): `scratch` vs `frozen` vs `finetune`
   - Axis B (Augmentation): `basic` vs `color` / `trivial` vs `cutmix`
   - Axis C (Loss): `ce` vs `ls` (label smoothing 0.1) vs `focal` (gamma=2.0)
   - Axis D (Sampler): `None` vs `balanced`
   - Axis F (EMA): `None` vs `0.999`
   Are all these options already implemented or what needs to be wired up?
4. Inspect `results.xlsx`: Does `results.xlsx` exist? What sheets exist (`Backbones`, `Training`)? What are the exact column headers required?
5. Inspect curve saving and checkpoint saving: Does `code/train.py` save to `curves/` and `runs/<exp_id>/seed0/`? What is the format of the curve plots (train/val loss, val macro-F1)?
6. Write your detailed findings to `.agents/teamwork/explorer_survey_3/analysis.md` and your handoff summary to `.agents/teamwork/explorer_survey_3/handoff.md`.
7. Send a message to the caller (parent) with a concise summary and pointer to your handoff.md.
Ensure you update your `progress.md` during execution. Do NOT modify source code.
