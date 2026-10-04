# Survey Task: Dataset, Splits & Zero-Leakage Verification

Read `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\ORIGINAL_REQUEST.md`.
Investigate `data/images` and `data/labels`, fold 0 CSV files (`train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`).
Analyze class distributions, sample counts, image paths/formats, label mappings, class imbalance (dominant Negative class), and verify strict zero-leakage constraints.
Write your structured findings and recommendations to `.agents/teamwork/explorer_survey_2/analysis.md` and complete with `handoff.md`.

## 2026-10-03T13:48:43Z
You are Explorer 2 (Dataset & Split Explorer) for the DeepWeeds Deep Learning Day 2 Lab project.
Your working directory is: `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\explorer_survey_2\`
Authoritative request is at: `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\ORIGINAL_REQUEST.md`

Your tasks:
1. Read `ORIGINAL_REQUEST.md`.
2. Inspect `data/images` and `data/labels`: Verify directory structure, image counts, image formats (e.g. .jpg, dimensions).
3. Inspect CSV files for fold 0: `train_subset0.csv`, `val_subset0.csv`, and check what splits exist in `data/labels/`.
4. Analyze class distribution: What are the 9 classes (Negative + 8 weed species)? How severe is class imbalance (Negative class percentage)?
5. Check dataset loading in `code/train.py` or dataset module: How are images transformed and loaded? What image resolution is used (224x224)?
6. VERIFY ZERO LEAKAGE RULES: Ensure `test_subset0.csv` is NOT used anywhere during training, validation, or model selection.
7. Write your detailed findings to `.agents/teamwork/explorer_survey_2/analysis.md` and your handoff summary to `.agents/teamwork/explorer_survey_2/handoff.md`.
8. Send a message to the caller (parent) with a concise summary and pointer to your handoff.md.
Ensure you update your `progress.md` during execution. Do NOT modify source code.
