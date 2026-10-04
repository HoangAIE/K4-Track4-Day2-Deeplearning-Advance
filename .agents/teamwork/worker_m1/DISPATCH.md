## 2026-10-03T14:08:14Z

# Milestone 1 Task: Step 1 Backbones Training, Benchmarking & Excel Logging

Authoritative Request: `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\ORIGINAL_REQUEST.md`
Project Blueprint: `d:\K4-Track4-Day2-Deeplearning-Advance\PROJECT.md`
Explorer Findings:
- `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\explorer_m1_1\handoff.md`
- `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\explorer_m1_2\handoff.md`
- `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\explorer_m1_3\handoff.md`

Your tasks:
1. Ensure Windows environment compatibility (`$env:PYTHONUTF8=1`, `num_workers=0` if needed for clean DataLoader operation).
2. Execute training for all 5 backbones under recipe T00 sequentially:
   - B01: `resnet50`
   - B02: `convnext_tiny`
   - B03: `swin_tiny_patch4_window7_224`
   - B04: `efficientnet_b0`
   - B05: `mobilenetv3_large_100`
   Parameters: AdamW, LR backbone 1e-4, LR head 1e-3, weight decay 0.05, 1 epoch warmup + cosine annealing, CE loss, AMP on, batch size 64, 12 epochs, seed 0.
3. Verify that checkpoints, configs, history, and summaries are generated in `runs/B0x/seed0/`.
4. Verify that training curves are saved in `curves/B0x_*.png` with dual axes (loss & validation metrics).
5. Run benchmark profiling for each model (batch size 1, 224x224, CUDA sync) to record params (M), GMACs, and inference latency (ms).
6. Create/update `results.xlsx` sheet `Backbones` with all 13 required columns and accurate numbers.
7. Perform quantitative trade-off analysis (Macro-F1 vs Latency vs Params) and declare the winning backbone for Step 2.
8. Document all commands, execution logs, and numbers in `.agents/teamwork/worker_m1/handoff.md`.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
Ensure you update your `progress.md` regularly. Maintain zero test set leakage (`test_subset0.csv` must NEVER be used).
