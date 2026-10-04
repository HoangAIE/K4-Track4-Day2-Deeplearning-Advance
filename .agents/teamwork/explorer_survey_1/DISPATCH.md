# Survey Task: Model, Architecture, Environment & Benchmark Investigation

Read `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\ORIGINAL_REQUEST.md`.
Investigate the Python/PyTorch environment, GPU hardware, `code/model.py`, `code/benchmark.py`, `code/train.py`, and `eval.py`.
Determine exact backbone model names, timm/torchvision availability, parameter/GMAC counting logic, benchmark latency setup, and verify readiness for B01..B05.
Write your structured findings and recommendations to `.agents/teamwork/explorer_survey_1/analysis.md` and complete with `handoff.md`.

## 2026-10-03T13:48:43Z
You are Explorer 1 (Architecture & Benchmark Explorer) for the DeepWeeds Deep Learning Day 2 Lab project.
Your working directory is: `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\explorer_survey_1\`
Authoritative request is at: `d:\K4-Track4-Day2-Deeplearning-Advance\.agents\teamwork\ORIGINAL_REQUEST.md`

Your tasks:
1. Read `ORIGINAL_REQUEST.md`.
2. Inspect the environment: Python version, PyTorch version, CUDA/GPU device info (GPU model, VRAM), installed libraries (timm, torchvision, openpyxl, etc.).
3. Inspect `code/model.py`: How are backbones created, adapted, and classified? Are B01 (`resnet50`), B02 (`convnext_tiny`), B03 (`swin_tiny_patch4_window7_224` or `vit_small_patch16_224`), B04 (`efficientnet_b0`), B05 (`mobilenetv3_large_100`) supported out of the box? Are pretrained weights available (local cache or timm download)?
4. Inspect `code/benchmark.py`: How does inference latency measurement work? Does it handle warmup, CUDA synchronization, batch size 1, resolution 224x224? Does it count params and GMACs accurately?
5. Inspect `code/train.py` and `eval.py`: How is evaluation executed? How are Macro-F1 and Top-1 calculated?
6. Write your detailed findings to `.agents/teamwork/explorer_survey_1/analysis.md` and your handoff summary to `.agents/teamwork/explorer_survey_1/handoff.md`.
7. Send a message to the caller (parent) with a concise summary and pointer to your handoff.md.
Ensure you update your `progress.md` during execution. Do NOT modify source code. Maintain zero test set leakage.
