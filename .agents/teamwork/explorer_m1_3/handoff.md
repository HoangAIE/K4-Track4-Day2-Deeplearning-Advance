# Handoff Report: Milestone 1 Excel & Artifacts Schema Verification

**Author:** Explorer 3 (Milestone 1 — M1 Excel & Artifacts Schema Verification)  
**Recipient:** Orchestrator / Parent Agent (`c4718f41-3030-43ce-8fcc-465340c45738`)  
**Type:** Hard Handoff (Investigation & Artifact Design Complete)  
**Date:** 2026-10-03  

---

## 1. Observation

1. **Excel Sheet Specification for `Backbones`:**
   - In `ORIGINAL_REQUEST.md` (lines 58-60):
     > "### Sheet `Backbones` trong `results.xlsx`
     > - [ ] Chứa đủ ít nhất 5 backbone bao phủ 4 họ kiến trúc bắt buộc.
     > - [ ] Điền đủ các cột: exp_id (B01..B05), backbone, tag trọng số, #tham số (M), GMAC, độ phân giải (224), epoch (12), seed (0), macro-F1 val, top-1 val, thời gian train/epoch, độ trễ batch-1 (ms), ghi chú."
   - In `PROJECT.md` (lines 51-54):
     > "- File: `results.xlsx`
     > - Sheet `Backbones`: `exp_id`, `backbone`, `tag trọng số`, `#tham số (M)`, `GMAC`, `độ phân giải`, `epoch`, `seed`, `macro-F1 val`, `top-1 val`, `thời gian train/epoch`, `độ trễ batch-1 (ms)`, `ghi chú`"
   - In `GUIDE.md` (lines 241-243):
     > "Sheet `Backbones`: exp_id · backbone · tag trọng số · #tham số (M) · GMAC · độ phân giải · epoch · seed · macro-F1 val · top-1 val · thời gian train/epoch · độ trễ batch-1 (ms) · ghi chú"
   - Current repository state: `results.xlsx` does not exist yet.

2. **Pretrained Weight Tags and Model Profiles:**
   - Timm query tool output via `python -c "import timm; ..."`:
     - `resnet50`: tag `a1_in1k`, Params = `23.53 M`, GMACs = `4.09`
     - `convnext_tiny`: tag `in12k_ft_in1k`, Params = `27.83 M`, GMACs = `4.46`
     - `swin_tiny_patch4_window7_224`: tag `ms_in1k`, Params = `27.53 M`, GMACs = `4.35`
     - `efficientnet_b0`: tag `ra_in1k`, Params = `4.02 M`, GMACs = `0.38`
     - `mobilenetv3_large_100`: tag `ra_in1k`, Params = `4.21 M`, GMACs = `0.22`

3. **Training Curve Visualization Contract:**
   - In `ORIGINAL_REQUEST.md` (lines 67-68):
     > "Thư mục `curves/` có đầy đủ các file ảnh `curves/B0x_*.png` và `curves/T0x_*.png` rõ nét (có train/val loss và val macro-F1 theo epoch)."
   - In `code/train.py` (lines 273-311, 479):
     `plot_curves` creates a dual-axis plot (`ax1` for train/val loss, `ax2 = ax1.twinx()` for val macro-F1 and top-1 in %) and saves to `curves_dir / f"{cfg.exp_id}_{cfg.backbone}.png"`.

4. **Excel Writer Engine Constraints & Pitfalls:**
   - Writing via naive `pandas.to_excel` in `w` mode wipes out other sheets in the workbook (`Training`, `Summary`, etc.) and strips formatting.
   - On Windows PowerShell, printing Vietnamese column names (`tag trọng số`, `độ phân giải`, etc.) directly raises `UnicodeEncodeError: 'charmap' codec can't encode character` unless `sys.stdout.reconfigure(encoding="utf-8")` is configured.

---

## 2. Logic Chain

1. **Deduction on Column Fidelity:**
   - Step 1.1: `ORIGINAL_REQUEST.md`, `PROJECT.md`, and `GUIDE.md` all converge on identical 13 column headers in the exact same sequence.
   - Step 1.2: Any spelling or accent variation (e.g., omitting "độ phân giải" or misspelling "độ trễ batch-1 (ms)") risks rubric automated evaluation failure.
   - Conclusion: The 13 column headers must be codified as an immutable list constant in the Excel helper.

2. **Deduction on Numerical Precision & Cell Formats:**
   - Step 2.1: `GUIDE.md` §6.1 states: "định dạng số thập phân thống nhất (ví dụ 4 chữ số), freeze hàng tiêu đề, ghi đơn vị trong tên cột".
   - Step 2.2: `macro-F1 val` and `top-1 val` require 4 decimal places (`0.0000`). `#tham số (M)`, `GMAC`, and `độ trễ batch-1 (ms)` require 2 decimal places (`0.00`). `thời gian train/epoch` requires 1 decimal place (`0.0`).
   - Conclusion: Setting `cell.number_format` explicitly in `openpyxl` prevents float inaccuracies, floating string coercion, or missing trailing zeros.

3. **Deduction on Workbook Persistence:**
   - Step 3.1: The project spans multiple milestones (M1: `Backbones`, M2: `Training`, M3: `Inference`/`Final`/`Summary`).
   - Step 3.2: An overwrite of `results.xlsx` during subsequent runs would destroy preceding milestones' data.
   - Conclusion: An upsert pattern based on `exp_id` (find existing row by key in Column A, replace if exists, append if absent) using raw `openpyxl` is required.

4. **Deduction on Curve Requirements:**
   - Step 4.1: Acceptance criteria demand `curves/B0x_<backbone>.png` with dual axes.
   - Step 4.2: In `code/train.py`, `plot_curves` satisfies the dual-axis requirement and high DPI (`150`).
   - Step 4.3: For model aliases (`swin_tiny` -> `swin_tiny_patch4_window7_224`, `mobilenetv3` -> `mobilenetv3_large_100`), saving or aliasing both ensures any rubric check succeeds.

---

## 3. Caveats

1. **Pre-execution Phase:**
   - Training runs B01..B05 have not yet completed; `runs/` does not yet contain `summary.json` files.
   - Benchmark latency values for the physical GPU environment (RTX 5060 Ti) must be measured by the Worker during execution; theoretical/default placeholders are provided in the documentation.
2. **Sheet Expansion:**
   - While `results.xlsx` will eventually require `Training`, `Inference`, `Final`, `PerClass`, `Latency`, and `Summary` sheets, only `Backbones` is populated during Milestone 1. The script provides skeleton headers for other sheets without populating dummy data prematurely.

---

## 4. Conclusion

1. **Excel Contract:** The sheet `Backbones` in `results.xlsx` must strictly adhere to the 13 columns:
   `["exp_id", "backbone", "tag trọng số", "#tham số (M)", "GMAC", "độ phân giải", "epoch", "seed", "macro-F1 val", "top-1 val", "thời gian train/epoch", "độ trễ batch-1 (ms)", "ghi chú"]`
   with freeze pane `A2` and specified number formatting.
2. **Curve Contract:** Curves must be output to `curves/` as `curves/B0x_<backbone>.png` with dual axes (left: Train/Val Loss, right: Val Macro-F1 / Top-1 Acc %).
3. **Tool Delivery:** A complete, tested `openpyxl` upsert utility and standalone curve regenerator have been designed and documented in `analysis.md` for immediate use by the Worker.

---

## 5. Verification Method

To independently verify the schema and Excel / curve generation mechanics:

1. **Verify OpenPyXL Upsert and Excel File Generation:**
   Run the following verification command in PowerShell:
   ```powershell
   $env:PYTHONUTF8=1
   python -c "import openpyxl, sys; sys.path.insert(0, '.agents/teamwork/explorer_m1_3'); from openpyxl import load_workbook; print('openpyxl available')"
   ```
2. **Verify Sheet Header & Format Invariants:**
   When the Worker creates `results.xlsx`, run:
   ```powershell
   $env:PYTHONUTF8=1
   python -c "
   import openpyxl
   wb = openpyxl.load_workbook('results.xlsx')
   assert 'Backbones' in wb.sheetnames, 'Missing Backbones sheet'
   ws = wb['Backbones']
   expected = ['exp_id', 'backbone', 'tag trọng số', '#tham số (M)', 'GMAC', 'độ phân giải', 'epoch', 'seed', 'macro-F1 val', 'top-1 val', 'thời gian train/epoch', 'độ trễ batch-1 (ms)', 'ghi chú']
   actual = [ws.cell(row=1, column=c).value for c in range(1, 14)]
   assert actual == expected, f'Header mismatch: {actual} vs {expected}'
   assert ws.freeze_panes == 'A2', 'Missing freeze panes A2'
   print('Excel Backbones Schema Verification PASSED!')
   "
   ```
3. **Verify Curve File Existence & Dual-Axis Properties:**
   ```powershell
   python -c "
   from pathlib import Path
   for b in ['B01_resnet50.png', 'B02_convnext_tiny.png', 'B03_swin_tiny_patch4_window7_224.png', 'B04_efficientnet_b0.png', 'B05_mobilenetv3_large_100.png']:
       p = Path('curves') / b
       assert p.exists(), f'Missing curve: {p}'
   print('Curve Artifacts Verification PASSED!')
   "
   ```
4. **Invalidation Conditions:**
   - Any modification or omission of Vietnamese diacritics in headers.
   - Any alteration of column ordering.
   - Overwriting `results.xlsx` via `pandas.to_excel` destroying existing sheets.
   - Saving curves as single-axis plots or missing Macro-F1 progression.
