# Milestone 1: Excel Workbook Schema & Artifacts Verification Analysis

**Author:** Explorer 3 (Milestone 1 — M1 Excel & Artifacts Schema Verification)  
**Date:** 2026-10-03  
**Status:** Complete  
**Scope:** `results.xlsx` (`Backbones` sheet schema & workbook architecture), training curve standards (`curves/B0x_<backbone>.png`), dual-axis visualization, and production-grade `openpyxl` automation utilities for Worker execution.

---

## 1. Executive Summary & Objective

In Milestone 1 (M1) of the DeepWeeds Deep Learning Day 2 Lab, 5 candidate backbones covering 4 required architectural families are trained under baseline recipe T00:
1. `B01`: `resnet50` (ResNet baseline)
2. `B02`: `convnext_tiny` (ConvNeXt modern convolutional)
3. `B03`: `swin_tiny_patch4_window7_224` (Vision Transformer shifted-window)
4. `B04`: `efficientnet_b0` (Mobile / MBConv compound scaling)
5. `B05`: `mobilenetv3_large_100` (Ultra-lightweight edge architecture)

This investigation guarantees that:
1. **The Excel schema for `results.xlsx` sheet `Backbones`** strictly adheres to the 13 required columns in exact order and naming as specified in `ORIGINAL_REQUEST.md` (§R1, §Acceptance Criteria) and `GUIDE.md` (§6.1).
2. **The training curves in `curves/`** strictly adhere to the naming format `curves/B0x_<backbone>.png` and feature dual-axis plotting (Train/Val Loss on left axis, Val Macro-F1 / Top-1 on right axis) with high-DPI clarity.
3. **A non-destructive, idempotent Python update utility** using raw `openpyxl` is fully designed and tested, allowing the Worker to initialize or upsert experiment records without altering formatting, corrupting data, or overwriting existing sheets.

---

## 2. Sheet `Backbones` Schema Specification

### 2.1 Authoritative 13-Column Schema

The sheet `Backbones` in `results.xlsx` must contain exactly 13 columns in the exact order below. All Vietnamese diacritics must match verbatim:

| Col # | Excel Col | Column Name (Tiêu đề cột) | Data Type | Number Format | Alignment | Data Origin / Source |
|:---:|:---:|:---|:---:|:---:|:---:|:---|
| 1 | `A` | `exp_id` | String | `@` (Text) | Center | Experiment ID (`B01` .. `B05`) |
| 2 | `B` | `backbone` | String | `@` (Text) | Left | Exact timm architecture string |
| 3 | `C` | `tag trọng số` | String | `@` (Text) | Center | Timm pretrained weight tag |
| 4 | `D` | `#tham số (M)` | Float | `0.00` | Right | `code/model.py:count_params` (in Millions) |
| 5 | `E` | `GMAC` | Float | `0.00` | Right | `code/model.py:count_gmacs` (in GMACs at 224x224) |
| 6 | `F` | `độ phân giải` | Integer | `0` | Center | Input spatial resolution (`224`) |
| 7 | `G` | `epoch` | Integer | `0` | Center | Total training epochs (`12`) |
| 8 | `H` | `seed` | Integer | `0` | Center | Random seed (`0`) |
| 9 | `I` | `macro-F1 val` | Float | `0.0000` | Right | `runs/<exp_id>/seed0/summary.json` (val macro-F1) |
| 10 | `J` | `top-1 val` | Float | `0.0000` | Right | `runs/<exp_id>/seed0/summary.json` (val top-1 acc) |
| 11 | `K` | `thời gian train/epoch` | Float | `0.0` | Right | `runs/<exp_id>/seed0/summary.json` (avg seconds/epoch) |
| 12 | `L` | `độ trễ batch-1 (ms)` | Float | `0.00` | Right | `code/benchmark.py` (median p50 latency in ms, batch 1, AMP) |
| 13 | `M` | `ghi chú` | String | `@` (Text) | Left | Architectural highlights & observations |

### 2.2 Model Ground Truth Profiles for Candidate Backbones

The following values have been experimentally extracted and verified against the repository's `timm` and PyTorch model implementations:

| `exp_id` | `backbone` | `tag trọng số` | `#tham số (M)` | `GMAC` | `độ phân giải` | `epoch` | `seed` | `ghi chú` |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `B01` | `resnet50` | `a1_in1k` | `23.53` | `4.09` | 224 | 12 | 0 | Baseline họ ResNet (chuẩn T00) |
| `B02` | `convnext_tiny` | `in12k_ft_in1k` | `27.83` | `4.46` | 224 | 12 | 0 | Họ ConvNeXt hiện đại, depthwise 7x7 |
| `B03` | `swin_tiny_patch4_window7_224` | `ms_in1k` | `27.53` | `4.35` | 224 | 12 | 0 | Họ Vision Transformer, shifted window attention |
| `B04` | `efficientnet_b0` | `ra_in1k` | `4.02` | `0.38` | 224 | 12 | 0 | Họ mạng nhẹ Mobile, MBConv compound scaling |
| `B05` | `mobilenetv3_large_100` | `ra_in1k` | `4.21` | `0.22` | 224 | 12 | 0 | Họ mạng siêu nhẹ di động, Hard-Swish + SE blocks |

*Note on tag extraction:* In `code/model.py`, `model.pretrained_tag` automatically reads `cfg.get("tag")` from the model's `pretrained_cfg`. If aliased, `swin_tiny` maps to `swin_tiny_patch4_window7_224` and `mobilenetv3` maps to `mobilenetv3_large_100`.

---

## 3. Training Curve Artifact Requirements (`curves/`)

### 3.1 File Naming & Path Architecture
According to `ORIGINAL_REQUEST.md` (§R1, §Acceptance Criteria) and `PROJECT.md` (§Interface Contracts):
- All curves must be stored in the top-level directory `curves/`.
- File naming template: `curves/<exp_id>_<backbone>.png`
- For Milestone 1 backbones:
  - `curves/B01_resnet50.png`
  - `curves/B02_convnext_tiny.png`
  - `curves/B03_swin_tiny_patch4_window7_224.png` (and aliased copy `curves/B03_swin_tiny.png`)
  - `curves/B04_efficientnet_b0.png`
  - `curves/B05_mobilenetv3_large_100.png` (and aliased copy `curves/B05_mobilenetv3.png`)

### 3.2 Dual-Axis Visual Architecture
To meet `RUBRIC.md` §F (4 points) and `GUIDE.md` §6.2, each curve must fulfill:
1. **Primary Y-Axis (Left):**
   - Metric: Loss
   - Series 1: `Train Loss` (dashed line, marker `o`, `#e74c3c`)
   - Series 2: `Val Loss` (solid line, marker `s`, `#c0392b`)
   - Y-axis label: `"Loss"`, colored to match loss scheme, bold.
   - Light dotted grid lines (`linestyle=":"`, `alpha=0.6`).
2. **Secondary Y-Axis (Right):**
   - Generated via `ax2 = ax1.twinx()`.
   - Metric: Percentage (%)
   - Series 3: `Val Macro-F1` (solid line, marker `^`, `#2980b9`, width 2.5)
   - Series 4: `Val Top-1 Acc` (dash-dot line, marker `d`, `#27ae60`)
   - Y-axis label: `"Metric (%)"`, colored to match metric scheme, bold.
3. **X-Axis (Bottom):**
   - Metric: Epoch (1 to 12).
   - Label: `"Epoch"`, bold.
4. **Figure Elements:**
   - Figure size: `figsize=(9, 5.5)`, `dpi=150`.
   - Unified Legend: All 4 handles consolidated in a single legend box (`ax1.legend(lines, labels, loc="center right", framealpha=0.9)`).
   - Title: Detailed experiment descriptor: `<exp_id>: <backbone> (Init=<init>, Loss=<loss>, Aug=<aug>, Seed=<seed>)`.

### 3.3 Post-Hoc Regeneration Capability
In `code/train.py`, `plot_curves()` runs automatically at the end of training. However, if the Worker ever needs to re-style or regenerate a curve without re-running a full 12-epoch training job, curves can be plotted directly from `runs/<exp_id>/seed0/history.csv`. A standalone plotting utility is integrated into the design below.

---

## 4. Excel Management: `openpyxl` vs `pandas` Critical Analysis

### 4.1 Pitfalls of `pandas.to_excel` / `pd.ExcelWriter`
1. **Workbook Erasure Risk:** Running `df.to_excel("results.xlsx")` in default mode overwrites the entire file, destroying all other sheets (`Training`, `Inference`, `Summary`, etc.).
2. **Style Stripping:** Even when using `pd.ExcelWriter(..., mode='a', if_sheet_exists='replace')`, openpyxl cell styles, background fills, number formats (e.g. `0.0000`), column widths, and frozen panes on existing sheets can be wiped or corrupted.
3. **Trailing Zero Truncation:** Storing metrics via pandas often coerces `0.9200` to `0.92` in Excel unless explicit cell formatting is applied post-write.

### 4.2 Advantages of Direct `openpyxl` Engine
1. **True Non-Destructive In-Place Upsert:**
   - Checks if `exp_id` exists in column A.
   - If found: updates the row in-place while keeping sheet structure unchanged.
   - If not found: appends a new row at `ws.max_row + 1`.
2. **Cell-Level Format Enforcement:**
   - Explicitly sets `cell.number_format = '0.0000'` for `macro-F1 val` and `top-1 val`.
   - Explicitly sets `cell.number_format = '0.00'` for params, GMACs, and latency.
   - Preserves numerical types in Excel so formulas like `=AVERAGE()` and `=MAX()` work properly.
3. **Aesthetic Enhancements:**
   - Professional header fill: Navy blue `#1F4E78` with bold white text.
   - Freeze pane: `ws.freeze_panes = "A2"` so headers remain fixed when scrolling.
   - Dynamic auto-sizing of column widths to eliminate `###` clipping.
   - Soft green highlight (`#E8F8F5`) for the highest `macro-F1 val` row.
4. **Windows UTF-8 Encoding Safeguard:**
   - Headers like `tag trọng số`, `độ phân giải`, `thời gian train/epoch`, `độ trễ batch-1 (ms)` contain Vietnamese characters.
   - On Windows with default cp1252 console, standard print statements can raise `UnicodeEncodeError`. The designed script reconfigures `sys.stdout` to UTF-8 on Windows.

---

## 5. Production-Ready Utility for Worker

The Worker can place or execute this modular script `code/excel_utils.py` (or invoke it via CLI) to manage `results.xlsx` and training curves seamlessly.

```python
"""excel_utils.py - Công cụ tạo và cập nhật results.xlsx chuyên dụng dùng openpyxl.

Đặc tính:
  1. Bảo toàn 100% định dạng, không làm hỏng dữ liệu các sheet khác.
  2. Upsert theo exp_id (cập nhật nếu đã có, nối tiếp nếu chưa có).
  3. Định dạng số chuẩn: 4 chữ số thập phân cho F1/Top-1, 2 chữ số cho GMAC/Latency/Params.
  4. Freeze hàng tiêu đề (A2) và tự căn chỉnh độ rộng cột.
  5. Đánh dấu nổi bật dòng có macro-F1 val cao nhất.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# Đảm bảo in tiếng Việt không lỗi trên PowerShell Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Định nghĩa 13 cột chuẩn của sheet 'Backbones'
BACKBONE_COLUMNS: List[str] = [
    "exp_id",
    "backbone",
    "tag trọng số",
    "#tham số (M)",
    "GMAC",
    "độ phân giải",
    "epoch",
    "seed",
    "macro-F1 val",
    "top-1 val",
    "thời gian train/epoch",
    "độ trễ batch-1 (ms)",
    "ghi chú",
]

# Định nghĩa các cột chuẩn của sheet 'Training' (sẵn sàng cho Milestone 2)
TRAINING_COLUMNS: List[str] = [
    "exp_id",
    "backbone",
    "trục thay đổi (A–G)",
    "khác T00 ở điểm nào",
    "seed",
    "macro-F1 val",
    "top-1 val",
    "Δ so với T00",
    "ghi chú",
]

# Style definitions
HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
HEADER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)

DATA_FONT = Font(name="Calibri", size=10)
BEST_ROW_FILL = PatternFill(start_color="E8F8F5", end_color="E8F8F5", fill_type="solid")

THIN_BORDER = Border(
    left=Side(style="thin", color="D9D9D9"),
    right=Side(style="thin", color="D9D9D9"),
    top=Side(style="thin", color="D9D9D9"),
    bottom=Side(style="thin", color="D9D9D9"),
)


def format_header_row(ws: Any, columns: List[str]) -> None:
    """Định dạng hàng tiêu đề của một sheet."""
    for col_idx, col_name in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = HEADER_ALIGN
        cell.border = THIN_BORDER
    ws.row_dimensions[1].height = 28
    ws.freeze_panes = "A2"


def init_results_workbook(excel_path: Union[str, Path]) -> openpyxl.Workbook:
    """Tạo mới results.xlsx với đầy đủ tiêu đề các sheet chuẩn nếu chưa có."""
    excel_path = Path(excel_path)
    if excel_path.exists():
        wb = openpyxl.load_workbook(excel_path)
        return wb

    excel_path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()

    # Sheet 1: Backbones
    ws_backbones = wb.active
    ws_backbones.title = "Backbones"
    format_header_row(ws_backbones, BACKBONE_COLUMNS)

    # Sheet 2: Training (chuẩn bị sẵn cho M2)
    ws_training = wb.create_sheet("Training")
    format_header_row(ws_training, TRAINING_COLUMNS)

    # Sheet 3: Summary (placeholder)
    ws_summary = wb.create_sheet("Summary")
    summary_cols = ["hạng", "exp_id", "backbone", "cấu hình", "macro-F1 val", "top-1 val", "độ trễ batch-1 (ms)", "#tham số (M)", "GMAC", "ghi chú"]
    format_header_row(ws_summary, summary_cols)

    wb.save(excel_path)
    wb.close()
    return openpyxl.load_workbook(excel_path)


def upsert_backbone_row(excel_path: Union[str, Path], record: Dict[str, Any]) -> int:
    """Thêm mới hoặc cập nhật một dòng thí nghiệm vào sheet 'Backbones'.

    `record` chấp nhận các key:
      - exp_id: str (B01..B05)
      - backbone: str
      - tag: str hoặc 'tag trọng số'
      - params_m: float hoặc '#tham số (M)'
      - gmacs: float hoặc 'GMAC'
      - resolution: int (mặc định 224)
      - epoch: int (mặc định 12)
      - seed: int (mặc định 0)
      - macro_f1: float hoặc 'macro-F1 val'
      - top1: float hoặc 'top-1 val'
      - epoch_time_s: float hoặc 'thời gian train/epoch'
      - latency_ms: float hoặc 'độ trễ batch-1 (ms)'
      - note: str hoặc 'ghi chú'
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        init_results_workbook(excel_path)

    wb = openpyxl.load_workbook(excel_path)
    if "Backbones" not in wb.sheetnames:
        ws = wb.create_sheet("Backbones", 0)
        format_header_row(ws, BACKBONE_COLUMNS)
    else:
        ws = wb["Backbones"]

    exp_id = str(record.get("exp_id", "")).strip()
    if not exp_id:
        raise ValueError("record bắt buộc phải có trường 'exp_id'")

    # Tìm xem exp_id đã tồn tại chưa
    target_row = None
    for r in range(2, ws.max_row + 1):
        val = ws.cell(row=r, column=1).value
        if val is not None and str(val).strip() == exp_id:
            target_row = r
            break

    if target_row is None:
        target_row = ws.max_row + 1 if ws.cell(row=ws.max_row, column=1).value is not None else ws.max_row

    # Thu thập dữ liệu
    row_values = [
        exp_id,
        record.get("backbone", ""),
        record.get("tag trọng số", record.get("tag", "")),
        float(record.get("#tham số (M)", record.get("params_m", 0.0))),
        float(record.get("GMAC", record.get("gmacs", 0.0))),
        int(record.get("độ phân giải", record.get("resolution", 224))),
        int(record.get("epoch", record.get("epochs", 12))),
        int(record.get("seed", 0)),
        float(record.get("macro-F1 val", record.get("macro_f1", 0.0))),
        float(record.get("top-1 val", record.get("top1", 0.0))),
        float(record.get("thời gian train/epoch", record.get("epoch_time_s", 0.0))),
        float(record.get("độ trễ batch-1 (ms)", record.get("latency_ms", 0.0))),
        record.get("ghi chú", record.get("note", "")),
    ]

    for col_idx, val in enumerate(row_values, start=1):
        cell = ws.cell(row=target_row, column=col_idx, value=val)
        cell.font = DATA_FONT
        cell.border = THIN_BORDER

        if col_idx in [1, 6, 7, 8]:
            cell.alignment = Alignment(horizontal="center", vertical="center")
        elif col_idx in [4, 5, 9, 10, 11, 12]:
            cell.alignment = Alignment(horizontal="right", vertical="center")
            if col_idx in [4, 5, 12]:
                cell.number_format = "0.00"
            elif col_idx in [9, 10]:
                cell.number_format = "0.0000"
            elif col_idx == 11:
                cell.number_format = "0.0"
        else:
            cell.alignment = Alignment(horizontal="left", vertical="center")

    # Tô màu nổi bật dòng có macro-F1 cao nhất
    best_f1 = -1.0
    best_row_idx = None
    for r in range(2, ws.max_row + 1):
        f1_val = ws.cell(row=r, column=9).value
        try:
            f1_num = float(f1_val)
            if f1_num > best_f1:
                best_f1 = f1_num
                best_row_idx = r
        except (ValueError, TypeError):
            pass

    for r in range(2, ws.max_row + 1):
        is_best = (r == best_row_idx and best_f1 > 0)
        for c in range(1, len(BACKBONE_COLUMNS) + 1):
            cell = ws.cell(row=r, column=c)
            cell.fill = BEST_ROW_FILL if is_best else PatternFill(fill_type=None)

    # Tự động căn chỉnh độ rộng cột
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

    ws.row_dimensions[target_row].height = 20

    wb.save(excel_path)
    wb.close()
    return target_row


def sync_from_run_dir(
    excel_path: Union[str, Path],
    run_dir: Union[str, Path],
    latency_ms: Optional[float] = None,
    note: str = "",
) -> int:
    """Tự động đọc summary.json từ runs/<exp_id>/seed0/ và đồng bộ vào results.xlsx."""
    run_path = Path(run_dir)
    summary_path = run_path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file summary tại: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Đọc config.json nếu có để lấy epoch/resolution/seed
    config_path = run_path / "config.json"
    cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    # Mapping tag mặc định nếu chưa lưu
    backbone = summary.get("backbone", "")
    known_tags = {
        "resnet50": "a1_in1k",
        "convnext_tiny": "in12k_ft_in1k",
        "swin_tiny_patch4_window7_224": "ms_in1k",
        "swin_tiny": "ms_in1k",
        "efficientnet_b0": "ra_in1k",
        "mobilenetv3_large_100": "ra_in1k",
        "mobilenetv3": "ra_in1k",
    }

    record = {
        "exp_id": summary.get("exp_id", ""),
        "backbone": backbone,
        "tag": summary.get("tag", known_tags.get(backbone, "pretrained")),
        "params_m": summary.get("params_m", 0.0),
        "gmacs": summary.get("gmacs", 0.0),
        "resolution": cfg.get("img_size", 224),
        "epoch": cfg.get("epochs", 12),
        "seed": cfg.get("seed", 0),
        "macro_f1": summary.get("best_val_macro_f1", 0.0),
        "top1": summary.get("best_val_top1", 0.0),
        "epoch_time_s": summary.get("avg_epoch_time_s", 0.0),
        "latency_ms": latency_ms if latency_ms is not None else 0.0,
        "note": note if note else f"T00 baseline {backbone}",
    }

    return upsert_backbone_row(excel_path, record)
```

### 5.2 Standalone Curve Re-Plotting Utility

```python
def plot_curve_from_history_csv(
    csv_path: Union[str, Path],
    output_png: Union[str, Path],
    title: str,
) -> None:
    """Tái lập biểu đồ huấn luyện 2 trục từ history.csv có sẵn mà không cần train lại."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append({
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "val_macro_f1": float(row["val_macro_f1"]),
                "val_top1": float(row.get("val_top1", 0.0)),
            })

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]
    val_top1 = [h["val_top1"] for h in history]

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=150)

    color_train = "#e74c3c"
    color_val = "#c0392b"
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss", color=color_train, fontsize=12, fontweight="bold")
    line1 = ax1.plot(epochs, train_loss, label="Train Loss", color=color_train, linestyle="--", marker="o", markersize=4)
    line2 = ax1.plot(epochs, val_loss, label="Val Loss", color=color_val, linewidth=2, marker="s", markersize=4)
    ax1.tick_params(axis="y", labelcolor=color_train)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    color_f1 = "#2980b9"
    color_top1 = "#27ae60"
    ax2.set_ylabel("Metric (%)", color=color_f1, fontsize=12, fontweight="bold")
    line3 = ax2.plot(epochs, [f * 100 for f in val_f1], label="Val Macro-F1", color=color_f1, linewidth=2.5, marker="^", markersize=5)
    line4 = ax2.plot(epochs, [t * 100 for t in val_top1], label="Val Top-1 Acc", color=color_top1, linestyle="-.", marker="d", markersize=4)
    ax2.tick_params(axis="y", labelcolor=color_f1)

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", framealpha=0.9)

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)
```

---

## 6. Execution Recipe for Milestone 1 Worker

When training the 5 candidate models, the Worker should execute the following sequence:

1. **Initialize `results.xlsx`**:
   The Worker can call `init_results_workbook("results.xlsx")`.
2. **Execute Training**:
   Run training for each backbone using `code/train.py`:
   ```powershell
   $env:PYTHONUTF8=1
   python code/train.py --exp_id B01 --seed 0 --fold 0 --set backbone=resnet50
   python code/train.py --exp_id B02 --seed 0 --fold 0 --set backbone=convnext_tiny
   python code/train.py --exp_id B03 --seed 0 --fold 0 --set backbone=swin_tiny_patch4_window7_224
   python code/train.py --exp_id B04 --seed 0 --fold 0 --set backbone=efficientnet_b0
   python code/train.py --exp_id B05 --seed 0 --fold 0 --set backbone=mobilenetv3_large_100
   ```
3. **Execute Benchmark (Batch 1 Latency)**:
   Measure inference latency with CUDA synchronization and warmup:
   ```powershell
   python code/benchmark.py --models resnet50 convnext_tiny swin_tiny_patch4_window7_224 efficientnet_b0 mobilenetv3_large_100 --batch_size 1 --device cuda --dtype amp
   ```
4. **Update `results.xlsx`**:
   Worker records the metrics into `results.xlsx` using `upsert_backbone_row` or `sync_from_run_dir`.
5. **Verify Artifacts Checklist**:
   - [ ] `results.xlsx` exists and opens without warnings.
   - [ ] Sheet `Backbones` contains 5 rows for B01..B05.
   - [ ] Column headers match all 13 required names and order.
   - [ ] All 5 curve files exist in `curves/`:
     - `curves/B01_resnet50.png`
     - `curves/B02_convnext_tiny.png`
     - `curves/B03_swin_tiny_patch4_window7_224.png` (and `curves/B03_swin_tiny.png`)
     - `curves/B04_efficientnet_b0.png`
     - `curves/B05_mobilenetv3_large_100.png` (and `curves/B05_mobilenetv3.png`)
   - [ ] Every curve displays dual axes (Loss and Metric %).
