"""update_excel.py - Cập nhật và bảo trì kết quả bảng tính results.xlsx bằng openpyxl.

Đảm bảo:
- Không ghi đè hay xóa các sheet khác trong workbook.
- Giữ đúng 13 cột của sheet Backbones theo quy chuẩn đề bài.
- Áp dụng number_format chuẩn (0.0000 cho F1/Top1, 0.00 cho tham số/GMAC/độ trễ).
- Freeze pane hàng tiêu đề A2, căn lề và màu sắc trực quan chuyên nghiệp.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

BACKBONE_COLUMNS = [
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

TRAINING_COLUMNS = [
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

HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
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
        return openpyxl.load_workbook(excel_path)

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


def upsert_backbone_row(excel_path: Union[str, Path], record: Dict[str, Any]) -> None:
    """Thêm mới hoặc cập nhật một dòng thí nghiệm vào sheet 'Backbones'."""
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
        max_len = 0
        for cell in col:
            val_str = str(cell.value or "")
            max_len = max(max_len, len(val_str.encode("utf-8", "ignore")))
        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

    wb.save(excel_path)
    wb.close()


def populate_backbones_from_runs(excel_path: Union[str, Path] = "results.xlsx",
                                 benchmark_file: Union[str, Path] = "runs/benchmark_m1.json") -> None:
    """Tự động đọc summary.json từ runs/B0x/seed0 và benchmark_m1.json để điền vào results.xlsx."""
    excel_path = Path(excel_path)
    init_results_workbook(excel_path)

    bench_dict = {}
    bench_p = Path(benchmark_file)
    if bench_p.exists():
        with open(bench_p, "r", encoding="utf-8") as f:
            bench_list = json.load(f)
            for item in bench_list:
                bench_dict[item["model"]] = item

    notes = {
        "B01": "Họ ResNet baseline (T00 recipe); hội tụ tại epoch 9, F1=80.65%, top-1=86.12%",
        "B02": "Họ ConvNeXt hiện đại; F1 cao nhất (96.04%), latency nhanh nhất (11.44ms). Chọn cho Bước 2",
        "B03": "Họ Vision Transformer (Swin); F1 rất cao (94.95%) nhưng train chậm nhất (96.8s/ep) và latency cao nhất (19.56ms)",
        "B04": "Họ mạng nhẹ Mobile MBConv; 4.02M params, 0.38 GMAC nhưng latency cao hơn ConvNeXt do depthwise conv",
        "B05": "Họ mạng siêu nhẹ MobileNetV3; GMAC thấp nhất (0.22), train nhanh nhất (52.5s/ep), F1=82.78% vượt EfficientNet",
    }

    for exp_id in ["B01", "B02", "B03", "B04", "B05"]:
        sum_p = Path(f"runs/{exp_id}/seed0/summary.json")
        if not sum_p.exists():
            print(f"Bỏ qua {exp_id} vì chưa có {sum_p}")
            continue

        with open(sum_p, "r", encoding="utf-8") as f:
            summary = json.load(f)

        backbone = summary["backbone"]
        bench_info = bench_dict.get(backbone, {})

DELTA_FORMAT = "+0.0000;-0.0000;0.0000"


def upsert_training_row(excel_path: Union[str, Path], record: Dict[str, Any]) -> None:
    """Thêm mới hoặc cập nhật một dòng thí nghiệm vào sheet 'Training' của results.xlsx.

    Đảm bảo:
    - Đúng 9 cột quy định.
    - Dòng T00 luôn được đặt hoặc duy trì ở vị trí hàng 2 với Δ = 0.0000.
    - Tự động tính Δ so với T00 nếu chưa có.
    - Định dạng số chuẩn 0.0000 và +0.0000;-0.0000;0.0000.
    - Tự động tô màu nổi bật #E8F8F5 cho dòng đạt macro-F1 cao nhất.
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        init_results_workbook(excel_path)

    wb = openpyxl.load_workbook(excel_path)
    if "Training" not in wb.sheetnames:
        ws = wb.create_sheet("Training")
        format_header_row(ws, TRAINING_COLUMNS)
    else:
        ws = wb["Training"]

    exp_id = str(record.get("exp_id", "")).strip()
    if not exp_id:
        raise ValueError("record bắt buộc phải có trường 'exp_id'")

    # 1. Tìm hoặc xác định chỉ số hàng
    target_row = None
    t00_f1 = 0.9604  # Giá trị chuẩn từ B02

    # Tìm hàng T00 trước để lấy giá trị chuẩn chính xác từ bảng nếu có
    for r in range(2, ws.max_row + 1):
        if str(ws.cell(row=r, column=1).value or "").strip() == "T00":
            val = ws.cell(row=r, column=6).value
            if val is not None:
                try:
                    t00_f1 = float(val)
                except (ValueError, TypeError):
                    pass
            break

    # Nếu đang ghi T00, ưu tiên hàng 2
    if exp_id == "T00":
        target_row = 2
    else:
        for r in range(2, ws.max_row + 1):
            val = ws.cell(row=r, column=1).value
            if val is not None and str(val).strip() == exp_id:
                target_row = r
                break
        if target_row is None:
            # Chèn tiếp vào cuối (hoặc hàng 3 nếu bảng mới có T00)
            target_row = ws.max_row + 1 if ws.cell(row=ws.max_row, column=1).value is not None else ws.max_row
            if target_row < 2:
                target_row = 2

    # 2. Chuẩn bị giá trị
    macro_f1 = float(record.get("macro-F1 val", record.get("macro_f1", 0.0)))
    top1 = float(record.get("top-1 val", record.get("top1", 0.0)))

    if exp_id == "T00":
        delta = 0.0000
    else:
        delta = float(record.get("Δ so với T00", round(macro_f1 - t00_f1, 4)))

    row_values = [
        exp_id,
        record.get("backbone", "convnext_tiny"),
        record.get("trục thay đổi (A–G)", record.get("axis", "")),
        record.get("khác T00 ở điểm nào", record.get("diff", "")),
        int(record.get("seed", 0)),
        macro_f1,
        top1,
        delta,
        record.get("ghi chú", record.get("note", "")),
    ]

    # 3. Ghi vào ô và định dạng
    for col_idx, val in enumerate(row_values, start=1):
        cell = ws.cell(row=target_row, column=col_idx, value=val)
        cell.font = DATA_FONT
        cell.border = THIN_BORDER

        if col_idx in [1, 3, 5]:
            cell.alignment = Alignment(horizontal="center", vertical="center")
        elif col_idx in [6, 7]:
            cell.alignment = Alignment(horizontal="right", vertical="center")
            cell.number_format = "0.0000"
        elif col_idx == 8:
            cell.alignment = Alignment(horizontal="right", vertical="center")
            cell.number_format = DELTA_FORMAT
        else:
            cell.alignment = Alignment(horizontal="left", vertical="center")

    # 4. Tô màu nổi bật dòng tốt nhất trong sheet Training
    best_f1 = -1.0
    best_row_idx = None
    for r in range(2, ws.max_row + 1):
        f1_val = ws.cell(row=r, column=6).value
        try:
            f1_num = float(f1_val)
            if f1_num > best_f1:
                best_f1 = f1_num
                best_row_idx = r
        except (ValueError, TypeError):
            pass

    for r in range(2, ws.max_row + 1):
        is_best = (r == best_row_idx and best_f1 > 0)
        for c in range(1, len(TRAINING_COLUMNS) + 1):
            cell = ws.cell(row=r, column=c)
            cell.fill = BEST_ROW_FILL if is_best else PatternFill(fill_type=None)

    # 5. Tự động căn chỉnh độ rộng cột hợp lý
    col_max_widths = {1: 12, 2: 18, 3: 22, 4: 45, 5: 10, 6: 15, 7: 15, 8: 16, 9: 55}
    for col_idx in range(1, len(TRAINING_COLUMNS) + 1):
        col_letter = get_column_letter(col_idx)
        max_len = 0
        for r in range(1, ws.max_row + 1):
            val_str = str(ws.cell(row=r, column=col_idx).value or "")
            max_len = max(max_len, len(val_str.encode("utf-8", "ignore")))
        target_w = max(max_len + 4, col_max_widths.get(col_idx, 12))
        ws.column_dimensions[col_letter].width = min(target_w, 65)

    ws.freeze_panes = "A2"
    wb.save(excel_path)
    wb.close()


def populate_training_from_runs(excel_path: Union[str, Path] = "results.xlsx",
                                runs_dir: Union[str, Path] = "runs") -> None:
    """Quét toàn bộ kết quả thí nghiệm T00..T10 từ runs/ và cập nhật sheet 'Training'."""
    excel_path = Path(excel_path)
    runs_path = Path(runs_dir)

    experiment_metadata = {
        "T00": {
            "axis": "Mốc chuẩn",
            "diff": "Công thức nền T00 chuẩn (finetune, basic aug, CE loss, sampler=None, ema=None)",
            "note_base": "Mốc chuẩn T00 (kế thừa từ B02: AdamW, lr=1e-4/1e-3, CE, 12 ep; F1=96.04%)",
            "fallback_run": "B02",
        },
        "T01": {
            "axis": "Trục A (Khởi tạo)",
            "diff": "Khởi tạo ngẫu nhiên từ đầu (init='scratch', không dùng pretrained weights)",
            "note_base": "Khởi tạo scratch trên DeepWeeds; đánh giá khả năng học từ đầu với 10k ảnh",
        },
        "T02": {
            "axis": "Trục A (Khởi tạo)",
            "diff": "Đóng băng backbone (init='frozen', chỉ huấn luyện linear head)",
            "note_base": "Đóng băng backbone; đánh giá chất lượng đặc trưng zero-shot của ImageNet",
        },
        "T03": {
            "axis": "Trục B (Augmentation)",
            "diff": "Tăng cường dữ liệu TrivialAugment (aug='trivial')",
            "note_base": "TrivialAugmentWide; đánh giá tác động của biến dạng hình học/màu sắc mạnh",
        },
        "T04": {
            "axis": "Trục B (Augmentation)",
            "diff": "Trộn mẫu CutMix (mix='cutmix', mix_alpha=1.0)",
            "note_base": "CutMix patch; buộc mạng học các đặc trưng cục bộ phân tán",
        },
        "T05": {
            "axis": "Trục C (Hàm loss)",
            "diff": "Làm mịn nhãn (loss='ls', label_smoothing=0.1)",
            "note_base": "Label smoothing 0.1; chống tự tin thái quá, cải thiện hiệu chuẩn",
        },
        "T06": {
            "axis": "Trục C (Hàm loss)",
            "diff": "Focal Loss (loss='focal', focal_gamma=2.0)",
            "note_base": "Focal Loss; giảm trọng số các mẫu Negative dễ, tập trung vào mẫu khó",
        },
        "T07": {
            "axis": "Trục D (Cân bằng mẫu)",
            "diff": "Bộ lấy mẫu cân bằng lớp (sampler='balanced')",
            "note_base": "WeightedRandomSampler; cân bằng tần suất xuất hiện giữa các lớp",
        },
        "T08": {
            "axis": "Trục F (Chính quy hóa)",
            "diff": "Trọng số trung bình động (ema_decay=0.999)",
            "note_base": "EMA 0.999; làm phẳng vùng cực tiểu và giảm phương sai tham số",
        },
        "T09": {
            "axis": "Kết hợp (Combo)",
            "diff": "Kết hợp tối ưu tương thích: loss='focal' (γ=2.0) + aug='color' + ema_decay=0.999",
            "note_base": "Best Recipe kết hợp các trục trực giao; cộng dồn lợi thế ổn định",
        },
        "T10": {
            "axis": "Kết hợp (Combo CutMix)",
            "diff": "Kết hợp không gian: mix='cutmix' + loss='focal' + ema_decay=0.999",
            "note_base": "Tổ hợp CutMix + Focal + EMA",
        },
    }

    t00_f1 = 0.9604
    # Ghi T00 trước
    t00_sum_p = runs_path / "T00" / "seed0" / "summary.json"
    if not t00_sum_p.exists():
        t00_sum_p = runs_path / "B02" / "seed0" / "summary.json"

    if t00_sum_p.exists():
        with open(t00_sum_p, "r", encoding="utf-8") as f:
            t00_data = json.load(f)
        t00_f1 = float(t00_data.get("best_val_macro_f1", 0.9604))
        t00_top1 = float(t00_data.get("best_val_top1", 0.9697))
        upsert_training_row(excel_path, {
            "exp_id": "T00",
            "backbone": t00_data.get("backbone", "convnext_tiny"),
            "axis": experiment_metadata["T00"]["axis"],
            "diff": experiment_metadata["T00"]["diff"],
            "seed": 0,
            "macro_f1": t00_f1,
            "top1": t00_top1,
            "Δ so với T00": 0.0000,
            "note": experiment_metadata["T00"]["note_base"],
        })
        print(f"Đã cập nhật T00 vào {excel_path} (F1={t00_f1:.4f})")

    # Quét T01..T10
    for exp_id in ["T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09", "T10"]:
        sum_p = runs_path / exp_id / "seed0" / "summary.json"
        if not sum_p.exists():
            continue

        with open(sum_p, "r", encoding="utf-8") as f:
            summary = json.load(f)

        meta = experiment_metadata.get(exp_id, {})
        macro_f1 = float(summary.get("best_val_macro_f1", 0.0))
        top1 = float(summary.get("best_val_top1", 0.0))
        delta = round(macro_f1 - t00_f1, 4)

        # Đánh giá so với ngưỡng nhiễu seed (0.0030)
        note_base = meta.get("note_base", "")
        if abs(delta) < 0.0030:
            noise_eval = "Không phân biệt được so với nhiễu seed (|Δ| < 0.0030)"
        elif delta >= 0.0030:
            noise_eval = f"Cải thiện có ý nghĩa thống kê (Δ = +{delta:.4f} ≥ +0.0030)"
        else:
            noise_eval = f"Suy giảm hiệu năng rõ rệt (Δ = {delta:.4f} ≤ -0.0030)"

        full_note = f"{note_base}; {noise_eval}"

        record = {
            "exp_id": exp_id,
            "backbone": summary.get("backbone", "convnext_tiny"),
            "axis": meta.get("axis", ""),
            "diff": meta.get("diff", ""),
            "seed": int(summary.get("seed", 0)),
            "macro_f1": macro_f1,
            "top1": top1,
            "Δ so với T00": delta,
            "note": full_note,
        }

        upsert_training_row(excel_path, record)
        print(f"Đã cập nhật {exp_id} vào {excel_path} (F1={macro_f1:.4f}, Δ={delta:+.4f})")


if __name__ == "__main__":
    populate_backbones_from_runs()
    populate_training_from_runs()
