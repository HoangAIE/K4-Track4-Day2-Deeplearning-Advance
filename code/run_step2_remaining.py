"""run_step2_remaining.py - Chạy các thí nghiệm ablation T04-T09 còn lại của Bước 2.
Backbone cố định: convnext_tiny. Mỗi thí nghiệm chỉ đổi đúng 1 yếu tố so với T00.
"""
import json
import os
import sys
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(CODE_DIR))

from train import Config, run


def check_done(exp_id: str) -> bool:
    """Kiểm tra xem thí nghiệm đã hoàn thành hay chưa."""
    summary_path = ROOT_DIR / "runs" / exp_id / "seed0" / "summary.json"
    return summary_path.exists()


def main():
    # Danh sách các thí nghiệm Bước 2 còn lại
    experiments = [
        # T04: Trục B (Augmentation) - CutMix
        Config(
            exp_id="T04",
            backbone="convnext_tiny",
            init="finetune",
            seed=0,
            mix="cutmix",
            mix_alpha=1.0,
            aug="basic",
            loss="ce",
        ),
        # T05: Trục C (Loss) - Label Smoothing
        Config(
            exp_id="T05",
            backbone="convnext_tiny",
            init="finetune",
            seed=0,
            loss="ls",
            label_smoothing=0.1,
            aug="basic",
        ),
        # T06: Trục C (Loss) - Focal Loss
        Config(
            exp_id="T06",
            backbone="convnext_tiny",
            init="finetune",
            seed=0,
            loss="focal",
            focal_gamma=2.0,
            aug="basic",
        ),
        # T07: Trục D (Sampler) - Balanced sampler
        Config(
            exp_id="T07",
            backbone="convnext_tiny",
            init="finetune",
            seed=0,
            sampler="balanced",
            aug="basic",
            loss="ce",
        ),
        # T08: Trục F (Regularization) - EMA
        Config(
            exp_id="T08",
            backbone="convnext_tiny",
            init="finetune",
            seed=0,
            ema_decay=0.999,
            aug="basic",
            loss="ce",
        ),
    ]

    results = []

    for cfg in experiments:
        if check_done(cfg.exp_id):
            print(f"\n{'='*70}")
            print(f"[SKIP] {cfg.exp_id} đã hoàn thành trước đó.")
            summary_path = ROOT_DIR / "runs" / cfg.exp_id / "seed0" / "summary.json"
            with open(summary_path) as f:
                s = json.load(f)
            results.append(s)
            print(f"  Val Macro-F1: {s['best_val_macro_f1']*100:.2f}%")
            continue

        print(f"\n{'='*70}")
        print(f"[START] Thí nghiệm {cfg.exp_id}")
        print(f"{'='*70}")
        try:
            summary = run(cfg)
            results.append(summary)
            print(f"\n[DONE] {cfg.exp_id}: Val Macro-F1 = {summary['best_val_macro_f1']*100:.2f}%")
        except Exception as e:
            print(f"\n[ERROR] {cfg.exp_id} failed: {e}")
            import traceback
            traceback.print_exc()

    # Sau khi hoàn tất T04-T08, thu thập kết quả tốt nhất để chạy T09 (combo)
    print(f"\n{'='*70}")
    print("[ANALYSIS] Phân tích kết quả để xác định thí nghiệm kết hợp T09...")
    print(f"{'='*70}")

    # Đọc tất cả kết quả T00-T08
    all_results = {}
    for tid in ["T00", "T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08"]:
        sp = ROOT_DIR / "runs" / tid / "seed0" / "summary.json"
        if sp.exists():
            with open(sp) as f:
                all_results[tid] = json.load(f)
            print(f"  {tid}: Val Macro-F1 = {all_results[tid]['best_val_macro_f1']*100:.2f}%")

    # Xác định yếu tố tốt nhất cho từng trục
    t00_f1 = all_results.get("T00", {}).get("best_val_macro_f1", 0)

    # Chọn các yếu tố tốt nhất (delta > 0 so với T00)
    best_aug = "basic"
    best_mix = None
    best_loss = "ce"
    best_ls = 0.0
    best_sampler = None
    best_ema = None

    # Trục B: TrivialAugment (T03) vs CutMix (T04)
    t03_f1 = all_results.get("T03", {}).get("best_val_macro_f1", 0)
    t04_f1 = all_results.get("T04", {}).get("best_val_macro_f1", 0)
    if t03_f1 > t00_f1:
        best_aug = "trivial"
    if t04_f1 > max(t00_f1, t03_f1):
        best_mix = "cutmix"
        best_aug = "basic"  # CutMix dùng basic aug

    # Trục C: Label Smoothing (T05) vs Focal (T06)
    t05_f1 = all_results.get("T05", {}).get("best_val_macro_f1", 0)
    t06_f1 = all_results.get("T06", {}).get("best_val_macro_f1", 0)
    if t05_f1 > t00_f1 and t05_f1 >= t06_f1:
        best_loss = "ls"
        best_ls = 0.1
    elif t06_f1 > t00_f1:
        best_loss = "focal"

    # Trục D: Balanced sampler (T07)
    t07_f1 = all_results.get("T07", {}).get("best_val_macro_f1", 0)
    if t07_f1 > t00_f1:
        best_sampler = "balanced"

    # Trục F: EMA (T08)
    t08_f1 = all_results.get("T08", {}).get("best_val_macro_f1", 0)
    if t08_f1 > t00_f1:
        best_ema = 0.999

    print(f"\n  Combo T09 sẽ dùng: aug={best_aug}, mix={best_mix}, loss={best_loss}, "
          f"ls={best_ls}, sampler={best_sampler}, ema={best_ema}")

    # Chạy T09 combo
    if not check_done("T09"):
        cfg_combo = Config(
            exp_id="T09",
            backbone="convnext_tiny",
            init="finetune",
            seed=0,
            aug=best_aug,
            mix=best_mix,
            loss=best_loss,
            label_smoothing=best_ls,
            sampler=best_sampler,
            ema_decay=best_ema,
        )
        print(f"\n{'='*70}")
        print(f"[START] Thí nghiệm T09 (Kết hợp tốt nhất)")
        print(f"{'='*70}")
        try:
            summary_combo = run(cfg_combo)
            print(f"\n[DONE] T09: Val Macro-F1 = {summary_combo['best_val_macro_f1']*100:.2f}%")
        except Exception as e:
            print(f"\n[ERROR] T09 failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[SKIP] T09 đã hoàn thành trước đó.")

    # Tổng hợp kết quả cuối cùng
    print(f"\n{'='*70}")
    print("TỔNG HỢP KẾT QUẢ BƯỚC 2 — ABLATION TRÊN convnext_tiny")
    print(f"{'='*70}")
    print(f"{'Mã':<6} {'Trục':<20} {'Thay đổi':<30} {'Val F1':>10} {'Delta':>10}")
    print("-" * 80)

    axis_desc = {
        "T00": ("Baseline", "finetune, CE, basic aug"),
        "T01": ("A (Init)", "scratch"),
        "T02": ("A (Init)", "frozen"),
        "T03": ("B (Aug)", "TrivialAugment"),
        "T04": ("B (Aug/Mix)", "CutMix alpha=1.0"),
        "T05": ("C (Loss)", "Label Smoothing eps=0.1"),
        "T06": ("C (Loss)", "Focal Loss gamma=2.0"),
        "T07": ("D (Sampler)", "Balanced Sampler"),
        "T08": ("F (Reg)", "EMA decay=0.999"),
        "T09": ("Combo", "Best combination"),
    }

    for tid in ["T00", "T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09"]:
        sp = ROOT_DIR / "runs" / tid / "seed0" / "summary.json"
        if sp.exists():
            with open(sp) as f:
                s = json.load(f)
            f1 = s['best_val_macro_f1'] * 100
            delta = (s['best_val_macro_f1'] - t00_f1) * 100
            axis, desc = axis_desc.get(tid, ("?", "?"))
            delta_str = f"{delta:+.2f}%" if tid != "T00" else "—"
            print(f"{tid:<6} {axis:<20} {desc:<30} {f1:>9.2f}% {delta_str:>10}")

    print(f"\n=== BƯỚC 2 HOÀN TẤT ===")


if __name__ == "__main__":
    main()
