"""run_ablation_batch.py - Điều phối thực thi chuỗi ablation T01..T09 theo chuẩn Bước 2."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

EXPERIMENTS = {
    "T01": [
        sys.executable, "-u", "code/train.py",
        "--exp_id", "T01",
        "--seed", "0",
        "--fold", "0",
        "--set", "backbone=convnext_tiny", "init=scratch", "num_workers=0"
    ],
    "T02": [
        sys.executable, "-u", "code/train.py",
        "--exp_id", "T02",
        "--seed", "0",
        "--fold", "0",
        "--set", "backbone=convnext_tiny", "init=frozen", "num_workers=0"
    ],
    "T03": [
        sys.executable, "-u", "code/train.py",
        "--exp_id", "T03",
        "--seed", "0",
        "--fold", "0",
        "--set", "backbone=convnext_tiny", "aug=trivial", "num_workers=0"
    ],
    "T04": [
        sys.executable, "-u", "code/train.py",
        "--exp_id", "T04",
        "--seed", "0",
        "--fold", "0",
        "--set", "backbone=convnext_tiny", "mix=cutmix", "num_workers=0"
    ],
    "T05": [
        sys.executable, "-u", "code/train.py",
        "--exp_id", "T05",
        "--seed", "0",
        "--fold", "0",
        "--set", "backbone=convnext_tiny", "loss=ls", "label_smoothing=0.1", "num_workers=0"
    ],
    "T06": [
        sys.executable, "-u", "code/train.py",
        "--exp_id", "T06",
        "--seed", "0",
        "--fold", "0",
        "--set", "backbone=convnext_tiny", "loss=focal", "focal_gamma=2.0", "num_workers=0"
    ],
    "T07": [
        sys.executable, "-u", "code/train.py",
        "--exp_id", "T07",
        "--seed", "0",
        "--fold", "0",
        "--set", "backbone=convnext_tiny", "sampler=balanced", "num_workers=0"
    ],
    "T08": [
        sys.executable, "-u", "code/train.py",
        "--exp_id", "T08",
        "--seed", "0",
        "--fold", "0",
        "--set", "backbone=convnext_tiny", "ema_decay=0.999", "num_workers=0"
    ],
    "T09": [
        sys.executable, "-u", "code/train.py",
        "--exp_id", "T09",
        "--seed", "0",
        "--fold", "0",
        "--set", "backbone=convnext_tiny", "aug=color", "loss=focal", "focal_gamma=2.0", "ema_decay=0.999", "num_workers=0"
    ],
}


def is_already_done(exp_id: str) -> bool:
    """Kiểm tra xem thí nghiệm đã hoàn thành đầy đủ chưa."""
    run_dir = ROOT_DIR / "runs" / exp_id / "seed0"
    summary_p = run_dir / "summary.json"
    ckpt_p = run_dir / "best_checkpoint.pt"
    hist_p = run_dir / "history.csv"
    val_logits_p = run_dir / "val_logits.npy"

    if summary_p.exists() and ckpt_p.exists() and hist_p.exists() and val_logits_p.exists():
        try:
            with open(summary_p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if "best_val_macro_f1" in d and d["best_val_macro_f1"] > 0:
                return True
        except Exception:
            pass
    return False


def update_progress_file(exp_id: str, status: str, details: str = "") -> None:
    """Cập nhật timestamp và tiến độ vào progress.md."""
    progress_file = ROOT_DIR / ".agents" / "teamwork" / "worker_m2" / "progress.md"
    if not progress_file.exists():
        return
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        content = progress_file.read_text(encoding="utf-8")
        # Update Last visited
        lines = content.splitlines()
        new_lines = []
        for line in lines:
            if line.startswith("**Last visited**:"):
                new_lines.append(f"**Last visited**: {now_str}")
            elif line.startswith("**Current Time**:"):
                new_lines.append(f"**Current Time**: {now_str}")
            else:
                new_lines.append(line)
        # Append progress update
        new_lines.append(f"- [{now_str}] **{exp_id}**: {status} {details}")
        progress_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"Lỗi cập nhật progress.md: {e}")


def run_experiment(exp_id: str, force: bool = False) -> bool:
    """Thực thi một thí nghiệm."""
    if exp_id not in EXPERIMENTS:
        print(f"Lỗi: Không tìm thấy định nghĩa thí nghiệm {exp_id}")
        return False

    if not force and is_already_done(exp_id):
        print(f"[{exp_id}] Đã hoàn thành trước đó -> Bỏ qua (dùng --force nếu muốn chạy lại).")
        return True

    cmd = EXPERIMENTS[exp_id]
    print(f"\n{'='*70}")
    print(f"BẮT ĐẦU THÍ NGHIỆM: {exp_id} lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Lệnh: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    update_progress_file(exp_id, "Bắt đầu chạy", f"Lệnh: {' '.join(cmd[2:])}")
    start_time = time.time()

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    rc = proc.wait()
    elapsed = time.time() - start_time

    if rc != 0:
        print(f"\n[LỖI] Thí nghiệm {exp_id} thất bại với exit code {rc} sau {elapsed:.1f}s!\n")
        update_progress_file(exp_id, f"THẤT BẠI (code {rc})", f"Thời gian: {elapsed:.1f}s")
        return False

    print(f"\n[THÀNH CÔNG] Thí nghiệm {exp_id} hoàn tất sau {elapsed:.1f}s ({elapsed/60:.2f} phút)!\n")

    # Đọc kết quả summary
    summary_p = ROOT_DIR / "runs" / exp_id / "seed0" / "summary.json"
    f1_str = "N/A"
    if summary_p.exists():
        with open(summary_p, "r", encoding="utf-8") as f:
            d = json.load(f)
        f1_str = f"F1={d.get('best_val_macro_f1', 0):.4f}, Top1={d.get('best_val_top1', 0):.4f}"

    update_progress_file(exp_id, "HOÀN TẤT", f"{f1_str}, Thời gian: {elapsed/60:.1f} phút")

    # Cập nhật kết quả vào results.xlsx ngay lập tức
    try:
        subprocess.run(
            [sys.executable, "code/update_excel.py"],
            cwd=str(ROOT_DIR),
            env=env,
            check=True
        )
        print(f"Đã cập nhật {exp_id} vào results.xlsx thành công!")
    except Exception as e:
        print(f"Cảnh báo: Không thể cập nhật results.xlsx: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Chạy tuần tự chuỗi ablation T01..T09")
    parser.add_argument("exps", nargs="*", default=["T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09"],
                        help="Danh sách thí nghiệm cần chạy (mặc định: T01..T09)")
    parser.add_argument("--force", action="store_true", help="Chạy lại kể cả khi đã có artifact")
    args = parser.parse_args()

    print(f"Danh sách thí nghiệm sẽ thực thi: {args.exps}")
    success_count = 0
    for exp_id in args.exps:
        ok = run_experiment(exp_id, force=args.force)
        if ok:
            success_count += 1
        else:
            print(f"Dừng chuỗi thực thi do thí nghiệm {exp_id} gặp lỗi!")
            sys.exit(1)

    print(f"\n{'='*70}")
    print(f"HOÀN TẤT TOÀN BỘ CHUỖI THÍ NGHIỆM: {success_count}/{len(args.exps)} thành công!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
