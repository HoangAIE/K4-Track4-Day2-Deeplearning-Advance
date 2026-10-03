"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H):
đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Đảm bảo import được eval.py từ root repo hoặc thư mục hiện tại
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import eval as ev
from dataset import (
    CLASS_NAMES,
    DeepWeedsDataset,
    build_transforms,
    check_split,
    load_split,
    make_loader,
)
from losses import build_criterion, class_weights, mix_batch, mixed_loss
from model import (
    build_model,
    count_gmacs,
    count_params,
    freeze_backbone,
    param_groups,
)


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: Optional[str] = None     # None | balanced
    mix: Optional[str] = None         # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: Optional[float] = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: Optional[float] = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số (xem model.param_groups)."""
    groups = param_groups(model, lr_backbone=cfg.lr_backbone, lr_head=cfg.lr_head, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55)."""
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    total_steps = max(1, cfg.epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        # Cosine giảm về 1e-4 thay vì 0 hoàn toàn để tránh gradient vanishing
        return max(1e-4, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W (slide trang 56)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone().detach()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def train_one_epoch(model: nn.Module, loader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, scaler,
                    cfg: Config, device: torch.device, ema: Optional[EMA] = None) -> dict:
    """Một epoch huấn luyện."""
    if cfg.init == "frozen":
        model.eval()
        classifier = model.get_classifier()
        if classifier is not None and isinstance(classifier, nn.Module):
            classifier.train()
    else:
        model.train()

    total_loss = 0.0
    num_samples = 0
    start_time = time.perf_counter()

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        # Mixup hoặc CutMix
        if cfg.mix in ("mixup", "cutmix"):
            images, targets = mix_batch(images, labels, alpha=cfg.mix_alpha, mode=cfg.mix)
            with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
                logits = model(images)
                loss = mixed_loss(criterion, logits, targets)
        else:
            with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

        optimizer.zero_grad()
        if cfg.amp and scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * batch_size
        num_samples += batch_size

    epoch_time = time.perf_counter() - start_time
    avg_loss = total_loss / max(1, num_samples)
    current_lr = optimizer.param_groups[0]["lr"]

    return {
        "train_loss": avg_loss,
        "lr": current_lr,
        "epoch_time_s": epoch_time,
    }


def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    """
    model.eval()
    all_filenames: List[str] = []
    all_y_true: List[int] = []
    all_logits: List[np.ndarray] = []
    total_loss = 0.0
    num_samples = 0

    with torch.inference_mode():
        for images, labels, fnames in loader:
            images = images.to(device, non_blocking=True)
            labels_dev = labels.to(device, non_blocking=True)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, labels_dev)

            total_loss += loss.item() * batch_size
            num_samples += batch_size

            all_filenames.extend(fnames)
            all_y_true.extend(labels.numpy().tolist())
            all_logits.append(logits.cpu().numpy())

    logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, ev.NUM_CLASSES))
    y_true_arr = np.array(all_y_true, dtype=np.int64)
    avg_loss = total_loss / max(1, num_samples)

    return all_filenames, y_true_arr, logits_arr, avg_loss


def plot_curves(history: List[dict], path: Union[str, Path], title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

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
    plt.savefig(path)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt."""
    set_seed(cfg.seed)
    output_dir = ROOT_DIR / run_dir(cfg) if not Path(cfg.out_dir).is_absolute() else run_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir_path = ROOT_DIR / cfg.pred_dir if not Path(cfg.pred_dir).is_absolute() else Path(cfg.pred_dir)
    pred_dir_path.mkdir(parents=True, exist_ok=True)
    curves_dir = ROOT_DIR / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ghi config.json
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    # 2. Đọc và kiểm tra split (S1-S6)
    labels_path = Path(cfg.labels_dir) if Path(cfg.labels_dir).exists() else ROOT_DIR / cfg.labels_dir
    images_path = Path(cfg.images_dir) if Path(cfg.images_dir).exists() else ROOT_DIR / cfg.images_dir
    train_df, val_df, test_df = load_split(labels_path, fold=cfg.fold)
    check_split(train_df, val_df, test_df, images_path)

    # 3. Dựng DataLoader
    train_tf = build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = build_transforms(train=False, img_size=cfg.img_size)

    train_loader = make_loader(
        train_df, images_path, transform=train_tf,
        batch_size=cfg.batch_size, train=True,
        sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = make_loader(
        val_df, images_path, transform=val_tf,
        batch_size=cfg.batch_size, train=False,
        sampler=None, num_workers=cfg.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Model
    model = build_model(
        cfg.backbone,
        pretrained=True,
        num_classes=ev.NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init,
    )
    model.to(device)

    n_params = count_params(model)
    gmacs = count_gmacs(model, img_size=cfg.img_size)

    # 5. Loss
    if cfg.loss in ("ce_weighted", "weighted"):
        train_counts = [int((train_df["Label"] == c).sum()) for c in range(ev.NUM_CLASSES)]
        w = class_weights(train_counts, beta=cfg.class_weight_beta or 0.0).to(device)
        criterion = build_criterion("ce_weighted", weight=w)
    elif cfg.loss in ("ls", "label_smoothing"):
        criterion = build_criterion("ls", smoothing=cfg.label_smoothing)
    elif cfg.loss == "focal":
        criterion = build_criterion("focal", gamma=cfg.focal_gamma)
    else:
        criterion = build_criterion("ce")

    val_criterion = nn.CrossEntropyLoss()

    # 6. Optimizer, Scheduler, Scaler, EMA
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device.type == "cuda") else None
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay is not None else None

    # 7. Vòng huấn luyện theo epoch
    best_macro_f1 = -1.0
    best_epoch = 0
    best_val_preds: Optional[dict] = None
    history: List[dict] = []
    best_ckpt_path = output_dir / "best_checkpoint.pt"

    print(f"\n[Bắt đầu] Thí nghiệm {cfg.exp_id} | Backbone={cfg.backbone} | Init={cfg.init} | Seed={cfg.seed} | Params={n_params}M | GMAC={gmacs}")

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(
            model=model, loader=train_loader, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            cfg=cfg, device=device, ema=ema
        )

        # Đánh giá trên VAL
        if ema is not None:
            ema.apply_shadow(model)

        val_fnames, val_true, val_logits, val_loss = evaluate(model, val_loader, val_criterion, device)

        if ema is not None:
            ema.restore(model)

        # Tính metric bằng eval.compute_metrics chuẩn repo
        val_probs = np.exp(val_logits - val_logits.max(axis=1, keepdims=True))
        val_probs /= val_probs.sum(axis=1, keepdims=True)
        val_ypred = val_probs.argmax(axis=1)

        val_metrics = ev.compute_metrics(val_true, val_ypred, val_probs)
        val_macro_f1 = float(val_metrics["macro_f1"])
        val_top1 = float(val_metrics["top1"])

        rec = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_top1": val_top1,
            "lr": train_stats["lr"],
            "epoch_time_s": train_stats["epoch_time_s"],
        }
        history.append(rec)

        print(f"Epoch {epoch:02d}/{cfg.epochs:02d} | Train Loss: {train_stats['train_loss']:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_macro_f1 * 100:.2f}% | Val Top-1: {val_top1 * 100:.2f}% | Time: {train_stats['epoch_time_s']:.1f}s")

        # Chọn checkpoint theo macro-F1 val (hòa thì lấy epoch sớm hơn)
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "macro_f1": val_macro_f1,
                "top1": val_top1,
                "cfg": dataclasses.asdict(cfg),
            }, best_ckpt_path)
            best_val_preds = {
                "filenames": val_fnames,
                "y_true": val_true,
                "y_pred": val_ypred,
                "probs": val_probs,
                "logits": val_logits,
            }

    # 8. Lưu kết quả tốt nhất trên VAL bằng eval.save_predictions
    if best_val_preds is not None:
        val_pred_file = pred_dir_path / f"{cfg.exp_id}_seed{cfg.seed}_val.csv"
        ev.save_predictions(val_pred_file, best_val_preds["filenames"], best_val_preds["y_true"], best_val_preds["probs"])
        np.save(output_dir / "val_logits.npy", best_val_preds["logits"])
        print(f"[Best Checkpoint] Epoch {best_epoch:02d} | Val Macro-F1: {best_macro_f1 * 100:.2f}% -> Lưu tại {val_pred_file}")

    # 9. Chỉ đánh giá trên TEST khi cờ save_test_predictions bật (ở Bước 4 chung kết)
    if cfg.save_test_predictions:
        print("[Đánh giá TEST] Chỉ chạy một lần trên toàn bộ test_subset0...")
        test_tf = build_transforms(train=False, img_size=cfg.img_size)
        test_loader = make_loader(test_df, images_path, transform=test_tf, batch_size=cfg.batch_size, train=False, num_workers=cfg.num_workers)

        # Nạp lại checkpoint tốt nhất
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_fnames, test_true, test_logits, _ = evaluate(model, test_loader, val_criterion, device)
        test_probs = np.exp(test_logits - test_logits.max(axis=1, keepdims=True))
        test_probs /= test_probs.sum(axis=1, keepdims=True)

        test_pred_file = pred_dir_path / f"{cfg.exp_id}_seed{cfg.seed}_test.csv"
        ev.save_predictions(test_pred_file, test_fnames, test_true, test_probs)
        np.save(output_dir / "test_logits.npy", test_logits)
        print(f"[TEST Đã Lưu] Dự đoán test ghi tại {test_pred_file}")

    # 10. Ghi history.csv và vẽ đường cong
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)

    curve_path = curves_dir / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_title = f"{cfg.exp_id}: {cfg.backbone} (Init={cfg.init}, Loss={cfg.loss}, Aug={cfg.aug}, Seed={cfg.seed})"
    plot_curves(history, curve_path, plot_title)

    avg_train_time = float(np.mean([h["epoch_time_s"] for h in history])) if history else 0.0
    summary = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "init": cfg.init,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_macro_f1, 4),
        "best_val_top1": round(float(history_df.loc[history_df['epoch'] == best_epoch, 'val_top1'].iloc[0]) if best_epoch > 0 else 0.0, 4),
        "avg_epoch_time_s": round(avg_train_time, 2),
        "params_m": n_params,
        "gmacs": gmacs,
        "curve_path": str(curve_path),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_overrides(pairs: List[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    cfg_fields = {f.name: f for f in dataclasses.fields(Config)}
    overrides = {}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Tham số không hợp lệ: '{pair}'. Cần cú pháp key=value.")
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()

        if key not in cfg_fields:
            raise KeyError(f"Trường '{key}' không tồn tại trong Config. Các trường hợp lệ: {list(cfg_fields.keys())}")

        field_type = cfg_fields[key].type

        # Xử lý giá trị None
        if val.lower() in ("none", "null"):
            overrides[key] = None
            continue

        # Ép kiểu dựa trên type annotation của trường (xử lý cả kiểu string từ from __future__ import annotations)
        t_str = str(field_type).lower()
        if "bool" in t_str:
            overrides[key] = val.lower() in ("true", "1", "yes")
        elif "int" in t_str and "float" not in t_str:
            overrides[key] = int(val)
        elif "float" in t_str:
            overrides[key] = float(val)
        elif field_type == bool:
            overrides[key] = val.lower() in ("true", "1", "yes")
        elif field_type == int:
            overrides[key] = int(val)
        elif field_type == float:
            overrides[key] = float(val)
        else:
            overrides[key] = val

    return overrides


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình DeepWeeds.")
    parser.add_argument("--set", nargs="*", default=[], help="Ghi đè siêu tham số theo cú pháp key=value")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    print(f"Cấu hình thí nghiệm: {cfg}")
    summary = run(cfg)
    print("\n=== HOÀN THÀNH THÍ NGHIỆM ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
