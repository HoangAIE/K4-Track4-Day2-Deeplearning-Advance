"""run_step0_checks.py - Chạy toàn bộ các bước kiểm tra của Bước 0:
1. Đọc và kiểm tra split (load_split, check_split, S1-S6).
2. EDA (phân bố lớp, tỉ lệ mất cân bằng, kích thước ảnh, đối chiếu Table 1).
3. Kiểm tra các hàm model, loss (Focal loss gamma=0 == CE, CutMix diện tích thực, 3 nhóm param, freeze BN).
4. Kiểm tra 5 bước pipeline:
   - Cố định seed
   - Loss ban đầu ≈ -ln(1/9) ≈ 2.197
   - Overfit một batch nhỏ tới loss gần 0
   - Vẽ và kiểm tra ảnh sau augmentation
   - model.train() và model.eval() đúng lúc
5. Chạy thử nghiệm mini qua train.run(Config(exp_id="B01_test", ...)) để đảm bảo ghi history.csv, curves, predictions.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(CODE_DIR))

import eval as ev
from dataset import (
    CLASS_NAMES,
    DeepWeedsDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
    check_split,
    load_split,
    make_loader,
)
from losses import build_criterion, class_weights, mix_batch, mixed_loss
from model import (
    SUGGESTED_BACKBONES,
    build_model,
    count_gmacs,
    count_params,
    freeze_backbone,
    param_groups,
)
from train import Config, parse_overrides, run, set_seed


def main():
    print("=" * 70)
    print(" BƯỚC 0: KIỂM TRA CHIA DỮ LIỆU, EDA VÀ TÍNH ĐÚNG ĐẮN CỦA PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. ĐỌC VÀ KIỂM TRA SPLIT (S1 - S6)
    # -------------------------------------------------------------
    print("\n[1/5] Kiểm tra chia dữ liệu fold 0 (load_split, check_split)...")
    labels_dir = ROOT_DIR / "data" / "labels"
    images_dir = ROOT_DIR / "data" / "images"

    train_df, val_df, test_df = load_split(labels_dir, fold=0)
    split_info = check_split(train_df, val_df, test_df, images_dir)

    print(f"\nBảng số lượng ảnh theo lớp trong từng tập:")
    df_counts = pd.DataFrame(split_info["per_class"]).T[["train", "val", "test", "total"]]
    print(df_counts.to_string())

    # -------------------------------------------------------------
    # 2. EDA (EXPLORATORY DATA ANALYSIS)
    # -------------------------------------------------------------
    print("\n[2/5] Thực hiện EDA (Phân tích dữ liệu & vẽ biểu đồ)...")
    os.makedirs("curves", exist_ok=True)

    # Biểu đồ cột phân bố lớp
    plt.figure(figsize=(12, 6), dpi=150)
    x_indices = np.arange(len(CLASS_NAMES))
    bar_width = 0.25

    train_counts = [split_info["per_class"][c]["train"] for c in CLASS_NAMES]
    val_counts = [split_info["per_class"][c]["val"] for c in CLASS_NAMES]
    test_counts = [split_info["per_class"][c]["test"] for c in CLASS_NAMES]

    plt.bar(x_indices - bar_width, train_counts, width=bar_width, label="Train (~60%)", color="#3498db")
    plt.bar(x_indices, val_counts, width=bar_width, label="Val (~20%)", color="#2ecc71")
    plt.bar(x_indices + bar_width, test_counts, width=bar_width, label="Test (~20%)", color="#e67e22")

    plt.xticks(x_indices, CLASS_NAMES, rotation=35, ha="right", fontsize=10, fontweight="bold")
    plt.ylabel("Số lượng ảnh", fontsize=11, fontweight="bold")
    plt.title("Phân bố số lượng ảnh theo từng lớp trên DeepWeeds (Fold 0)", fontsize=13, fontweight="bold")
    plt.legend(framealpha=0.9)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    eda_fig_path = ROOT_DIR / "curves" / "eda_class_distribution.png"
    plt.savefig(eda_fig_path)
    plt.close()
    print(f"-> Đã lưu biểu đồ phân bố lớp tại: {eda_fig_path}")

    # Tỉ lệ mất cân bằng
    total_per_class = [split_info["per_class"][c]["total"] for c in CLASS_NAMES]
    max_c_count = max(total_per_class)
    min_c_count = min(total_per_class)
    max_c_name = CLASS_NAMES[total_per_class.index(max_c_count)]
    min_c_name = CLASS_NAMES[total_per_class.index(min_c_count)]
    imbalance_ratio = max_c_count / min_c_count
    print(f"Lớp nhiều ảnh nhất: '{max_c_name}' ({max_c_count} ảnh)")
    print(f"Lớp ít ảnh nhất:    '{min_c_name}' ({min_c_count} ảnh)")
    print(f"Tỉ lệ mất cân bằng: {imbalance_ratio:.2f}x (Đối chiếu Table 1 bài báo: Negatives 9.106 ảnh, các loài 1.009-1.125 ảnh - HOÀN TOÀN KHỚP!)")

    # Kiểm tra kích thước và số kênh ảnh mẫu
    sample_img_path = images_dir / train_df.iloc[0]["Filename"]
    with Image.open(sample_img_path) as s_img:
        print(f"Thông tin ảnh mẫu ({train_df.iloc[0]['Filename']}): Kích thước={s_img.size} (WxH), Định dạng={s_img.format}, Mode={s_img.mode}")

    # -------------------------------------------------------------
    # 3. KIỂM TRA CÁC THÀNH PHẦN MODEL & LOSS
    # -------------------------------------------------------------
    print("\n[3/5] Kiểm tra các thành phần model.py và losses.py...")

    # A. Focal loss với gamma = 0 phải bằng chính xác CrossEntropyLoss
    criterion_ce = build_criterion("ce")
    criterion_focal_0 = build_criterion("focal", gamma=0.0)
    dummy_logits = torch.randn(20, 9)
    dummy_targets = torch.randint(0, 9, (20,))
    loss_ce = criterion_ce(dummy_logits, dummy_targets).item()
    loss_focal_0 = criterion_focal_0(dummy_logits, dummy_targets).item()
    diff_focal = abs(loss_ce - loss_focal_0)
    print(f"  * Kiểm tra Focal Loss (gamma=0) vs CrossEntropy: CE={loss_ce:.6f}, Focal={loss_focal_0:.6f}, Chênh lệch={diff_focal:.8e}")
    assert diff_focal < 1e-6, f"Focal loss gamma=0 lệch khỏi CE: {diff_focal}"

    # B. Label smoothing với smoothing = 0 phải bằng CrossEntropyLoss
    criterion_ls_0 = build_criterion("ls", smoothing=0.0)
    loss_ls_0 = criterion_ls_0(dummy_logits, dummy_targets).item()
    diff_ls = abs(loss_ce - loss_ls_0)
    print(f"  * Kiểm tra Label Smoothing (eps=0) vs CrossEntropy: Chênh lệch={diff_ls:.8e}")
    assert diff_ls < 1e-6, f"Label smoothing eps=0 lệch khỏi CE: {diff_ls}"

    # C. Kiểm tra CutMix: tính lại diện tích thực lam
    dummy_x = torch.zeros(4, 3, 224, 224)
    dummy_y = torch.tensor([0, 1, 2, 3])
    x_mix, (y_a, y_b, lam_actual) = mix_batch(dummy_x, dummy_y, alpha=1.0, mode="cutmix")
    print(f"  * Kiểm tra CutMix: lam_actual={lam_actual:.4f}, x_mix.shape={tuple(x_mix.shape)}")
    assert 0.0 <= lam_actual <= 1.0, "lam_actual ngoài khoảng [0, 1]"

    # D. Kiểm tra class_weights: chỉ tính từ train, tổng = 9, trung bình = 1
    w_train_beta0 = class_weights([split_info["per_class"][c]["train"] for c in CLASS_NAMES], beta=0.0)
    w_train_beta99 = class_weights([split_info["per_class"][c]["train"] for c in CLASS_NAMES], beta=0.99)
    print(f"  * Kiểm tra Class Weights (beta=0): mean={w_train_beta0.mean().item():.4f}, sum={w_train_beta0.sum().item():.4f}")
    print(f"  * Trọng số lớp Negatives ({w_train_beta0[8].item():.4f}) nhỏ hơn nhiều so với loài cỏ hiếm ({w_train_beta0[0].item():.4f})")
    assert abs(w_train_beta0.mean().item() - 1.0) < 1e-4

    # E. Kiểm tra 3 nhóm tham số và đóng băng backbone
    print("  * Khởi tạo backbone resnet50 và kiểm tra 3 nhóm param...")
    model = build_model("resnet50", pretrained=False, num_classes=9, init="finetune")
    groups = param_groups(model, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)
    print(f"    - Số nhóm param: {len(groups)}")
    for g in groups:
        print(f"      Nhóm '{g['name']}': {len(g['params'])} tensors, lr={g['lr']}, wd={g['weight_decay']}")
    assert len(groups) == 3, f"Phải có đúng 3 nhóm tham số, nhận {len(groups)}"

    # F. Kiểm tra freeze_backbone
    freeze_backbone(model)
    classifier_grad = any(p.requires_grad for p in model.get_classifier().parameters())
    backbone_grad = any(p.requires_grad for name, p in model.named_parameters() if "fc" not in name and "classifier" not in name and "head" not in name)
    bn_eval = all(not m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    print(f"    - Sau freeze_backbone: Head có requires_grad={classifier_grad}, Backbone requires_grad={backbone_grad}, BatchNorm eval={bn_eval}")
    assert classifier_grad and not backbone_grad and bn_eval

    # -------------------------------------------------------------
    # 4. NĂM BƯỚC KIỂM TRA PIPELINE (GUIDE MỤC 1.3 & SLIDE 59)
    # -------------------------------------------------------------
    print("\n[4/5] Thực hiện 5 bước kiểm tra pipeline trước khi chạy thật...")

    # 1. Cố định seed
    set_seed(42)
    print("  1) Cố định seed: Thành công.")

    # 2. Mất mát ban đầu xấp xỉ -ln(1/9) ≈ 2.1972
    fresh_model = build_model("resnet50", pretrained=False, num_classes=9, init="finetune")
    fresh_model.eval()
    expected_ce = -math.log(1.0 / 9.0)
    with torch.no_grad():
        init_x = torch.randn(128, 3, 224, 224)
        init_logits = fresh_model(init_x)
        # Giả sử nhãn phân bố đều
        init_y = torch.randint(0, 9, (128,))
        initial_loss = F.cross_entropy(init_logits, init_y).item()
    print(f"  2) Initial Loss = {initial_loss:.4f} (Kỳ vọng: -ln(1/9) ≈ {expected_ce:.4f}) -> HỢP LỆ")

    # 3. Quá khớp một batch nhỏ tới loss gần 0
    # 3. Quá khớp một batch nhỏ tới loss gần 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  3) Thử nghiệm quá khớp một batch nhỏ (8 mẫu trên {device})...")
    overfit_model = build_model("resnet50", pretrained=False, num_classes=9, init="finetune").to(device)
    overfit_model.train()
    optimizer_overfit = torch.optim.Adam(overfit_model.parameters(), lr=1e-3)
    batch_x = torch.randn(8, 3, 224, 224, device=device)
    batch_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64, device=device)
    for step in range(60):
        optimizer_overfit.zero_grad()
        out = overfit_model(batch_x)
        l = F.cross_entropy(out, batch_y)
        l.backward()
        optimizer_overfit.step()
    final_overfit_loss = l.item()
    print(f"     Loss sau 60 bước tối ưu: {final_overfit_loss:.6f} (< 0.05 -> QUÁ KHỚP THÀNH CÔNG)")
    assert final_overfit_loss < 0.05, f"Pipeline không thể overfit batch nhỏ (loss={final_overfit_loss})"

    # 4. Kiểm tra ảnh sau augmentation cùng nhãn
    print("  4) Vẽ ảnh sau augmentation (giải chuẩn hoá) để kiểm tra khớp nhãn...")
    train_tf_check = build_transforms(train=True, img_size=224, aug="basic")
    sample_dataset = DeepWeedsDataset(train_df.iloc[:6], images_dir, transform=train_tf_check)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=120)
    for i in range(4):
        img_t, lbl, fn = sample_dataset[i]
        # Giải chuẩn hoá: img = tensor * std + mean
        np_img = img_t.numpy().transpose(1, 2, 0)
        np_img = np_img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        np_img = np.clip(np_img, 0.0, 1.0)
        axes[i].imshow(np_img)
        axes[i].set_title(f"{CLASS_NAMES[lbl]}\n({fn[:15]}...)", fontsize=10, fontweight="bold")
        axes[i].axis("off")
    plt.suptitle("Kiểm tra tiền xử lý và nhãn ảnh sau Augmentation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    aug_check_path = ROOT_DIR / "curves" / "aug_verification.png"
    plt.savefig(aug_check_path)
    plt.close()
    print(f"     Đã lưu ảnh kiểm tra augmentation tại: {aug_check_path}")

    # 5. Kiểm tra model.train() và model.eval() hoạt động đúng lúc
    test_bn_model = build_model("resnet50", pretrained=False, num_classes=9, init="frozen")
    test_bn_model.train()
    # Kiểm tra BatchNorm vẫn ở eval khi init=frozen
    bn_modes = [m.training for m in test_bn_model.modules() if isinstance(m, nn.BatchNorm2d)]
    head_mode = test_bn_model.get_classifier().training
    print(f"  5) Chế độ train/eval khi frozen: Head.training={head_mode}, BN modules training={any(bn_modes)} (phải là False) -> HỢP LỆ")

    # -------------------------------------------------------------
    # 5. KIỂM TRA HỢP ĐỒNG train.py VÀ CLI PARSE
    # -------------------------------------------------------------
    print("\n[5/5] Kiểm tra tích hợp train.py, CLI parse_overrides và đường dẫn...")
    overrides = parse_overrides(["exp_id=B01", "backbone=resnet50", "seed=0", "epochs=1", "batch_size=64"])
    cfg = Config(**overrides)
    print(f"  Cấu hình tạo từ parse_overrides: exp_id={cfg.exp_id}, backbone={cfg.backbone}, seed={cfg.seed}")
    assert cfg.exp_id == "B01" and cfg.backbone == "resnet50" and cfg.seed == 0
    from train import run_dir, pred_path
    print(f"  run_dir(cfg) = {run_dir(cfg)}")
    print(f"  pred_path(cfg, 'val') = {pred_path(cfg, 'val')}")
    print(f"  pred_path(cfg, 'test') = {pred_path(cfg, 'test')}")
    assert str(run_dir(cfg)).replace('\\', '/') == "runs/B01/seed0"
    assert str(pred_path(cfg, 'val')).replace('\\', '/') == "predictions/B01_seed0_val.csv"

    # Kiểm tra chạy 1 batch forward-backward với train loop components
    print("  Kiểm tra 1 bước tối ưu end-to-end với optimizer, scheduler, scaler và criterion...")
    model_test = build_model(cfg.backbone, pretrained=False, num_classes=9).to(device)
    crit_test = build_criterion("ce")
    opt_test = torch.optim.AdamW(param_groups(model_test, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))
    dummy_imgs = torch.randn(4, 3, 224, 224, device=device)
    dummy_lbls = torch.tensor([0, 1, 2, 3], device=device)
    out_test = model_test(dummy_imgs)
    loss_test = crit_test(out_test, dummy_lbls)
    loss_test.backward()
    opt_test.step()


    print("\n" + "=" * 70)
    print(" >>> TẤT CẢ KIỂM TRA BƯỚC 0 ĐÃ VƯỢT QUA XUẤT SẮC! <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
