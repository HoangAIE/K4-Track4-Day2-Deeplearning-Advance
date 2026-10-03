"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2:
  - label smoothing (trang 56)
  - focal loss (trang 57)
  - Mixup/CutMix (trang 48)
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw) -> nn.Module:
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted"."""
    kind = kind.lower()
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind in ("ls", "label_smoothing"):
        smoothing = float(kw.get("smoothing", 0.1))
        return LabelSmoothingCE(smoothing=smoothing)
    elif kind == "focal":
        gamma = float(kw.get("gamma", 2.0))
        alpha = kw.get("alpha", None)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind in ("ce_weighted", "weighted"):
        weight = kw.get("weight", None)
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Không hỗ trợ loss loại: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.smoothing <= 0.0:
            return F.cross_entropy(logits, target)

        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # NLL loss cho true class
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        # Smooth loss là trung bình âm log xác suất mọi lớp
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth_loss
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t) (slide trang 57).

    Khi gamma = 0 và alpha = None: chính xác bằng cross-entropy thông thường.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[Union[float, torch.Tensor, Sequence[float]]] = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if isinstance(alpha, (int, float)):
                self.alpha = float(alpha)
            else:
                self.register_buffer("alpha_weight", torch.as_tensor(alpha, dtype=torch.float32))
                self.alpha = "tensor"
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Lấy log(p_t) và p_t
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            if isinstance(self.alpha, float):
                loss = self.alpha * loss
            elif self.alpha == "tensor":
                alpha_w = self.alpha_weight.to(logits.device)
                loss = alpha_w[target] * loss

        return loss.mean()


def class_weights(counts: Sequence[int], beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN (không dùng val hay test).

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    num_classes = len(counts_arr)

    if beta <= 0.0:
        # Tỉ lệ nghịch với số lượng mẫu
        weights = 1.0 / np.maximum(counts_arr, 1.0)
    else:
        # Effective number of samples
        effective_num = 1.0 - np.power(beta, counts_arr)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)

    # Chuẩn hoá để trung bình cộng trọng số bằng 1 (tổng bằng num_classes)
    weights = weights * (num_classes / np.sum(weights))
    return torch.tensor(weights, dtype=torch.float32)


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0,
              mode: str = "cutmix") -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, rồi điều chỉnh lam theo
      DIỆN TÍCH THỰC của hộp sau khi cắt ở biên ảnh (slide trang 48)
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]
    """
    if alpha > 0.0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)

    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        return x_mixed, (y_a, y_b, lam)

    elif mode == "cutmix":
        h, w = x.shape[2], x.shape[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # Toạ độ tâm ngẫu nhiên
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]

        # Điều chỉnh lam theo DIỆN TÍCH THỰC sau khi clip biên
        actual_area = (bbx2 - bbx1) * (bby2 - bby1)
        lam_actual = 1.0 - float(actual_area) / float(w * h)

        return x_mixed, (y_a, y_b, lam_actual)

    else:
        raise ValueError(f"Không hỗ trợ mix mode: {mode}")


def mixed_loss(criterion: Callable, logits: torch.Tensor,
               targets: Tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
