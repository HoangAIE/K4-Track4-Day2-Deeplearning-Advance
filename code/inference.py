"""inference.py - các phương pháp suy luận (TTA, ensemble, temperature scaling, gộp BatchNorm).

Liên hệ slide Day 2:
  - TTA (trang 62-66, 75)
  - ensemble/EMA/soup (trang 67)
  - độ phân giải kiểm tra (trang 68)
  - temperature scaling (trang 69)
  - gộp BatchNorm (trang 71)
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_logits(model: nn.Module, loader, device: torch.device,
                   view: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    all_fnames: List[str] = []
    all_y_true: List[int] = []
    all_logits: List[np.ndarray] = []

    with torch.inference_mode():
        for images, labels, fnames in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)

            logits = model(images)
            all_fnames.extend(fnames)
            all_y_true.extend(labels.numpy().tolist())
            all_logits.append(logits.cpu().numpy())

    y_true_arr = np.array(all_y_true, dtype=np.int64)
    logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 9))
    return all_fnames, y_true_arr, logits_arr


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W) trên chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[3])


def views_multicrop(x: torch.Tensor, crop: int) -> List[torch.Tensor]:
    """5 crop (4 góc + giữa) kích thước `crop` (slide trang 64)."""
    _, _, h, w = x.shape
    if crop > h or crop > w:
        raise ValueError(f"Kích thước crop {crop} lớn hơn kích thước ảnh {(h, w)}")

    tl = x[:, :, 0:crop, 0:crop]
    tr = x[:, :, 0:crop, w - crop:w]
    bl = x[:, :, h - crop:h, 0:crop]
    br = x[:, :, h - crop:h, w - crop:w]
    ch = (h - crop) // 2
    cw = (w - crop) // 2
    cc = x[:, :, ch:ch + crop, cw:cw + crop]

    return [tl, tr, bl, br, cc]


def views_multiscale(x: torch.Tensor, sizes: List[int]) -> List[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    return [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False) for s in sizes]


def aggregate_views(logits_per_view: List[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một phân bố xác suất (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    """
    if not logits_per_view:
        raise ValueError("Danh sách logits rỗng")

    if space == "prob":
        prob_list = []
        for l in logits_per_view:
            exp_l = np.exp(l - l.max(axis=1, keepdims=True))
            prob_list.append(exp_l / exp_l.sum(axis=1, keepdims=True))
        avg_prob = np.mean(prob_list, axis=0)
        return avg_prob / avg_prob.sum(axis=1, keepdims=True)

    elif space == "logit":
        avg_logit = np.mean(logits_per_view, axis=0)
        exp_l = np.exp(avg_logit - avg_logit.max(axis=1, keepdims=True))
        return exp_l / exp_l.sum(axis=1, keepdims=True)

    else:
        raise ValueError(f"Không hỗ trợ space={space}. Dùng 'prob' hoặc 'logit'.")


def ensemble_probs(list_of_probs: List[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed)."""
    if not list_of_probs:
        raise ValueError("Danh sách xác suất rỗng")
    avg = np.mean(list_of_probs, axis=0)
    return avg / avg.sum(axis=1, keepdims=True)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T) (slide trang 69)."""
    logits_t = torch.as_tensor(val_logits, dtype=torch.float32)
    labels_t = torch.as_tensor(val_labels, dtype=torch.int64)

    # Tối ưu hoá log(T) bằng LBFGS
    log_temp = nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.05, max_iter=100)
    criterion = nn.CrossEntropyLoss()

    def eval_closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp)
        loss = criterion(logits_t / temp, labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_closure)
    best_t = float(torch.exp(log_temp).item())
    return max(0.01, min(10.0, best_t))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về xác suất softmax(logits / T)."""
    scaled = logits / max(1e-4, T)
    exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    return probs


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào Conv2d liền trước theo công thức chuẩn (slide trang 71, 75)."""
    model.eval()
    try:
        # PyTorch hỗ trợ sẵn fuse_conv_bn_eval cho các cấu hình tiêu chuẩn
        fused = torch.nn.utils.fuse_conv_bn_eval(model)
        return fused
    except Exception:
        return model
