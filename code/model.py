"""model.py - tạo backbone qua timm, đóng băng, 3 nhóm tham số, đếm params/GMAC.

Quy tắc:
  - 3 nhóm tham số (slide trang 52)
  - Đóng băng backbone: BatchNorm phải ở chế độ eval
  - Giao diện giữ nguyên theo docstring starter
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

SUGGESTED_BACKBONES: Dict[str, str] = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    if timm is None:
        raise ImportError("Cần cài đặt timm (`pip install timm`).")

    model_name = SUGGESTED_BACKBONES.get(name, name)
    use_pretrained = (init != "scratch") and pretrained

    try:
        model = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    except Exception as e:
        if "resnet50" in model_name:
            import torchvision.models as tv_models
            weights = tv_models.ResNet50_Weights.DEFAULT if use_pretrained else None
            tv_model = tv_models.resnet50(weights=weights)
            in_features = tv_model.fc.in_features
            tv_model.fc = nn.Linear(in_features, num_classes)
            tv_model.get_classifier = lambda: tv_model.fc
            tv_model.default_cfg = {"tag": "torchvision_resnet50"}
            model = tv_model
        else:
            raise e
    model._init_mode = init
    model._backbone_name = model_name

    # Lưu tag cấu hình pretrained
    cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", None)
    if cfg and isinstance(cfg, dict):
        model.pretrained_tag = cfg.get("tag", cfg.get("architecture", model_name))
    else:
        model.pretrained_tag = model_name if use_pretrained else "scratch"

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ classifier head.

    Lưu ý quan trọng: Khi backbone đóng băng, toàn bộ BatchNorm của backbone
    phải được giữ ở chế độ eval() trong suốt quá trình train để không cập nhật running stats.
    """
    for param in model.parameters():
        param.requires_grad = False

    classifier = model.get_classifier()
    if classifier is not None:
        if isinstance(classifier, nn.Module):
            for param in classifier.parameters():
                param.requires_grad = True
        elif isinstance(classifier, nn.Parameter):
            classifier.requires_grad = True

    # Đảm bảo BatchNorm ở chế độ eval và không bị model.train() kích hoạt lại
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.eval()

    orig_train = model.train
    def frozen_train(mode: bool = True):
        orig_train(mode)
        if mode:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                    m.eval()
        return model

    model.train = frozen_train


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> List[Dict]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    1. Backbone weights có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    2. Norm và bias của backbone (ndim <= 1): lr = lr_backbone, weight_decay = 0.0
    3. Head mới: lr = lr_head, weight_decay = weight_decay
    Bỏ qua các tham số requires_grad == False.
    """
    classifier = model.get_classifier()
    head_param_ids = set()
    if classifier is not None:
        if isinstance(classifier, nn.Module):
            head_param_ids = {id(p) for p in classifier.parameters()}
        elif isinstance(classifier, nn.Parameter):
            head_param_ids = {id(classifier)}

    backbone_decay = []
    backbone_no_decay = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if id(param) in head_param_ids:
            head_params.append(param)
        elif param.ndim > 1:
            backbone_decay.append(param)
        else:
            # bias hoặc norm 1D
            backbone_no_decay.append(param)

    groups = []
    if backbone_decay:
        groups.append({
            "params": backbone_decay,
            "lr": lr_backbone,
            "weight_decay": weight_decay,
            "name": "backbone_decay",
        })
    if backbone_no_decay:
        groups.append({
            "params": backbone_no_decay,
            "lr": lr_backbone,
            "weight_decay": 0.0,
            "name": "backbone_no_decay",
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": lr_head,
            "weight_decay": weight_decay,
            "name": "head",
        })

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total = sum(p.numel() for p in model.parameters())
    return round(total / 1e6, 3)


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (Multiply-Accumulate operations, không phải FLOPs 2x).

    Sử dụng forward hook chuẩn để tính chính xác cho Conv2d, Linear.
    """
    total_macs = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, input, output):
        nonlocal total_macs
        # output shape: (B, C_out, H_out, W_out)
        out = output[0] if isinstance(output, tuple) else output
        b, c_out, h_out, w_out = out.shape
        c_in = module.in_channels
        k_h, k_w = module.kernel_size
        groups = module.groups
        macs_per_elem = (c_in // groups) * k_h * k_w
        total_macs += c_out * h_out * w_out * macs_per_elem

    def linear_hook(module: nn.Linear, input, output):
        nonlocal total_macs
        # input shape: (*, in_features)
        inp = input[0]
        num_tokens = inp.numel() // module.in_features
        total_macs += num_tokens * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    dummy_input = torch.zeros(1, 3, img_size, img_size, device=device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(dummy_input)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    gmacs = total_macs / 1e9
    return round(gmacs, 3)
