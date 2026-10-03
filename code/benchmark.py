"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() trước và sau mỗi lượt đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, phiên bản PyTorch
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn


def bench(fn: Callable[[], None], warmup: int = 10, iters: int = 100,
          sync: Optional[Callable[[], None]] = None) -> dict:
    """Đo thời gian chạy hàm `fn()` theo mili-giây (ms)."""
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    # 2. Đo đạc
    durations_ms = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        durations_ms.append((t1 - t0) * 1000.0)

    arr = np.array(durations_ms)
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "mean": round(float(np.mean(arr)), 2),
        "std": round(float(np.std(arr, ddof=1)), 2) if len(arr) > 1 else 0.0,
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int, img_size: int,
                   dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên."""
    model.eval()
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(dev)

    if dtype == "fp16" and dev.type == "cuda":
        model = model.half()
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, dtype=torch.float16, device=dev)
    else:
        dummy_input = torch.randn(batch_size, 3, img_size, img_size, dtype=torch.float32, device=dev)

    sync_fn = torch.cuda.synchronize if dev.type == "cuda" else None

    if dtype == "amp" and dev.type == "cuda":
        def forward_fn():
            with torch.inference_mode(), torch.autocast(device_type="cuda"):
                model(dummy_input)
    else:
        def forward_fn():
            with torch.inference_mode():
                model(dummy_input)

    res = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    gpu_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    p50_ms = res["p50"]
    images_per_s = round(batch_size / (p50_ms / 1000.0), 1) if p50_ms > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": p50_ms,
        "p95": res["p95"],
        "p99": res["p99"],
        "mean": res["mean"],
        "images_per_s": images_per_s,
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1,
                img_size: int = 224, device: str = "cuda", **kw) -> dict:
    """Đo độ trễ của TTA với k_views."""
    report = latency_report(model, batch_size=batch_size, img_size=img_size, device=device, **kw)
    report["p50_tta"] = round(report["p50"] * k_views, 2)
    report["p95_tta"] = round(report["p95"] * k_views, 2)
    report["p99_tta"] = round(report["p99"] * k_views, 2)
    report["k_views"] = k_views
    return report
