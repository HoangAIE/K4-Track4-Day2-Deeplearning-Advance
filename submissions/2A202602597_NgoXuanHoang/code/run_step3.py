# -*- coding: utf-8 -*-
"""run_step3.py - Buoc 3: phuong phap suy luan + do do tre tren T04 (convnext_tiny CutMix).

Base model co dinh: runs/T04/seed0/best_checkpoint.pt (Val Macro-F1=97.19%).
- I00: 1-view @224 (moc)
- I01: TTA hflip K=2
- I02: TTA 5-crop K=5 (input 256, crop 224)
- I03: so sanh gop prob vs logit (tu I01/I02, numpy)
- I04: do phan giai kiem tra 224/256/288/320
- I05: ensemble (T04+T03+T07) va (T04+B03)
- I06: model soup (trung binh trong so T04+T03)
- I07: temperature scaling + ECE (fit T tren val)
- I08: FP16 + kiem tra fuse BN (ConvNeXt dung LayerNorm -> N/A)
- Latency: fp32/amp/fp16, batch 1 & 32, p50/p95/p99, warmup+sync.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms as T

ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(CODE_DIR))

import eval as ev
from dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms, load_split, make_loader
from model import build_model
from inference import (
    aggregate_views,
    apply_temperature,
    ensemble_probs,
    fit_temperature,
    fuse_conv_bn,
    view_hflip,
    view_identity,
    views_multicrop,
)
from benchmark import latency_report

BASE_EXP = "T04"
BASE_CKPT = ROOT_DIR / "runs" / BASE_EXP / "seed0" / "best_checkpoint.pt"
OUT_JSON = ROOT_DIR / "runs" / "inference" / "inference_results.json"
BATCH = 64


def softmax_np(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def metrics_from_logits(y_true: np.ndarray, logits: np.ndarray):
    probs = softmax_np(logits)
    y_pred = probs.argmax(axis=1)
    m = ev.compute_metrics(y_true, y_pred, probs)
    ece = float(ev.ece_score(probs, y_true, bins=15))
    return {
        "macro_f1": float(m["macro_f1"]),
        "top1": float(m["top1"]),
        "ece": ece,
    }


def load_base_model(device: torch.device, dtype_fp16: bool = False) -> nn.Module:
    model = build_model("convnext_tiny", pretrained=False, num_classes=9)
    ckpt = torch.load(BASE_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    if dtype_fp16:
        model = model.half()
    return model


def run_loader_logits(model, loader, device, view=None, fp16: bool = False):
    """Chay het loader, tra ve (fnames, y_true, logits). Ho tro view tensor-level."""
    from inference import predict_logits
    with torch.inference_mode():
        return predict_logits(model, loader, device, view=view)


def eval_5crop(model, loader256, device):
    """TTA 5-crop: moi batch 256x256 -> 5 crop 224, forward tung crop, gop theo batch."""
    model.eval()
    per_view_logits: list[list[np.ndarray]] = [[] for _ in range(5)]
    all_fnames: list[str] = []
    all_y: list[int] = []
    with torch.inference_mode():
        for images, labels, fnames in loader256:
            images = images.to(device, non_blocking=True)
            crops = views_multicrop(images, 224)
            for i, c in enumerate(crops):
                logits = model(c)
                per_view_logits[i].append(logits.cpu().numpy())
            all_fnames.extend(fnames)
            all_y.extend(labels.numpy().tolist())
    views = [np.concatenate(v, axis=0) for v in per_view_logits]
    return all_fnames, np.array(all_y, dtype=np.int64), views


def res_transform(n: int):
    """Transform eval xac dinh tai do phan giai n (mirrors dataset.build_transforms)."""
    if n == 224:
        return build_transforms(train=False, img_size=224)  # giong he I00: Resize256+CenterCrop224
    if n == 256:
        return T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    side = int(round(n * 1.14))  # ~ ti le resize/crop nhu pipeline 224 (256/224)
    return T.Compose([
        T.Resize(side),
        T.CenterCrop(n),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'})")
    labels_dir = ROOT_DIR / "data" / "labels"
    images_dir = ROOT_DIR / "data" / "images"
    train_df, val_df, _ = load_split(labels_dir, fold=0)

    results: dict = {"base": BASE_EXP, "entries": {}}

    def record(iid: str, method: str, ckpt_used: str, k: int, met: dict,
               lat: dict | None = None, cost: str = "", note: str = ""):
        results["entries"][iid] = {
            "exp_id": iid, "method": method, "checkpoint": ckpt_used, "K": k,
            **met, "latency": lat or {}, "cost": cost, "note": note,
        }
        print(f"[{iid}] {method} | F1={met['macro_f1']*100:.2f}% Top1={met['top1']*100:.2f}% "
              f"ECE={met['ece']:.4f} | {note}", flush=True)

    # ---------- I00: 1-view @224 ----------
    model = load_base_model(device)
    val_loader = make_loader(val_df, images_dir, transform=build_transforms(train=False, img_size=224),
                             batch_size=BATCH, train=False, sampler=None, num_workers=0)
    fnames0, y0, logits0 = run_loader_logits(model, val_loader, device)
    m00 = metrics_from_logits(y0, logits0)
    record("I00", "1-view 224 (moc)", f"{BASE_EXP} best ckpt", 1, m00, cost="1x",
           note="Resize256+CenterCrop224, FP32, model.eval()+inference_mode")

    # ---------- I01: TTA hflip K=2 ----------
    _, _, logits_id = run_loader_logits(model, val_loader, device, view=view_identity)
    _, _, logits_hf = run_loader_logits(model, val_loader, device, view=view_hflip)
    prob01 = aggregate_views([logits_id, logits_hf], space="prob")
    logit01 = aggregate_views([logits_id, logits_hf], space="logit")
    # metrics dung: argmax/softmax tren prob gop
    y01p = prob01.argmax(axis=1)
    mm = ev.compute_metrics(y0, y01p, prob01)
    m01_prob = {"macro_f1": float(mm["macro_f1"]), "top1": float(mm["top1"]),
                "ece": float(ev.ece_score(prob01, y0, bins=15))}
    y01l = logit01.argmax(axis=1)
    mml = ev.compute_metrics(y0, y01l, logit01)
    m01_logit = {"macro_f1": float(mml["macro_f1"]), "top1": float(mml["top1"]),
                 "ece": float(ev.ece_score(logit01, y0, bins=15))}
    record("I01", "TTA hflip K=2 (gop prob)", f"{BASE_EXP} best ckpt", 2, m01_prob, cost="~2x",
           note="Trung binh softmax 2 view (goc + lat ngang)")
    results["entries"]["I03a"] = {"exp_id": "I03a", "method": "I01 gop logit (so sanh)",
                                  "macro_f1": m01_logit["macro_f1"], "top1": m01_logit["top1"],
                                  "ece": m01_logit["ece"], "K": 2,
                                  "note": "Trung binh logit roi softmax; so voi I01-gop-prob"}

    # ---------- I02: TTA 5-crop K=5 ----------
    loader256 = make_loader(val_df, images_dir, transform=build_transforms(train=False, img_size=256),
                            batch_size=BATCH, train=False, sampler=None, num_workers=0)
    fn2, y2, views5 = eval_5crop(model, loader256, device)
    assert (y2 == y0).all(), "Thu tu val khac nhau giua 2 loader!"
    prob02 = aggregate_views(views5, space="prob")
    logit02 = aggregate_views(views5, space="logit")
    y02p = prob02.argmax(axis=1)
    mm2 = ev.compute_metrics(y2, y02p, prob02)
    m02_prob = {"macro_f1": float(mm2["macro_f1"]), "top1": float(mm2["top1"]),
                "ece": float(ev.ece_score(prob02, y2, bins=15))}
    record("I02", "TTA 5-crop K=5 (gop prob)", f"{BASE_EXP} best ckpt", 5, m02_prob, cost="~5x",
           note="5 crop 224 tu input 256 (4 goc + giua)")
    y02l = logit02.argmax(axis=1)
    mm2l = ev.compute_metrics(y2, y02l, logit02)
    results["entries"]["I03b"] = {"exp_id": "I03b", "method": "I02 gop logit (so sanh)",
                                  "macro_f1": float(mm2l["macro_f1"]), "top1": float(mm2l["top1"]),
                                  "ece": float(ev.ece_score(logit02, y2, bins=15)), "K": 5,
                                  "note": "Trung binh logit roi softmax; so voi I02-gop-prob"}

    # ---------- I04: do phan giai ----------
    for n in [224, 256, 288, 320]:
        loader_n = make_loader(val_df, images_dir, transform=res_transform(n),
                               batch_size=BATCH, train=False, sampler=None, num_workers=0)
        _, yn, logits_n = run_loader_logits(model, loader_n, device)
        mn = metrics_from_logits(yn, logits_n)
        tag = " (moc, trung I00)" if n == 224 else ""
        record(f"I04_{n}", f"Do phan giai test {n}{tag}", f"{BASE_EXP} best ckpt", 1, mn,
               cost="1x" if n <= 256 else "~1.7-2x FLOPs",
               note=f"Eval xac dinh tai {n}; {n}224: Resize256+CenterCrop224" if n == 224
               else f"Resize canh {int(round(n*1.14))}+CenterCrop{n} (FixRes probe)")

    # ---------- I05: ensemble tu logit luu san ----------
    def load_val_logits(exp: str):
        p = ROOT_DIR / "runs" / exp / "seed0" / "val_logits.npy"
        df = pd.read_csv(ROOT_DIR / "predictions" / f"{exp}_seed0_val.csv")
        order = df["Filename"].tolist()
        return np.load(p), df["Label"].to_numpy(dtype=np.int64) if "Label" in df.columns else None, order, df

    logits_T04, _, order_T04, dfT04 = load_val_logits("T04")
    y_ref = dfT04["y_true"].to_numpy(dtype=np.int64) if "y_true" in dfT04.columns else None
    if y_ref is None:
        # y_true tu loader (y0) theo thu tu -- can anh xa Filename
        y_ref = y0
    ens_members = {}
    for exp in ["T03", "T07", "B03"]:
        lg, _, order_e, dfe = load_val_logits(exp)
        idx = [order_e.index(f) for f in order_T04]  # can chinh thu tu theo Filename
        ens_members[exp] = lg[idx]

    probs_T04 = softmax_np(logits_T04)
    # I05a: ensemble 3 convnext (T04+T03+T07)
    ens3 = ensemble_probs([probs_T04, softmax_np(ens_members["T03"]), softmax_np(ens_members["T07"])])
    y_e3 = ens3.argmax(axis=1)
    mme3 = ev.compute_metrics(y_ref, y_e3, ens3)
    record("I05a", "Ensemble 3 convnext (T04+T03+T07)", "val_logits T04/T03/T07", 3,
           {"macro_f1": float(mme3["macro_f1"]), "top1": float(mme3["top1"]),
            "ece": float(ev.ece_score(ens3, y_ref, bins=15))}, cost="3x",
           note="Trung binh xac suat; cung kien truc, khac recipe")
    # I05b: ensemble lien kien truc (T04+B03 swin)
    ens2x = ensemble_probs([probs_T04, softmax_np(ens_members["B03"])])
    y_ex = ens2x.argmax(axis=1)
    mmex = ev.compute_metrics(y_ref, y_ex, ens2x)
    record("I05b", "Ensemble lien kien truc (T04+B03-swin)", "val_logits T04/B03", 2,
           {"macro_f1": float(mmex["macro_f1"]), "top1": float(mmex["top1"]),
            "ece": float(ev.ece_score(ens2x, y_ref, bins=15))}, cost="~2x (arch khac nhau)",
           note="Trung binh xac suat convnext+swin; da dang kien truc")

    # ---------- I06: soup T04+T03 ----------
    mA = build_model("convnext_tiny", pretrained=False, num_classes=9)
    mB = build_model("convnext_tiny", pretrained=False, num_classes=9)
    mA.load_state_dict(torch.load(ROOT_DIR / "runs/T04/seed0/best_checkpoint.pt", map_location="cpu")["model_state_dict"])
    mB.load_state_dict(torch.load(ROOT_DIR / "runs/T03/seed0/best_checkpoint.pt", map_location="cpu")["model_state_dict"])
    soup_sd = {k: (mA.state_dict()[k].float() + mB.state_dict()[k].float()) / 2.0 for k in mA.state_dict()}
    soup = build_model("convnext_tiny", pretrained=False, num_classes=9)
    soup.load_state_dict(soup_sd)
    soup.to(device).eval()
    _, ys, logits_soup = run_loader_logits(soup, val_loader, device)
    ms = metrics_from_logits(ys, logits_soup)
    record("I06", "Model soup (T04+T03)/2", "trung binh trong so T04+T03", 1, ms, cost="1x",
           note="Uniform soup 2 finetune cung init; suy luan 1-view nhu I00")
    del soup, mA, mB

    # ---------- I07: temperature scaling ----------
    T_star = fit_temperature(logits0, y0)
    probs_cal = apply_temperature(logits0, T_star)
    y_cal = probs_cal.argmax(axis=1)
    mmc = ev.compute_metrics(y0, y_cal, probs_cal)
    unchanged = bool((y_cal == probs.argmax(axis=1) if (probs := softmax_np(logits0)).any() else y_cal).all())
    record("I07", f"Temperature scaling T={T_star:.3f}", f"{BASE_EXP} best ckpt (fit T tren val)", 1,
           {"macro_f1": float(mmc["macro_f1"]), "top1": float(mmc["top1"]),
            "ece": float(ev.ece_score(probs_cal, y0, bins=15))}, cost="1x",
           note=f"ECE truoc={m00['ece']:.4f} -> sau={float(ev.ece_score(probs_cal, y0, bins=15)):.4f}; "
                f"argmax doi={int((y_cal != softmax_np(logits0).argmax(axis=1)).sum())}/3501 (ly thuyet=0)")
    results["temperature"] = {"T": T_star, "ece_before": m00["ece"],
                              "ece_after": float(ev.ece_score(probs_cal, y0, bins=15))}

    # ---------- I08: FP16 + fuse BN ----------
    model_fp16 = load_base_model(device, dtype_fp16=True)
    loader_fp = make_loader(val_df, images_dir, transform=build_transforms(train=False, img_size=224),
                            batch_size=BATCH, train=False, sampler=None, num_workers=0)
    # DataLoader tra tensor float32 -> doi sang half sau khi len device (qua view fn)
    def to_half(x: torch.Tensor) -> torch.Tensor:
        return x.half()
    _, yf, logits_fp16 = run_loader_logits(model_fp16, loader_fp, device, view=to_half)
    mf = metrics_from_logits(yf, logits_fp16)
    record("I08", "FP16 inference", f"{BASE_EXP} best ckpt (.half())", 1, mf, cost="1x",
           note=f"F1 doi {mf['macro_f1']-m00['macro_f1']:+.4f} vs FP32; xem sheet Latency ve toc do")
    # fuse BN check
    model_f32 = load_base_model(device)
    n_bn = sum(1 for m in model_f32.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)))
    try:
        fused = fuse_conv_bn(model_f32)
        is_same = fused is model_f32
    except Exception as e:
        is_same, fused = True, model_f32
        print(f"fuse error (du kien voi ConvNeXt): {e}")
    with torch.inference_mode():
        _, _, lg_orig = run_loader_logits(model_f32, val_loader, device)
        _, _, lg_fused = run_loader_logits(fused, val_loader, device)
    maxdiff = float(np.max(np.abs(lg_orig - lg_fused)))
    results["fuse_bn"] = {"num_bn_layers": n_bn, "is_noop": bool(is_same), "max_abs_diff": maxdiff}
    print(f"[I08-fuse] ConvNeXt BatchNorm layers: {n_bn}; fuse la no-op: {is_same}; max|diff|={maxdiff:.2e}", flush=True)

    # ---------- Latency ----------
    lat_entries: dict = {}

    def fresh_probe() -> nn.Module:
        m = build_model("convnext_tiny", pretrained=False, num_classes=9)
        m.eval()
        return m

    lat_entries["L_fp32_b1"] = latency_report(fresh_probe(), batch_size=1, img_size=224, dtype="fp32",
                                              device="cuda" if device.type == "cuda" else "cpu",
                                              warmup=20, iters=100)
    lat_entries["L_amp_b1"] = latency_report(fresh_probe(), batch_size=1, img_size=224, dtype="amp",
                                             device="cuda" if device.type == "cuda" else "cpu",
                                             warmup=20, iters=100)
    lat_entries["L_fp16_b1"] = latency_report(fresh_probe(), batch_size=1, img_size=224, dtype="fp16",
                                              device="cuda" if device.type == "cuda" else "cpu",
                                              warmup=20, iters=100)
    lat_entries["L_fp32_b32"] = latency_report(fresh_probe(), batch_size=32, img_size=224, dtype="fp32",
                                               device="cuda" if device.type == "cuda" else "cpu",
                                               warmup=20, iters=100)
    lat_entries["L_fp16_b32"] = latency_report(fresh_probe(), batch_size=32, img_size=224, dtype="fp16",
                                               device="cuda" if device.type == "cuda" else "cpu",
                                               warmup=20, iters=100)
    lat_entries["L_fp32_b1_288"] = latency_report(fresh_probe(), batch_size=1, img_size=288, dtype="fp32",
                                                  device="cuda" if device.type == "cuda" else "cpu",
                                                  warmup=20, iters=100)
    lat_entries["L_fp32_b1_320"] = latency_report(fresh_probe(), batch_size=1, img_size=320, dtype="fp32",
                                                  device="cuda" if device.type == "cuda" else "cpu",
                                                  warmup=20, iters=100)
    results["latency"] = lat_entries
    for k, v in lat_entries.items():
        print(f"[{k}] p50={v['p50']}ms p95={v['p95']}ms p99={v['p99']}ms img/s={v['images_per_s']} "
              f"(gpu={v['gpu']}, dtype={v['dtype']}, batch={v['batch']}, size={v['img_size']}, torch={v['torch']})",
              flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDa luu {OUT_JSON}")

    # Gan latency batch-1 fp32/fp16 vao cac entry de ve scatter
    p_b1 = lat_entries["L_fp32_b1"]
    p_f16 = lat_entries["L_fp16_b1"]
    lat_map = {
        "I00": (p_b1["p50"], p_b1["p95"], p_b1["p99"]),
        "I01": (round(p_b1["p50"] * 2, 2), round(p_b1["p95"] * 2, 2), round(p_b1["p99"] * 2, 2)),
        "I02": (round(p_b1["p50"] * 5, 2), round(p_b1["p95"] * 5, 2), round(p_b1["p99"] * 5, 2)),
        "I04_224": (p_b1["p50"], p_b1["p95"], p_b1["p99"]),
        "I04_256": (p_b1["p50"], p_b1["p95"], p_b1["p99"]),
        "I04_288": (lat_entries["L_fp32_b1_288"]["p50"], lat_entries["L_fp32_b1_288"]["p95"],
                    lat_entries["L_fp32_b1_288"]["p99"]),
        "I04_320": (lat_entries["L_fp32_b1_320"]["p50"], lat_entries["L_fp32_b1_320"]["p95"],
                    lat_entries["L_fp32_b1_320"]["p99"]),
        "I05a": (round(p_b1["p50"] * 3, 2), round(p_b1["p95"] * 3, 2), round(p_b1["p99"] * 3, 2)),
        "I05b": (round(p_b1["p50"] * 2, 2), round(p_b1["p95"] * 2, 2), round(p_b1["p99"] * 2, 2)),
        "I06": (p_b1["p50"], p_b1["p95"], p_b1["p99"]),
        "I07": (p_b1["p50"], p_b1["p95"], p_b1["p99"]),
        "I08": (p_f16["p50"], p_f16["p95"], p_f16["p99"]),
    }
    for iid, (a, b, c) in lat_map.items():
        if iid in results["entries"]:
            results["entries"][iid]["latency"] = {"p50": a, "p95": b, "p99": c}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Scatter F1 vs p50
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    pts = [("I00", "1-view", "C0"), ("I01", "TTA-hflip x2", "C1"), ("I02", "TTA-5crop x5", "C1"),
           ("I04_256", "res-256", "C2"), ("I04_288", "res-288", "C2"), ("I04_320", "res-320", "C2"),
           ("I05a", "ens-3cvnx", "C3"), ("I05b", "ens-cvnx+swin", "C3"),
           ("I06", "soup", "C4"), ("I07", f"T={results['temperature']['T']:.2f}", "C4"),
           ("I08", "FP16", "C5")]
    for iid, label, col in pts:
        e = results["entries"].get(iid)
        if not e or not e.get("latency"):
            continue
        ax.scatter(e["latency"]["p50"], e["macro_f1"] * 100, s=90, label=f"{iid} {label}")
        ax.annotate(iid, (e["latency"]["p50"], e["macro_f1"] * 100),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axvline(100, color="red", linestyle="--", linewidth=1.2, label="p95<=100ms (I5)")
    ax.set_xlabel("Do tre batch-1 p50 (ms, FP32 tru khi ghi chu)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Val Macro-F1 (%)", fontsize=11, fontweight="bold")
    ax.set_title("Buoc 3: danh doi chinh xac - do tre (base T04 convnext_tiny CutMix)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    scatter_path = ROOT_DIR / "curves" / "inference_accuracy_vs_latency.png"
    plt.savefig(scatter_path)
    plt.close(fig)
    print(f"Da ve {scatter_path}")


if __name__ == "__main__":
    main()
