# -*- coding: utf-8 -*-
"""Phan tich loi test: ma tran nham lan + cap lop nham nhieu nhat + anh loi mau (F01 seed0)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
import eval as ev

CLASS_NAMES = list(ev.CLASS_NAMES)
print("Classes:", CLASS_NAMES)

# Ma tran nham lan tong hop (sum over seeds) tu eval_out
cm_sum = pd.read_csv(ROOT_DIR / "eval_out/F01_confusion_sum.csv", index_col=0)
cm = cm_sum.to_numpy(dtype=int)
print("CM shape:", cm.shape, "total:", cm.sum())

# Ve heatmap
fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(9), CLASS_NAMES, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(9), CLASS_NAMES, fontsize=8)
ax.set_xlabel("Nhan du doan", fontweight="bold")
ax.set_ylabel("Nhan that", fontweight="bold")
ax.set_title("Ma tran nham lan TEST - F01 (convnext_tiny CutMix, sum 3 seeds)", fontweight="bold")
for i in range(9):
    for j in range(9):
        v = cm[i, j]
        if v > 0:
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                    color="white" if v > cm.max() / 2 else "black")
plt.colorbar(im, ax=ax, label="So anh")
plt.tight_layout()
plt.savefig(ROOT_DIR / "curves/confusion_matrix_F01_test.png")
plt.close(fig)

# Cap nham nhieu nhat (off-diagonal)
pairs = [(cm[i, j], i, j) for i in range(9) for j in range(9) if i != j]
pairs.sort(reverse=True)
print("Top nham lan:")
for v, i, j in pairs[:8]:
    print(f"  {CLASS_NAMES[i]} -> {CLASS_NAMES[j]}: {v} anh "
          f"({v/cm[i].sum()*100:.1f}% cua lop {CLASS_NAMES[i]})")

# Anh loi mau: moi cap top-3 lay 2 anh (F01 seed0)
df = pd.read_csv(ROOT_DIR / "predictions/F01_seed0_test.csv")
err = df[df["y_true"] != df["y_pred"]].copy()
print(f"Seed0: {len(err)}/{len(df)} anh sai ({len(err)/len(df)*100:.2f}%)")
top3 = [(i, j) for _, i, j in pairs[:3]]
fig2, axes = plt.subplots(3, 4, figsize=(12, 9), dpi=120)
for r, (ti, pi) in enumerate(top3):
    sub = err[(err["y_true"] == ti) & (err["y_pred"] == pi)].head(4)
    for c in range(4):
        ax = axes[r][c]
        ax.axis("off")
        if c < len(sub):
            img = Image.open(ROOT_DIR / "data/images" / sub.iloc[c]["Filename"]).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"{sub.iloc[c]['Filename']}\nthat:{CLASS_NAMES[ti]} -> doan:{CLASS_NAMES[pi]}",
                         fontsize=7)
        if c == 0:
            ax.set_ylabel(f"{CLASS_NAMES[ti]}→{CLASS_NAMES[pi]}", fontsize=9, fontweight="bold")
plt.suptitle("Mau anh bi doan sai - F01 seed0 test", fontweight="bold")
plt.tight_layout()
plt.savefig(ROOT_DIR / "curves/error_samples_F01.png")
plt.close(fig2)
print("Saved curves/confusion_matrix_F01_test.png, curves/error_samples_F01.png")
