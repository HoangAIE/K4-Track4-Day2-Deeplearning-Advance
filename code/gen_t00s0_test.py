# -*- coding: utf-8 -*-
"""Tao predictions/T00_seed0_test.csv tu checkpoint san co (eval thuan tuy, khong train lai).

Dung dung pipeline danh gia test cua train.py: build_transforms(eval,224) + evaluate + save_predictions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(CODE_DIR))

import eval as ev
from dataset import build_transforms, load_split, make_loader
from model import build_model
from train import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model("convnext_tiny", pretrained=False, num_classes=9)
ckpt = torch.load(ROOT_DIR / "runs/T00/seed0/best_checkpoint.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()
print(f"Loaded T00 seed0 ckpt (epoch={ckpt['epoch']}, val F1={ckpt['macro_f1']*100:.2f}%)")

_, _, test_df = load_split(ROOT_DIR / "data/labels", fold=0)
test_loader = make_loader(test_df, ROOT_DIR / "data/images",
                          transform=build_transforms(train=False, img_size=224),
                          batch_size=64, train=False, sampler=None, num_workers=0)
fnames, y_true, logits, _ = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
probs = np.exp(logits - logits.max(axis=1, keepdims=True))
probs /= probs.sum(axis=1, keepdims=True)

out = ROOT_DIR / "predictions" / "T00_seed0_test.csv"
ev.save_predictions(out, fnames, y_true, probs)
np.save(ROOT_DIR / "runs/T00/seed0/test_logits.npy", logits)
print(f"Saved {out} ({len(fnames)} samples)")
m = ev.compute_metrics(y_true, probs.argmax(axis=1), probs)
print(f"Sanity (train.py-style): macro-F1={m['macro_f1']*100:.2f}% top1={m['top1']*100:.2f}%")
