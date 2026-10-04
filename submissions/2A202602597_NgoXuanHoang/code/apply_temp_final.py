# -*- coding: utf-8 -*-
"""Fit T tren val moi seed F01 (val-only), ap dung sang test, luu file cal + giu ban uncal."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(CODE_DIR))

import eval as ev
from inference import apply_temperature, fit_temperature

for seed in [0, 1, 2]:
    val_logits = np.load(ROOT_DIR / f"runs/F01/seed{seed}/val_logits.npy")
    df_val = pd.read_csv(ROOT_DIR / f"predictions/F01_seed{seed}_val.csv")
    y_val = df_val["y_true"].to_numpy(dtype=np.int64)
    assert len(y_val) == len(val_logits), (len(y_val), len(val_logits))

    T = fit_temperature(val_logits, y_val)
    probs_val = np.exp(val_logits - val_logits.max(axis=1, keepdims=True))
    probs_val /= probs_val.sum(axis=1, keepdims=True)
    ece_before = float(ev.ece_score(probs_val, y_val, bins=15))
    probs_val_cal = apply_temperature(val_logits, T)
    ece_after = float(ev.ece_score(probs_val_cal, y_val, bins=15))

    test_logits = np.load(ROOT_DIR / f"runs/F01/seed{seed}/test_logits.npy")
    df_test = pd.read_csv(ROOT_DIR / f"predictions/F01_seed{seed}_test.csv")
    assert len(df_test) == len(test_logits)
    y_test = df_test["y_true"].to_numpy(dtype=np.int64)
    fnames = df_test["Filename"].tolist()

    # Giu ban uncal cho grade I4(a)
    shutil.copyfile(ROOT_DIR / f"predictions/F01_seed{seed}_test.csv",
                    ROOT_DIR / f"predictions/F01_seed{seed}_uncal_test.csv")
    probs_cal = apply_temperature(test_logits, T)
    ev.save_predictions(ROOT_DIR / f"predictions/F01_seed{seed}_test.csv",
                        fnames, y_test, probs_cal)
    nflip = int((probs_cal.argmax(axis=1) != df_test["y_pred"].to_numpy()).sum())
    print(f"[F01 seed{seed}] T={T:.4f} | val ECE {ece_before:.4f}->{ece_after:.4f} | "
          f"test flips vs uncal: {nflip}/{len(df_test)}", flush=True)
print("OK: F01_seed{k}_test.csv = calibrated; F01_seed{k}_uncal_test.csv = giu cho grade.")
