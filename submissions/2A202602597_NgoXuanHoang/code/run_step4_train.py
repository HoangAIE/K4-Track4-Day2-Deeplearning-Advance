# -*- coding: utf-8 -*-
"""run_step4_train.py - Chung ket Buoc 4: huan luyen F01 (recipe T04) va T00 baseline.

F01 = convnext_tiny, finetune, basic aug, CE, mix=cutmix(a=1.0), 12 ep (chot tu val).
T00 = cong thuc nen (finetune, basic aug, CE), 12 ep.
Moi run: save_test_predictions=True (chi mo test 1 lan/seed de bao cao).
Usage: python code/run_step4_train.py F01 0 1 2
       python code/run_step4_train.py T00 1 2
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(CODE_DIR))

from train import Config, run


def main():
    exp_id = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]]
    for seed in seeds:
        if exp_id == "F01":
            cfg = Config(
                exp_id="F01", backbone="convnext_tiny", init="finetune",
                seed=seed, aug="basic", loss="ce", mix="cutmix", mix_alpha=1.0,
                epochs=12, batch_size=64, num_workers=0,
                save_test_predictions=True,
            )
        elif exp_id == "T00":
            cfg = Config(
                exp_id="T00", backbone="convnext_tiny", init="finetune",
                seed=seed, aug="basic", loss="ce",
                epochs=12, batch_size=64, num_workers=0,
                save_test_predictions=True,
            )
        else:
            raise ValueError(f"exp_id khong ho tro: {exp_id}")
        print(f"\n{'='*70}\n[STEP4-START] {exp_id} seed={seed}\n{'='*70}", flush=True)
        summary = run(cfg)
        print(f"\n[STEP4-DONE] {exp_id} seed={seed}: "
              f"Val F1={summary['best_val_macro_f1']*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
