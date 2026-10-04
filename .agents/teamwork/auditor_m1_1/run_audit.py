import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score

# Add code directory to path
repo_root = Path("d:/K4-Track4-Day2-Deeplearning-Advance")
sys.path.insert(0, str(repo_root / "code"))

from model import build_model, count_params, count_gmacs
from dataset import DeepWeedsDataset, build_transforms

print("=== STARTING ADVANCED FORENSIC INTEGRITY AUDIT ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Auditing device: {device}")

val_df = pd.read_csv(repo_root / "data/labels/val_subset0.csv")
y_true = val_df["Label"].values
val_filenames = val_df["Filename"].values
assert len(y_true) == 3501, f"Expected 3501 validation samples, got {len(y_true)}"

val_tf = build_transforms(train=False, img_size=224)
val_ds = DeepWeedsDataset(val_df, repo_root / "data/images", transform=val_tf)

audit_report = {}

for exp in ["B01", "B02", "B03", "B04", "B05"]:
    print(f"\n==========================================")
    print(f"       FORENSIC AUDITING: {exp}")
    print(f"==========================================")
    
    ckpt_path = repo_root / f"runs/{exp}/seed0/best_checkpoint.pt"
    summary_path = repo_root / f"runs/{exp}/seed0/summary.json"
    history_path = repo_root / f"runs/{exp}/seed0/history.csv"
    logits_path = repo_root / f"runs/{exp}/seed0/val_logits.npy"
    config_path = repo_root / f"runs/{exp}/seed0/config.json"

    # 1. File existence
    for p in [ckpt_path, summary_path, history_path, logits_path, config_path]:
        assert p.exists(), f"Missing file: {p}"
    
    with open(summary_path) as f:
        summary = json.load(f)
    with open(config_path) as f:
        config = json.load(f)
    hist_df = pd.read_csv(history_path)
    logits = np.load(logits_path)

    # 2. Config & Leakage check
    print(f"[Check 1: Leakage Flag in Config]")
    print(f"  save_test_predictions = {config.get('save_test_predictions')}")
    assert config.get("save_test_predictions") is False, f"Test set evaluation flag was active in {exp}!"

    # 3. History CSV analysis
    print(f"[Check 2: History & Dynamics]")
    assert len(hist_df) == 12, f"Expected 12 epochs in history, got {len(hist_df)}"
    best_row = hist_df.loc[hist_df["val_macro_f1"].idxmax()]
    best_ep = int(best_row["epoch"])
    best_f1_hist = float(best_row["val_macro_f1"])
    best_top1_hist = float(best_row["val_top1"])
    print(f"  Best epoch in history: {best_ep}")
    print(f"  Best val F1 in history: {best_f1_hist:.6f}")
    print(f"  Best top-1 in history: {best_top1_hist:.6f}")
    assert best_ep == summary["best_epoch"], f"Summary best_epoch mismatch: {best_ep} vs {summary['best_epoch']}"
    assert abs(best_f1_hist - summary["best_val_macro_f1"]) < 1e-4, "Summary F1 mismatch"
    assert abs(best_top1_hist - summary["best_val_top1"]) < 1e-4, "Summary Top1 mismatch"

    train_loss_start = hist_df["train_loss"].iloc[0]
    train_loss_end = hist_df["train_loss"].iloc[-1]
    print(f"  Loss progression: Ep1={train_loss_start:.4f} -> Ep12={train_loss_end:.4f}")
    assert train_loss_end < train_loss_start, "Train loss failed to converge/decrease"
    
    avg_ep_time = hist_df["epoch_time_s"].mean()
    print(f"  Avg epoch time: {avg_ep_time:.2f}s")
    assert avg_ep_time > 30.0, f"Abnormally rapid epoch time: {avg_ep_time}s"

    # 4. Logits and ground truth metric recalculation
    print(f"[Check 3: Logits & Ground Truth Recalculation]")
    assert logits.shape == (3501, 9), f"Invalid logits shape: {logits.shape}"
    assert not np.isnan(logits).any(), "NaN in logits"
    assert not np.isinf(logits).any(), "Inf in logits"

    y_pred = np.argmax(logits, axis=1)
    recalc_f1 = float(f1_score(y_true, y_pred, average="macro"))
    recalc_top1 = float(accuracy_score(y_true, y_pred))
    print(f"  Recalculated Macro-F1 from raw logits: {recalc_f1:.6f}")
    print(f"  Recalculated Top-1 from raw logits:    {recalc_top1:.6f}")
    assert abs(recalc_f1 - summary["best_val_macro_f1"]) < 1e-4, "Recalculated F1 does not match summary"
    assert abs(recalc_top1 - summary["best_val_top1"]) < 1e-4, "Recalculated Top-1 does not match summary"

    # 5. Checkpoint weights and training delta verification
    print(f"[Check 4: Checkpoint Tensor Genuine Training Check]")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_sd = ckpt["model_state_dict"]
    
    # Check keys
    print(f"  Keys in checkpoint: {list(ckpt.keys())}")
    assert "model_state_dict" in ckpt
    assert ckpt["epoch"] == best_ep

    # Compare with fresh pre-trained architecture weights to prove weights actually moved!
    fresh_model = build_model(summary["backbone"], pretrained=True, num_classes=9)
    fresh_sd = fresh_model.state_dict()

    param_deltas = []
    weight_norms = []
    for k in ckpt_sd.keys():
        if k in fresh_sd and ckpt_sd[k].is_floating_point():
            w_trained = ckpt_sd[k].float()
            w_fresh = fresh_sd[k].float()
            delta = torch.norm(w_trained - w_fresh).item()
            param_deltas.append(delta)
            weight_norms.append(torch.norm(w_trained).item())

    total_delta = sum(param_deltas)
    mean_delta = np.mean(param_deltas)
    max_delta = np.max(param_deltas)
    print(f"  Trained vs Pretrained weight delta: total_L2={total_delta:.4f}, mean_L2={mean_delta:.4f}, max_L2={max_delta:.4f}")
    assert total_delta > 1.0, f"Checkpoint has zero/negligible difference from pre-trained model! Possible dummy weights: delta={total_delta}"

    # 6. Empirical Inference Verification: Checkpoint outputs vs val_logits.npy
    print(f"[Check 5: Empirical Model Inference vs val_logits.npy]")
    fresh_model.load_state_dict(ckpt_sd)
    fresh_model.to(device)
    fresh_model.eval()
    
    sample_indices = [0, 10, 50, 100, 500]
    sample_images = torch.stack([val_ds[i][0] for i in sample_indices]).to(device)
    
    with torch.no_grad():
        inferred_logits = fresh_model(sample_images).cpu().numpy()
    
    expected_logits = logits[sample_indices]
    logit_diff = np.max(np.abs(inferred_logits - expected_logits))
    print(f"  Max absolute difference between model inference and saved val_logits: {logit_diff:.6e}")
    # Verify predictions match exactly
    inferred_preds = np.argmax(inferred_logits, axis=1)
    expected_preds = np.argmax(expected_logits, axis=1)
    print(f"  Inferred classes: {inferred_preds.tolist()}")
    print(f"  Expected classes: {expected_preds.tolist()}")
    assert np.array_equal(inferred_preds, expected_preds), f"Predictions mismatch: {inferred_preds} vs {expected_preds}"
    assert logit_diff < 0.1, f"Logit difference too large: {logit_diff}"

    audit_report[exp] = {
        "backbone": summary["backbone"],
        "best_epoch": best_ep,
        "summary_val_f1": summary["best_val_macro_f1"],
        "recalculated_val_f1": round(recalc_f1, 6),
        "summary_val_top1": summary["best_val_top1"],
        "recalculated_val_top1": round(recalc_top1, 6),
        "avg_epoch_time_s": round(avg_ep_time, 2),
        "weight_delta_L2": round(total_delta, 4),
        "inference_logits_max_diff": float(logit_diff),
        "test_leakage_flag": config.get("save_test_predictions"),
        "verdict": "CLEAN",
    }

print("\n=======================================================")
print("  ALL 5 BACKBONES VERIFIED EMPIRICALLY WITH ZERO ERROR ")
print("=======================================================")

with open(repo_root / ".agents/teamwork/auditor_m1_1/audit_empirical_results.json", "w") as f:
    json.dump(audit_report, f, indent=2)
print("Saved audit_empirical_results.json")
