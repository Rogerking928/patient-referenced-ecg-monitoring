"""Stage 2: the three-model comparison plus the reference baselines.

Every model is trained on the SAME patient-level split and evaluated on the
same held-out subjects, with subject-cluster bootstrap confidence intervals.
"""
import time
import numpy as np
import pandas as pd
import torch

from hemo import config, experiments as E, models as M, train as T
from hemo.config import out

torch.set_num_threads(14)
cfg = config.load()
log = E.log_to(out(cfg, "run_main.log"))
t00 = time.time()

ctx = E.Ctx(cfg)
log(f"windows train/val/test = {len(ctx.tr)}/{len(ctx.va)}/{len(ctx.te)}  "
    f"subjects = {ctx.meta.groupby('split').subject.nunique().to_dict()}")
log(f"features = {len(ctx.names)}")

rows, preds = [], {}


def record(label, pred, meta, extra=None):
    sc = E.eval_pred(pred, meta, cfg, label, extra)
    rows.append(sc)
    preds[label] = pred
    log(f"[{time.time()-t00:6.0f}s] {label:26s} stageAUC {sc['stage_auroc_macro']:.3f} "
        f"[{sc.get('stage_auroc_macro_lo', np.nan):.3f},{sc.get('stage_auroc_macro_hi', np.nan):.3f}]  "
        f"bACC {sc['stage_bacc']:.3f}  volR2 {sc['volume_r2']:.3f}  "
        f"decAUC {sc['decomp_auroc']:.3f}  resR2 {sc['reserve_r2']:.3f}")
    pd.DataFrame(rows).to_csv(out(cfg, "results_main.csv"), index=False)


# ------------------------------------------------------------------ 1. handcrafted
log("\n=== 1. Handcrafted physiological features + gradient boosting ===")
b_all, p, mte = E.run_boosted(ctx, label="handcrafted")
record("handcrafted_xgb", p, mte, {"n_features": len(ctx.names)})
import joblib
joblib.dump(b_all, out(cfg, "model_handcrafted.joblib"))

# ------------------------------------------------------------------ 2. raw waveform
log("\n=== 2. Raw waveform ResNet1D ===")
net = M.WaveNet(len(cfg.simulation.leads), cfg)
log(f"    parameters: {M.count_params(net):,}")
net, p, mte = E.run_torch(ctx, net, log=log, seed=cfg.run.seed % 2**31)
record("raw_waveform_cnn", p, mte, {"n_params": M.count_params(net)})
torch.save(net.state_dict(), out(cfg, "model_cnn.pt"))

# ------------------------------------------------------------------ 3. hybrid
log("\n=== 3. Hybrid (waveform embedding + handcrafted features) ===")
hyb = M.HybridNet(len(cfg.simulation.leads), ctx.A.shape[1], cfg)
hyb, p, mte = E.run_torch(ctx, hyb, log=log, seed=cfg.run.seed % 2**31)
record("hybrid", p, mte, {"n_params": M.count_params(hyb)})
torch.save(hyb.state_dict(), out(cfg, "model_hybrid.pt"))

# ------------------------------------------------------------------ 4. like-for-like neural on features
log("\n=== 4. Feature MLP (same head, no waveform) ===")
fm = M.FeatMLP(ctx.A.shape[1], cfg)
fm, p, mte = E.run_torch(ctx, fm, log=log, epochs=40, seed=cfg.run.seed % 2**31)
record("handcrafted_mlp", p, mte)

# ------------------------------------------------------------------ 5. reference baselines
log("\n=== 5. Reference baselines ===")
for label, keep, drop in [
    ("baseline_heart_rate_only", E.RATE_FEATURES, None),
    ("baseline_rr_series_only", E.HRV_ONLY, None),
    ("morphology_only_no_rr", None, E.HRV_ONLY),
    ("no_rate_features", None, E.RATE_FEATURES),
]:
    cols, names = ctx.feat_cols(keep=keep, drop=drop)
    _, p, mte = E.run_boosted(ctx, cols=cols, label=label)
    record(label, p, mte, {"n_features": len(names)})

np.savez_compressed(out(cfg, "predictions_main.npz"),
                    **{f"{k}__{t}": v[t] for k, v in preds.items() for t in T.TASKS},
                    test_index=ctx.te)
ctx.meta.iloc[ctx.te].to_parquet(out(cfg, "meta_test.parquet"))

# confusion matrices for the three headline models
cms = {k: T.confusion(preds[k], mte) for k in
       ["handcrafted_xgb", "raw_waveform_cnn", "hybrid"] if k in preds}
np.savez(out(cfg, "confusions.npz"), **cms)

log(f"\nMAIN_COMPLETE  {time.time()-t00:.0f}s")
