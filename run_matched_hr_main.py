"""Matched-heart-rate analysis for the haemorrhage arm, plus a verified rate-blind
feature set.

The feature ablations turned out not to remove heart rate at all: it is recoverable
from every "rate-removed" representation at R^2 > 0.96 (see ablation_completeness.csv).
Two definitional routes are responsible -- the beat count over a fixed-length window
is heart rate, and QT together with QTc determines RR exactly -- and one physiological
route is unavoidable, because rate and morphology are both downstream of the same
latent state.

Ablating features therefore cannot answer "is this more than heart rate".  Matching on
heart rate can.  This script does both:

  1. re-scores the already-trained models inside narrow heart-rate bands
  2. builds a feature set with every definitionally rate-determined feature removed,
     verifies how much heart rate is left in it, and re-runs the tasks on it
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from hemo import config, experiments as E, features as FE, train as T
from hemo.config import out

cfg = config.load()
log = E.log_to(out(cfg, "run_matched_hr.log"))
ctx = E.Ctx(cfg)

P = np.load(out(cfg, "predictions_main.npz"))
mte = pd.read_parquet(out(cfg, "meta_test.parquet")).reset_index(drop=True)
models = sorted({k.split("__")[0] for k in P.files if "__" in k})

BANDS = [(50, 70), (70, 85), (85, 100), (100, 125), (125, 170)]

# =================================================================== 1 matched bands
log("=== matched heart-rate bands (models unchanged, test set restricted) ===")
rows = []
for lo, hi in BANDS:
    m = ((mte.hr_true >= lo) & (mte.hr_true < hi)).to_numpy()
    if m.sum() < 150 or mte[m].stage.nunique() < 3:
        log(f"    HR {lo}-{hi}: skipped (n={m.sum()}, {mte[m].stage.nunique()} states)")
        continue
    for mod in models:
        pred = {t: P[f"{mod}__{t}"][m] for t in T.TASKS}
        try:
            s = T.score_all(pred, mte[m])
        except Exception:
            continue
        s.update(model=mod, hr_lo=lo, hr_hi=hi, n_band=int(m.sum()),
                 n_states=int(mte[m].stage.nunique()),
                 hr_sd_in_band=float(mte[m].hr_true.std()),
                 volume_sd_in_band=float(mte[m].volume_lost.std()))
        rows.append(s)
    b = [r for r in rows if r["hr_lo"] == lo]
    log(f"    HR {lo:3d}-{hi:3d} bpm  n={m.sum():4d}  {mte[m].stage.nunique()} states  "
        + "  ".join(f"{r['model'][:12]}: stageAUC {r['stage_auroc_macro']:.3f}/volR2 "
                    f"{r['volume_r2']:+.3f}" for r in b
                    if r["model"] in ("handcrafted_xgb", "raw_waveform_cnn", "hybrid",
                                      "baseline_heart_rate_only")))
mh = pd.DataFrame(rows)
mh.to_csv(out(cfg, "matched_hr_main.csv"), index=False)

# =================================================================== 2 rate-blind set
log("\n=== a feature set with every definitionally rate-determined feature removed ===")
# HRV_KEYS covers everything computed from the RR series (including hr_mean and
# rr_kept_frac); n_beats is the beat count over a fixed window, i.e. heart rate;
# QT and QTc jointly determine RR; T duration and the QT variability measures are
# defined against RR as well; the RSA-derived respiratory estimate uses the tachogram.
DEFINITIONAL = set(FE.HRV_KEYS) | {"n_beats", "resp_rate_rsa", "rsa_strength"}
for lead in cfg.simulation.leads:
    DEFINITIONAL |= {f"{lead}_{k}" for k in ["qt", "qtc", "qt_sd", "qtv_index", "t_dur"]}
blind = [c for c in ctx.names if c not in DEFINITIONAL]
log(f"    kept {len(blind)}/{len(ctx.names)} features")

import xgboost as xgb
hr = ctx.meta.hr_true.to_numpy()


def hr_recovery(cols, tag):
    r = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.85,
                         colsample_bytree=0.75, n_jobs=cfg.run.n_jobs, tree_method="hist",
                         random_state=0, early_stopping_rounds=40)
    r.fit(ctx.A[np.ix_(ctx.tr, cols)], hr[ctx.tr],
          eval_set=[(ctx.A[np.ix_(ctx.va, cols)], hr[ctx.va])], verbose=False)
    p = r.predict(ctx.A[np.ix_(ctx.te, cols)])
    v = float(r2_score(hr[ctx.te], p))
    log(f"    heart rate recoverable from {tag}: R2 {v:+.3f}, MAE {np.mean(np.abs(hr[ctx.te]-p)):.1f} bpm")
    return v, float(np.mean(np.abs(hr[ctx.te] - p)))


ci = [ctx.names.index(c) for c in blind]
r2_blind, mae_blind = hr_recovery(ci, "the rate-blind set")

_, pred, m2 = E.run_boosted(ctx, cols=ci, label="rate_blind")
sc = E.eval_pred(pred, m2, cfg, "handcrafted_rate_blind",
                 {"experiment": "C5b_rate_blind_verified", "n_features": len(blind),
                  "hr_recovery_r2": r2_blind, "hr_recovery_mae": mae_blind})
log(f"    rate-blind model: stageAUC {sc['stage_auroc_macro']:.3f} volR2 {sc['volume_r2']:+.3f} "
    f"decAUC {sc['decomp_auroc']:.3f} resR2 {sc['reserve_r2']:+.3f}")

old = out(cfg, "results_controls.csv")
df = pd.DataFrame([sc])
if old.exists():
    df = pd.concat([pd.read_csv(old), df], ignore_index=True)
df.to_csv(old, index=False)
pd.DataFrame([dict(representation="rate-attenuated feature set",
                   hr_recovery_r2=r2_blind, hr_mae_bpm=mae_blind)]).to_csv(
    out(cfg, "ablation_completeness_blind.csv"), index=False)
print("MATCHED_HR_COMPLETE")
