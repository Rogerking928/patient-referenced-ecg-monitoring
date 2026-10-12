"""Supplementary negative controls.  Both first-pass controls landed above chance,
for two different and instructive reasons.

C2 (permute the sample order inside a window) reached 0.69-0.76.  Permutation destroys
every temporal structure but preserves the window's amplitude histogram exactly, and
QRS voltage is a genuine function of preload -- so the residual should be amplitude,
not leakage.  C2b and C2c test that.

C3 (permute whole label blocks between subjects) reached 0.59-0.60.  That control keeps
each subject's label trajectory intact, so the *time position* of a label is preserved:
early windows are still mostly "baseline" and late windows still mostly "shock", in both
the real and the permuted assignment.  The ECG is time-dependent for the same reason, so
a residual association survives the permutation.  C3b removes it.

  C2b  permute AND standardise each window   -> amplitude scale removed too
  C2c  replace the window with Gaussian noise of the same mean and variance
  C3b  permute the label vector completely, destroying the time alignment as well
"""
import time
import numpy as np
import pandas as pd

from hemo import config, experiments as E, features as FE, train as T, views as V
from hemo.config import out

cfg = config.load()
log = E.log_to(out(cfg, "run_controls.log"))
ctx = E.Ctx(cfg)
t0 = time.time()
rows = []


class StdShuffledView(V._BaseView):
    def __init__(self, X, seed=0):
        super().__init__(X)
        self.seed = seed

    def _one(self, k):
        x = np.asarray(self.X[k], np.float32)
        rng = np.random.default_rng(self.seed * 7919 + k)
        x = x[:, rng.permutation(x.shape[1])]
        return ((x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)).astype(np.float32)


class MomentNoiseView(V._BaseView):
    def __init__(self, X, seed=0):
        super().__init__(X)
        self.seed = seed

    def _one(self, k):
        x = np.asarray(self.X[k], np.float32)
        rng = np.random.default_rng(self.seed * 104729 + k)
        return (rng.standard_normal(x.shape).astype(np.float32)
                * x.std(1, keepdims=True) + x.mean(1, keepdims=True)).astype(np.float32)


for exp, label, view in [
    ("C2b_shuffled_standardised", "handcrafted_xgb_shuffled_standardised", StdShuffledView(ctx.X, 5)),
    ("C2c_moment_matched_noise", "handcrafted_xgb_moment_noise", MomentNoiseView(ctx.X, 6)),
]:
    Fv = FE.extract_batch(view, cfg)
    Av, _, _ = T.prep_features(Fv, ctx.tr)
    _, p, mte = E.run_boosted(ctx, A_override=Av, label=label)
    sc = E.eval_pred(p, mte, cfg, label, {"experiment": exp})
    rows.append(sc)
    log(f"[{time.time()-t0:6.0f}s] {exp:28s} {label:38s} stageAUC {sc['stage_auroc_macro']:.3f} "
        f"volR2 {sc['volume_r2']:+.3f} decAUC {sc['decomp_auroc']:.3f}")

# ---------------------------------------------------------------- C3b full label permutation
import torch
from hemo import models as M
torch.set_num_threads(14)
rng = np.random.default_rng(cfg.run.seed + 4242)
meta_rand = ctx.meta.copy()
perm = rng.permutation(len(meta_rand))
for c in ["stage", "volume_lost", "reserve", "decomp_soon", "at_risk", "tte", "tte_event"]:
    meta_rand[c] = ctx.meta[c].to_numpy()[perm]
_, p, mte = E.run_boosted(ctx, meta_override=meta_rand, label="fullrand")
sc = E.eval_pred(p, mte, cfg, "handcrafted_xgb_fully_random_labels",
                 {"experiment": "C3b_fully_random_labels"})
rows.append(sc)
log(f"[{time.time()-t0:6.0f}s] C3b_fully_random_labels      handcrafted_xgb  "
    f"stageAUC {sc['stage_auroc_macro']:.3f} volR2 {sc['volume_r2']:+.3f} "
    f"decAUC {sc['decomp_auroc']:.3f}")

net = M.WaveNet(len(cfg.simulation.leads), cfg)
net, p, mte = E.run_torch(ctx, net, meta_override=meta_rand, log=log, epochs=8)
sc = E.eval_pred(p, mte, cfg, "raw_waveform_cnn_fully_random_labels",
                 {"experiment": "C3b_fully_random_labels"})
rows.append(sc)
log(f"[{time.time()-t0:6.0f}s] C3b_fully_random_labels      raw_waveform_cnn "
    f"stageAUC {sc['stage_auroc_macro']:.3f} volR2 {sc['volume_r2']:+.3f} "
    f"decAUC {sc['decomp_auroc']:.3f}")

# ---------------------------------------------------------------- ablation completeness
# The rate ablations only mean something if heart rate really is gone.  Beat-wise
# resampling removes the interval but not the rate-adaptation of repolarisation, and
# the "no rate features" set still contains QT and QTc.  Measure the leak instead of
# assuming it away: try to recover heart rate from each ablated representation.
import xgboost as xgb
from sklearn.metrics import r2_score

leak = []
hr = ctx.meta.hr_true.to_numpy()


def hr_recovery(Amat, cols, tag):
    ci = np.arange(Amat.shape[1]) if cols is None else np.asarray(cols)
    r = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.85,
                         colsample_bytree=0.75, n_jobs=cfg.run.n_jobs, tree_method="hist",
                         random_state=0, early_stopping_rounds=40)
    r.fit(Amat[np.ix_(ctx.tr, ci)], hr[ctx.tr],
          eval_set=[(Amat[np.ix_(ctx.va, ci)], hr[ctx.va])], verbose=False)
    p = r.predict(Amat[np.ix_(ctx.te, ci)])
    v = float(r2_score(hr[ctx.te], p))
    leak.append(dict(representation=tag, hr_recovery_r2=v,
                     hr_mae_bpm=float(np.mean(np.abs(hr[ctx.te] - p)))))
    log(f"    heart rate recoverable from {tag:34s} R2 {v:+.3f}")
    return v


hr_recovery(ctx.A, None, "the full feature set")
RATE = ["hr_mean", "hr_sd", "rr_mean"]
hr_recovery(ctx.A, [ctx.names.index(c) for c in ctx.names if c not in RATE],
            "the feature set minus rate features")
hr_recovery(ctx.A, [ctx.names.index(c) for c in ctx.names if c not in FE.HRV_KEYS],
            "morphology features only")
rn = out(cfg, "X_ratenorm.npy")
if rn.exists():
    Xrn = np.load(rn, mmap_mode="r")
    Frn = FE.extract_batch(Xrn, cfg)
    Arn, _, _ = T.prep_features(Frn, ctx.tr)
    hr_recovery(Arn, None, "the rate-normalised waveform")
pd.DataFrame(leak).to_csv(out(cfg, "ablation_completeness.csv"), index=False)

# ---------------------------------------------------------------- C9b fair personal baseline
# The delta features are defined against each subject's own early windows, so for those
# windows the delta is identically zero and the baseline class becomes trivially
# identifiable.  Re-score both models on the windows that did NOT contribute to the
# reference, which is the only comparison that means anything.
BASE_MAX = 7.6
later = (ctx.meta.t_min.to_numpy() > BASE_MAX)
te_later = np.array([i for i in ctx.te if later[i]])
tr_later = np.array([i for i in ctx.tr if later[i]])
va_later = np.array([i for i in ctx.va if later[i]])
Dlt = E.subject_baseline_delta(ctx)
for tag, Amat in [("population model (absolute features)", ctx.A),
                  ("personal baseline (delta features)", Dlt),
                  ("absolute + delta", np.concatenate([ctx.A, Dlt], axis=1))]:
    _, p, mte = E.run_boosted(ctx, A_override=Amat, tr=tr_later, va=va_later, te=te_later,
                              label=tag)
    sc = E.eval_pred(p, mte, cfg, tag, {"experiment": "C9b_personal_baseline_reference_excluded"})
    rows.append(sc)
    log(f"[{time.time()-t0:6.0f}s] C9b (reference windows excluded) {tag:38s} "
        f"stageAUC {sc['stage_auroc_macro']:.3f} volR2 {sc['volume_r2']:+.3f} "
        f"decAUC {sc['decomp_auroc']:.3f} resR2 {sc['reserve_r2']:+.3f} n={sc['n']}")

old = out(cfg, "results_controls.csv")
df = pd.DataFrame(rows)
if old.exists():
    df = pd.concat([pd.read_csv(old), df], ignore_index=True)
df.to_csv(old, index=False)
print("C2B_COMPLETE")
