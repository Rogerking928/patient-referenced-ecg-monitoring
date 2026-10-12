"""Does the personal-baseline advantage survive a realistic baseline?

In the main analysis a subject's reference windows come from the same recording, the
same electrodes and the same session, which is the best case and not a realistic one.
Here the reference is degraded the way a real prior recording would be: a different
electrode placement and gain, a different device sampling rate, a change of posture
and breathing, a drug change between the two sessions, and a few ectopic beats.

Reference strategies compared:
    population model (no personal reference)
    one prior recording
    median of three, median of five prior recordings
    exponentially weighted running reference
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import signal as sps
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score

from hemo import config, experiments as E, features as FE, train as T
from hemo.config import out

cfg = config.load()
log = E.log_to(out(cfg, "run_baseline_robustness.log"))
O = lambda p: out(cfg, p)
t00 = time.time()

X = np.load(O("X_main.npy"), mmap_mode="r")
meta = pd.read_parquet(O("meta_main.parquet")).reset_index(drop=True)
F = pd.read_parquet(O("features_main.parquet"))
names = list(F.columns)
BASE_MAX = 7.6
is_base = (meta.t_min <= BASE_MAX).to_numpy()
log(f"{is_base.sum()} reference windows, {(~is_base).sum()} later windows")


# ------------------------------------------------------------------ degradations
def degrade(x, rng, level=1.0):
    """Everything that changes between two real recording sessions."""
    x = np.asarray(x, np.float32).copy()
    fs = cfg.simulation.fs
    # electrode gain and placement: per-lead scale and a small cross-lead mixing
    g = 1.0 + level * rng.normal(0, 0.18, (x.shape[0], 1))
    mix = level * rng.normal(0, 0.12)
    x = x * g
    if x.shape[0] > 1:
        x = np.stack([x[0] + mix * x[1], x[1] + mix * x[0]])
    # a different device: resample through another rate and back
    alt = int(rng.choice([100, 128, 200, 500]))
    n = x.shape[1]
    x = sps.resample(sps.resample(x, int(n * alt / fs), axis=1), n, axis=1).astype(np.float32)
    # posture and breathing: extra baseline wander at a different frequency
    tt = np.arange(n) / fs
    f_new = rng.uniform(0.10, 0.45)
    x = x + level * rng.uniform(0.02, 0.12) * np.sin(2 * np.pi * f_new * tt + rng.uniform(0, 6.3))
    # a few ectopic beats: blank and displace short segments
    if rng.random() < 0.35 * level:
        for _ in range(rng.integers(1, 4)):
            c = int(rng.integers(int(0.5 * fs), n - int(0.5 * fs)))
            w = int(rng.uniform(0.12, 0.30) * fs)
            x[:, c:c + w] = x[:, max(c - 3 * w, 0):max(c - 3 * w, 0) + w][:, :x[:, c:c + w].shape[1]] * 1.4
    # additional recording noise
    x = x + level * rng.uniform(0.004, 0.03) * rng.standard_normal(x.shape).astype(np.float32)
    return x


# features of the degraded reference windows
fp = O("features_degraded_baseline.parquet")
if fp.exists():
    Fd = pd.read_parquet(fp)
else:
    idx = np.where(is_base)[0]
    from joblib import Parallel, delayed

    def work(ii):
        rows = []
        for k in ii:
            rng = np.random.default_rng(20260805 + int(k))
            rows.append(FE.extract_window(degrade(X[k], rng), cfg))
        return rows
    chunks = np.array_split(idx, 48)
    res = Parallel(n_jobs=cfg.run.n_jobs)(delayed(work)(c) for c in chunks if len(c))
    Fd = pd.DataFrame([r for c in res for r in c], columns=names, index=idx)
    Fd.to_parquet(fp)
log(f"[{time.time()-t00:.0f}s] degraded reference features {Fd.shape}")

A, mu, sd = T.prep_features(F, np.where(meta.split == "train")[0])
Ad = A.copy()
Ad[Fd.index.to_numpy()] = np.nan_to_num(Fd.to_numpy(np.float32),
                                        nan=np.nanmedian(A, axis=0))

tr = np.where((meta.split == "train") & ~is_base)[0]
va = np.where((meta.split == "val") & ~is_base)[0]
te = np.where((meta.split == "test") & ~is_base)[0]
XGBP = dict(n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.85,
            colsample_bytree=0.75, n_jobs=cfg.run.n_jobs, tree_method="hist", random_state=0)
rows = []


def reference_matrix(Asrc, how):
    """Build the per-window reference from the (possibly degraded) baseline windows."""
    R = np.zeros_like(Asrc)
    for s, g in meta.groupby("subject").groups.items():
        ix = np.asarray(g)
        b = ix[is_base[ix]]
        if len(b) == 0:
            R[ix] = Asrc[ix].mean(0)
            continue
        if how == "one":
            ref = Asrc[b[0]]
        elif how == "median3":
            ref = np.median(Asrc[b[:3]], axis=0)
        elif how == "median5":
            ref = np.median(Asrc[b[:5]], axis=0)
        elif how == "ewma":
            w = 0.6 ** np.arange(len(b))[::-1]
            ref = (Asrc[b] * w[:, None]).sum(0) / w.sum()
        R[ix] = ref
    return R


def evaluate(Xf, tag, extra=None):
    r = dict(strategy=tag)
    if extra:
        r.update(extra)
    y = meta.stage.to_numpy()
    cnt = np.bincount(y[tr], minlength=5).astype(float)
    wg = (cnt.sum() / (5 * np.maximum(cnt, 1))) ** 0.5
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=5,
                            eval_metric="mlogloss", early_stopping_rounds=40, **XGBP)
    clf.fit(Xf[tr], y[tr], sample_weight=wg[y[tr]], eval_set=[(Xf[va], y[va])], verbose=False)
    P = clf.predict_proba(Xf[te])
    r["stage_auroc_macro"] = float(np.mean([roc_auc_score((y[te] == k).astype(int), P[:, k])
                                            for k in range(5) if (y[te] == k).sum() > 3]))
    r["stage_bacc"] = balanced_accuracy_score(y[te], P.argmax(1))
    for task, col in [("volume", "volume_lost"), ("reserve", "reserve")]:
        reg = xgb.XGBRegressor(objective="reg:squarederror", early_stopping_rounds=40, **XGBP)
        reg.fit(Xf[tr], meta[col].to_numpy()[tr],
                eval_set=[(Xf[va], meta[col].to_numpy()[va])], verbose=False)
        r[f"{task}_r2"] = float(r2_score(meta[col].to_numpy()[te], reg.predict(Xf[te])))
    at = meta.at_risk.to_numpy().astype(bool)
    yd = meta.decomp_soon.to_numpy()
    itr, iva, ite = tr[at[tr]], va[at[va]], te[at[te]]
    c2 = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                           early_stopping_rounds=40,
                           scale_pos_weight=float((1 - yd[itr]).sum() / max(yd[itr].sum(), 1)),
                           **XGBP)
    c2.fit(Xf[itr], yd[itr], eval_set=[(Xf[iva], yd[iva])], verbose=False)
    r["decomp_auroc"] = float(roc_auc_score(yd[ite], c2.predict_proba(Xf[ite])[:, 1]))
    rows.append(r)
    pd.DataFrame(rows).to_csv(O("baseline_robustness.csv"), index=False)
    log(f"[{time.time()-t00:6.0f}s] {tag:52s} stageAUC {r['stage_auroc_macro']:.3f}  "
        f"volR2 {r['volume_r2']:+.3f}  decAUC {r['decomp_auroc']:.3f}  "
        f"resR2 {r['reserve_r2']:+.3f}")


log("\n=== reference strategies, clean and degraded ===")
evaluate(A, "population model (no personal reference)", dict(reference="none", degraded=0))
for how, label in [("one", "one prior recording"), ("median3", "median of three"),
                   ("median5", "median of five"), ("ewma", "exponentially weighted")]:
    for src, dtag in [(A, 0), (Ad, 1)]:
        R = reference_matrix(src, how)
        cond = "same session" if dtag == 0 else "different session, device and electrodes"
        evaluate(A - R, f"{label} ({cond})", dict(reference=how, degraded=dtag))
        evaluate(np.hstack([A, A - R]),
                 f"absolute + {label} ({cond})", dict(reference=how + "+abs", degraded=dtag))

log("\nBASELINE_ROBUSTNESS_COMPLETE")
