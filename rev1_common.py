"""Shared pieces for the first-revision analyses (PMEA-107380).

Every revision script builds the same four representations that the manuscript
compares, from the same 81 handcrafted features, and scores them with the same
gradient-boosting settings as run_longitudinal.py:

    population      a single 30 s window, absolute values
    reference       the window as a change from the subject's own reference
    trajectory      absolute window + slope + curvature over the preceding ten minutes
    reference+traj  change from reference + slope + curvature

The reference is passed in, so the same scorer serves the oracle pre-onset
reference, a reference taken from whatever the first observed window happens to
be, and a rolling reference that needs no knowledge of onset at all.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, roc_auc_score

from hemo import features as FE, simulate, train as T
from hemo.config import Cfg

N_JOBS = 14


def xgbp(n_jobs=N_JOBS):
    return dict(n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.85,
                colsample_bytree=0.75, n_jobs=n_jobs, tree_method="hist", random_state=0)


def build(cfg, subj_hook=None, x_hook=None, seed=None, n_jobs=N_JOBS):
    """Simulate a world, render it and extract features.  Nothing is saved."""
    from hemo import subjects as SM, physiology as phys
    subj = SM.assign_splits(cfg, SM.make_cohort(cfg, seed=seed), seed=seed)
    if subj_hook is not None:
        subj = subj_hook(subj)
    Z, D, t, bh = phys.simulate_cohort(cfg, subj, seed=seed)
    lab = phys.stage_labels(cfg, subj, Z, D, t, bh)
    viab = phys.viability_mask(cfg, D)
    X, meta = simulate.build_dataset(cfg, subj, Z, D, t, lab, viab, save=False)
    if x_hook is not None:
        X = x_hook(X, meta)
    F = FE.extract_batch(X, cfg, n_jobs=n_jobs)
    del X
    return meta.reset_index(drop=True), F, subj


def standardise(F, meta):
    tr = np.where(meta.split == "train")[0]
    A, mu, sd = T.prep_features(F, tr)
    return np.clip((A - mu) / sd, -8, 8), mu, sd


def sort_by_subject(meta, A):
    o = meta.sort_values(["subject", "win"]).index.to_numpy()
    return meta.loc[o].reset_index(drop=True), A[o]


def step_minutes(cfg):
    n = int(cfg.simulation.n_windows)
    return (cfg.simulation.window_last_min - cfg.simulation.window_first_min) / (n - 1)


def history_index(meta, n_back):
    """(n, n_back) row indices of the windows ending at each row, padded with the
    earliest available window of the same subject.  meta must be sorted."""
    out = np.zeros((len(meta), n_back), int)
    for _, g in meta.groupby("subject", sort=False).indices.items():
        idx = np.sort(np.asarray(g))
        for p in range(len(idx)):
            out[idx[p]] = [idx[max(p - j, 0)] for j in range(n_back)][::-1]
    return out


def derivatives(A, H, step_min):
    """Least-squares slope and a three-point curvature over the history H."""
    nb = H.shape[1]
    S = A[H]                                            # (n, nb, f)
    tt = np.arange(nb) * step_min
    slope = np.einsum("ijk,j->ik", S - S.mean(1, keepdims=True), tt - tt.mean()) / \
        max(((tt - tt.mean()) ** 2).sum(), 1e-9)
    curv = S[:, -1] - 2 * S[:, nb // 2] + S[:, 0]
    return slope, curv


def representations(A, meta, ref, step_min, horizon_min=10.0):
    """ref: (n, f) reference vector for every row (NaN rows = no reference)."""
    nb = max(int(round(horizon_min / step_min)) + 1, 2)
    H = history_index(meta, nb)
    slope, curv = derivatives(A, H, step_min)
    D = A - ref
    return {"population": A,
            "reference": D,
            "trajectory": np.hstack([A, slope, curv]),
            "reference+trajectory": np.hstack([D, slope, curv])}


def first_window_reference(A, meta, max_t=None):
    """Mean of the subject's windows with t_min <= max_t; if max_t is None, the
    subject's first observed window."""
    ref = np.zeros_like(A)
    for s, g in meta.groupby("subject").groups.items():
        ix = np.sort(np.asarray(g))
        if max_t is None:
            b = ix[:1]
        else:
            b = ix[meta.t_min.to_numpy()[ix] <= max_t]
            if not len(b):
                b = ix[:1]
        ref[ix] = A[b].mean(0)
    return ref


def score(Xf, meta, tr, va, te, n_jobs=N_JOBS, horizons=(5.0,), want=("stage", "volume",
                                                                       "decomp", "h")):
    """Same targets and settings as run_longitudinal.py.  Returns metrics and the
    held-out predictions (for bootstrap)."""
    P = xgbp(n_jobs)
    r, pr = {}, {}
    y = meta.stage.to_numpy()
    if "stage" in want:
        cnt = np.bincount(y[tr], minlength=5).astype(float)
        wgt = (cnt.sum() / (5 * np.maximum(cnt, 1))) ** 0.5
        clf = xgb.XGBClassifier(objective="multi:softprob", num_class=5,
                                eval_metric="mlogloss", early_stopping_rounds=40, **P)
        clf.fit(Xf[tr], y[tr], sample_weight=wgt[y[tr]], eval_set=[(Xf[va], y[va])],
                verbose=False)
        pr["stage"] = clf.predict_proba(Xf[te])
        r["stage_auroc_macro"] = macro_auc(y[te], pr["stage"])
    if "volume" in want:
        reg = xgb.XGBRegressor(objective="reg:squarederror", early_stopping_rounds=40, **P)
        v = meta.volume_lost.to_numpy()
        reg.fit(Xf[tr], v[tr], eval_set=[(Xf[va], v[va])], verbose=False)
        pr["volume"] = reg.predict(Xf[te])
        r["volume_r2"] = float(r2_score(v[te], pr["volume"]))
    at = meta.at_risk.to_numpy().astype(bool)
    if "decomp" in want:
        yd = meta.decomp_soon.to_numpy()
        a, b, c = tr[at[tr]], va[at[va]], te[at[te]]
        m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              early_stopping_rounds=40,
                              scale_pos_weight=float((1 - yd[a]).sum() / max(yd[a].sum(), 1)),
                              **P)
        m.fit(Xf[a], yd[a], eval_set=[(Xf[b], yd[b])], verbose=False)
        pr["decomp"] = (c, m.predict_proba(Xf[c])[:, 1])
        r["decomp_auroc"] = float(roc_auc_score(yd[c], pr["decomp"][1])) \
            if 0 < yd[c].sum() < len(c) else np.nan
    if "h" in want:
        for h in horizons:
            yy, usable = horizon_label(meta, h)
            a, b, c = tr[usable[tr]], va[usable[va]], te[usable[te]]
            if yy[c].sum() < 5 or yy[a].sum() < 5:
                r[f"decomp_auroc_{h:.0f}min"] = np.nan
                continue
            m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                  early_stopping_rounds=40,
                                  scale_pos_weight=float((1 - yy[a]).sum() / max(yy[a].sum(), 1)),
                                  **P)
            m.fit(Xf[a], yy[a], eval_set=[(Xf[b], yy[b])], verbose=False)
            p = m.predict_proba(Xf[c])[:, 1]
            pr[f"h{h:.0f}"] = (c, p, m, b, m.predict_proba(Xf[b])[:, 1])
            r[f"decomp_auroc_{h:.0f}min"] = float(roc_auc_score(yy[c], p))
    return r, pr


def horizon_label(meta, h):
    at = meta.at_risk.to_numpy().astype(bool)
    yy = ((meta.tte_event == 1) & (meta.tte <= h)).to_numpy().astype(int)
    usable = ~((meta.tte_event == 0) & (meta.tte < h)).to_numpy() & at
    return yy, usable


def macro_auc(y, P):
    """One-vs-rest AUC per state, unweighted mean over states with more than three
    positive and three negative held-out windows (the manuscript's definition)."""
    a = [roc_auc_score((y == k).astype(int), P[:, k]) for k in range(P.shape[1])
         if (y == k).sum() > 3 and (y != k).sum() > 3]
    return float(np.mean(a)) if a else np.nan


def cluster_boot_diff(groups, fn_a, fn_b, n_boot=400, seed=20261012):
    """Paired subject-cluster bootstrap of fn_b(ix) - fn_a(ix)."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    u = np.unique(g)
    rows = {s: np.where(g == s)[0] for s in u}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(u, len(u), replace=True)
        ix = np.concatenate([rows[s] for s in pick])
        try:
            out.append(fn_b(ix) - fn_a(ix))
        except ValueError:
            continue
    out = np.asarray(out)
    out = out[np.isfinite(out)]
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if len(out) > 20 \
        else (np.nan, np.nan)


def deep(cfg):
    return Cfg(copy.deepcopy(dict(cfg)))


def stamp(t0):
    return f"[{time.time() - t0:6.0f}s]"
