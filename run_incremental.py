"""What does the electrocardiogram add to what is already known?

The comparison that has been made so far is ECG against no-ECG, and the no-ECG side wins:
subject metadata together with the measured heart rate beats every waveform model. That
is the wrong comparison for a monitoring claim. The clinical question is incremental:
given the elapsed time, the heart rate and the patient's recorded characteristics, does
the waveform add anything?

    delta = performance(metadata + heart rate + ECG) - performance(metadata + heart rate)

reported with a paired cluster bootstrap so the interval accounts for the fact that
windows from one subject are not independent. Three settings:

    1. the whole held-out set
    2. the held-out set matched on elapsed time and heart rate
    3. the external intraoperative cohort, where the equivalent of "metadata" is the
       recorded anaesthetic and vasopressor state

A positive point estimate whose interval crosses zero is reported as crossing zero.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from hemo import config, experiments as E, train as T
from hemo.config import out

cfg = config.load()
log = E.log_to(out(cfg, "run_incremental.log"))
O = lambda p: out(cfg, p)
HERE = Path(__file__).resolve().parent
t00 = time.time()
NBOOT = int(cfg.controls.bootstrap_n)

meta = pd.read_parquet(O("meta_main.parquet")).reset_index(drop=True)
F = pd.read_parquet(O("features_main.parquet"))
A_ecg, mu, sd = T.prep_features(F, np.where(meta.split == "train")[0])
A_ecg = np.clip((A_ecg - mu) / sd, -8, 8)

m = meta.copy()
m["frac_trajectory"] = m.groupby("subject").t_min.transform(lambda s: s / max(s.max(), 1e-9))
m["traj_end"] = m.groupby("subject").t_min.transform("max")
m["censored"] = (~m.groupby("subject").tte_event.transform("max").astype(bool)).astype(int)
m["phen_id"] = pd.Categorical(m.phenotype).codes
m["hr_meas"] = F.hr_mean.to_numpy()
# Two metadata baselines. The prospective one is the honest comparator: everything in it
# is knowable at the moment the window is recorded. The oracle one adds the variables
# that encode when the trajectory ends, which is close to knowing the answer -- it is
# reported so that the gap between the two is visible rather than buried.
META_PROSPECTIVE = ["t_min", "win", "phen_id", "age", "sex", "hr0", "map0",
                    "beta_blocked", "hr_meas"]
META_ORACLE = META_PROSPECTIVE + ["frac_trajectory", "traj_end", "censored"]
A_meta = np.nan_to_num(m[META_PROSPECTIVE].to_numpy(np.float32), nan=-99.0)
A_oracle = np.nan_to_num(m[META_ORACLE].to_numpy(np.float32), nan=-99.0)
A_both = np.hstack([A_meta, A_ecg]).astype(np.float32)

tr = np.where(m.split == "train")[0]
va = np.where(m.split == "val")[0]
te = np.where(m.split == "test")[0]
XGBP = dict(n_estimators=500, learning_rate=0.06, max_depth=5, subsample=0.85,
            colsample_bytree=0.75, n_jobs=cfg.run.n_jobs, tree_method="hist", random_state=0)


def fit_predict(A):
    """One fit per task; returns held-out predictions aligned to `te`."""
    P = {}
    y = m.stage.to_numpy()
    cnt = np.bincount(y[tr], minlength=5).astype(float)
    wg = (cnt.sum() / (5 * np.maximum(cnt, 1))) ** 0.5
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=5, eval_metric="mlogloss",
                            early_stopping_rounds=40, **XGBP)
    clf.fit(A[tr], y[tr], sample_weight=wg[y[tr]], eval_set=[(A[va], y[va])], verbose=False)
    P["stage"] = clf.predict_proba(A[te])
    for task, col in [("volume", "volume_lost"), ("reserve", "reserve")]:
        reg = xgb.XGBRegressor(objective="reg:squarederror", early_stopping_rounds=40, **XGBP)
        reg.fit(A[tr], m[col].to_numpy()[tr], eval_set=[(A[va], m[col].to_numpy()[va])],
                verbose=False)
        P[task] = reg.predict(A[te])
    at = m.at_risk.to_numpy().astype(bool)
    yd = m.decomp_soon.to_numpy()
    itr, iva = tr[at[tr]], va[at[va]]
    c2 = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                           early_stopping_rounds=40,
                           scale_pos_weight=float((1 - yd[itr]).sum() / max(yd[itr].sum(), 1)),
                           **XGBP)
    c2.fit(A[itr], yd[itr], eval_set=[(A[iva], yd[iva])], verbose=False)
    P["decomp"] = c2.predict_proba(A[te])[:, 1]      # scored on the at-risk subset below
    return P


def metrics(P, rows):
    """Metrics over a subset of positions within the held-out block."""
    y = m.stage.to_numpy()[te][rows]
    out_ = {}
    aucs = [roc_auc_score((y == k).astype(int), P["stage"][rows, k])
            for k in range(5) if 3 < (y == k).sum() < len(y)]
    out_["stage_auroc_macro"] = float(np.mean(aucs)) if aucs else np.nan
    out_["stage_bacc"] = balanced_accuracy_score(y, P["stage"][rows].argmax(1))
    for task, col in [("volume", "volume_lost"), ("reserve", "reserve")]:
        out_[f"{task}_r2"] = float(r2_score(m[col].to_numpy()[te][rows], P[task][rows]))
    at = m.at_risk.to_numpy().astype(bool)[te][rows]
    yd = m.decomp_soon.to_numpy()[te][rows]
    out_["decomp_auroc"] = (float(roc_auc_score(yd[at], P["decomp"][rows][at]))
                            if at.sum() > 20 and 0 < yd[at].sum() < at.sum() else np.nan)
    out_["n_decomp_events"] = int(yd[at].sum())
    return out_


log("fitting the three feature sets")
PRED = {}
for name, A in [("metadata + heart rate", A_meta), ("ECG only", A_ecg),
                ("metadata + heart rate + ECG", A_both),
                ("oracle metadata + heart rate", A_oracle)]:
    PRED[name] = fit_predict(A)
    log(f"[{time.time()-t00:5.0f}s] fitted {name}")

# ------------------------------------------------------------------ matched subset
mte = m.iloc[te].reset_index(drop=True)
mte["_hrbin"] = pd.qcut(mte.hr_true, 4, labels=False, duplicates="drop")
mte["_tbin"] = pd.qcut(mte.t_min, 4, labels=False, duplicates="drop")
rng = np.random.default_rng(1)
keep = []
for _, gc in mte.groupby(mte._hrbin.astype(str) + "_" + mte._tbin.astype(str)):
    cnt = gc.stage.value_counts()
    if len(cnt) < 3 or cnt.min() < 5:
        continue
    n = int(cnt.min())
    for _, gl in gc.groupby("stage"):
        keep.append(gl.sample(n, random_state=int(rng.integers(1e6))).index.to_numpy())
matched = np.concatenate(keep) if keep else np.array([], int)
log(f"time+rate-matched subset: {len(matched)} of {len(te)} held-out windows")

SETTINGS = [("all held-out windows", np.arange(len(te))), ("time + rate matched", matched)]
TASKS = ["stage_auroc_macro", "volume_r2", "reserve_r2", "decomp_auroc"]
subj_te = m.subject.to_numpy()[te]

rows = []
for sname, rowsel in SETTINGS:
    if not len(rowsel):
        continue
    base = metrics(PRED["metadata + heart rate"], rowsel)
    full = metrics(PRED["metadata + heart rate + ECG"], rowsel)
    ecg = metrics(PRED["ECG only"], rowsel)
    orc = metrics(PRED["oracle metadata + heart rate"], rowsel)
    # paired cluster bootstrap of the difference, resampling held-out subjects
    subs = np.unique(subj_te[rowsel])
    by_sub = {s: rowsel[subj_te[rowsel] == s] for s in subs}
    boot = {k: [] for k in TASKS}
    brng = np.random.default_rng(20260802)
    for _ in range(NBOOT):
        pick = brng.choice(subs, len(subs), replace=True)
        ix = np.concatenate([by_sub[s] for s in pick])
        try:
            b = metrics(PRED["metadata + heart rate"], ix)
            f = metrics(PRED["metadata + heart rate + ECG"], ix)
        except ValueError:
            continue
        for k in TASKS:
            if np.isfinite(b[k]) and np.isfinite(f[k]):
                boot[k].append(f[k] - b[k])
    for k in TASKS:
        d = np.array(boot[k], float)
        rows.append(dict(
            setting=sname, n_windows=int(len(rowsel)), n_subjects=int(len(subs)),
            task=k,
            metadata_hr=base[k], ecg_only=ecg[k], metadata_hr_plus_ecg=full[k],
            oracle_metadata_hr=orc[k],
            delta=full[k] - base[k],
            delta_lo=float(np.percentile(d, 2.5)) if len(d) > 20 else np.nan,
            delta_hi=float(np.percentile(d, 97.5)) if len(d) > 20 else np.nan,
            n_boot=int(len(d)), n_decomp_events=base["n_decomp_events"]))
        r = rows[-1]
        log(f"    {sname:22s} {k:18s} base {r['metadata_hr']:+.3f} -> "
            f"{r['metadata_hr_plus_ecg']:+.3f}   delta {r['delta']:+.3f} "
            f"[{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]")

pd.DataFrame(rows).to_csv(O("incremental_value.csv"), index=False)

# ------------------------------------------------------------------ external cohort
log("\n=== incremental value in the external intraoperative cohort ===")
OV = HERE / "out_vitaldb2"
rows2 = []
if (OV / "features_vitaldb.parquet").exists():
    dv = pd.concat([pd.read_parquet(OV / "meta_vitaldb.parquet").reset_index(drop=True),
                    pd.read_parquet(OV / "features_vitaldb.parquet").add_prefix("f_")], axis=1)
    bc = "f_II_beat_corr" if "f_II_beat_corr" in dv.columns else "f_beat_corr"
    ok = (dv.f_n_beats.between(10, 70) & (dv[bc] > 0.75) & dv.f_hr_mean.between(30, 180)
          & dv.f_rr_kept_frac.gt(0.6) & dv.art_mbp.between(30, 160))
    dv = dv[ok].reset_index(drop=True)
    FE_ = [c for c in dv.columns if c.startswith("f_")]
    b = dv[dv.window == "-5min"].groupby("caseid")[FE_].median()
    dv = dv[dv.caseid.isin(b.index)].reset_index(drop=True)
    for c in FE_:
        dv["d_" + c] = dv[c].to_numpy() - b.loc[dv.caseid, c].to_numpy()
    dv["hypotension"] = (dv.art_mbp < 65).astype(int)
    dv["sex_n"] = (dv.sex.astype(str).str.upper().str[0] == "M").astype(float)

    def prep(cols):
        Ax = dv[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        Ax = np.nan_to_num(Ax, nan=0.0, posinf=0.0, neginf=0.0)
        return (Ax - Ax.mean(0)) / (Ax.std(0) + 1e-9)

    CONF = [c for c in ["mac", "ppf20_ce", "phen_rate", "nepi_rate", "bt", "age", "sex_n",
                        "t_rel_min", "f_hr_mean"] if c in dv.columns]
    CNF, DEL = prep(CONF), prep(["d_" + c for c in FE_])
    yv, gv = dv.hypotension.to_numpy(), dv.caseid.to_numpy()
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)

    def cvp(A):
        return cross_val_predict(LogisticRegression(max_iter=5000, C=0.05), A, yv,
                                 cv=cv, groups=gv, method="predict_proba")[:, 1]

    p_base, p_full = cvp(CNF), cvp(np.hstack([CNF, DEL]))
    cases = np.unique(gv)
    brng = np.random.default_rng(20260802)
    bw, bc_ = [], []
    for _ in range(NBOOT):
        pick = brng.choice(cases, len(cases), replace=True)
        ix = np.concatenate([np.where(gv == c)[0] for c in pick])
        if len(np.unique(yv[ix])) < 2:
            continue
        bw.append(roc_auc_score(yv[ix], p_full[ix]) - roc_auc_score(yv[ix], p_base[ix]))
        # case level: one risk per operation, labelled by whether it ever went hypotensive
        cl = pd.DataFrame(dict(case=gv[ix], y=yv[ix], b=p_base[ix], f=p_full[ix]))
        agg = cl.groupby("case").agg(y=("y", "max"), b=("b", "median"), f=("f", "median"))
        if agg.y.nunique() > 1:
            bc_.append(roc_auc_score(agg.y, agg.f) - roc_auc_score(agg.y, agg.b))
    agg0 = pd.DataFrame(dict(case=gv, y=yv, b=p_base, f=p_full)).groupby("case").agg(
        y=("y", "max"), b=("b", "median"), f=("f", "median"))
    for lvl, a_b, a_f, bd in [
            ("window level (case-cluster bootstrap)", roc_auc_score(yv, p_base),
             roc_auc_score(yv, p_full), np.array(bw)),
            ("case level (median risk per operation)", roc_auc_score(agg0.y, agg0.b),
             roc_auc_score(agg0.y, agg0.f), np.array(bc_))]:
        rows2.append(dict(cohort="VitalDB, concurrent hypotension", level=lvl,
                          n_windows=int(len(yv)), n_cases=int(len(cases)),
                          n_pos_windows=int(yv.sum()), n_pos_cases=int(agg0.y.sum()),
                          confounders=float(a_b), confounders_plus_ecg=float(a_f),
                          delta=float(a_f - a_b),
                          delta_lo=float(np.percentile(bd, 2.5)) if len(bd) > 20 else np.nan,
                          delta_hi=float(np.percentile(bd, 97.5)) if len(bd) > 20 else np.nan,
                          n_boot=int(len(bd))))
        r = rows2[-1]
        log(f"    {lvl:42s} {r['confounders']:.3f} -> {r['confounders_plus_ecg']:.3f}   "
            f"delta {r['delta']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]")
    pd.DataFrame(rows2).to_csv(O("incremental_vitaldb.csv"), index=False)
else:
    log("    external cohort features not on disk; skipped")

log(f"\nINCREMENTAL_COMPLETE [{time.time()-t00:.0f}s]")
