"""Three experiments that attack the result rather than extend it.

  L1  metadata-only baselines: how much of each outcome is obtainable from time,
      trajectory progress, phenotype or heart rate WITHOUT any waveform
  L2  strict matching: re-score the trained models on subsets in which the states
      have the same distribution of time position, and (for the aetiology task)
      the same heart rate, pressure, lactate and reserve
  L3  the cleanest binary question in the project: haemorrhage-related preload loss
      versus pure sympathetic tachycardia, in pairs matched on age, sex, heart rate
      and respiratory rate
"""
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from hemo import config, experiments as E, multi, physiology as phys, train as T
from hemo.config import Cfg, out

cfg = config.load()
mcfg = Cfg(multi.merge_cfg(cfg, yaml.safe_load(open("configs/multi.yaml"))))
log = E.log_to(out(cfg, "run_leakage.log"))
O = lambda p: out(cfg, p)
from pathlib import Path
OM = lambda p: Path(__file__).resolve().parent / "out_multi" / p

meta = pd.read_parquet(O("meta_main.parquet"))
subj = pd.read_parquet(O("subjects.parquet"))
F = pd.read_parquet(O("features_main.parquet"))
# L2 needs the trained models' saved predictions; L1 and L3 do not, so the script
# can also be run before the main comparison exists.
HAVE_PRED = O("predictions_main.npz").exists() and O("meta_test.parquet").exists()
P = np.load(O("predictions_main.npz")) if HAVE_PRED else None
mte = (pd.read_parquet(O("meta_test.parquet")).reset_index(drop=True)
       if HAVE_PRED else None)
R = pd.read_csv(O("results_main.csv")) if O("results_main.csv").exists() else None

XGBC = dict(n_estimators=350, max_depth=4, learning_rate=0.07, subsample=0.85,
            colsample_bytree=0.8, n_jobs=cfg.run.n_jobs, tree_method="hist", random_state=0)

# =================================================================== L1
log("=== L1  metadata-only baselines: what is obtainable without the waveform ===")
m = meta.merge(subj[["subject", "onset_min", "phenotype", "autonomic_gain",
                     "cardiac_reserve", "withdraw_at", "hemorrhage_rate"]],
               on="subject", how="left", suffixes=("", "_s"))
m["time_since_onset"] = np.where(np.isfinite(m.onset_min) & (m.onset_min < 1e8),
                                 m.t_min - m.onset_min, -99.0)
traj = m.groupby("subject").t_min.max().rename("traj_end")
m = m.merge(traj, on="subject")
m["frac_trajectory"] = m.t_min / m.traj_end
m["censored"] = (m.traj_end < cfg.simulation.window_last_min - 1e-6).astype(int)
m["phen_id"] = pd.Categorical(m.phenotype).codes
m["hr_meas"] = F.hr_mean.to_numpy()

tr = np.where(m.split == "train")[0]
va = np.where(m.split == "val")[0]
te = np.where(m.split == "test")[0]

SETS = {
    "absolute time only": ["t_min"],
    "window index only": ["win"],
    "fraction of trajectory completed": ["frac_trajectory"],
    "time since onset (oracle)": ["time_since_onset"],
    "trajectory duration + censoring (oracle)": ["traj_end", "censored"],
    "phenotype only": ["phen_id"],
    "measured heart rate only": ["hr_meas"],
    "heart rate + time": ["hr_meas", "t_min"],
    "all metadata, no ECG": ["t_min", "win", "frac_trajectory", "traj_end", "censored",
                             "phen_id", "age", "sex", "hr0", "map0", "beta_blocked"],
    "all metadata + heart rate": ["t_min", "win", "frac_trajectory", "traj_end", "censored",
                                  "phen_id", "age", "sex", "hr0", "map0", "beta_blocked",
                                  "hr_meas"],
}
rows = []
for name, cols in SETS.items():
    A = m[cols].to_numpy(np.float32)
    A = np.nan_to_num(A, nan=-99.0)
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=5,
                            eval_metric="mlogloss", early_stopping_rounds=30, **XGBC)
    cnt = np.bincount(m.stage.to_numpy()[tr], minlength=5).astype(float)
    wgt = (cnt.sum() / (5 * np.maximum(cnt, 1))) ** 0.5
    clf.fit(A[tr], m.stage.to_numpy()[tr], sample_weight=wgt[m.stage.to_numpy()[tr]],
            eval_set=[(A[va], m.stage.to_numpy()[va])], verbose=False)
    Pm = clf.predict_proba(A[te])
    y = m.stage.to_numpy()[te]
    aucs = [roc_auc_score((y == k).astype(int), Pm[:, k]) for k in range(5)
            if (y == k).sum() > 3]
    r = dict(features=name, n_features=len(cols),
             stage_auroc_macro=float(np.mean(aucs)),
             stage_bacc=balanced_accuracy_score(y, Pm.argmax(1)))
    for task, col in [("volume", "volume_lost"), ("reserve", "reserve")]:
        reg = xgb.XGBRegressor(objective="reg:squarederror", early_stopping_rounds=30, **XGBC)
        reg.fit(A[tr], m[col].to_numpy()[tr], eval_set=[(A[va], m[col].to_numpy()[va])],
                verbose=False)
        r[f"{task}_r2"] = float(r2_score(m[col].to_numpy()[te], reg.predict(A[te])))
    at = m.at_risk.to_numpy().astype(bool)
    c2 = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                           early_stopping_rounds=30, **XGBC)
    yy = m.decomp_soon.to_numpy()
    c2.fit(A[tr[at[tr]]], yy[tr[at[tr]]], eval_set=[(A[va[at[va]]], yy[va[at[va]]])],
           verbose=False)
    pv = c2.predict_proba(A[te[at[te]]])[:, 1]
    r["decomp_auroc"] = float(roc_auc_score(yy[te[at[te]]], pv))
    rows.append(r)
    log(f"    {name:42s} stageAUC {r['stage_auroc_macro']:.3f}  volR2 {r['volume_r2']:+.3f}  "
        f"decAUC {r['decomp_auroc']:.3f}  resR2 {r['reserve_r2']:+.3f}")
for mdl in (["handcrafted_xgb", "raw_waveform_cnn", "hybrid"] if R is not None else []):
    rr = R[R.model == mdl].iloc[0]
    rows.append(dict(features=f"ECG: {mdl}", n_features=len(F.columns),
                     stage_auroc_macro=rr.stage_auroc_macro, stage_bacc=rr.stage_bacc,
                     volume_r2=rr.volume_r2, reserve_r2=rr.reserve_r2,
                     decomp_auroc=rr.decomp_auroc))
pd.DataFrame(rows).to_csv(O("leakage_baselines.csv"), index=False)

# =================================================================== L2 time-matched
log("\n=== L2  matching the states on their position in the recording ===")
if not HAVE_PRED:
    log("    skipped: no saved predictions yet")


def match_on(df, key, by, n_bins=8, seed=0):
    """Resample so that every level of `by` has the same distribution of `key`."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["_bin"] = pd.qcut(d[key], n_bins, labels=False, duplicates="drop")
    keep = []
    for b, gb in d.groupby("_bin"):
        counts = gb[by].value_counts()
        if len(counts) < 2:
            continue
        n = int(counts.min())
        if n < 5:
            continue
        for lvl, gl in gb.groupby(by):
            keep.append(gl.sample(n, random_state=int(rng.integers(1e6))).index.to_numpy())
    return np.concatenate(keep) if keep else np.array([], int)


rows = []
idx = match_on(mte, "t_min", "stage", n_bins=8) if HAVE_PRED else np.array([], int)
if len(idx):
    sub = mte.loc[idx]
    pos = mte.index.get_indexer(idx)
    log(f"    time-matched subset: {len(idx)} windows, "
        f"{sub.stage.nunique()} states, per-state time medians "
        f"{sub.groupby('stage').t_min.median().round(1).to_dict()}")
    for mdl in ["handcrafted_xgb", "raw_waveform_cnn", "hybrid",
                "baseline_heart_rate_only"]:
        pred = {t: P[f"{mdl}__{t}"][pos] for t in T.TASKS}
        s = T.score_all(pred, sub)
        s.update(model=mdl, subset="time-matched", n=len(idx))
        rows.append(s)
        log(f"      {mdl:26s} stageAUC {s['stage_auroc_macro']:.3f}  "
            f"volR2 {s['volume_r2']:+.3f}  decAUC {s['decomp_auroc']:.3f}")
# joint time + heart-rate matching
mte2 = mte.copy() if HAVE_PRED else pd.DataFrame(columns=["hr_true", "t_min", "stage"])
keep = []
if HAVE_PRED:
    mte2["_hrbin"] = pd.qcut(mte2.hr_true, 4, labels=False, duplicates="drop")
    mte2["_tbin"] = pd.qcut(mte2.t_min, 4, labels=False, duplicates="drop")
    mte2["_cell"] = mte2._hrbin.astype(str) + "_" + mte2._tbin.astype(str)
    rng = np.random.default_rng(1)
    for c, gc in mte2.groupby("_cell"):
        cnt = gc.stage.value_counts()
        if len(cnt) < 3 or cnt.min() < 5:
            continue
        n = int(cnt.min())
        for lvl, gl in gc.groupby("stage"):
            keep.append(gl.sample(n, random_state=int(rng.integers(1e6))).index.to_numpy())
if keep:
    idx = np.concatenate(keep)
    sub = mte.loc[idx]
    pos = mte.index.get_indexer(idx)
    log(f"    time+rate-matched subset: {len(idx)} windows, {sub.stage.nunique()} states")
    for mdl in ["handcrafted_xgb", "raw_waveform_cnn", "hybrid",
                "baseline_heart_rate_only"]:
        pred = {t: P[f"{mdl}__{t}"][pos] for t in T.TASKS}
        s = T.score_all(pred, sub)
        s.update(model=mdl, subset="time+rate-matched", n=len(idx))
        rows.append(s)
        log(f"      {mdl:26s} stageAUC {s['stage_auroc_macro']:.3f}  "
            f"volR2 {s['volume_r2']:+.3f}  decAUC {s['decomp_auroc']:.3f}")
pd.DataFrame(rows).to_csv(O("matched_time.csv"), index=False)

# =================================================================== L2b aetiology at equal severity
log("\n=== L2b  aetiology discrimination at equal severity ===")
if OM("meta_multi.parquet").exists():
    mm = pd.read_parquet(OM("meta_multi.parquet"))
    Fm = pd.read_parquet(OM("features_multi.parquet"))
    Am, mu_m, sd_m = T.prep_features(Fm, np.where(mm.split == "train")[0])
    CIRC = ["haemorrhage", "sepsis", "cardiogenic", "hypoxaemia"]
    sel = mm[mm.aetiology_name.isin(CIRC)].copy()
    log(f"    circulatory aetiologies only: {len(sel)} windows, "
        + str(sel.aetiology_name.value_counts().to_dict()))
    rows2 = []
    for tag, keys in [("unmatched", []),
                      ("matched on heart rate", ["hr_true"]),
                      ("matched on heart rate + mean pressure", ["hr_true", "map_frac"]),
                      ("matched on rate, pressure, lactate and reserve",
                       ["hr_true", "map_frac", "lactate", "reserve"])]:
        d = sel.copy()
        if keys:
            nb = 3 if len(keys) > 2 else 4
            cell = None
            for k in keys:
                try:
                    b = pd.qcut(d[k], nb, labels=False, duplicates="drop").astype(str)
                except ValueError:
                    b = pd.Series("0", index=d.index)
                cell = b if cell is None else cell + "_" + b
            d["_cell"] = cell
            keep = []
            rng2 = np.random.default_rng(3)
            for c, gc in d.groupby("_cell"):
                cnt = gc.aetiology_name.value_counts()
                if len(cnt) < 3 or cnt.min() < 4:
                    continue
                n = int(cnt.min())
                for lvl, gl in gc.groupby("aetiology_name"):
                    keep.append(gl.sample(n, random_state=int(rng2.integers(1e6))).index.to_numpy())
            if not keep:
                log(f"    {tag}: no usable cells")
                continue
            d = d.loc[np.concatenate(keep)]
        if d.aetiology_name.nunique() < 3 or len(d) < 200:
            log(f"    {tag}: too few windows ({len(d)})")
            continue
        y = pd.Categorical(d.aetiology_name, categories=CIRC).codes
        Ad = Am[d.index.to_numpy()]
        pr = cross_val_predict(
            xgb.XGBClassifier(objective="multi:softprob", num_class=len(CIRC),
                              eval_metric="mlogloss", n_estimators=300, max_depth=4,
                              learning_rate=0.07, n_jobs=cfg.run.n_jobs,
                              tree_method="hist", random_state=0),
            Ad, y, cv=GroupKFold(5), groups=d.subject.to_numpy(), method="predict_proba")
        aucs = [roc_auc_score((y == k).astype(int), pr[:, k]) for k in range(len(CIRC))
                if (y == k).sum() > 3 and (y != k).sum() > 3]
        r = dict(subset=tag, n=len(d), classes=int(d.aetiology_name.nunique()),
                 aet_auroc_macro=float(np.mean(aucs)),
                 aet_bacc=balanced_accuracy_score(y, pr.argmax(1)),
                 hr_sd=float(d.hr_true.std()), map_sd=float(d.map_frac.std()))
        for k, c_ in enumerate(CIRC):
            if (y == k).sum() > 3 and (y != k).sum() > 3:
                r[f"auroc_{c_}"] = roc_auc_score((y == k).astype(int), pr[:, k])
        rows2.append(r)
        log(f"    {tag:46s} n={len(d):5d}  aetAUC {r['aet_auroc_macro']:.3f}  "
            f"bACC {r['aet_bacc']:.3f}  (HR sd {r['hr_sd']:.1f})")
    pd.DataFrame(rows2).to_csv(O("matched_severity_aetiology.csv"), index=False)

# =================================================================== L3 stressor vs haemorrhage
log("\n=== L3  haemorrhage versus pure sympathetic tachycardia, matched pairs ===")
mm = meta.merge(subj[["subject", "stressor_onset_min", "stressor_dur_min"]],
                on="subject", how="left")
in_stress = ((mm.stressor == 1) & (mm.t_min >= mm.stressor_onset_min)
             & (mm.t_min < mm.stressor_onset_min + mm.stressor_dur_min)).to_numpy()
bleeding_comp = ((mm.bleeder == 1) & (mm.stage.isin([1, 2]))).to_numpy()
Fx = F.to_numpy(np.float32)
Fx = np.nan_to_num(Fx, nan=np.nanmedian(Fx, axis=0))
S = mm.assign(_i=np.arange(len(mm)))
A_ = S[in_stress]
B_ = S[bleeding_comp]
log(f"    pool: {len(A_)} stressor windows, {len(B_)} compensated-haemorrhage windows")

rng = np.random.default_rng(11)
pairs = []
usedB = set()
Bv = B_[["age", "sex", "hr_true", "resp", "_i", "subject"]].to_numpy()
for _, a in A_.iterrows():
    cand = np.where((np.abs(Bv[:, 0] - a.age) <= 8) & (Bv[:, 1] == a.sex)
                    & (np.abs(Bv[:, 2] - a.hr_true) <= 4)
                    & (np.abs(Bv[:, 3] - a.resp) <= 2.5)
                    & ~np.isin(Bv[:, 4], list(usedB)))[0]
    if not len(cand):
        continue
    j = int(cand[rng.integers(len(cand))])
    usedB.add(Bv[j, 4])
    pairs.append((int(a._i), int(Bv[j, 4]), int(a.subject), int(Bv[j, 5])))
log(f"    matched {len(pairs)} pairs on age (±8 y), sex, heart rate (±4 bpm) and "
    f"respiratory rate (±2.5/min)")

if len(pairs) >= 60:
    ia = np.array([p[0] for p in pairs])
    ib = np.array([p[1] for p in pairs])
    idx = np.r_[ia, ib]
    y = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
    grp = np.r_[[p[2] for p in pairs], [p[3] for p in pairs]]
    d = mm.iloc[idx]
    log(f"    matched set: HR {d[y==0].hr_true.mean():.1f} vs {d[y==1].hr_true.mean():.1f} bpm, "
        f"preload {d[y==0].preload.mean():.2f} vs {d[y==1].preload.mean():.2f}")
    out_rows = []
    for tag, cols in [("all features", list(range(Fx.shape[1]))),
                      ("morphology only", [i for i, c in enumerate(F.columns)
                                           if c not in E.HRV_ONLY]),
                      ("heart rate only", [list(F.columns).index("hr_mean")])]:
        pr = cross_val_predict(
            xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              n_estimators=300, max_depth=4, learning_rate=0.07,
                              n_jobs=cfg.run.n_jobs, tree_method="hist", random_state=0),
            Fx[np.ix_(idx, cols)], y, cv=GroupKFold(5), groups=grp,
            method="predict_proba")[:, 1]
        a = roc_auc_score(y, pr)
        out_rows.append(dict(features=tag, n_pairs=len(pairs), auroc=a,
                             hr_diff=float(d[y == 1].hr_true.mean() - d[y == 0].hr_true.mean()),
                             preload_diff=float(d[y == 1].preload.mean() - d[y == 0].preload.mean())))
        log(f"      {tag:20s} AUROC {a:.3f}")
    pd.DataFrame(out_rows).to_csv(O("stressor_vs_haemorrhage.csv"), index=False)
else:
    log("    too few matched pairs to test")

log("\nLEAKAGE_MATCHING_COMPLETE")
