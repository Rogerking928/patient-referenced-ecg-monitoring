"""External proof-of-concept on real intraoperative recordings (VitalDB).

This is the only part of the project that uses real physiology with a real change in
circulating volume.  It is a within-patient, longitudinal design: each patient's own
pre-incision recording is the reference, and the question is whether the change in
the electrocardiogram over the next ninety minutes differs between patients who lose
half a litre of blood or more and matched patients who lose almost none.

What it cannot do is prove the simulation right.  Intraoperative recordings are
confounded by anaesthetic depth, vasopressors, temperature, positioning and surgical
stimulus, all of which are recorded here and none of which the simulation contains.
The analysis therefore reports the crude association, the association after adjusting
for the recorded confounders, and the confounders' own associations, side by side.
"""
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GroupKFold, StratifiedGroupKFold,
                                     cross_val_predict)

from hemo import config, features as FE

cfg = config.load()
HERE = Path(__file__).resolve().parent
OUT = HERE / "out_vitaldb"
lcfg = copy.deepcopy(cfg)
lcfg["simulation"]["leads"] = ["II"]          # VitalDB gives lead II only
lcfg["simulation"]["fs"] = 250
lcfg["simulation"]["window_s"] = 30.0

X = np.load(OUT / "X_vitaldb.npy")
meta = pd.read_parquet(OUT / "meta_vitaldb.parquet").reset_index(drop=True)
print(f"{X.shape} windows from {meta.caseid.nunique()} cases "
      f"({meta[meta.window=='-5min'].arm.value_counts().to_dict()} with a baseline)")

# ------------------------------------------------------------------ features
fp = OUT / "features_vitaldb.parquet"
if fp.exists():
    F = pd.read_parquet(fp)
else:
    F = FE.extract_batch(X[:, None, :], lcfg, n_jobs=8)
    F.to_parquet(fp)
print(f"features {F.shape}, NaN {F.isna().mean().mean()*100:.1f}%")

d = pd.concat([meta, F.add_prefix("f_")], axis=1)
d["is_bleed"] = (d.arm == "bleeding").astype(int)

# quality control, the same gates used on PTB-XL
bc = "f_II_beat_corr" if "f_II_beat_corr" in d.columns else "f_beat_corr"
ok = (d.f_n_beats.between(10, 70) & (d[bc] > 0.75)
      & d.f_hr_mean.between(30, 180) & d.f_rr_kept_frac.gt(0.6))
print(f"quality control keeps {ok.mean()*100:.0f}% of windows "
      f"({(~ok).sum()} dropped)")
d = d[ok].reset_index(drop=True)

base = d[d.window == "-5min"].set_index("caseid")
later = d[d.window != "-5min"].copy()
have_base = later.caseid.isin(base.index)
later = later[have_base].reset_index(drop=True)
print(f"{later.caseid.nunique()} cases have a usable baseline and at least one later window")

FEATS = [c for c in d.columns if c.startswith("f_")]
for c in FEATS:
    later["d_" + c] = later[c].to_numpy() - base.loc[later.caseid, c].to_numpy()
later["d_map"] = later.art_mbp.to_numpy() - base.loc[later.caseid, "art_mbp"].to_numpy()
later["d_hr_mon"] = later.hr.to_numpy() - base.loc[later.caseid, "hr"].to_numpy()

# ------------------------------------------------------------------ description
print("\n=== cohort ===")
cc = later.groupby("arm").agg(cases=("caseid", "nunique"), windows=("caseid", "size"),
                              age=("age", "mean"), ebl=("ebl", "median"),
                              crystalloid=("crystalloid", "median"),
                              d_map=("d_map", "mean"), d_hr=("d_hr_mon", "mean"))
print(cc.round(1).to_string())
rows = [dict(quantity=k, **{a: float(v) for a, v in cc[k].items()}) for k in cc.columns]
pd.DataFrame(rows).to_csv(OUT / "cohort_summary.csv", index=False)

# ------------------------------------------------------------------ per-feature test
print("\n=== change from each patient's own baseline, bleeding vs control ===")
res = []
for c in FEATS:
    a = later.loc[later.is_bleed == 1, "d_" + c].dropna()
    b = later.loc[later.is_bleed == 0, "d_" + c].dropna()
    if len(a) < 20 or len(b) < 20:
        continue
    u = stats.mannwhitneyu(a, b)
    pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-12
    res.append(dict(feature=c.replace("f_", ""), n_bleed=len(a), n_ctrl=len(b),
                    median_bleed=float(a.median()), median_ctrl=float(b.median()),
                    smd=float((a.mean() - b.mean()) / pooled), p=float(u.pvalue)))
res = pd.DataFrame(res).sort_values("p")
from statsmodels.stats.multitest import multipletests
res["q"] = multipletests(res.p, method="fdr_bh")[1]
res.to_csv(OUT / "feature_deltas.csv", index=False)
print(res.head(12)[["feature", "median_bleed", "median_ctrl", "smd", "p", "q"]]
      .to_string(index=False))
print(f"\n{(res.q < 0.05).sum()}/{len(res)} feature changes differ between the arms "
      f"at a false-discovery rate of 5%")

# ------------------------------------------------------------------ multivariable
print("\n=== can the ECG change alone separate the arms? (case-level cross-validation) ===")
DCOLS = ["d_" + c for c in FEATS]
Xd = later[DCOLS].to_numpy(np.float64)
Xd = np.nan_to_num(Xd, nan=0.0, posinf=0.0, neginf=0.0)
Xd = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-9)
y = later.is_bleed.to_numpy()
grp = later.caseid.to_numpy()
later["sex_n"] = (later.sex.astype(str).str.upper().str[0] == "M").astype(float)
CONF = ["mac", "ppf20_ce", "phen_rate", "nepi_rate", "bt", "t_rel_min", "age", "sex_n",
        "weight", "asa"]
cc_ = [c for c in CONF if c in later.columns]
conf = later[cc_].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
print(f"    confounders adjusted for: {', '.join(cc_)}")
conf = np.nan_to_num(conf, nan=0.0)
conf = (conf - conf.mean(0)) / (conf.std(0) + 1e-9)

out = []
for tag, Xin in [("ECG change only", Xd),
                 ("recorded confounders only", conf),
                 ("ECG change + confounders", np.hstack([Xd, conf])),
                 ("monitor haemodynamics only (MAP and HR change)",
                  np.nan_to_num(later[["d_map", "d_hr_mon"]].to_numpy(np.float64)))]:
    # the folds must be stratified as well as grouped: with plain GroupKFold a fold
    # can end up almost entirely one arm, and the model then inverts systematically
    p = cross_val_predict(LogisticRegression(max_iter=4000, C=0.1), Xin, y,
                          cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                          groups=grp, method="predict_proba")[:, 1]
    a = roc_auc_score(y, p)
    # case level: one prediction per patient
    cl = pd.DataFrame({"caseid": grp, "p": p, "y": y}).groupby("caseid").mean()
    ac = roc_auc_score(cl.y, cl.p)
    out.append(dict(predictors=tag, n_windows=len(y), window_auroc=a, case_auroc=ac))
    print(f"    {tag:46s} window AUROC {a:.3f}   case-level AUROC {ac:.3f}")
pd.DataFrame(out).to_csv(OUT / "discrimination.csv", index=False)

# ------------------------------------------------------------------ dose response
print("\n=== is the ECG change graded with the amount of blood lost? ===")
cl = later.groupby("caseid").agg(ebl=("ebl", "first"), arm=("arm", "first"),
                                 d_map=("d_map", "mean"),
                                 **{c: (c, "mean") for c in DCOLS})
rows = []
for c in DCOLS:
    v = cl[c].to_numpy()
    m = np.isfinite(v)
    if m.sum() < 30:
        continue
    r = stats.spearmanr(np.log1p(cl.ebl.to_numpy()[m]), v[m])
    rows.append(dict(feature=c.replace("d_f_", ""), rho=float(r.statistic),
                     p=float(r.pvalue), n=int(m.sum())))
dr = pd.DataFrame(rows).sort_values("p")
dr["q"] = multipletests(dr.p, method="fdr_bh")[1]
dr.to_csv(OUT / "dose_response.csv", index=False)
print(dr.head(10).to_string(index=False))
print(f"\n{(dr.q < 0.05).sum()}/{len(dr)} features show a graded association with "
      f"log blood loss at a false-discovery rate of 5%")

# ------------------------------------------------------------------ continuous target
# The arm comparison is confounded by treatment: the bleeding arm received far more
# fluid, which is exactly what a theatre does. A label-free test asks whether the ECG
# change tracks the *concurrent* haemodynamic change in the same patient.
print("\n=== does the ECG change track the concurrent haemodynamic change? ===")
later["d_pp"] = ((later.art_sbp - later.art_dbp).to_numpy()
                 - (base.loc[later.caseid, "art_sbp"] - base.loc[later.caseid, "art_dbp"]).to_numpy())
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
rows = []
for tgt, name in [("d_map", "change in mean arterial pressure"),
                  ("d_pp", "change in pulse pressure"),
                  ("d_hr_mon", "change in monitor heart rate")]:
    yv = later[tgt].to_numpy(np.float64)
    m = np.isfinite(yv)
    if m.sum() < 60:
        continue
    for tag, Xin in [("ECG change", Xd), ("recorded confounders", conf),
                     ("ECG change + confounders", np.hstack([Xd, conf]))]:
        pr = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 20)), Xin[m], yv[m],
                               cv=GroupKFold(5), groups=grp[m])
        r2 = float(r2_score(yv[m], pr))
        rho = float(stats.spearmanr(yv[m], pr).statistic)
        rows.append(dict(target=name, predictors=tag, n=int(m.sum()), r2=r2, spearman=rho))
        print(f"    {name:36s} from {tag:26s} R2 {r2:+.3f}  rho {rho:+.3f}")
pd.DataFrame(rows).to_csv(OUT / "continuous_targets.csv", index=False)

print("\nVITALDB_ANALYSIS_COMPLETE")
