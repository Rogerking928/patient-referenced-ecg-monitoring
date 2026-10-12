"""The reframed external question: circulatory state, not aetiology.

The arm comparison in run_vitaldb_analysis.py was null, and the reason is visible in
the data -- the bleeding arm received three times as much fluid, so the volume deficit
was corrected as it was created.  That confound cannot be removed by analysis.

This asks a question the theatre cannot confound in the same way: given a thirty
second electrocardiogram, is this patient hypotensive *right now*?  The ground truth
is an arterial line, the comparison is within patient, and heart rate is included as
a competitor rather than assumed away.
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict

from hemo import config, features as FE

cfg = config.load()
HERE = Path(__file__).resolve().parent
OUT = HERE / "out_vitaldb2"
if not (OUT / "X_vitaldb.npy").exists():
    OUT = HERE / "out_vitaldb"
lcfg = copy.deepcopy(cfg)
lcfg["simulation"]["leads"] = ["II"]
lcfg["simulation"]["fs"] = 250
lcfg["simulation"]["window_s"] = 30.0

X = np.load(OUT / "X_vitaldb.npy")
meta = pd.read_parquet(OUT / "meta_vitaldb.parquet").reset_index(drop=True)
fp = OUT / "features_vitaldb.parquet"
F = pd.read_parquet(fp) if fp.exists() else FE.extract_batch(X[:, None, :], lcfg, n_jobs=10)
if not fp.exists():
    F.to_parquet(fp)
d = pd.concat([meta, F.add_prefix("f_")], axis=1)
print(f"{X.shape} windows from {d.caseid.nunique()} cases")

bc = "f_II_beat_corr" if "f_II_beat_corr" in d.columns else "f_beat_corr"
ok = (d.f_n_beats.between(10, 70) & (d[bc] > 0.75) & d.f_hr_mean.between(30, 180)
      & d.f_rr_kept_frac.gt(0.6) & d.art_mbp.between(30, 160))
print(f"quality control keeps {ok.mean()*100:.0f}% ({(~ok).sum()} dropped)")
d = d[ok].reset_index(drop=True)

FEATS = [c for c in d.columns if c.startswith("f_")]
base = d[d.window == "-5min"].groupby("caseid")[FEATS].median()
d = d[d.caseid.isin(base.index)].reset_index(drop=True)
for c in FEATS:
    d["d_" + c] = d[c].to_numpy() - base.loc[d.caseid, c].to_numpy()

d["hypotension"] = (d.art_mbp < 65).astype(int)
d["sex_n"] = (d.sex.astype(str).str.upper().str[0] == "M").astype(float)
print(f"{len(d)} windows, {d.caseid.nunique()} cases; "
      f"hypotension in {d.hypotension.sum()} windows "
      f"({d.hypotension.mean()*100:.0f}%) across {d[d.hypotension==1].caseid.nunique()} cases")


def prep(cols):
    A = d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return (A - A.mean(0)) / (A.std(0) + 1e-9)


ABS = prep(FEATS)
DEL = prep(["d_" + c for c in FEATS])
HRO = prep(["f_hr_mean"])
HRD = prep(["f_hr_mean", "d_f_hr_mean"])
CONF = [c for c in ["mac", "ppf20_ce", "phen_rate", "nepi_rate", "bt", "age", "sex_n",
                    "t_rel_min"] if c in d.columns]
CNF = prep(CONF)
grp = d.caseid.to_numpy()

SETS = [("heart rate only", HRO),
        ("heart rate, absolute and change", HRD),
        ("all ECG features, absolute", ABS),
        ("all ECG features, change from own baseline", DEL),
        ("all ECG features, absolute + change", np.hstack([ABS, DEL])),
        ("recorded confounders only", CNF),
        ("ECG (absolute + change) + confounders", np.hstack([ABS, DEL, CNF]))]

print("\n=== concurrent hypotension (mean arterial pressure < 65 mmHg) ===")
rows = []
y = d.hypotension.to_numpy()
for tag, A in SETS:
    p = cross_val_predict(LogisticRegression(max_iter=5000, C=0.05), A, y,
                          cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
                          groups=grp, method="predict_proba")[:, 1]
    a = roc_auc_score(y, p)
    rows.append(dict(target="hypotension (MAP < 65)", predictors=tag,
                     n=len(y), n_pos=int(y.sum()), auroc=float(a)))
    print(f"    {tag:46s} AUROC {a:.3f}")

print("\n=== mean arterial pressure as a continuous target ===")
for tag, A in SETS:
    yv = d.art_mbp.to_numpy(np.float64)
    p = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 24)), A, yv,
                          cv=GroupKFold(5), groups=grp)
    rows.append(dict(target="mean arterial pressure (mmHg)", predictors=tag, n=len(yv),
                     n_pos=np.nan, auroc=np.nan, r2=float(r2_score(yv, p)),
                     mae=float(np.mean(np.abs(yv - p))),
                     spearman=float(stats.spearmanr(yv, p).statistic)))
    print(f"    {tag:46s} R2 {r2_score(yv, p):+.3f}  MAE {np.mean(np.abs(yv-p)):.1f} mmHg  "
          f"rho {stats.spearmanr(yv, p).statistic:+.3f}")

print("\n=== positive control: recover the monitor's own heart rate from the waveform ===")
yv = pd.to_numeric(d.hr, errors="coerce").to_numpy(np.float64)
m = np.isfinite(yv)
p = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 24)), ABS[m], yv[m],
                      cv=GroupKFold(5), groups=grp[m])
print(f"    monitor heart rate from ECG features: R2 {r2_score(yv[m], p):+.3f}, "
      f"MAE {np.mean(np.abs(yv[m]-p)):.1f} bpm")
rows.append(dict(target="monitor heart rate (positive control)",
                 predictors="all ECG features, absolute", n=int(m.sum()), n_pos=np.nan,
                 auroc=np.nan, r2=float(r2_score(yv[m], p)),
                 mae=float(np.mean(np.abs(yv[m] - p)))))
pd.DataFrame(rows).to_csv(OUT / "state_detection.csv", index=False)

print("\n=== which single feature changes track the pressure, within patient ===")
res = []
for c in FEATS:
    v = d["d_" + c].to_numpy(np.float64)
    m = np.isfinite(v)
    if m.sum() < 100:
        continue
    dm = d.art_mbp.to_numpy()[m] - base.index.map(lambda x: np.nan) if False else None
    r = stats.spearmanr(v[m], d.art_mbp.to_numpy()[m])
    res.append(dict(feature=c.replace("f_", ""), rho=float(r.statistic),
                    p=float(r.pvalue), n=int(m.sum())))
res = pd.DataFrame(res).sort_values("p")
from statsmodels.stats.multitest import multipletests
res["q"] = multipletests(res.p, method="fdr_bh")[1]
res.to_csv(OUT / "feature_vs_pressure.csv", index=False)
print(res.head(10).to_string(index=False))
print(f"\n{(res.q < 0.05).sum()}/{len(res)} feature changes track mean arterial pressure "
      f"at a false-discovery rate of 5%")
print("VITALDB_STATE_COMPLETE")
