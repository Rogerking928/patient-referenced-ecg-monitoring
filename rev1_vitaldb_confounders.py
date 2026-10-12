"""Reviewer 1 comment 4 and Reviewer 2 major comment 8: anaesthetic, ventilatory and
surgical confounding of the external hypotension result.

The analysis set is rebuilt exactly as in run_vitaldb_state.py (same quality control,
same pre-incision reference, 714 windows from 77 operations).  For every window the
concurrent anaesthetic and ventilatory state is read from the local copy of the
VitalDB files (same 30 s interval as the electrocardiogram):

    volatile agent (MAC), propofol and remifentanil effect-site concentrations,
    bispectral index, phenylephrine and noradrenaline infusion rates,
    PEEP, tidal volume, ventilator rate, end-tidal CO2, mean airway pressure,
    temperature, elapsed time from incision,
and per operation: crystalloid and colloid volume, estimated blood loss, operation
type.  Patient position and bolus drugs are not recorded in VitalDB.

Three analyses:
  1. incremental value of the patient-referenced ECG over the extended set,
     window and operation level, operation-cluster bootstrap;
  2. the out-of-fold predictions of the manuscript's three models evaluated within
     strata (volatile vs intravenous maintenance, vasopressor infusion running or
     not, early vs late surgery, PEEP, depth of anaesthesia);
  3. the patient-referenced ECG change with every feature residualised on the
     concurrent change in anaesthetic and ventilatory state, so that only the part
     of the ECG change those variables do not explain is left.
"""
import copy
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_rev1"
VDIR = Path(os.environ.get("VITALDB_DIR", HERE / "data" / "VitalDB"))   # cases.csv + vital/<caseid>.vital
t00 = time.time()
LOG = open(OUT / "vitaldb_confounders.log", "w")


def log(m):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


# ------------------------------------------------------------------ analysis set
meta = pd.read_parquet(HERE / "out_vitaldb2/meta_vitaldb.parquet").reset_index(drop=True)
F = pd.read_parquet(HERE / "out_vitaldb2/features_vitaldb.parquet")
d = pd.concat([meta, F.add_prefix("f_")], axis=1)
bc = "f_II_beat_corr" if "f_II_beat_corr" in d.columns else "f_beat_corr"
ok = (d.f_n_beats.between(10, 70) & (d[bc] > 0.75) & d.f_hr_mean.between(30, 180)
      & d.f_rr_kept_frac.gt(0.6) & d.art_mbp.between(30, 160))
d = d[ok].reset_index(drop=True)
FEATS = [c for c in d.columns if c.startswith("f_")]
base = d[d.window == "-5min"].groupby("caseid")[FEATS].median()
d = d[d.caseid.isin(base.index)].reset_index(drop=True)
for c in FEATS:
    d["d_" + c] = d[c].to_numpy() - base.loc[d.caseid, c].to_numpy()
d["hypotension"] = (d.art_mbp < 65).astype(int)
d["sex_n"] = (d.sex.astype(str).str.upper().str[0] == "M").astype(float)
d["t_from_incision_min"] = d.t_rel_min / 60.0          # stored in seconds
log(f"{len(d)} windows, {d.caseid.nunique()} operations, {d.hypotension.sum()} hypotensive "
    f"windows in {d[d.hypotension == 1].caseid.nunique()} operations")
assert len(d) == 714 and d.caseid.nunique() == 77

# ------------------------------------------------------------------ concurrent state
TRK = {"mac": "Primus/MAC", "ppf_ce": "Orchestra/PPF20_CE", "rftn_ce": "Orchestra/RFTN20_CE",
       "bis": "BIS/BIS", "phen": "Orchestra/PHEN_RATE", "nepi": "Orchestra/NEPI_RATE",
       "peep": "Primus/PEEP_MBAR", "tv": "Primus/TV", "vent_rr": "Primus/RR_CO2",
       "etco2": "Primus/ETCO2", "mawp": "Primus/MAWP_MBAR", "temp": "Solar8000/BT",
       "mbp_check": "Solar8000/ART_MBP"}
cases = pd.read_csv(VDIR / "cases.csv").set_index("caseid")
cache = OUT / "vitaldb_concurrent_state.parquet"
if cache.exists():
    S = pd.read_parquet(cache)
else:
    rec = []
    for cid, g in d.groupby("caseid"):
        vf = vitaldb.VitalFile(str(VDIR / "vital" / f"{cid}.vital"), list(TRK.values()))
        x = vf.to_numpy(list(TRK.values()), 1)
        op0 = float(cases.loc[cid, "opstart"])
        for i, w in g.iterrows():
            t0 = int(round(op0 + w.t_rel_min))
            seg = x[max(t0, 0):t0 + 30]
            with np.errstate(all="ignore"):
                vals = np.nanmean(seg, axis=0) if len(seg) else np.full(len(TRK), np.nan)
            rec.append(dict(row=i, **dict(zip(TRK, vals))))
        # reference window
        t0 = int(round(op0 - 300))
        seg = x[max(t0, 0):t0 + 30]
        with np.errstate(all="ignore"):
            vals = np.nanmean(seg, axis=0)
        rec.append(dict(row=-int(cid), **dict(zip(TRK, vals))))
    S = pd.DataFrame(rec)
    S.to_parquet(cache)
ref_state = S[S.row < 0].assign(caseid=lambda s: -s.row).set_index("caseid")
S = S[S.row >= 0].set_index("row").sort_index()
for k in TRK:
    d["s_" + k] = S[k].to_numpy()
    d["ds_" + k] = d["s_" + k] - ref_state.loc[d.caseid, k].to_numpy()
chk = np.nanmax(np.abs(d.s_mbp_check - d.art_mbp))
log(f"alignment check: max |MAP from file - MAP in analysis set| = {chk:.2f} mmHg")
for k in ["phen", "nepi"]:
    d["s_" + k] = d["s_" + k].fillna(0.0)
d["vasopressor_running"] = ((d.s_phen > 0) | (d.s_nepi > 0)).astype(int)
d["volatile"] = (d.s_mac.fillna(0) >= 0.3).astype(int)
d["tiva"] = ((d.s_ppf_ce.fillna(0) >= 0.5) & (d.volatile == 0)).astype(int)
cs = cases.loc[d.caseid]
d["crystalloid_case"] = cs.intraop_crystalloid.to_numpy()
d["colloid_case"] = cs.intraop_colloid.to_numpy()
d["ebl_case"] = cs.intraop_ebl.to_numpy()
d["optype_id"] = pd.Categorical(cs.optype.to_numpy()).codes
avail = {k: float(d["s_" + k].notna().mean()) for k in TRK}
log("track availability (fraction of windows): "
    + ", ".join(f"{k} {v*100:.0f}%" for k, v in avail.items()))


def prep(cols):
    """Identical to run_vitaldb_state.py / run_incremental.py: missing -> 0, then z-score."""
    A = d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    return (A - A.mean(0)) / (A.std(0) + 1e-9)


y = d.hypotension.to_numpy()
grp = d.caseid.to_numpy()
CV = StratifiedGroupKFold(5, shuffle=True, random_state=0)


def cvp(A):
    return cross_val_predict(LogisticRegression(max_iter=5000, C=0.05), A, y, cv=CV,
                             groups=grp, method="predict_proba")[:, 1]


ABS = prep(FEATS)
DEL = prep(["d_" + c for c in FEATS])
HRO = prep(["f_hr_mean"])
P = {"heart rate only": cvp(HRO), "absolute ECG features": cvp(ABS),
     "patient-referenced ECG features": cvp(DEL)}
for k, p in P.items():
    log(f"  reproduced: {k:34s} AUROC {roc_auc_score(y, p):.3f}")

# ------------------------------------------------------------------ 1. incremental
CONF_MS = ["mac", "ppf20_ce", "phen_rate", "nepi_rate", "bt", "age", "sex_n", "t_rel_min",
           "f_hr_mean"]
CONF_EXT = (["s_" + k for k in ["mac", "ppf_ce", "rftn_ce", "bis", "phen", "nepi", "peep", "tv",
                                "vent_rr", "etco2", "mawp", "temp"]]
            + ["ds_" + k for k in ["mac", "ppf_ce", "rftn_ce", "bis", "peep", "tv", "vent_rr",
                                   "etco2", "mawp"]]
            + ["t_from_incision_min", "age", "sex_n", "f_hr_mean", "crystalloid_case",
               "colloid_case", "ebl_case", "optype_id"])
rows = []


def incremental(tag, base_cols):
    B = prep(base_cols)
    pb, pf = cvp(B), cvp(np.hstack([B, DEL]))
    cases_ = np.unique(grp)
    rng = np.random.default_rng(20260802)   # same resamples as run_incremental.py
    bw, bc_ = [], []
    for _ in range(400):
        pick = rng.choice(cases_, len(cases_), replace=True)
        ix = np.concatenate([np.where(grp == c)[0] for c in pick])
        if len(np.unique(y[ix])) < 2:
            continue
        bw.append(roc_auc_score(y[ix], pf[ix]) - roc_auc_score(y[ix], pb[ix]))
        cl = pd.DataFrame(dict(case=grp[ix], y=y[ix], b=pb[ix], f=pf[ix]))
        a = cl.groupby("case").agg(y=("y", "max"), b=("b", "median"), f=("f", "median"))
        if a.y.nunique() > 1:
            bc_.append(roc_auc_score(a.y, a.f) - roc_auc_score(a.y, a.b))
    a0 = pd.DataFrame(dict(c=grp, y=y, b=pb, f=pf)).groupby("c").agg(
        y=("y", "max"), b=("b", "median"), f=("f", "median"))
    for lvl, ab, af, bd in [("window", roc_auc_score(y, pb), roc_auc_score(y, pf), bw),
                            ("operation", roc_auc_score(a0.y, a0.b), roc_auc_score(a0.y, a0.f), bc_)]:
        lo, hi = np.percentile(bd, [2.5, 97.5])
        rows.append(dict(analysis="incremental", baseline=tag, level=lvl, base_auroc=ab,
                         plus_ecg_auroc=af, delta=af - ab, lo=lo, hi=hi, n_boot=len(bd),
                         n_windows=len(y), n_pos=int(y.sum())))
        log(f"  {tag:58s} {lvl:9s} {ab:.3f} -> {af:.3f}  delta {af-ab:+.3f} ({lo:+.3f} to {hi:+.3f})")


log("\n=== 1. incremental value of the patient-referenced ECG ===")
incremental("manuscript set (MAC, propofol, vasopressors, temperature, time, age, sex, HR)",
            CONF_MS)
incremental("extended set (+ remifentanil, BIS, ventilation, their changes, fluids, EBL, "
            "operation type)", CONF_EXT)
incremental("extended concurrent set only (no end-of-case fluid, EBL or operation type)",
            [c for c in CONF_EXT if c not in ("crystalloid_case", "colloid_case", "ebl_case",
                                              "optype_id")])

# ------------------------------------------------------------------ 2. strata
log("\n=== 2. stratified evaluation of the out-of-fold predictions ===")
tfi = d.t_from_incision_min
STRATA = [("volatile maintenance (MAC >= 0.3)", d.volatile == 1),
          ("intravenous maintenance (propofol Ce >= 0.5, no volatile)", d.tiva == 1),
          ("no vasopressor infusion running", d.vasopressor_running == 0),
          ("first 40 min after incision", tfi <= 40),
          ("after 40 min", tfi > 40),
          ("PEEP >= 5 cmH2O", d.s_peep >= 5),
          ("PEEP < 5 cmH2O", d.s_peep < 5),
          ("BIS 40-60", d.s_bis.between(40, 60)),
          ("BIS outside 40-60 or missing", ~d.s_bis.between(40, 60))]
for lab, mk in STRATA:
    mk = mk.fillna(False).to_numpy(bool)
    n_pos, n = int(y[mk].sum()), int(mk.sum())
    r = dict(analysis="stratum", stratum=lab, n_windows=n, n_pos=n_pos,
             n_operations=int(len(np.unique(grp[mk]))))
    for k, p in P.items():
        r[k] = roc_auc_score(y[mk], p[mk]) if 5 <= n_pos < n else np.nan
    rows.append(r)
    log(f"  {lab:58s} n={n:3d} pos={n_pos:2d}  ref {r['patient-referenced ECG features']:.3f}  "
        f"abs {r['absolute ECG features']:.3f}  HR {r['heart rate only']:.3f}")

# ------------------------------------------------------------------ 3. residualised
log("\n=== 3. ECG change with the concurrent anaesthetic/ventilatory change removed ===")
RES_ON = ["ds_" + k for k in ["mac", "ppf_ce", "rftn_ce", "bis", "peep", "tv", "vent_rr",
                              "etco2", "mawp", "temp"]] + ["t_from_incision_min"]
Z = prep(RES_ON)
R = DEL - LinearRegression().fit(Z, DEL).predict(Z)
R = (R - R.mean(0)) / (R.std(0) + 1e-9)
expl = 1 - R.var(0).mean() / DEL.var(0).mean() if False else \
    float(1 - ((DEL - LinearRegression().fit(Z, DEL).predict(Z)).var(0) / DEL.var(0)).mean())
pr = cvp(R)
pz = cvp(Z)
for tag, p in [("patient-referenced ECG, residualised", pr),
               ("the anaesthetic/ventilatory changes alone", pz)]:
    rows.append(dict(analysis="residualised", model=tag, auroc=roc_auc_score(y, p),
                     variance_explained=expl))
    log(f"  {tag:46s} AUROC {roc_auc_score(y, p):.3f}")
log(f"  mean fraction of ECG-change variance explained by those changes: {expl:.3f}")

pd.DataFrame(rows).to_csv(OUT / "vitaldb_confounders.csv", index=False)
d[["caseid", "window", "t_from_incision_min", "hypotension", "volatile", "tiva",
   "vasopressor_running"] + ["s_" + k for k in TRK]].to_csv(
    OUT / "vitaldb_window_state.csv", index=False)
log(f"VITALDB_CONFOUNDERS_COMPLETE {time.time()-t00:.0f}s")
