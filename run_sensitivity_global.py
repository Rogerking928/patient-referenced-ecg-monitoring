"""Global sensitivity analysis over the physiological assumptions.

Every result in this project is conditional on a set of coupling coefficients that
were chosen by hand.  A single SHAP figure says which features a model used; it says
nothing about how much of the plausible parameter space the conclusion survives in.

A Latin hypercube is drawn over the couplings that carry the argument, each sample is
a complete world -- cohort, trajectories, waveforms, features, model -- and the answer
reported is the fraction of the sampled space in which the pre-declared criteria hold,
plus the first-order effect of each parameter.

Two knock-out arms are included: the preload pathway removed, and everything except
the autonomic pathway removed.
"""
import copy
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from scipy.stats import qmc, spearmanr
from sklearn.metrics import r2_score, roc_auc_score

from hemo import (config, features as FE, generators as G, physiology as phys,
                  simulate, subjects as SM)
from hemo.config import Cfg

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_sens"
OUT.mkdir(exist_ok=True)
N_SAMPLES = int(os.environ.get("SENS_N", 48))
N_SUBJ = int(os.environ.get("SENS_SUBJ", 150))
t00 = time.time()

BASE = config.load(overrides="configs/v2.yaml")
BASE = Cfg(copy.deepcopy(dict(BASE)))
BASE["cohort"]["n_subjects"] = N_SUBJ
BASE["run"]["n_jobs"] = 1                       # the outer loop is parallel instead

# parameter, path, multiplicative range around the default
KNOBS = [
    ("preload -> QRS amplitude", "ecg.coupling.qrs_amp_preload", (0.15, 2.2)),
    ("preload -> P amplitude", "ecg.coupling.p_amp_preload", (0.15, 2.2)),
    ("respiration -> R amplitude", "ecg.ramp_resp_gain", (0.15, 2.5)),
    ("metabolic stress -> ST/T", "ecg.coupling.t_amp_metab", (0.2, 2.0)),
    ("sympathetic -> heart rate", "physiology.hr_symp_gain", (0.5, 1.7)),
    ("sympathetic -> QT variability", "ecg.coupling.qtv_symp", (0.2, 2.5)),
    ("baroreflex gain", "physiology.baro_gain", (0.5, 1.8)),
    ("reserve depletion (withdrawal depth)", "physiology.withdraw_depth", (0.4, 1.3)),
    ("blood volume -> stroke volume (Starling)", "physiology.starling_hill", (0.6, 1.6)),
    ("beta-blocker response", "physiology.hr_symp_gain", None),      # coupled below
    ("recording noise", "ecg.emg_mv_range", (0.4, 3.0)),
    ("between-subject morphology spread", "ecg.rsa_gain", (0.4, 2.2)),
]
KNOBS = [k for k in KNOBS if k[2] is not None]
CRIT = dict(reserve_r2=0.50, decomp_auroc=0.75, volume_r2=0.40)


def setp(c, path, v):
    parts = path.split(".")
    node = c
    for p in parts[:-1]:
        node = node[p]
    cur = node[parts[-1]]
    node[parts[-1]] = [float(x) * v for x in cur] if isinstance(cur, list) else float(cur) * v


def one_world(vals, seed, knockout=None):
    c = Cfg(copy.deepcopy(dict(BASE)))
    c["run"]["seed"] = 30000 + seed
    for (name, path, _), v in zip(KNOBS, vals):
        setp(c, path, v)
    if knockout == "no preload pathway":
        c["ecg"]["coupling"]["qrs_amp_preload"] = 0.0
        c["ecg"]["coupling"]["p_amp_preload"] = 0.0
        c["ecg"]["ramp_resp_gain"] = 0.0
    elif knockout == "autonomic pathway only":
        c["ecg"]["coupling"]["qrs_amp_preload"] = 0.0
        c["ecg"]["coupling"]["p_amp_preload"] = 0.0
        c["ecg"]["ramp_resp_gain"] = 0.0
        c["ecg"]["coupling"]["t_amp_metab"] = 0.0
        c["ecg"]["coupling"]["qrs_dur_metab"] = 0.0
        c["ecg"]["coupling"]["qt_metab"] = 0.0
        c["ecg"]["coupling"]["st_metab"] = 0.0
    try:
        subj, Z, D, t, lab, viab = simulate.run_world(c)
        X, meta = simulate.build_dataset(c, subj, Z, D, t, lab, viab, save=False)
        if len(meta) < 900 or meta.stage.nunique() < 3:
            return None
        F = FE.extract_batch(X, c, n_jobs=1)
        A = F.to_numpy(np.float32)
        med = np.nanmedian(A, axis=0)
        ix = np.where(~np.isfinite(A))
        A[ix] = np.take(med, ix[1])
        tr = np.where(meta.split != "test")[0]
        te = np.where(meta.split == "test")[0]
        p = dict(n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.85,
                 colsample_bytree=0.8, n_jobs=1, tree_method="hist", random_state=0)
        r = dict(seed=seed, knockout=knockout or "none", n_windows=len(meta))
        for task, col in [("volume", "volume_lost"), ("reserve", "reserve")]:
            reg = xgb.XGBRegressor(objective="reg:squarederror", **p)
            reg.fit(A[tr], meta[col].to_numpy()[tr])
            r[f"{task}_r2"] = float(r2_score(meta[col].to_numpy()[te], reg.predict(A[te])))
        at = meta.at_risk.to_numpy().astype(bool)
        yd = meta.decomp_soon.to_numpy()
        itr, ite = tr[at[tr]], te[at[te]]
        if yd[itr].sum() > 8 and yd[ite].sum() > 4:
            c2 = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                   scale_pos_weight=float((1 - yd[itr]).sum() / max(yd[itr].sum(), 1)),
                                   **p)
            c2.fit(A[itr], yd[itr])
            r["decomp_auroc"] = float(roc_auc_score(yd[ite], c2.predict_proba(A[ite])[:, 1]))
        else:
            r["decomp_auroc"] = np.nan
        y = meta.stage.to_numpy()
        cnt = np.bincount(y[tr], minlength=5).astype(float)
        wg = (cnt.sum() / (5 * np.maximum(cnt, 1))) ** 0.5
        clf = xgb.XGBClassifier(objective="multi:softprob", num_class=5,
                                eval_metric="mlogloss", **p)
        clf.fit(A[tr], y[tr], sample_weight=wg[y[tr]])
        P = clf.predict_proba(A[te])
        r["stage_auroc_macro"] = float(np.mean([roc_auc_score((y[te] == k).astype(int), P[:, k])
                                                for k in range(5) if (y[te] == k).sum() > 3]))
        for (name, _, _), v in zip(KNOBS, vals):
            r["x_" + name] = float(v)
        r["passes"] = int(all(np.isfinite(r.get(k, np.nan)) and r[k] >= v
                              for k, v in CRIT.items()))
        return r
    except Exception as e:
        return dict(seed=seed, knockout=knockout or "none", error=str(e)[:120])


samp = qmc.LatinHypercube(d=len(KNOBS), seed=11).random(N_SAMPLES)
lo = np.array([k[2][0] for k in KNOBS])
hi = np.array([k[2][1] for k in KNOBS])
VALS = lo + samp * (hi - lo)
print(f"{N_SAMPLES} parameter sets x {N_SUBJ} subjects over {len(KNOBS)} couplings",
      flush=True)

jobs = [delayed(one_world)(VALS[i], i) for i in range(N_SAMPLES)]
jobs += [delayed(one_world)(np.ones(len(KNOBS)), 900 + j, k)
         for j, k in enumerate(["none", "no preload pathway", "autonomic pathway only"])]
res = Parallel(n_jobs=12, verbose=5)(jobs)
res = [r for r in res if r is not None]
df = pd.DataFrame(res)
df.to_csv(OUT / "sensitivity_samples.csv", index=False)

ok = df[(df.knockout == "none") & df.get("passes", pd.Series(dtype=float)).notna()]
print(f"\n[{time.time()-t00:.0f}s] {len(ok)} usable parameter sets", flush=True)
summ = []
for k, v in CRIT.items():
    frac = float((ok[k] >= v).mean())
    summ.append(dict(criterion=f"{k} >= {v}", fraction_of_parameter_space=frac,
                     median=float(ok[k].median()),
                     p10=float(ok[k].quantile(0.1)), p90=float(ok[k].quantile(0.9))))
    print(f"    {k:20s} >= {v:.2f} in {frac*100:.0f}% of the sampled space "
          f"(median {ok[k].median():.3f}, 10-90th {ok[k].quantile(.1):.3f}-{ok[k].quantile(.9):.3f})")
summ.append(dict(criterion="all three simultaneously",
                 fraction_of_parameter_space=float(ok.passes.mean()),
                 median=np.nan, p10=np.nan, p90=np.nan))
print(f"    all three simultaneously: {ok.passes.mean()*100:.0f}%")
pd.DataFrame(summ).to_csv(OUT / "sensitivity_summary.csv", index=False)

eff = []
for name, _, _ in KNOBS:
    col = "x_" + name
    if col not in ok:
        continue
    for tgt in ["reserve_r2", "decomp_auroc", "volume_r2", "stage_auroc_macro"]:
        m = ok[[col, tgt]].dropna()
        if len(m) < 10:
            continue
        eff.append(dict(parameter=name, outcome=tgt,
                        spearman=float(spearmanr(m[col], m[tgt]).statistic),
                        p=float(spearmanr(m[col], m[tgt]).pvalue)))
eff = pd.DataFrame(eff)
eff.to_csv(OUT / "sensitivity_effects.csv", index=False)
print("\n  strongest first-order effects:")
for _, r in eff.reindex(eff.spearman.abs().sort_values(ascending=False).index).head(8).iterrows():
    print(f"    {r.parameter:42s} -> {r.outcome:18s} rho {r.spearman:+.2f}")

ko = df[df.knockout != "none"]
if len(ko):
    print("\n  pathway knock-outs (default parameters):")
    for _, r in ko.iterrows():
        print(f"    {r.knockout:26s} reserve R2 {r.get('reserve_r2', np.nan):+.3f}  "
              f"volume R2 {r.get('volume_r2', np.nan):+.3f}  "
              f"decomp AUROC {r.get('decomp_auroc', np.nan):.3f}")
    ko.to_csv(OUT / "sensitivity_knockouts.csv", index=False)
print("SENSITIVITY_COMPLETE")
