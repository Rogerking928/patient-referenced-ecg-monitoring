"""Time-to-event evaluation, computed from the saved predictions.

The multi-task head predicts P(decompensation within a fixed horizon).  That is a
binary summary; this script scores the same predictions as a *risk* against the
observed time-to-decompensation with right-censoring, which is the form the
question was actually asked in.

  * Harrell's concordance index over comparable at-risk window pairs
  * time-dependent AUROC at several horizons
  * observed decompensation rate by predicted-risk decile (calibration)
  * lead time: how far ahead of the mean-pressure fall the alarm is available
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hemo import config, train as T
from hemo.config import out

cfg = config.load()
O = lambda *p: out(cfg, *p)

P = np.load(O("predictions_main.npz"))
mte = pd.read_parquet(O("meta_test.parquet")).reset_index(drop=True)
models = sorted({k.split("__")[0] for k in P.files if "__" in k})


def concordance(risk, time, event):
    """Harrell's C over pairs where the ordering of event times is known."""
    risk = np.asarray(risk, float)
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    n = len(risk)
    # pair (i, j) is comparable if i had the event and t_i < t_j
    ti = time[:, None]
    tj = time[None, :]
    comp = event[:, None] & (ti < tj)
    np.fill_diagonal(comp, False)
    if comp.sum() == 0:
        return np.nan, 0
    ri = risk[:, None]
    rj = risk[None, :]
    conc = (ri > rj) & comp
    ties = (ri == rj) & comp
    return float((conc.sum() + 0.5 * ties.sum()) / comp.sum()), int(comp.sum())


at = mte.at_risk.to_numpy().astype(bool)
sub = mte[at].reset_index(drop=True)
rows = []
for m in models:
    risk = P[f"{m}__decomp"][at]
    c, npairs = concordance(risk, sub.tte.to_numpy(), sub.tte_event.to_numpy())
    r = dict(model=m, c_index=c, n_comparable_pairs=npairs, n_at_risk=int(at.sum()))
    for h in [5.0, 10.0, 15.0, 20.0, 30.0]:
        y = ((sub.tte_event == 1) & (sub.tte <= h)).to_numpy().astype(int)
        # exclude windows censored before the horizon: their status at h is unknown
        usable = ~((sub.tte_event == 0) & (sub.tte < h)).to_numpy()
        if y[usable].sum() > 5 and (1 - y[usable]).sum() > 5:
            r[f"auroc_{h:.0f}min"] = roc_auc_score(y[usable], risk[usable])
            r[f"prevalence_{h:.0f}min"] = float(y[usable].mean())
    rows.append(r)
tte = pd.DataFrame(rows)
tte.to_csv(O("tte_results.csv"), index=False)
print(tte.round(3).to_string(index=False))

# ---------------------------------------------------------------- calibration
best = tte.sort_values("c_index", ascending=False).model.iloc[0]
risk = P[f"{best}__decomp"][at]
dec = pd.qcut(risk, 10, labels=False, duplicates="drop")
cal = pd.DataFrame({"decile": dec, "risk": risk,
                    "observed": ((sub.tte_event == 1) & (sub.tte <= cfg.tasks.decomp_horizon_min)).to_numpy()})
calg = cal.groupby("decile").agg(mean_predicted=("risk", "mean"),
                                 observed_rate=("observed", "mean"),
                                 n=("observed", "size")).reset_index()
calg["model"] = best
calg.to_csv(O("tte_calibration.csv"), index=False)
print("\ncalibration (best model =", best, ")")
print(calg.round(3).to_string(index=False))

# ---------------------------------------------------------------- lead time
w = np.load(O("world.npz"))
lab, D, t = w["lab"], w["D"], w["t"]
subj = pd.read_parquet(O("subjects.parquet"))
# the stage labels are read off a trailing 60 s mean, so the pressure-fall time it is
# compared against must be too -- an instantaneous threshold is tripped by the few
# percent of beat-to-beat variation long before any sustained fall
from hemo.physiology import _trailing_mean
kk = max(int(cfg.staging.stage_average_s / cfg.simulation.dt_s), 1)
MAPF = _trailing_mean(D[:, :, 2] / subj.map0.to_numpy()[:, None], kk)
rows = []
for s in mte.subject.unique():
    i = int(np.where(subj.subject.to_numpy() == s)[0][0])
    bad = np.where((lab[i] == 2) | (lab[i] == 3))[0]
    if len(bad) == 0:
        continue
    t_dec = float(t[bad[0]])
    below = np.where(MAPF[i] < 0.90)[0]
    t_map = float(t[below[0]]) if len(below) else np.nan
    shock = np.where(lab[i] == 3)[0]
    t_shock = float(t[shock[0]]) if len(shock) else np.nan
    rows.append(dict(subject=int(s), t_early_decomp=t_dec, t_map_below_90pct=t_map,
                     t_overt_shock=t_shock,
                     lead_over_map=t_map - t_dec if np.isfinite(t_map) else np.nan,
                     lead_over_shock=t_shock - t_dec if np.isfinite(t_shock) else np.nan))
lt = pd.DataFrame(rows)
lt.to_csv(O("lead_time.csv"), index=False)
print(f"\nlead time of the early-decompensation label over a SUSTAINED 10% fall in mean pressure: "
      f"median {lt.lead_over_map.median():.1f} min (IQR "
      f"{lt.lead_over_map.quantile(0.25):.1f} to {lt.lead_over_map.quantile(0.75):.1f}), "
      f"n={lt.lead_over_map.notna().sum()}")
print(f"lead time over overt shock: median {lt.lead_over_shock.median():.1f} min, "
      f"n={lt.lead_over_shock.notna().sum()}")
print("TTE_COMPLETE")
