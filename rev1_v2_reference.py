"""Reviewer 2, major comments 5 and 6 (questions 2): a personal reference without
knowing when deterioration began, in a world where elapsed time is not a shortcut.

The v2 world (configs/v2.yaml, built by run_v2.sh) simulates four hours and observes
each subject through a 90-minute window placed at a random point in it.  Haemorrhage
can start at any time, including before the recording starts; bleed rates are far more
dispersed; a third of spontaneous stops re-bleed.  Nothing in the study design tells
the monitor when a bleed began.

Two references that need no knowledge of onset are compared with the population model:
  first window   the subject's first observed window, whatever state it is in
  rolling        the median of the subject's windows 15 to 30 minutes earlier,
                 updated continuously (the first window until that history exists)
and results are split by whether the first window was already recorded after the
bleed had begun.  The prospective metadata + heart-rate baseline and the incremental
value of the waveform over it are recomputed in this world.
"""
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, roc_auc_score

import rev1_common as RC
from hemo import config, train as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/out_rev1"
os.makedirs(OUT, exist_ok=True)
cfg = config.load(overrides="configs/v2.yaml")
t00 = time.time()
LOG = open(f"{OUT}/v2_reference.log", "w")


def log(m):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


meta = pd.read_parquet(f"{HERE}/out_v2/meta_main.parquet").reset_index(drop=True)
F = pd.read_parquet(f"{HERE}/out_v2/features_main.parquet")
subj = pd.read_parquet(f"{HERE}/out_v2/subjects.parquet").set_index("subject")
A, _, _ = RC.standardise(F, meta)
hr_meas = F.hr_mean.to_numpy()
o = meta.sort_values(["subject", "win"]).index.to_numpy()
meta, A, hr_meas = meta.loc[o].reset_index(drop=True), A[o], hr_meas[o]
step = RC.step_minutes(cfg)
log(f"v2 world: {meta.subject.nunique()} subjects, {len(meta)} windows, {step:.1f} min apart")

# ------------------------------------------------------------------ references
first = meta.groupby("subject").head(1).set_index("subject")
onset_obs = subj.onset_min - subj.obs_start_min            # onset in recording time
contam = pd.Series(False, index=first.index)
b = subj.loc[first.index]
contam[:] = ((b.bleeder == 1) & (onset_obs.loc[first.index] < first.t_min + 0.5)).to_numpy()
not_base = first.stage != 0
meta["ref_after_onset"] = meta.subject.map(contam).astype(int)
meta["ref_not_baseline"] = meta.subject.map(not_base).astype(int)
for sp in ["train", "test"]:
    s = first[first.split == sp]
    log(f"  {sp}: {len(s)} subjects; first window recorded after bleeding began in "
        f"{contam.loc[s.index].sum()} ({contam.loc[s.index].mean()*100:.0f}%), "
        f"first window not in the baseline state in {not_base.loc[s.index].sum()} "
        f"({not_base.loc[s.index].mean()*100:.0f}%)")

ref_first = RC.first_window_reference(A, meta, max_t=None)
ref_roll = np.zeros_like(A)
tmin = meta.t_min.to_numpy()
for s, g in meta.groupby("subject", sort=False).indices.items():
    ix = np.sort(np.asarray(g))
    for k in ix:
        w = ix[(tmin[ix] <= tmin[k] - 15.0) & (tmin[ix] >= tmin[k] - 30.0)]
        ref_roll[k] = np.median(A[w], axis=0) if len(w) else A[ix[0]]

base = RC.representations(A, meta, ref_first, step)
roll = RC.representations(A, meta, ref_roll, step)
REPS = {"population (no reference)": base["population"],
        "first observed window as reference": base["reference"],
        "rolling reference, 15-30 min earlier": roll["reference"],
        "trajectory (absolute + slope + curvature)": base["trajectory"],
        "first-window reference + trajectory": base["reference+trajectory"],
        "rolling reference + trajectory": roll["reference+trajectory"]}

tr, va, te = (np.where(meta.split == s)[0] for s in ["train", "val", "test"])
grp_te = meta.subject.to_numpy()[te]

# ------------------------------------------------------------------ prospective metadata
m = meta.copy()
m["phen_id"] = pd.Categorical(m.phenotype).codes
m["hr_meas"] = hr_meas
META = ["t_min", "phen_id", "age", "sex", "hr0", "map0", "beta_blocked", "hr_meas"]
AM = np.nan_to_num(m[META].to_numpy(np.float32), nan=-99.0)
REPS_INC = {"elapsed time only": m[["t_min"]].to_numpy(np.float32),
            "prospective metadata + heart rate": AM,
            "metadata + heart rate + ECG (first-window reference + trajectory)":
                np.hstack([AM, base["reference+trajectory"]]).astype(np.float32)}

rows, PRED = [], {}
for name, Xf in {**REPS, **REPS_INC}.items():
    r, p = RC.score(Xf, meta, tr, va, te)
    PRED[name] = p
    rows.append(dict(representation=name, subset="all test subjects", **r))
    log(f"{RC.stamp(t00)} {name:66s} state {r['stage_auroc_macro']:.3f}  "
        f"vol {r['volume_r2']:+.3f}  dec {r['decomp_auroc']:.3f}  "
        f"dec5 {r.get('decomp_auroc_5min', np.nan):.3f}")


# ------------------------------------------------------------------ subgroup metrics
def sub_metrics(p, mask_te):
    out = {}
    y = meta.stage.to_numpy()[te]
    out["stage_auroc_macro"] = RC.macro_auc(y[mask_te], p["stage"][mask_te])
    v = meta.volume_lost.to_numpy()[te]
    out["volume_r2"] = float(r2_score(v[mask_te], p["volume"][mask_te]))
    for key, lab in [("decomp", None), ("h5", 5.0)]:
        c, pp = p[key][0], p[key][1]
        keep = np.isin(c, te[mask_te])
        yy = meta.decomp_soon.to_numpy()[c] if lab is None else RC.horizon_label(meta, lab)[0][c]
        nm = "decomp_auroc" if lab is None else "decomp_auroc_5min"
        out[nm] = float(roc_auc_score(yy[keep], pp[keep])) if 0 < yy[keep].sum() < keep.sum() \
            else np.nan
        out["n_events_" + nm] = int(yy[keep].sum())
    return out


for lab, col, val in [("first window after bleed onset", "ref_after_onset", 1),
                      ("first window before bleed onset or no bleed", "ref_after_onset", 0),
                      ("first window not in the baseline state", "ref_not_baseline", 1)]:
    mk = meta[col].to_numpy()[te] == val
    log(f"\n  subset: {lab} ({mk.sum()} windows, "
        f"{meta.subject.to_numpy()[te][mk].tolist() and len(set(meta.subject.to_numpy()[te][mk]))} subjects)")
    for name in REPS:
        r = sub_metrics(PRED[name], mk)
        rows.append(dict(representation=name, subset=lab, **r))
        log(f"    {name:44s} state {r['stage_auroc_macro']:.3f}  vol {r['volume_r2']:+.3f}  "
            f"dec5 {r['decomp_auroc_5min']:.3f} ({r['n_events_decomp_auroc_5min']} events)")

# ------------------------------------------------------------------ paired bootstrap
log("\n  paired subject-cluster bootstrap on the whole test set")
y_te = meta.stage.to_numpy()[te]
v_te = meta.volume_lost.to_numpy()[te]
pos = {s: np.where(grp_te == s)[0] for s in np.unique(grp_te)}


def mk_fn(name, task):
    p = PRED[name]
    if task == "state":
        return lambda ix: RC.macro_auc(y_te[ix], p["stage"][ix])
    if task == "volume":
        return lambda ix: r2_score(v_te[ix], p["volume"][ix])
    key = "h5"
    c, pp = p[key][0], p[key][1]
    yy = RC.horizon_label(meta, 5.0)[0][c]
    where = {k: i for i, k in enumerate(te)}
    loc = np.array([where[k] for k in c])

    pos_h = {}
    for j, l in enumerate(loc):
        pos_h.setdefault(int(l), []).append(j)
    empty = np.zeros(0, int)
    pos_h = {k: np.asarray(v) for k, v in pos_h.items()}

    def f(ix):
        sel = np.concatenate([pos_h.get(int(i), empty) for i in ix])
        return roc_auc_score(yy[sel], pp[sel])
    return f


boot = []
for a, b_, task in [("population (no reference)", "first observed window as reference", "state"),
                    ("population (no reference)", "first observed window as reference", "volume"),
                    ("population (no reference)", "rolling reference, 15-30 min earlier", "state"),
                    ("population (no reference)", "rolling reference, 15-30 min earlier", "volume"),
                    ("population (no reference)", "trajectory (absolute + slope + curvature)", "dec5"),
                    ("first observed window as reference", "first-window reference + trajectory", "dec5"),
                    ("prospective metadata + heart rate",
                     "metadata + heart rate + ECG (first-window reference + trajectory)", "state"),
                    ("prospective metadata + heart rate",
                     "metadata + heart rate + ECG (first-window reference + trajectory)", "volume"),
                    ("prospective metadata + heart rate",
                     "metadata + heart rate + ECG (first-window reference + trajectory)", "dec5")]:
    fa, fb = mk_fn(a, task), mk_fn(b_, task)
    full = np.arange(len(te))
    rng = np.random.default_rng(20261012)
    subs = np.array(list(pos))
    ds = []
    for _ in range(400):
        pick = rng.choice(subs, len(subs), replace=True)
        ix = np.concatenate([pos[s] for s in pick])
        try:
            ds.append(fb(ix) - fa(ix))
        except ValueError:
            pass
    d0 = fb(full) - fa(full)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    boot.append(dict(comparator=a, representation=b_, task=task, delta=float(d0),
                     lo=float(lo), hi=float(hi), n_boot=len(ds)))
    log(f"    {task:6s} {b_[:52]:52s} vs {a[:34]:34s} {d0:+.3f} ({lo:+.3f} to {hi:+.3f})")

pd.DataFrame(rows).to_csv(f"{OUT}/v2_reference.csv", index=False)
pd.DataFrame(boot).to_csv(f"{OUT}/v2_reference_boot.csv", index=False)
log(f"\nV2_REFERENCE_COMPLETE {RC.stamp(t00)}")
