"""Reviewer 2, major comment 11 and minor comments 14.6 and 14.7.

1. QT correction.  Bazett's formula over-corrects at fast rates, and haemorrhage
   spans a wide rate range.  The QTc features are recomputed with Fridericia and
   with the Framingham linear correction (the uncorrected QT and RR are both stored,
   so the substitution is exact), and with every QTc removed; the four
   representations are rescored.  The residual rate dependence of each correction
   is reported as its correlation with heart rate.
2. Lead configuration.  Lead II only (the configuration of the external cohort),
   V5 only, and both.
3. A fresh test cohort.  The held-out subjects of the main world were used for
   every analysis in the manuscript.  A new cohort of 200 subjects is simulated with a
   seed that has never been used, the models are trained on the original training
   subjects exactly as before, and the headline comparisons are evaluated on it once.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

import rev1_common as RC
from hemo import config, train as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/out_rev1"
cfg = config.load()
STEP = RC.step_minutes(cfg)
t00 = time.time()
LOG = open(f"{OUT}/qtc_leads_fresh.log", "w")


def log(m):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


meta0 = pd.read_parquet(f"{HERE}/out/meta_main.parquet").reset_index(drop=True)
F0 = pd.read_parquet(f"{HERE}/out/features_main.parquet")


def variant(F):
    A, _, _ = RC.standardise(F, meta0)
    meta, A = RC.sort_by_subject(meta0, A)
    ref = RC.first_window_reference(A, meta, max_t=7.6)
    return meta, RC.representations(A, meta, ref, STEP)


def qtc_variants(F):
    out = {}
    for name in ["bazett (manuscript)", "fridericia", "framingham", "no QTc"]:
        G = F.copy()
        for ld in ["II", "V5"]:
            qt, qtc = G[f"{ld}_qt"], G[f"{ld}_qtc"]
            rr = (qt / qtc) ** 2                       # the RR the extractor used, in s
            if name == "fridericia":
                G[f"{ld}_qtc"] = qt / np.cbrt(rr)
            elif name == "framingham":
                G[f"{ld}_qtc"] = qt + 154.0 * (1 - rr)
            elif name == "no QTc":
                G = G.drop(columns=[f"{ld}_qtc"])
        out[name] = G
    return out


rows = []
log("=== 1. QT correction ===")
base_w = meta0.stage.to_numpy() == 0
for name, G in qtc_variants(F0).items():
    if "II_qtc" in G:
        rho = np.corrcoef(G.loc[base_w, "II_qtc"].fillna(G.II_qtc.median()),
                          G.loc[base_w, "hr_mean"].fillna(G.hr_mean.median()))[0, 1]
        rho_all = np.corrcoef(G["II_qtc"].fillna(G.II_qtc.median()),
                              G["hr_mean"].fillna(G.hr_mean.median()))[0, 1]
    else:
        rho = rho_all = np.nan
    meta, reps = variant(G)
    tr, va, te = (np.where(meta.split == s)[0] for s in ["train", "val", "test"])
    for rep in ["population", "reference", "trajectory"]:
        r, _ = RC.score(reps[rep], meta, tr, va, te)
        rows.append(dict(analysis="QT correction", variant=name, representation=rep,
                         qtc_hr_corr_baseline=rho, qtc_hr_corr_all=rho_all, **r))
        log(f"{RC.stamp(t00)} {name:20s} {rep:12s} r(QTc,HR) baseline {rho:+.2f} all {rho_all:+.2f}"
            f"  state {r['stage_auroc_macro']:.3f} vol {r['volume_r2']:+.3f} "
            f"dec {r['decomp_auroc']:.3f} dec5 {r['decomp_auroc_5min']:.3f}")
    pd.DataFrame(rows).to_csv(f"{OUT}/qtc_leads.csv", index=False)

log("\n=== 2. lead configuration ===")
shared = [c for c in F0.columns if not c.startswith(("II_", "V5_"))]
for name, cols in [("leads II and V5 (manuscript)", list(F0.columns)),
                   ("lead II only", shared + [c for c in F0.columns if c.startswith("II_")]),
                   ("lead V5 only", shared + [c for c in F0.columns if c.startswith("V5_")])]:
    meta, reps = variant(F0[cols])
    tr, va, te = (np.where(meta.split == s)[0] for s in ["train", "val", "test"])
    for rep in ["population", "reference", "trajectory"]:
        r, _ = RC.score(reps[rep], meta, tr, va, te)
        rows.append(dict(analysis="lead configuration", variant=name, representation=rep,
                         n_features=len(cols), **r))
        log(f"{RC.stamp(t00)} {name:30s} {rep:12s} state {r['stage_auroc_macro']:.3f} "
            f"vol {r['volume_r2']:+.3f} dec {r['decomp_auroc']:.3f} "
            f"dec5 {r['decomp_auroc_5min']:.3f}")
    pd.DataFrame(rows).to_csv(f"{OUT}/qtc_leads.csv", index=False)

# ------------------------------------------------------------------ fresh test cohort
log("\n=== 3. a fresh, never-used test cohort ===")
c = RC.deep(cfg)
c["cohort"]["n_subjects"] = 200
FRESH_SEED = 20261012
mf, Ff, _ = RC.build(c, seed=FRESH_SEED)
mf = mf.copy()
mf["subject"] = mf.subject + 100000
mf["split"] = "fresh"
log(f"{RC.stamp(t00)} fresh cohort: {mf.subject.nunique()} subjects, {len(mf)} windows, "
    f"seed {FRESH_SEED}")
tr0 = np.where(meta0.split == "train")[0]
A0, mu, sd = T.prep_features(F0, tr0)
A0 = np.clip((A0 - mu) / sd, -8, 8)
Af = Ff[F0.columns].to_numpy(np.float32).copy()
med = np.nanmedian(F0.to_numpy(np.float32)[tr0], axis=0)
ix = np.where(~np.isfinite(Af))
Af[ix] = np.take(med, ix[1])
Af = np.clip((Af - mu) / sd, -8, 8)
m0, A0s = RC.sort_by_subject(meta0, A0)
m1, A1s = RC.sort_by_subject(mf, Af)
keep0 = m0.split.isin(["train", "val"]).to_numpy()
meta = pd.concat([m0[keep0], m1], ignore_index=True)
A = np.vstack([A0s[keep0], A1s])
ref = RC.first_window_reference(A, meta, max_t=7.6)
reps = RC.representations(A, meta, ref, STEP)
tr, va, te = (np.where(meta.split == s)[0] for s in ["train", "val", "fresh"])
fresh_rows, PRED = [], {}
for rep in ["population", "reference", "trajectory", "reference+trajectory"]:
    r, p = RC.score(reps[rep], meta, tr, va, te)
    PRED[rep] = p
    fresh_rows.append(dict(representation=rep, n_subjects=int(meta.iloc[te].subject.nunique()),
                           n_windows=len(te), **r))
    log(f"{RC.stamp(t00)} fresh | {rep:22s} state {r['stage_auroc_macro']:.3f} "
        f"vol {r['volume_r2']:+.3f} dec {r['decomp_auroc']:.3f} dec5 {r['decomp_auroc_5min']:.3f}")

g = meta.subject.to_numpy()[te]
y = meta.stage.to_numpy()[te]
v = meta.volume_lost.to_numpy()[te]
yy = RC.horizon_label(meta, 5.0)[0]
pos = {s: np.where(g == s)[0] for s in np.unique(g)}


def f_state(rep):
    return lambda ix: RC.macro_auc(y[ix], PRED[rep]["stage"][ix])


def f_vol(rep):
    return lambda ix: r2_score(v[ix], PRED[rep]["volume"][ix])


def f_dec5(rep):
    c_, p_ = PRED[rep]["h5"][0], PRED[rep]["h5"][1]
    where = {k: i for i, k in enumerate(te)}
    loc = np.array([where[k] for k in c_])
    m_ = {}
    for j, l in enumerate(loc):
        m_.setdefault(int(l), []).append(j)
    m_ = {k: np.asarray(w) for k, w in m_.items()}
    e = np.zeros(0, int)
    return lambda ix: roc_auc_score(yy[c_][np.concatenate([m_.get(int(i), e) for i in ix])],
                                    p_[np.concatenate([m_.get(int(i), e) for i in ix])])


boot = []
rng = np.random.default_rng(20261013)
subs = np.array(list(pos))
picks = [np.concatenate([pos[s] for s in rng.choice(subs, len(subs), replace=True)])
         for _ in range(400)]
full = np.arange(len(te))
for task, fa, fb, a, b in [
        ("state", f_state("population"), f_state("reference"), "population", "reference"),
        ("volume", f_vol("population"), f_vol("reference"), "population", "reference"),
        ("dec5", f_dec5("population"), f_dec5("reference"), "population", "reference"),
        ("dec5", f_dec5("population"), f_dec5("trajectory"), "population", "trajectory"),
        ("volume", f_vol("population"), f_vol("trajectory"), "population", "trajectory")]:
    ds = []
    for ix in picks:
        try:
            ds.append(fb(ix) - fa(ix))
        except ValueError:
            pass
    d0 = fb(full) - fa(full)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    boot.append(dict(task=task, comparator=a, representation=b, delta=float(d0),
                     lo=float(lo), hi=float(hi), n_boot=len(ds)))
    log(f"    fresh | {task:6s} {b} vs {a}: {d0:+.3f} ({lo:+.3f} to {hi:+.3f})")
pd.DataFrame(fresh_rows).to_csv(f"{OUT}/fresh_test.csv", index=False)
pd.DataFrame(boot).to_csv(f"{OUT}/fresh_test_boot.csv", index=False)
log(f"QTC_LEADS_FRESH_COMPLETE {RC.stamp(t00)}")
