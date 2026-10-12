"""Reviewer 2, major comment 1 and question 1: how much of the central result is
written into the observation model?

Two parts.

A.  Alternative observation models, one structural change at a time, each a
    complete new world (cohort, trajectories, waveforms, features, models):
      - QRS and P amplitude only weakly dependent on preload (couplings x0.25)
      - no preload pathway at all (QRS, P and respiratory-modulation couplings = 0)
      - respiratory QRS modulation set by ventilation, independent of preload
      - amplitude dominated by electrode position: an independent gain per lead
        and per window (log-normal, CV 0.25), applied to every window including
        the reference
      - autonomic-to-morphology couplings with doubled between-subject spread
        and a random sign for every subject
      - a myocardial-disease subgroup (30%): preload-to-QRS coupling reversed and
        halved, QRS widened by 30 ms, cardiac reserve x0.7
B.  A Latin hypercube over the population means of the latent-to-waveform
    couplings (x0.25 to x2) and over their between-subject spread (x0.5 to x2).

For every world the four representations of the manuscript are scored with the
same model settings: population, personal reference, trajectory, and both.
The reported quantities are the gains that carry the central claim.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr

import rev1_common as RC
from hemo import config

OUT = RC.__file__.rsplit("/", 1)[0] + "/out_rev1"
os.makedirs(OUT, exist_ok=True)
N_SUBJ = int(os.environ.get("ALT_SUBJ", 600))
N_LHS = int(os.environ.get("ALT_LHS", 24))
LHS_SUBJ = int(os.environ.get("ALT_LHS_SUBJ", 400))
t00 = time.time()
BASE = config.load()


def log(m):
    print(m, flush=True)
    with open(f"{OUT}/alt_observation.log", "a") as fh:
        fh.write(m + "\n")


def scale_cpl(c, key, k):
    c["ecg"]["coupling"][key] = float(c["ecg"]["coupling"][key]) * k


def electrode_gain(X, meta, cv=0.25, seed=5):
    rng = np.random.default_rng(seed)
    s = np.sqrt(np.log1p(cv ** 2))
    g = rng.lognormal(-0.5 * s ** 2, s, size=(X.shape[0], X.shape[1], 1)).astype(np.float32)
    X *= g
    return X


def autonomic_spread(subj, seed=6):
    rng = np.random.default_rng(seed)
    for k in ["p_amp_symp", "qrs_amp_symp", "t_amp_symp", "p_dur_symp", "pr_symp"]:
        col = "c_" + k
        mu = float(BASE.ecg.coupling[k])
        subj[col] = rng.choice([-1.0, 1.0], len(subj)) * np.abs(
            rng.normal(mu, 2 * (abs(mu) * 1.15 + 0.05), len(subj)))
    return subj


def myocardial(subj, seed=7):
    rng = np.random.default_rng(seed)
    m = rng.random(len(subj)) < 0.30
    subj.loc[m, "c_qrs_amp_preload"] *= -0.5
    subj.loc[m, "qrs0"] = np.clip(subj.loc[m, "qrs0"] + 0.030, 0.062, 0.18)
    subj.loc[m, "cardiac_reserve"] *= 0.7
    subj["myocardial_disease"] = m.astype(int)
    return subj


def world(tag, cfg, subj_hook=None, x_hook=None, n_subj=N_SUBJ, extra=None):
    c = RC.deep(cfg)
    c["cohort"]["n_subjects"] = n_subj
    t0 = time.time()
    meta, F, _ = RC.build(c, subj_hook=subj_hook, x_hook=x_hook)
    A, _, _ = RC.standardise(F, meta)
    meta, A = RC.sort_by_subject(meta, A)
    ref = RC.first_window_reference(A, meta, max_t=7.6)      # pre-onset, as in the paper
    reps = RC.representations(A, meta, ref, RC.step_minutes(c))
    tr, va, te = (np.where(meta.split == s)[0] for s in ["train", "val", "test"])
    row = dict(scenario=tag, n_subjects=n_subj, n_windows=len(meta),
               n_test_windows=len(te), **(extra or {}))
    for name, Xf in reps.items():
        r, _ = RC.score(Xf, meta, tr, va, te)
        for k, v in r.items():
            row[f"{name}:{k}"] = v
    for k in ["stage_auroc_macro", "volume_r2", "decomp_auroc_5min"]:
        p = row[f"population:{k}"]
        row[f"gain_reference:{k}"] = row[f"reference:{k}"] - p
        row[f"gain_trajectory:{k}"] = row[f"trajectory:{k}"] - p
    row["division_of_labour"] = int(
        row["gain_reference:volume_r2"] > row["gain_trajectory:volume_r2"]
        and row["gain_trajectory:decomp_auroc_5min"] > row["gain_reference:decomp_auroc_5min"])
    log(f"{RC.stamp(t00)} {tag:58s} state {row['population:stage_auroc_macro']:.3f}->"
        f"{row['reference:stage_auroc_macro']:.3f}  vol {row['population:volume_r2']:+.3f}->"
        f"{row['reference:volume_r2']:+.3f}/{row['trajectory:volume_r2']:+.3f}  "
        f"dec5 {row['population:decomp_auroc_5min']:.3f}->"
        f"{row['reference:decomp_auroc_5min']:.3f}/{row['trajectory:decomp_auroc_5min']:.3f}"
        f"  DoL={row['division_of_labour']}  ({time.time()-t0:.0f}s)")
    return row


def cfg_with(fn):
    c = RC.deep(BASE)
    fn(c)
    return c


SCEN = [
    ("default observation model", BASE, None, None),
    ("QRS and P amplitude weakly preload-dependent (x0.25)",
     cfg_with(lambda c: (scale_cpl(c, "qrs_amp_preload", 0.25),
                         scale_cpl(c, "p_amp_preload", 0.25))), None, None),
    ("no preload pathway (QRS, P, respiratory modulation)",
     cfg_with(lambda c: (scale_cpl(c, "qrs_amp_preload", 0.0),
                         scale_cpl(c, "p_amp_preload", 0.0),
                         c["ecg"].__setitem__("ramp_preload_slope", 0.0))), None, None),
    ("respiratory modulation set by ventilation, not preload",
     cfg_with(lambda c: (c["ecg"].__setitem__("ramp_preload_slope", 0.0),
                         c["ecg"].__setitem__("ramp_base", 0.80))), None, None),
    ("amplitude dominated by electrode position (gain CV 0.25 per window)",
     BASE, None, electrode_gain),
    ("autonomic-morphology couplings: doubled spread, random sign",
     BASE, autonomic_spread, None),
    ("myocardial-disease subgroup (30%)", BASE, myocardial, None),
]

rows = []
part = sys.argv[1] if len(sys.argv) > 1 else "all"
if part in ("all", "A"):
    log(f"=== A. alternative observation models, {N_SUBJ} subjects each ===")
    done = {}
    if os.path.exists(f"{OUT}/alt_observation.csv"):           # resume after an interruption
        for r in pd.read_csv(f"{OUT}/alt_observation.csv").to_dict("records"):
            done[r["scenario"]] = r
    for tag, c, sh, xh in SCEN:
        if tag in done:
            rows.append(done[tag])
            log(f"   (kept from an earlier run) {tag}")
            continue
        rows.append(world(tag, c, sh, xh))
        pd.DataFrame(rows).to_csv(f"{OUT}/alt_observation.csv", index=False)

if part in ("all", "B"):
    KN = [("preload -> QRS amplitude", "qrs_amp_preload"),
          ("preload -> P amplitude", "p_amp_preload"),
          ("respiratory QRS modulation", "ramp_resp_gain"),
          ("metabolic stress -> T amplitude", "t_amp_metab"),
          ("metabolic stress -> QRS duration", "qrs_dur_metab"),
          ("metabolic stress -> QT", "qt_metab"),
          ("between-subject coupling spread", None)]
    U = qmc.LatinHypercube(d=len(KN), seed=11).random(N_LHS)
    lo = np.array([0.25] * 6 + [0.5])
    hi = np.array([2.0] * 6 + [2.0])
    V = lo + U * (hi - lo)
    log(f"\n=== B. Latin hypercube, {N_LHS} worlds x {LHS_SUBJ} subjects ===")
    lhs = []
    prev = {}
    if os.path.exists(f"{OUT}/alt_observation_lhs.csv"):
        for r in pd.read_csv(f"{OUT}/alt_observation_lhs.csv").to_dict("records"):
            prev[r["scenario"]] = r
    for i, v in enumerate(V):
        if f"LHS {i+1:02d}" in prev:
            lhs.append(prev[f"LHS {i+1:02d}"])
            continue
        def fn(c, v=v):
            for (name, key), k in zip(KN[:6], v[:6]):
                if key == "ramp_resp_gain":
                    c["ecg"]["ramp_resp_gain"] = float(c["ecg"]["ramp_resp_gain"]) * k
                else:
                    scale_cpl(c, key, k)
        spread = float(v[6])

        def sh(subj, spread=spread, c=None):
            for key, mu in BASE.ecg.coupling.items():
                col = "c_" + key
                if col not in subj or key == "qt_rate_alpha":
                    continue
                if float(mu) > 0 and (subj[col] > 0).all():
                    base_mu = subj[col].mean()
                    subj[col] = base_mu * (subj[col] / base_mu) ** spread
                else:
                    subj[col] = subj[col].mean() + spread * (subj[col] - subj[col].mean())
            return subj
        c = cfg_with(fn)
        extra = {"x_" + n: float(k) for (n, _), k in zip(KN, v)}
        try:
            lhs.append(world(f"LHS {i+1:02d}", c, sh, None, n_subj=LHS_SUBJ, extra=extra))
        except Exception as e:                                   # noqa: BLE001
            log(f"LHS {i+1:02d} failed: {e}")
        pd.DataFrame(lhs).to_csv(f"{OUT}/alt_observation_lhs.csv", index=False)
    d = pd.DataFrame(lhs)
    log(f"\nDivision of labour holds in {d.division_of_labour.mean()*100:.0f}% "
        f"of {len(d)} sampled worlds")
    for g in ["gain_reference:stage_auroc_macro", "gain_reference:volume_r2",
              "gain_trajectory:decomp_auroc_5min"]:
        log(f"  {g:40s} median {d[g].median():+.3f}  range {d[g].min():+.3f} to {d[g].max():+.3f}"
            f"  >0 in {(d[g] > 0).mean()*100:.0f}%")
    eff = []
    for n, _ in KN:
        for g in ["gain_reference:volume_r2", "gain_trajectory:decomp_auroc_5min",
                  "reference:stage_auroc_macro"]:
            rho = spearmanr(d["x_" + n], d[g])
            eff.append(dict(parameter=n, quantity=g, spearman=float(rho.statistic),
                            p=float(rho.pvalue)))
    pd.DataFrame(eff).to_csv(f"{OUT}/alt_observation_lhs_effects.csv", index=False)
log("ALT_OBSERVATION_COMPLETE")
