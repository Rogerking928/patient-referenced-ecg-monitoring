"""Calibrate the generator against PTB-XL, and measure the gap before and after.

The gap is measured with the same feature extractor on both sides, so any bias the
delineator has cancels: what is left is a difference in the signals themselves.
Calibration is a fixed-point loop on the generative parameter distributions -- the
synthetic marginal is measured, compared with the real marginal, and the generative
mean is nudged, three or four times until it stops moving.  Nothing is fitted to a
label, so this cannot manufacture a result; it can only make the synthetic ECGs look
more like real ones.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hemo import (config, features as FE, generators as G, physiology as phys,
                  ptbxl_bridge as PB, subjects as SM)
from hemo.config import Cfg

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_calib"
OUT.mkdir(exist_ok=True)
cfg0 = config.load()
FS_R, SEC = 100, 10.0
N_CMP = 1400
t00 = time.time()

# parameters that are adjusted, and where they live in the configuration
KNOBS = [("ecg.p_dur_s", "II_p_dur", 1000.0),
         ("ecg.t_dur_s", "II_t_dur", 1000.0),
         ("ecg.pr_s", "II_pr_int", 1000.0),
         ("ecg.qrs_dur_s", "II_qrs_dur", 1000.0),
         ("ecg.rr_jitter", "pnn20", None),
         ("ecg.rsa_gain", "rmssd", None),
         ("ecg.t_amp_mv", "II_t_amp", 1.0),
         ("ecg.r_amp_mv", "II_r_amp", 1.0)]


def getp(c, path):
    a, b = path.split(".")
    return c[a][b]


def setp(c, path, v):
    a, b = path.split(".")
    c[a][b] = float(v)


def synth_features(c, gen="gaussian", n_subj=110, seed=5):
    """Render a small resting cohort with the current parameters and measure it."""
    s = SM.assign_splits(c, SM.make_cohort(c, seed=seed))
    Z, D, t, bh = phys.simulate_cohort(c, s, seed=seed)
    ts = t * 60.0
    w = int(c.simulation.window_s / c.simulation.dt_s)
    waves = []
    rng = np.random.default_rng(seed)
    for i in range(len(s)):
        k0 = 30                              # a resting window, before any onset
        waves.append(G.render_window(c, s.iloc[i], ts[k0:k0 + w], Z[i, k0:k0 + w],
                                     D[i, k0:k0 + w], rng, generator=gen))
    X = np.stack(waves)
    Xv = PB.synthetic_to_ptbxl_view(c, X, fs_out=FS_R, seconds=SEC)
    Xv = PB.preprocess(Xv, FS_R)
    q = PB.quality_control(Xv, FS_R)
    Xv = Xv[q["pass"].to_numpy()]
    return FE.extract_batch(Xv, PB.matched_cfg(c, FS_R, SEC), n_jobs=8)


# ------------------------------------------------------------------ real target
print("loading PTB-XL reference", flush=True)
Xr, mr = PB.load_ptbxl(cfg0, n=2600, seed=4242, only_normal=True)
Xr = PB.preprocess(Xr, FS_R)
q = PB.quality_control(Xr, FS_R)
Xr = Xr[q["pass"].to_numpy()][:N_CMP]
Fr = FE.extract_batch(Xr, PB.matched_cfg(cfg0, FS_R, SEC), n_jobs=8)
print(f"[{time.time()-t00:.0f}s] real reference: {Fr.shape}", flush=True)
target = Fr.median(numeric_only=True)

rows = []


def evaluate(c, gen, tag):
    Fs = synth_features(c, gen=gen)
    smd = PB.standardised_mean_difference(Fr, Fs)
    auc, _ = PB.domain_discriminability(Fr.fillna(Fr.median()), Fs.fillna(Fr.median()))
    r = dict(stage=tag, generator=gen, n_synth=len(Fs),
             median_abs_smd=float(smd.smd.abs().median()),
             frac_smd_gt_0p5=float((smd.smd.abs() > 0.5).mean()),
             domain_auroc=auc)
    for _, key, _sc in KNOBS:
        r["real_" + key] = float(target.get(key, np.nan))
        r["synth_" + key] = float(Fs[key].median()) if key in Fs else np.nan
    rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "calibration.csv", index=False)
    print(f"[{time.time()-t00:6.0f}s] {tag:22s} {gen:10s} median|SMD| "
          f"{r['median_abs_smd']:.2f}  >0.5SD {r['frac_smd_gt_0p5']*100:4.0f}%  "
          f"domain AUROC {auc:.3f}", flush=True)
    return Fs, smd


# ------------------------------------------------------------------ before
for gen in ["gaussian", "dynamical", "empirical"]:
    evaluate(cfg0, gen, "before calibration")

# ------------------------------------------------------------------ calibrate
c = Cfg(copy.deepcopy(dict(cfg0)))
hist = []
for it in range(4):
    Fs = synth_features(c, gen="gaussian")
    step = {}
    for path, key, _sc in KNOBS:
        if key not in Fs or not np.isfinite(target.get(key, np.nan)):
            continue
        s_med, r_med = float(Fs[key].median()), float(target[key])
        if not np.isfinite(s_med) or abs(s_med) < 1e-9:
            continue
        ratio = np.clip(r_med / s_med, 0.6, 1.7)
        cur = getp(c, path)
        new = float(np.clip(cur * (1 + 0.75 * (ratio - 1)),
                            0.25 * getp(cfg0, path), 4.0 * getp(cfg0, path)))
        setp(c, path, new)
        step[path] = dict(measured=s_med, target=r_med, ratio=float(ratio),
                          value=new)
    hist.append(dict(iteration=it, **{k: v["value"] for k, v in step.items()}))
    print(f"[{time.time()-t00:6.0f}s] calibration pass {it+1}: "
          + ", ".join(f"{k.split('.')[-1]} {v['measured']:.1f}->{v['target']:.1f}"
                      for k, v in list(step.items())[:5]), flush=True)
pd.DataFrame(hist).to_csv(OUT / "calibration_path.csv", index=False)
with open(OUT / "calibrated_params.json", "w") as fh:
    json.dump({p: getp(c, p) for p, _, _ in KNOBS}, fh, indent=2)
# write it out as an override file so any other script can use the calibrated world
ov = {"ecg": {}}
for p, _, _ in KNOBS:
    ov["ecg"][p.split(".")[1]] = getp(c, p)
with open(HERE / "configs" / "calibrated.yaml", "w") as fh:
    fh.write("# Generator parameters fitted to PTB-XL marginals by run_calibration.py.\n"
             "# No outcome label is involved; only the resting morphology distributions.\n")
    yaml.safe_dump(ov, fh)

# ------------------------------------------------------------------ after
for gen in ["gaussian", "dynamical", "empirical"]:
    evaluate(c, gen, "after calibration")

d = pd.DataFrame(rows)
b = d[d.stage == "before calibration"].set_index("generator")
a = d[d.stage == "after calibration"].set_index("generator")
print("\n=== summary ===")
for g in b.index:
    print(f"  {g:10s} median|SMD| {b.loc[g,'median_abs_smd']:.2f} -> "
          f"{a.loc[g,'median_abs_smd']:.2f}   domain AUROC "
          f"{b.loc[g,'domain_auroc']:.3f} -> {a.loc[g,'domain_auroc']:.3f}")
print("CALIBRATION_COMPLETE")
