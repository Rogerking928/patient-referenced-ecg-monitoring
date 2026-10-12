"""Stage 3: the control battery.

  C1  patient-level split audit
  C2  negative control -- ECG sample order shuffled
  C3  negative control -- randomised labels
  C4  heart-rate ablation (features removed / waveform rate-normalised)
  C5  RR series only vs full morphology
  C6  noise robustness (test-time distribution shift)
  C7  medication / confounder perturbation (re-simulated physiology)
  C8  leave-one-phenotype-out
  C9  subject-specific baseline normalisation vs population model
"""
import time
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from hemo import (config, experiments as E, features as FE, models as M,
                  physiology as phys, simulate, train as T, views as V)
from hemo.config import out

torch.set_num_threads(14)
cfg = config.load()
log = E.log_to(out(cfg, "run_controls.log"))
t00 = time.time()
ctx = E.Ctx(cfg)
rows = []


def record(exp, label, pred, meta, extra=None, boot=True):
    sc = E.eval_pred(pred, meta, cfg, label, extra, boot=boot)
    sc["experiment"] = exp
    rows.append(sc)
    pd.DataFrame(rows).to_csv(out(cfg, "results_controls.csv"), index=False)
    log(f"[{time.time()-t00:6.0f}s] {exp:22s} {label:26s} stageAUC {sc['stage_auroc_macro']:.3f}  "
        f"bACC {sc['stage_bacc']:.3f}  volR2 {sc['volume_r2']:+.3f}  "
        f"decAUC {sc['decomp_auroc']:.3f}  resR2 {sc['reserve_r2']:+.3f}  n={sc['n']}")


# =================================================================== C1 split audit
log("\n=== C1  patient-level split audit ===")
g = ctx.meta.groupby("subject").split.nunique()
ov = int((g > 1).sum())
sub_split = ctx.meta.groupby("split").subject.nunique().to_dict()
log(f"    subjects appearing in >1 split: {ov}   subjects per split: {sub_split}")
pd.DataFrame([{"experiment": "C1_split_audit", "subjects_in_multiple_splits": ov,
               **{f"n_subjects_{k}": v for k, v in sub_split.items()},
               "n_windows_train": len(ctx.tr), "n_windows_val": len(ctx.va),
               "n_windows_test": len(ctx.te)}]).to_csv(out(cfg, "c1_split_audit.csv"), index=False)
assert ov == 0

# =================================================================== C2 shuffled ECG
log("\n=== C2  negative control: ECG sample order shuffled ===")
Xsh = V.ShuffledView(ctx.X, seed=11)
Fsh = FE.extract_batch(Xsh, cfg)
Ash, mush, sdsh = T.prep_features(Fsh, ctx.tr)
_, p, mte = E.run_boosted(ctx, A_override=Ash, label="shuffled")
record("C2_shuffled_ecg", "handcrafted_xgb_shuffled", p, mte)

net = M.WaveNet(len(cfg.simulation.leads), cfg)
net, p, mte = E.run_torch(ctx, net, X_override=Xsh, log=log, epochs=8)
record("C2_shuffled_ecg", "raw_waveform_cnn_shuffled", p, mte)
del net

# =================================================================== C3 random labels
log("\n=== C3  negative control: randomised labels ===")
rng = np.random.default_rng(cfg.run.seed + 99)
meta_rand = ctx.meta.copy()
# permute labels ACROSS subjects but keep each subject's own label block intact,
# so the control retains the within-subject correlation structure
subs = meta_rand.subject.unique()
perm = rng.permutation(subs)
mapping = dict(zip(subs, perm))
cols = ["stage", "volume_lost", "reserve", "decomp_soon", "at_risk", "tte", "tte_event"]
src = meta_rand.set_index(["subject", "win"])[cols]
new = []
for s, w in zip(meta_rand.subject, meta_rand.win):
    key = (mapping[s], w)
    new.append(src.loc[key] if key in src.index else src.loc[(s, w)])
meta_rand[cols] = pd.DataFrame(new).to_numpy()
meta_rand["stage"] = meta_rand["stage"].astype(int)
for c in ["decomp_soon", "at_risk", "tte_event"]:
    meta_rand[c] = meta_rand[c].astype(int)

_, p, mte = E.run_boosted(ctx, meta_override=meta_rand, label="randlab")
record("C3_random_labels", "handcrafted_xgb_randomlabel", p, mte)
net = M.WaveNet(len(cfg.simulation.leads), cfg)
net, p, mte = E.run_torch(ctx, net, meta_override=meta_rand, log=log, epochs=8)
record("C3_random_labels", "raw_waveform_cnn_randomlabel", p, mte)
del net

# =================================================================== C4/C5 rate ablation
log("\n=== C4  rate-normalised waveform (heart rate removed from the signal) ===")
fs = cfg.simulation.fs
path_rn = out(cfg, "X_ratenorm.npy")
if not path_rn.exists():
    Xrn = np.lib.format.open_memmap(path_rn, mode="w+", dtype=np.float32, shape=ctx.X.shape)
    chunks = np.array_split(np.arange(len(ctx.X)), 64)
    Xsrc = ctx.X                      # bind the memmap only, not the whole context

    def work(ix):
        return ix, np.stack([E.rate_normalise(np.asarray(Xsrc[k]), fs) for k in ix])
    for ix, arr in Parallel(n_jobs=cfg.run.n_jobs, verbose=0)(delayed(work)(c) for c in chunks):
        Xrn[ix] = arr
    Xrn.flush()
    del Xrn
Xrn = np.load(path_rn, mmap_mode="r")
log(f"    rate-normalised waveform cached at {path_rn}")

net = M.WaveNet(len(cfg.simulation.leads), cfg)
net, p, mte = E.run_torch(ctx, net, X_override=Xrn, log=log)
record("C4_rate_ablation", "raw_waveform_cnn_rate_removed", p, mte)
del net

# feature-side counterparts (fast)
for label, keep, drop in [("morphology_only_no_rr", None, E.HRV_ONLY),
                          ("rr_series_only", E.HRV_ONLY, None),
                          ("no_rate_features", None, E.RATE_FEATURES)]:
    cols_i, names = ctx.feat_cols(keep=keep, drop=drop)
    _, p, mte = E.run_boosted(ctx, cols=cols_i, label=label)
    record("C5_rr_vs_morphology", "handcrafted_" + label, p, mte, {"n_features": len(names)})

# =================================================================== C6 noise robustness
log("\n=== C6  noise robustness (test-time only, models unchanged) ===")
w = np.load(out(cfg, "world.npz"))
Z, D, tt, lab, viab = w["Z"], w["D"], w["t"], w["lab"], w["viab"]
te_sub = np.where(ctx.subj.split == "test")[0]

import joblib
b_all = joblib.load(out(cfg, "model_handcrafted.joblib"))
cnn = M.WaveNet(len(cfg.simulation.leads), cfg)
cnn.load_state_dict(torch.load(out(cfg, "model_cnn.pt")))
hyb = M.HybridNet(len(cfg.simulation.leads), ctx.A.shape[1], cfg)
hyb.load_state_dict(torch.load(out(cfg, "model_hybrid.pt")))


def eval_external(Xe, Fe, me, tag, extra=None):
    Ae = Fe.to_numpy(np.float32).copy()
    ix = np.where(~np.isfinite(Ae))
    med = np.nanmedian(ctx.A[ctx.tr], axis=0)
    Ae[ix] = np.take(med, ix[1])
    record(tag, "handcrafted_xgb", T.predict_boosted(b_all, Ae), me, extra)
    ds = T.WinDataset(Xe, Ae, me, np.arange(len(me)), ctx.mu, ctx.sd, ctx.wave_scale)
    record(tag, "raw_waveform_cnn", T.predict_dataset(cnn, ds), me, extra)
    record(tag, "hybrid", T.predict_dataset(hyb, ds), me, extra)


for k in cfg.controls.noise_multipliers:
    Xe, me = simulate.build_dataset(cfg, ctx.subj, Z, D, tt, lab, viab, tag=f"noise{k}",
                                    noise_scale=float(k), seed_offset=int(1000 * k),
                                    subject_idx=te_sub, save=False)
    Fe = FE.extract_batch(Xe, cfg)
    eval_external(Xe, Fe, me, "C6_noise", {"noise_multiplier": float(k)})
    del Xe, Fe

# =================================================================== C7 medication perturbation
log("\n=== C7  medication / confounder perturbation (physiology re-simulated) ===")
for tag, mut in [
    ("all_beta_blocked", dict(beta_blocked=1, bb_strength=0.65)),
    ("none_beta_blocked", dict(beta_blocked=0, bb_strength=0.0)),
    ("all_vasodilated", dict(on_vasodilator=1)),
]:
    sp = ctx.subj.copy()
    for k, v in mut.items():
        sp[k] = v
    if "on_vasodilator" in mut:
        sp["map0"] = sp["map0"] * 0.94
    Zp, Dp, tp, bhp = phys.simulate_cohort(cfg, sp)
    lp = phys.stage_labels(cfg, sp, Zp, Dp, tp, bhp)
    vp = phys.viability_mask(cfg, Dp)
    Xe, me = simulate.build_dataset(cfg, sp, Zp, Dp, tp, lp, vp, tag=tag,
                                    subject_idx=te_sub, save=False)
    Fe = FE.extract_batch(Xe, cfg)
    dist = me.stage.value_counts(normalize=True).sort_index().round(3).to_dict()
    log(f"    {tag}: stage distribution {dist}")
    eval_external(Xe, Fe, me, "C7_medication", {"perturbation": tag})
    del Xe, Fe, Zp, Dp

# =================================================================== C8 leave-one-phenotype-out
log("\n=== C8  leave-one-phenotype-out ===")
rng2 = np.random.default_rng(cfg.run.seed + 5)
for held in ctx.subj.phenotype.unique():
    is_held = ctx.meta.phenotype.values == held
    te_i = np.where(is_held)[0]
    pool = ctx.meta.subject.values[~is_held]
    us = np.unique(pool)
    rng2.shuffle(us)
    va_s = set(us[: max(int(0.2 * len(us)), 1)])
    tr_i = np.where((~is_held) & ~ctx.meta.subject.isin(va_s).values)[0]
    va_i = np.where((~is_held) & ctx.meta.subject.isin(va_s).values)[0]
    ex = {"held_out_phenotype": held, "n_test_subjects": int(ctx.meta.iloc[te_i].subject.nunique())}

    _, p, mte = E.run_boosted(ctx, tr=tr_i, va=va_i, te=te_i, label=f"lopo_{held}")
    record("C8_leave_phenotype_out", f"handcrafted_xgb__{held}", p, mte, ex)
    net = M.WaveNet(len(cfg.simulation.leads), cfg)
    net, p, mte = E.run_torch(ctx, net, tr=tr_i, va=va_i, te=te_i, log=log, epochs=10)
    record("C8_leave_phenotype_out", f"raw_waveform_cnn__{held}", p, mte, ex)
    del net

# =================================================================== C9 personal baseline
log("\n=== C9  subject-specific baseline normalisation vs population model ===")
Dlt = E.subject_baseline_delta(ctx)
_, p, mte = E.run_boosted(ctx, A_override=Dlt, label="delta")
record("C9_personal_baseline", "handcrafted_xgb_delta_only", p, mte)
Both = np.concatenate([ctx.A, Dlt], axis=1)
_, p, mte = E.run_boosted(ctx, A_override=Both, label="both")
record("C9_personal_baseline", "handcrafted_xgb_absolute_plus_delta", p, mte,
       {"n_features": Both.shape[1]})
np.save(out(cfg, "delta_features.npy"), Dlt)

log(f"\nCONTROLS_COMPLETE  {time.time()-t00:.0f}s")
