"""Cross-generator validation.

The same 560 synthetic subjects, the same latent trajectories, the same per-beat
morphology parameters, rendered three ways:

  A  gaussian   sums of Gaussian waves            (the original generator)
  B  dynamical  a McSharry-style limit-cycle ODE
  C  empirical  real PTB-XL median beats, warped to the requested intervals

A model trained on one and tested on another cannot be reading that generator's
private syntax.  The train/test matrix is A->A, A->B, A->C, B->B, C->C, A+B->C and
A+B+C->all, on the handcrafted and the raw-waveform families.

Success criteria, fixed before running (as proposed): after a generator change the
model should still reach state macro AUROC > 0.70, reserve R2 > 0.40 and imminent
decompensation AUROC > 0.70.
"""
import itertools
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from hemo import (config, experiments as E, features as FE, generators as G,
                  models as M, simulate, subjects as SM, physiology as phys, train as T)

os.environ.setdefault("HEMO_CFG", "configs/xgen.yaml")
torch.set_num_threads(14)
cfg = config.load()
HERE = Path(__file__).resolve().parent
OUT = Path(cfg.run.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
O = lambda p: OUT / p
log = E.log_to(O("run_cross_generator.log"))
GENS = ["gaussian", "dynamical", "empirical"]
t00 = time.time()

CRIT = dict(stage_auroc_macro=0.70, reserve_r2=0.40, decomp_auroc=0.70)


# =================================================================== build
def render_subject(cfg, row, t_s, Zi, Di, starts, n_samp, seed, gen):
    rng = np.random.default_rng(seed)
    w = int(cfg.simulation.window_s / cfg.simulation.dt_s)
    xs = np.empty((len(starts), len(cfg.simulation.leads), n_samp), np.float32)
    for j, k0 in enumerate(starts):
        xs[j] = G.render_window(cfg, row, t_s[k0:k0 + w], Zi[k0:k0 + w], Di[k0:k0 + w],
                                rng, generator=gen)
    return xs


if not O("meta.parquet").exists():
    log("=== building one world, rendering it three ways ===")
    subj, Z, D, t, lab, viab = simulate.run_world(cfg)
    subj.to_parquet(O("subjects.parquet"))
    wlen = int(cfg.simulation.window_s / cfg.simulation.dt_s)
    n_samp = int(round(cfg.simulation.window_s * cfg.simulation.fs))
    tte, event = phys.time_to_decomp(cfg, lab, t)
    hz = cfg.tasks.decomp_horizon_min
    rows, keep = [], []
    for i in range(len(subj)):
        off = float(subj.obs_start_min.iloc[i])
        wt = simulate.window_times(cfg, off)
        st = np.searchsorted(t, wt)
        mid = np.clip(st + wlen // 2, 0, len(t) - 1)
        for j, (k0, km) in enumerate(zip(st, mid)):
            if not viab[i, k0:k0 + wlen].all():
                continue
            cur = int(lab[i, km])
            rows.append(dict(subject=int(subj.subject.iloc[i]), row_i=i, win=j,
                             t_min=float(wt[j] - off), stage=cur,
                             volume_lost=float(D[i, km, 6]), reserve=float(D[i, km, 4]),
                             hr_true=float(D[i, km, 0]), preload=float(D[i, km, 7]),
                             map_frac=float(D[i, km, 2] / subj.map0.iloc[i]),
                             tte=float(tte[i, km]), tte_event=int(event[i, km]),
                             decomp_soon=int((cur < 2 or cur == 4) and event[i, km] == 1
                                             and tte[i, km] <= hz),
                             at_risk=int(cur < 2 or cur == 4)))
            keep.append((i, k0))
    meta = pd.DataFrame(rows).merge(
        subj[["subject", "split", "phenotype", "age", "sex", "hr0", "map0"]],
        on="subject", how="left")
    kp = pd.DataFrame(keep, columns=["row_i", "k0"])
    meta.to_parquet(O("meta.parquet"))
    log(f"    {len(meta)} windows, states "
        + str(meta.stage.value_counts(normalize=True).round(3).sort_index().to_dict()))
    ts = t * 60.0
    for gen in GENS:
        if O(f"X_{gen}.npy").exists():
            continue
        t0 = time.time()
        jobs = [delayed(render_subject)(cfg, subj.iloc[i], ts, Z[i], D[i],
                                        kp.loc[kp.row_i == i, "k0"].to_numpy(), n_samp,
                                        cfg.run.seed + 900 + i, gen)
                for i in range(len(subj))]
        chunks = Parallel(n_jobs=cfg.run.n_jobs)(jobs)
        X = np.concatenate([c for c in chunks if len(c)]).astype(np.float32)
        assert len(X) == len(meta)
        np.save(O(f"X_{gen}.npy"), X)
        F = FE.extract_batch(X, cfg)
        F.to_parquet(O(f"features_{gen}.parquet"))
        log(f"    {gen:10s} rendered {X.shape} + features in {time.time()-t0:.0f}s")
        del X, F

meta = pd.read_parquet(O("meta.parquet"))
Xs = {g: np.load(O(f"X_{g}.npy"), mmap_mode="r") for g in GENS}
Fs = {g: pd.read_parquet(O(f"features_{g}.parquet")) for g in GENS}
names = list(Fs["gaussian"].columns)
tr = np.where(meta.split == "train")[0]
va = np.where(meta.split == "val")[0]
te = np.where(meta.split == "test")[0]
log(f"windows train/val/test = {len(tr)}/{len(va)}/{len(te)}")

# how different are the three generators, measured the same way as the reality gap
from hemo.ptbxl_bridge import domain_discriminability
gaps = []
for a, b in itertools.combinations(GENS, 2):
    auc, _ = domain_discriminability(Fs[a].fillna(Fs[a].median()),
                                     Fs[b].fillna(Fs[a].median()))
    gaps.append(dict(a=a, b=b, discriminability_auroc=auc))
    log(f"    {a} vs {b}: a classifier separates them at AUROC {auc:.3f}")
pd.DataFrame(gaps).to_csv(O("generator_gaps.csv"), index=False)

# =================================================================== matrix
A = {}
med = np.nanmedian(Fs["gaussian"].to_numpy(np.float32)[tr], axis=0)
for g in GENS:
    a = Fs[g].to_numpy(np.float32).copy()
    ix = np.where(~np.isfinite(a))
    a[ix] = np.take(med, ix[1])
    A[g] = a
mu = np.concatenate([A[g][tr] for g in GENS]).mean(0)
sd = np.concatenate([A[g][tr] for g in GENS]).std(0) + 1e-6

TRAIN_SETS = [("gaussian",), ("dynamical",), ("empirical",),
              ("gaussian", "dynamical"), tuple(GENS)]
rows = []


def record(family, train_on, test_on, pred, m, extra=None):
    s = T.score_all(pred, m)
    s.update(family=family, train_on="+".join(train_on), test_on=test_on,
             transfer=("within" if test_on in train_on else "across"))
    if extra:
        s.update(extra)
    s["passes_criteria"] = int(all(s.get(k, -9) >= v for k, v in CRIT.items()))
    rows.append(s)
    pd.DataFrame(rows).to_csv(O("cross_generator.csv"), index=False)
    log(f"[{time.time()-t00:6.0f}s] {family:12s} train {'+'.join(train_on):24s} -> "
        f"test {test_on:10s} stageAUC {s['stage_auroc_macro']:.3f} volR2 "
        f"{s['volume_r2']:+.3f} decAUC {s['decomp_auroc']:.3f} resR2 "
        f"{s['reserve_r2']:+.3f} {'PASS' if s['passes_criteria'] else 'fail'}")


log("\n=== handcrafted features + gradient boosting ===")
for ts_ in TRAIN_SETS:
    Atr = np.concatenate([A[g][tr] for g in ts_])
    Ava = np.concatenate([A[g][va] for g in ts_])
    mtr = pd.concat([meta.iloc[tr]] * len(ts_), ignore_index=True)
    mva = pd.concat([meta.iloc[va]] * len(ts_), ignore_index=True)
    b = T.fit_boosted(cfg, Atr, mtr, Ava, mva, None, seed=0)
    for g in GENS:
        record("handcrafted", ts_, g, T.predict_boosted(b, A[g][te]), meta.iloc[te])

log("\n=== raw waveform (1D ResNet) ===")


class MultiGenDS(torch.utils.data.Dataset):
    def __init__(self, gens, idx, scale):
        self.gens, self.idx, self.scale = gens, np.asarray(idx), scale
        self.m = pd.concat([meta.iloc[self.idx]] * len(gens), ignore_index=True)
        self.n = len(self.idx)
        self.y_stage = self.m.stage.to_numpy(np.int64)
        self.y_vol = self.m.volume_lost.to_numpy(np.float32)
        self.y_dec = self.m.decomp_soon.to_numpy(np.float32)
        self.y_res = self.m.reserve.to_numpy(np.float32)
        self.mask_dec = self.m.at_risk.to_numpy(np.float32)

    def __len__(self):
        return self.n * len(self.gens)

    def __getitem__(self, i):
        g = self.gens[i // self.n]
        k = self.idx[i % self.n]
        x = np.asarray(Xs[g][k], np.float32) / self.scale
        return (torch.from_numpy(x), torch.zeros(1), self.y_stage[i],
                self.y_vol[i], self.y_dec[i], self.y_res[i], self.mask_dec[i])


scale = float(np.abs(np.asarray(Xs["gaussian"][tr[:400]])).std())
for ts_ in TRAIN_SETS:
    torch.manual_seed(cfg.run.seed % 2 ** 31)
    net = M.WaveNet(len(cfg.simulation.leads), cfg)
    net = T.fit_torch(net, cfg, MultiGenDS(ts_, tr, scale), MultiGenDS(ts_, va, scale),
                      log=log)
    for g in GENS:
        pred = T.predict_dataset(net, MultiGenDS((g,), te, scale))
        record("raw_waveform", ts_, g, pred, meta.iloc[te])
    del net

log(f"\nCROSS_GENERATOR_COMPLETE  {time.time()-t00:.0f}s")
