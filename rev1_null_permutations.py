"""How far a single full label permutation strays from chance.

Supplementary Table S4 reports one full permutation per model; the raw-waveform
network landed below chance (decompensation 0.443) and the handcrafted model above it
(0.550), each with a subject-bootstrap interval excluding 0.5.  That interval resamples
test subjects for one fixed permutation and one fitted model, so it is not the null
distribution.  Repeat the permutation with new seeds to measure the real spread.

    python3 rev1_null_permutations.py [n_xgb] [n_cnn]
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

from hemo import config, experiments as E, models as M
from hemo.config import out

N_XGB = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_CNN = int(sys.argv[2]) if len(sys.argv) > 2 else 8
cfg = config.load()
cfg.run.n_jobs = 8
torch.set_num_threads(8)
log = E.log_to(out(cfg, "..", "out_rev1", "null_permutations.log"))
ctx = E.Ctx(cfg)
dst = out(cfg, "..", "out_rev1", "null_permutations.csv")
COLS = ["stage", "volume_lost", "reserve", "decomp_soon", "at_risk", "tte", "tte_event"]
rows, t0 = [], time.time()


def permuted(seed):
    rng = np.random.default_rng(cfg.run.seed + 4242 + 1000 * seed)
    m = ctx.meta.copy()
    p = rng.permutation(len(m))
    for c in COLS:
        m[c] = ctx.meta[c].to_numpy()[p]
    return m


def keep(model, seed, sc):
    rows.append(dict(model=model, seed=seed, **{k: sc[k] for k in
                ["stage_auroc_macro", "volume_r2", "decomp_auroc", "reserve_r2", "n"]}))
    pd.DataFrame(rows).to_csv(dst, index=False)
    log(f"[{time.time()-t0:6.0f}s] {model:18s} seed {seed:2d} stage {sc['stage_auroc_macro']:.3f} "
        f"vol {sc['volume_r2']:+.3f} dec {sc['decomp_auroc']:.3f} res {sc['reserve_r2']:+.3f}")


for s in range(N_XGB):
    _, p, mte = E.run_boosted(ctx, meta_override=permuted(s), label=f"null{s}")
    keep("handcrafted_xgb", s, E.eval_pred(p, mte, cfg, "null", {}))
for s in range(N_CNN):
    net = M.WaveNet(len(cfg.simulation.leads), cfg)
    net, p, mte = E.run_torch(ctx, net, meta_override=permuted(s), log=log, epochs=8)
    keep("raw_waveform_cnn", s, E.eval_pred(p, mte, cfg, "null", {}))
    del net
log("NULL_COMPLETE")
