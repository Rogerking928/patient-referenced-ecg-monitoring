"""Experiment definitions: the three-model comparison and the control battery."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import features as FE, models as M, physiology as phys, simulate, train as T
from .config import out as out_path

RATE_FEATURES = ["hr_mean", "hr_sd", "rr_mean"]
HRV_ONLY = FE.HRV_KEYS                       # everything derived from the RR series
QC_KEYS = FE.RESP_KEYS


def log_to(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg):
        print(msg, flush=True)
        with open(p, "a") as fh:
            fh.write(str(msg) + "\n")
    return _log


# --------------------------------------------------------------------- context
class Ctx:
    """Everything the experiments share: data, splits, standardisation."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.X, self.meta = simulate.load(cfg)
        self.F = pd.read_parquet(out_path(cfg, "features_main.parquet"))
        self.names = list(self.F.columns)
        self.subj = pd.read_parquet(out_path(cfg, "subjects.parquet"))
        self.tr = np.where(self.meta.split == "train")[0]
        self.va = np.where(self.meta.split == "val")[0]
        self.te = np.where(self.meta.split == "test")[0]
        self.A, self.mu, self.sd = T.prep_features(self.F, self.tr)
        self.wave_scale = float(np.abs(np.asarray(self.X[self.tr[:600]])).std())
        # sanity: no subject may appear in two splits
        g = self.meta.groupby("subject").split.nunique()
        assert g.max() == 1, "patient-level split violated"

    def ds(self, idx, **kw):
        return T.WinDataset(self.X, self.A, self.meta, idx, self.mu, self.sd,
                            self.wave_scale, **kw)

    def feat_cols(self, keep=None, drop=None):
        cols = list(self.names)
        if keep is not None:
            cols = [c for c in cols if c in keep]
        if drop is not None:
            cols = [c for c in cols if c not in drop]
        return [self.names.index(c) for c in cols], cols


# --------------------------------------------------------------------- helpers
def eval_pred(pred, meta, cfg, label, extra=None, boot=True):
    sc = T.score_all(pred, meta)
    if boot:
        ci = T.bootstrap_ci(pred, meta, n_boot=cfg.controls.bootstrap_n)
        for k, (lo, hi) in ci.items():
            sc[k + "_lo"], sc[k + "_hi"] = lo, hi
    sc["model"] = label
    if extra:
        sc.update(extra)
    return sc


def run_boosted(ctx, cols=None, tr=None, va=None, te=None, meta_override=None,
                A_override=None, seed=0, label="boosted"):
    cfg = ctx.cfg
    A = ctx.A if A_override is None else A_override
    meta = ctx.meta if meta_override is None else meta_override
    tr = ctx.tr if tr is None else tr
    va = ctx.va if va is None else va
    te = ctx.te if te is None else te
    ci = np.arange(A.shape[1]) if cols is None else np.asarray(cols)
    b = T.fit_boosted(cfg, A[np.ix_(tr, ci)], meta.iloc[tr], A[np.ix_(va, ci)], meta.iloc[va],
                      None, seed=seed)
    pred = T.predict_boosted(b, A[np.ix_(te, ci)])
    return b, pred, meta.iloc[te]


def run_torch(ctx, net, tr=None, va=None, te=None, epochs=None, log=None, seed=0,
              shuffle_wave=False, meta_override=None, X_override=None, A_override=None):
    cfg = ctx.cfg
    tr = ctx.tr if tr is None else tr
    va = ctx.va if va is None else va
    te = ctx.te if te is None else te
    X = ctx.X if X_override is None else X_override
    A = ctx.A if A_override is None else A_override
    meta = ctx.meta if meta_override is None else meta_override
    mk = lambda ix, sh: T.WinDataset(X, A, meta, ix, ctx.mu, ctx.sd, ctx.wave_scale,
                                     shuffle_wave=sh, rng_seed=seed)
    net = T.fit_torch(net, cfg, mk(tr, shuffle_wave), mk(va, shuffle_wave),
                      epochs=epochs, log=log, seed=seed)
    pred = T.predict_dataset(net, mk(te, shuffle_wave))
    return net, pred, meta.iloc[te]


# --------------------------------------------------------------------- rate stripping
def rate_normalise(x, fs, target_rr=0.8, n_beats=None):
    """Resample every beat to a fixed cycle length.

    Morphology is preserved; instantaneous rate and its variability are removed.
    Used for the "does the model only read heart rate?" ablation.
    """
    from scipy.signal import resample
    lead0 = np.asarray(x[0], float)
    pk = FE.detect_rpeaks(lead0, fs)
    L = x.shape[1]
    tgt = int(round(target_rr * fs))
    if len(pk) < 4:
        return np.asarray(x, np.float32)
    out = np.zeros_like(np.asarray(x, np.float32))
    pos = 0
    for a, b in zip(pk[:-1], pk[1:]):
        if b - a < int(0.2 * fs):
            continue
        seg = np.asarray(x[:, a:b], float)
        r = resample(seg, tgt, axis=1)
        n = min(tgt, L - pos)
        if n <= 0:
            break
        out[:, pos:pos + n] = r[:, :n]
        pos += n
    if pos < L:                     # tile to fill
        out[:, pos:] = out[:, max(pos - tgt, 0):max(pos - tgt, 0) + (L - pos)]
    return out


class MappedView:
    """Lazy element-wise transform of a big array, for the ablation datasets."""

    def __init__(self, X, fn):
        self.X, self.fn = X, fn
        self.shape = X.shape

    def __len__(self):
        return len(self.X)

    def __getitem__(self, k):
        if isinstance(k, (int, np.integer)):
            return self.fn(np.asarray(self.X[k], np.float32))
        return np.stack([self.fn(np.asarray(self.X[i], np.float32)) for i in np.atleast_1d(k)])


class ShuffledView(MappedView):
    def __init__(self, X, seed=0):
        rng = np.random.default_rng(seed)
        perms = {}

        def fn(x):
            n = x.shape[1]
            if n not in perms:
                perms[n] = rng.permutation(n)
            return x[:, perms[n]]
        super().__init__(X, fn)


# --------------------------------------------------------------------- baseline normalisation
def subject_baseline_delta(ctx, baseline_max_min=7.6):
    """Feature matrix expressed as a change from each subject's own early windows.

    The early windows precede every haemorrhage onset in the configuration, so
    this is a legitimate personal reference, not leakage of the outcome.
    """
    A = ctx.A.copy()
    meta = ctx.meta
    base_mask = meta.t_min.to_numpy() <= baseline_max_min
    D = np.zeros_like(A)
    for s, g in meta.groupby("subject").groups.items():
        ix = np.asarray(g)
        b = ix[base_mask[ix]]
        ref = A[b].mean(0) if len(b) else A[ix].mean(0)
        D[ix] = A[ix] - ref
    return D
