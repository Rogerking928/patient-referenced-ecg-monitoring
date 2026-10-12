"""Training loops and evaluation metrics for the four prediction tasks."""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, r2_score, roc_auc_score)
from scipy.stats import spearmanr

from . import models as M

TASKS = ["stage", "volume", "decomp", "reserve"]


# --------------------------------------------------------------------- data
class WinDataset(torch.utils.data.Dataset):
    def __init__(self, X, F, meta, idx, feat_mu=None, feat_sd=None, wave_scale=None,
                 shuffle_wave=False, rng_seed=0):
        self.X, self.F, self.idx = X, F, np.asarray(idx)
        self.m = meta.iloc[self.idx].reset_index(drop=True)
        self.feat_mu, self.feat_sd = feat_mu, feat_sd
        self.wave_scale = wave_scale
        self.shuffle_wave = shuffle_wave
        self.rng = np.random.default_rng(rng_seed)
        self.y_stage = self.m.stage.to_numpy(np.int64)
        self.y_vol = self.m.volume_lost.to_numpy(np.float32)
        self.y_dec = self.m.decomp_soon.to_numpy(np.float32)
        self.y_res = self.m.reserve.to_numpy(np.float32)
        self.mask_dec = self.m.at_risk.to_numpy(np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k = self.idx[i]
        x = np.asarray(self.X[k], np.float32)
        if self.shuffle_wave:
            perm = self.rng.permutation(x.shape[1])
            x = x[:, perm]
        if self.wave_scale is not None:
            x = x / self.wave_scale
        f = np.zeros(1, np.float32)
        if self.F is not None:
            f = (self.F[k] - self.feat_mu) / self.feat_sd
            f = np.clip(f, -8, 8).astype(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(f), self.y_stage[i],
                self.y_vol[i], self.y_dec[i], self.y_res[i], self.mask_dec[i])


def prep_features(F, tr_idx):
    """Median-impute and standardise using training rows only."""
    A = F.to_numpy(np.float32).copy()
    med = np.nanmedian(A[tr_idx], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    ix = np.where(~np.isfinite(A))
    A[ix] = np.take(med, ix[1])
    mu = A[tr_idx].mean(0)
    sd = A[tr_idx].std(0) + 1e-6
    return A, mu, sd


# --------------------------------------------------------------------- loss
def multitask_loss(out, batch, w, class_w):
    _, _, ys, yv, yd, yr, md = batch
    ce = nn.functional.cross_entropy(out["stage"], ys, weight=class_w)
    mse_v = nn.functional.mse_loss(out["volume"], yv)
    mse_r = nn.functional.mse_loss(out["reserve"], yr)
    bce = nn.functional.binary_cross_entropy_with_logits(out["decomp"], yd, reduction="none")
    bce = (bce * md).sum() / (md.sum() + 1e-6)
    return w.stage * ce + w.volume * mse_v + w.decomp * bce + w.reserve * mse_r


# --------------------------------------------------------------------- loop
def fit_torch(model, cfg, ds_tr, ds_va, epochs=None, log=None, seed=0, lr=None):
    torch.manual_seed(seed)
    c = cfg.models.cnn
    epochs = epochs or c.epochs
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=c.batch_size, shuffle=True,
                                        num_workers=0, drop_last=True)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=128, shuffle=False)

    cnt = np.bincount(ds_tr.y_stage, minlength=M.N_STAGES).astype(float)
    class_w = torch.tensor((cnt.sum() / (M.N_STAGES * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=lr or c.lr, weight_decay=c.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=(lr or c.lr) * 3,
                                                total_steps=epochs * max(len(dl_tr), 1),
                                                pct_start=0.3)
    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for b in dl_tr:
            opt.zero_grad()
            out = model(b[0], b[1])
            loss = multitask_loss(out, b, cfg.models.loss_weights, class_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            tot += loss.item()
        pred = predict_torch(model, dl_va)
        sc = score_all(pred, ds_va.m)
        # Composite criterion across all four tasks: selecting on the stage
        # AUROC alone discards checkpoints that are far better at the
        # regression heads, and with a one-cycle schedule the useful epochs
        # are the annealing ones at the end.
        crit = float(np.mean([sc["stage_auroc_macro"],
                              np.clip(sc["volume_r2"], 0, 1),
                              np.nan_to_num(sc["decomp_auroc"], nan=0.5),
                              np.clip(sc["reserve_r2"], 0, 1)]))
        msg = (f"    ep{ep+1:02d} loss {tot/max(len(dl_tr),1):.4f} "
               f"val stageAUC {sc['stage_auroc_macro']:.3f} volR2 {sc['volume_r2']:.3f} "
               f"decAUC {sc['decomp_auroc']:.3f} resR2 {sc['reserve_r2']:.3f} "
               f"crit {crit:.4f} ({time.time()-t0:.0f}s)")
        if log:
            log(msg)
        if crit > best + 1e-4:
            best, best_state, bad = crit, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= c.patience and ep + 1 >= int(0.75 * epochs):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_torch(model, dl):
    model.eval()
    P = {k: [] for k in TASKS}
    for b in dl:
        o = model(b[0], b[1])
        P["stage"].append(torch.softmax(o["stage"], 1).numpy())
        P["volume"].append(o["volume"].numpy())
        P["decomp"].append(torch.sigmoid(o["decomp"]).numpy())
        P["reserve"].append(o["reserve"].numpy())
    return {k: np.concatenate(v) for k, v in P.items()}


def predict_dataset(model, ds, bs=128):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
    return predict_torch(model, dl)


# --------------------------------------------------------------------- metrics
def score_all(pred, meta):
    m = meta.reset_index(drop=True)
    s = {}
    y = m.stage.to_numpy()
    P = pred["stage"]
    present = np.unique(y)
    aucs = []
    for k in range(M.N_STAGES):
        if k in present and (y == k).sum() > 3 and (y != k).sum() > 3:
            a = roc_auc_score((y == k).astype(int), P[:, k])
            s[f"stage_auroc_{k}"] = a
            aucs.append(a)
    s["stage_auroc_macro"] = float(np.mean(aucs)) if aucs else np.nan
    s["stage_bacc"] = balanced_accuracy_score(y, P.argmax(1))
    s["stage_acc"] = float((y == P.argmax(1)).mean())

    v, pv = m.volume_lost.to_numpy(), pred["volume"]
    s["volume_r2"] = r2_score(v, pv)
    s["volume_mae"] = float(np.mean(np.abs(v - pv)))
    s["volume_rho"] = float(spearmanr(v, pv).statistic)

    r, pr = m.reserve.to_numpy(), pred["reserve"]
    s["reserve_r2"] = r2_score(r, pr)
    s["reserve_mae"] = float(np.mean(np.abs(r - pr)))
    s["reserve_rho"] = float(spearmanr(r, pr).statistic)

    at = m.at_risk.to_numpy().astype(bool)
    yd = m.decomp_soon.to_numpy()[at]
    pd_ = pred["decomp"][at]
    if yd.sum() > 3 and (1 - yd).sum() > 3:
        s["decomp_auroc"] = roc_auc_score(yd, pd_)
        s["decomp_auprc"] = average_precision_score(yd, pd_)
        s["decomp_prev"] = float(yd.mean())
    else:
        s["decomp_auroc"] = s["decomp_auprc"] = s["decomp_prev"] = np.nan
    s["n"] = len(m)
    return s


def bootstrap_ci(pred, meta, n_boot=400, seed=0, keys=("stage_auroc_macro", "volume_r2",
                                                       "decomp_auroc", "reserve_r2")):
    """Cluster bootstrap: resample SUBJECTS, not windows."""
    rng = np.random.default_rng(seed)
    m = meta.reset_index(drop=True)
    subs = m.subject.unique()
    by = {s: np.where(m.subject.values == s)[0] for s in subs}
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(subs, len(subs), replace=True)
        ix = np.concatenate([by[s] for s in pick])
        sub_pred = {k: v[ix] for k, v in pred.items()}
        try:
            sc = score_all(sub_pred, m.iloc[ix])
        except Exception:
            continue
        for k in keys:
            if np.isfinite(sc.get(k, np.nan)):
                acc[k].append(sc[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in acc.items() if v}


def confusion(pred, meta, normalize="true"):
    return confusion_matrix(meta.stage.to_numpy(), pred["stage"].argmax(1),
                            labels=list(range(M.N_STAGES)), normalize=normalize)


# --------------------------------------------------------------------- boosted
def fit_boosted(cfg, Ftr, mtr, Fva, mva, feature_names, seed=0):
    """XGBoost, one model per task, sharing the same feature matrix."""
    import xgboost as xgb
    p = dict(cfg.models.handcrafted.xgb)
    common = dict(n_jobs=cfg.run.n_jobs, random_state=seed, tree_method="hist", **p)

    out = {}
    cnt = np.bincount(mtr.stage.to_numpy(), minlength=M.N_STAGES).astype(float)
    w = (cnt.sum() / (M.N_STAGES * np.maximum(cnt, 1))) ** 0.5
    sw = w[mtr.stage.to_numpy()]
    out["stage"] = xgb.XGBClassifier(objective="multi:softprob", num_class=M.N_STAGES,
                                     eval_metric="mlogloss", early_stopping_rounds=40, **common)
    out["stage"].fit(Ftr, mtr.stage, sample_weight=sw, eval_set=[(Fva, mva.stage)], verbose=False)

    for task, col in [("volume", "volume_lost"), ("reserve", "reserve")]:
        r = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse",
                             early_stopping_rounds=40, **common)
        r.fit(Ftr, mtr[col], eval_set=[(Fva, mva[col])], verbose=False)
        out[task] = r

    atr, ava = mtr.at_risk.to_numpy().astype(bool), mva.at_risk.to_numpy().astype(bool)
    c = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          early_stopping_rounds=40,
                          scale_pos_weight=float((1 - mtr.decomp_soon[atr]).sum()
                                                 / max(mtr.decomp_soon[atr].sum(), 1)), **common)
    c.fit(Ftr[atr], mtr.decomp_soon[atr], eval_set=[(Fva[ava], mva.decomp_soon[ava])], verbose=False)
    out["decomp"] = c
    return out


def predict_boosted(bundle, Fx):
    return dict(stage=bundle["stage"].predict_proba(Fx),
                volume=bundle["volume"].predict(Fx),
                reserve=bundle["reserve"].predict(Fx),
                decomp=bundle["decomp"].predict_proba(Fx)[:, 1])
