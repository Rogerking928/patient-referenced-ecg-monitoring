"""Trajectory models, and the ordinal structure of the state.

Two things are being tested.

First, whether a single thirty-second window is the right unit at all.  Acute volume
loss is a process, and the rate and direction of change may carry more than the
absolute value.  Windows are therefore assembled into per-subject sequences and the
same targets are predicted from one window, from five minutes, from ten minutes, from
the change since the subject's own first window, and from the slope and curvature of
that change.

Second, the five states are an ordered sequence cut out of a continuum, and treating
them as unordered categories throws that away.  An ordinal (cumulative-link) model is
compared with the multiclass one on the same features.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score

from hemo import config, experiments as E, train as T
from hemo.config import out

cfg = config.load()
log = E.log_to(out(cfg, "run_longitudinal.log"))
O = lambda p: out(cfg, p)
t00 = time.time()

meta = pd.read_parquet(O("meta_main.parquet")).reset_index(drop=True)
F = pd.read_parquet(O("features_main.parquet"))
A, mu, sd = T.prep_features(F, np.where(meta.split == "train")[0])
A = np.clip((A - mu) / sd, -8, 8)
names = list(F.columns)
n_win = int(cfg.simulation.n_windows)
step_min = (cfg.simulation.window_last_min - cfg.simulation.window_first_min) / (n_win - 1)
log(f"{len(meta)} windows, {meta.subject.nunique()} subjects, "
    f"{step_min:.1f} min between windows")

# ------------------------------------------------------------------ sequences
meta = meta.sort_values(["subject", "win"]).reset_index(drop=True)
order = meta.index.to_numpy()
A = A[order] if len(A) == len(meta) else A
sub = meta.subject.to_numpy()
starts = {s: np.where(sub == s)[0] for s in meta.subject.unique()}


def history(k, n_back):
    """Indices of the n_back windows ending at row k, padded with the earliest."""
    s = sub[k]
    idx = starts[s]
    pos = int(np.where(idx == k)[0][0])
    take = [idx[max(pos - j, 0)] for j in range(n_back)][::-1]
    return np.array(take)


REPS = {}
REPS["single window"] = A
for lab, nb in [("5 minute trajectory", max(int(round(5 / step_min)) + 1, 2)),
                ("10 minute trajectory", max(int(round(10 / step_min)) + 1, 2))]:
    H = np.stack([A[history(k, nb)] for k in range(len(meta))])          # (n, nb, f)
    REPS[lab] = H
    log(f"    {lab}: {nb} windows ({(nb-1)*step_min:.0f} min of history)")

base_ref = np.zeros_like(A)
for s, idx in starts.items():
    base_ref[idx] = A[idx[0]]
REPS["change from the subject's first window"] = A - base_ref

nb10 = REPS["10 minute trajectory"].shape[1]
H = REPS["10 minute trajectory"]
tt = np.arange(nb10) * step_min
slope = np.einsum("ijk,j->ik", H - H.mean(1, keepdims=True), tt - tt.mean()) / \
    max(((tt - tt.mean()) ** 2).sum(), 1e-9)
curv = H[:, -1] - 2 * H[:, nb10 // 2] + H[:, 0]
REPS["value + slope + curvature"] = np.hstack([A, slope, curv])

tr = np.where(meta.split == "train")[0]
va = np.where(meta.split == "val")[0]
te = np.where(meta.split == "test")[0]
XGBP = dict(n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.85,
            colsample_bytree=0.75, n_jobs=cfg.run.n_jobs, tree_method="hist",
            random_state=0)

rows = []


def score_boosted(Xf, tag):
    Xf = Xf.reshape(len(Xf), -1)
    r = dict(representation=tag, n_features=Xf.shape[1])
    y = meta.stage.to_numpy()
    cnt = np.bincount(y[tr], minlength=5).astype(float)
    wgt = (cnt.sum() / (5 * np.maximum(cnt, 1))) ** 0.5
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=5,
                            eval_metric="mlogloss", early_stopping_rounds=40, **XGBP)
    clf.fit(Xf[tr], y[tr], sample_weight=wgt[y[tr]], eval_set=[(Xf[va], y[va])], verbose=False)
    P = clf.predict_proba(Xf[te])
    r["stage_auroc_macro"] = float(np.mean([roc_auc_score((y[te] == k).astype(int), P[:, k])
                                            for k in range(5) if (y[te] == k).sum() > 3]))
    r["stage_bacc"] = balanced_accuracy_score(y[te], P.argmax(1))
    for task, col in [("volume", "volume_lost"), ("reserve", "reserve")]:
        reg = xgb.XGBRegressor(objective="reg:squarederror", early_stopping_rounds=40, **XGBP)
        reg.fit(Xf[tr], meta[col].to_numpy()[tr],
                eval_set=[(Xf[va], meta[col].to_numpy()[va])], verbose=False)
        r[f"{task}_r2"] = float(r2_score(meta[col].to_numpy()[te], reg.predict(Xf[te])))
    at = meta.at_risk.to_numpy().astype(bool)
    yd = meta.decomp_soon.to_numpy()
    c2 = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                           early_stopping_rounds=40,
                           scale_pos_weight=float((1 - yd[tr[at[tr]]]).sum()
                                                  / max(yd[tr[at[tr]]].sum(), 1)), **XGBP)
    c2.fit(Xf[tr[at[tr]]], yd[tr[at[tr]]],
           eval_set=[(Xf[va[at[va]]], yd[va[at[va]]])], verbose=False)
    r["decomp_auroc"] = float(roc_auc_score(yd[te[at[te]]],
                                            c2.predict_proba(Xf[te[at[te]]])[:, 1]))
    # horizons
    for h in [5.0, 15.0]:
        yy = ((meta.tte_event == 1) & (meta.tte <= h)).to_numpy().astype(int)
        usable = ~((meta.tte_event == 0) & (meta.tte < h)).to_numpy() & at
        i_tr, i_va, i_te = tr[usable[tr]], va[usable[va]], te[usable[te]]
        if yy[i_te].sum() < 5:
            continue
        cH = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                               early_stopping_rounds=40,
                               scale_pos_weight=float((1 - yy[i_tr]).sum()
                                                      / max(yy[i_tr].sum(), 1)), **XGBP)
        cH.fit(Xf[i_tr], yy[i_tr], eval_set=[(Xf[i_va], yy[i_va])], verbose=False)
        r[f"decomp_auroc_{h:.0f}min"] = float(
            roc_auc_score(yy[i_te], cH.predict_proba(Xf[i_te])[:, 1]))
    rows.append(r)
    pd.DataFrame(rows).to_csv(O("longitudinal.csv"), index=False)
    log(f"[{time.time()-t00:6.0f}s] {tag:38s} stageAUC {r['stage_auroc_macro']:.3f}  "
        f"volR2 {r['volume_r2']:+.3f}  decAUC {r['decomp_auroc']:.3f}  "
        f"resR2 {r['reserve_r2']:+.3f}")


log("\n=== how much history helps ===")
for tag, Xf in REPS.items():
    score_boosted(Xf, tag)

# ------------------------------------------------------------------ recurrent model
log("\n=== a recurrent model over the trajectory ===")


class GRUNet(nn.Module):
    def __init__(self, n_feat, hidden=96):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.15))
        self.stage = nn.Linear(64, 5)
        self.vol = nn.Linear(64, 1)
        self.res = nn.Linear(64, 1)
        self.dec = nn.Linear(64, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        z = self.head(h[:, -1])
        return (self.stage(z), self.vol(z).squeeze(-1), self.res(z).squeeze(-1),
                self.dec(z).squeeze(-1))


Hseq = REPS["10 minute trajectory"].astype(np.float32)
y_s = torch.tensor(meta.stage.to_numpy(np.int64))
y_v = torch.tensor(meta.volume_lost.to_numpy(np.float32))
y_r = torch.tensor(meta.reserve.to_numpy(np.float32))
y_d = torch.tensor(meta.decomp_soon.to_numpy(np.float32))
m_d = torch.tensor(meta.at_risk.to_numpy(np.float32))
Xt = torch.tensor(Hseq)
cnt = np.bincount(meta.stage.to_numpy()[tr], minlength=5).astype(float)
cw = torch.tensor((cnt.sum() / (5 * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32)
torch.manual_seed(0)
net = GRUNet(Hseq.shape[2])
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
best, best_state = -np.inf, None
for ep in range(30):
    net.train()
    perm = np.random.default_rng(ep).permutation(tr)
    for i in range(0, len(perm), 256):
        j = perm[i:i + 256]
        opt.zero_grad()
        ps, pv, pr, pd_ = net(Xt[j])
        bce = nn.functional.binary_cross_entropy_with_logits(pd_, y_d[j], reduction="none")
        loss = (nn.functional.cross_entropy(ps, y_s[j], weight=cw)
                + 0.6 * nn.functional.mse_loss(pv, y_v[j])
                + 0.6 * nn.functional.mse_loss(pr, y_r[j])
                + 0.6 * (bce * m_d[j]).sum() / (m_d[j].sum() + 1e-6))
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        ps, pv, pr, pd_ = net(Xt[va])
    yv = meta.stage.to_numpy()[va]
    au = np.mean([roc_auc_score((yv == k).astype(int),
                                torch.softmax(ps, 1).numpy()[:, k])
                  for k in range(5) if (yv == k).sum() > 3])
    crit = au + np.clip(r2_score(meta.reserve.to_numpy()[va], pr.numpy()), 0, 1)
    if crit > best:
        best, best_state = crit, {k: v.clone() for k, v in net.state_dict().items()}
net.load_state_dict(best_state)
net.eval()
with torch.no_grad():
    ps, pv, pr, pd_ = net(Xt[te])
yv = meta.stage.to_numpy()[te]
at = meta.at_risk.to_numpy().astype(bool)
r = dict(representation="recurrent model over the 10 minute trajectory",
         n_features=Hseq.shape[1] * Hseq.shape[2],
         stage_auroc_macro=float(np.mean([roc_auc_score((yv == k).astype(int),
                                                        torch.softmax(ps, 1).numpy()[:, k])
                                          for k in range(5) if (yv == k).sum() > 3])),
         stage_bacc=balanced_accuracy_score(yv, torch.softmax(ps, 1).numpy().argmax(1)),
         volume_r2=float(r2_score(meta.volume_lost.to_numpy()[te], pv.numpy())),
         reserve_r2=float(r2_score(meta.reserve.to_numpy()[te], pr.numpy())),
         decomp_auroc=float(roc_auc_score(meta.decomp_soon.to_numpy()[te[at[te]]],
                                          torch.sigmoid(pd_).numpy()[at[te]])))
rows.append(r)
pd.DataFrame(rows).to_csv(O("longitudinal.csv"), index=False)
log(f"[{time.time()-t00:6.0f}s] {r['representation']:38s} "
    f"stageAUC {r['stage_auroc_macro']:.3f}  volR2 {r['volume_r2']:+.3f}  "
    f"decAUC {r['decomp_auroc']:.3f}  resR2 {r['reserve_r2']:+.3f}")

# ------------------------------------------------------------------ ordinal
log("\n=== treating the states as ordered rather than unordered ===")
ORD = [0, 1, 2, 3]          # baseline < compensated < early decompensation < shock
mask = np.isin(meta.stage.to_numpy(), ORD)
yo = meta.stage.to_numpy()
res_ord = []
for tag, Xf in [("single window", A), ("10 minute trajectory", Hseq.reshape(len(Hseq), -1))]:
    Xf = np.asarray(Xf, np.float32)
    itr, iva, ite = tr[mask[tr]], va[mask[va]], te[mask[te]]
    # cumulative-link: K-1 binary problems P(stage > k), then differenced
    cum = np.zeros((len(ite), len(ORD) - 1))
    for k in range(len(ORD) - 1):
        yk = (yo > k).astype(int)
        c = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              early_stopping_rounds=40,
                              scale_pos_weight=float((1 - yk[itr]).sum() / max(yk[itr].sum(), 1)),
                              **XGBP)
        c.fit(Xf[itr], yk[itr], eval_set=[(Xf[iva], yk[iva])], verbose=False)
        cum[:, k] = c.predict_proba(Xf[ite])[:, 1]
    cum = np.maximum.accumulate(cum[:, ::-1], axis=1)[:, ::-1]   # enforce monotonicity
    P = np.zeros((len(ite), len(ORD)))
    P[:, 0] = 1 - cum[:, 0]
    for k in range(1, len(ORD) - 1):
        P[:, k] = cum[:, k - 1] - cum[:, k]
    P[:, -1] = cum[:, -1]
    P = np.clip(P, 1e-6, None)
    P /= P.sum(1, keepdims=True)
    yt = yo[ite]
    from scipy.stats import kendalltau
    exp_stage = P @ np.arange(len(ORD))
    res_ord.append(dict(representation=tag, model="ordinal cumulative-link",
                        macro_auroc=float(np.mean([roc_auc_score((yt == k).astype(int), P[:, k])
                                                   for k in range(len(ORD)) if (yt == k).sum() > 3])),
                        bacc=balanced_accuracy_score(yt, P.argmax(1)),
                        kendall_tau=float(kendalltau(yt, exp_stage).statistic),
                        mae_in_stages=float(np.mean(np.abs(yt - exp_stage)))))
    # multiclass comparison on the same rows
    cnt = np.bincount(yo[itr], minlength=len(ORD)).astype(float)
    wg = (cnt.sum() / (len(ORD) * np.maximum(cnt, 1))) ** 0.5
    mc = xgb.XGBClassifier(objective="multi:softprob", num_class=len(ORD),
                           eval_metric="mlogloss", early_stopping_rounds=40, **XGBP)
    mc.fit(Xf[itr], yo[itr], sample_weight=wg[yo[itr]],
           eval_set=[(Xf[iva], yo[iva])], verbose=False)
    Pm = mc.predict_proba(Xf[ite])
    em = Pm @ np.arange(len(ORD))
    res_ord.append(dict(representation=tag, model="multiclass",
                        macro_auroc=float(np.mean([roc_auc_score((yt == k).astype(int), Pm[:, k])
                                                   for k in range(len(ORD)) if (yt == k).sum() > 3])),
                        bacc=balanced_accuracy_score(yt, Pm.argmax(1)),
                        kendall_tau=float(kendalltau(yt, em).statistic),
                        mae_in_stages=float(np.mean(np.abs(yt - em)))))
    for r_ in res_ord[-2:]:
        log(f"    {r_['representation']:24s} {r_['model']:24s} macroAUC {r_['macro_auroc']:.3f}  "
            f"bACC {r_['bacc']:.3f}  Kendall tau {r_['kendall_tau']:.3f}  "
            f"MAE {r_['mae_in_stages']:.3f} states")
pd.DataFrame(res_ord).to_csv(O("ordinal.csv"), index=False)
log("\nLONGITUDINAL_COMPLETE")
