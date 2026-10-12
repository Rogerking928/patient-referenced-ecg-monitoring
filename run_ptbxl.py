"""Stage 5: the public-data arm (PTB-XL).

  P1  preprocessing + quality control pipeline on real ECGs
  P2  matched-view comparison of synthetic and real feature distributions
  P3  domain discriminability -- how separable are they, as a number
  P4  a real-data encoder, and whether pretraining on it transfers to the
      synthetic haemodynamic task

No haemorrhage claim is made anywhere in this stage; PTB-XL has no such labels.
"""
import time
import numpy as np
import pandas as pd
import torch

from hemo import (config, experiments as E, features as FE, models as M,
                  ptbxl_bridge as PB, simulate, train as T)
from hemo.config import out

torch.set_num_threads(14)
cfg = config.load()
log = E.log_to(out(cfg, "run_ptbxl.log"))
t00 = time.time()
FS_R = 100
SEC = 10.0
mcfg = PB.matched_cfg(cfg, fs=FS_R, window_s=SEC)

# =================================================================== P1 load + QC
log("=== P1  PTB-XL loading, preprocessing and quality control ===")
Xr, mr = PB.load_ptbxl(cfg, n=6000, seed=cfg.run.seed)
log(f"    loaded {Xr.shape} records, {mr.is_norm.mean()*100:.1f}% fully normal")
Xr = PB.preprocess(Xr, FS_R)
qr = PB.quality_control(Xr, FS_R)
log(f"    QC pass {qr['pass'].mean()*100:.1f}%  "
    f"(beats {qr.pass_beats.mean()*100:.1f}%, beat consistency "
    f"{qr.pass_beat_corr.mean()*100:.1f}%, flat {qr.pass_flat.mean()*100:.1f}%, "
    f"amp {qr.pass_amp.mean()*100:.1f}%); median beat correlation "
    f"{qr.beat_corr.median():.3f}, median high-frequency SNR {qr.snr_db.median():.0f} dB "
    f"(the latter does not gate -- the 100 Hz release is already anti-alias filtered)")
qr.assign(**{c: mr[c].values for c in ["age", "sex", "is_norm", "strat_fold", "device"]}) \
  .to_csv(out(cfg, "ptbxl_qc.csv"), index=False)

keep = qr["pass"].to_numpy()
Xr_ok, mr_ok = Xr[keep], mr[keep].reset_index(drop=True)

# =================================================================== P2 matched comparison
log("\n=== P2  matched-view feature comparison ===")
Xs, ms = simulate.load(cfg)
base = np.where((ms.stage == 0).to_numpy() & (ms.t_min < 7.6).to_numpy())[0]
rng = np.random.default_rng(0)
base = rng.choice(base, min(2500, len(base)), replace=False)
Xs_v = PB.synthetic_to_ptbxl_view(cfg, [Xs[k] for k in base], fs_out=FS_R, seconds=SEC)
Xs_v = PB.preprocess(Xs_v, FS_R)
qs = PB.quality_control(Xs_v, FS_R)
log(f"    synthetic QC pass {qs['pass'].mean()*100:.1f}%")
Xs_ok = Xs_v[qs["pass"].to_numpy()]

n_cmp = min(len(Xr_ok), len(Xs_ok), 2200)
Fr = FE.extract_batch(Xr_ok[:n_cmp], mcfg)
Fs = FE.extract_batch(Xs_ok[:n_cmp], mcfg)
smd = PB.standardised_mean_difference(Fr, Fs)
smd.to_csv(out(cfg, "ptbxl_smd.csv"), index=False)
big = smd.reindex(smd.smd.abs().sort_values(ascending=False).index)
log(f"    |SMD| > 0.5 in {(smd.smd.abs() > 0.5).sum()}/{len(smd)} features; "
    f"median |SMD| {smd.smd.abs().median():.2f}")
log(f"    largest gaps: {big.head(8)[['feature','real_mean','synth_mean','smd']].to_string(index=False)}")

# =================================================================== P3 domain gap
log("\n=== P3  domain discriminability ===")
auc_f, _ = PB.domain_discriminability(Fr.fillna(Fr.median()), Fs.fillna(Fr.median()))
log(f"    handcrafted-feature space: real vs synthetic AUROC = {auc_f:.3f} "
    f"(0.5 = indistinguishable)")

# =================================================================== P4 encoder + transfer
log("\n=== P4  PTB-XL encoder and transfer probe ===")
y_norm = mr_ok.is_norm.to_numpy().astype(np.float32)
age = mr_ok.age.to_numpy(np.float32)
age = np.clip(age, 15, 95)
fold = mr_ok.strat_fold.to_numpy()
tr = np.where(fold <= 8)[0]
va = np.where(fold == 9)[0]
te = np.where(fold == 10)[0]
scale = float(np.abs(Xr_ok[tr[:500]]).std())


class PTBNet(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.models.cnn
        self.enc = M.Encoder1D(2, c.base_channels, c.n_blocks, c.kernel, c.dropout, c.embed_dim)
        self.norm = torch.nn.Linear(c.embed_dim, 1)
        self.age = torch.nn.Linear(c.embed_dim, 1)

    def forward(self, x):
        h = self.enc(x)
        return self.norm(h).squeeze(-1), self.age(h).squeeze(-1), h


def batches(idx, bs, shuffle=True, seed=0):
    idx = np.array(idx)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield (torch.tensor(Xr_ok[j] / scale, dtype=torch.float32),
               torch.tensor(y_norm[j]), torch.tensor((age[j] - 60) / 20.0))


net = PTBNet(cfg)
opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
from sklearn.metrics import roc_auc_score
best, best_state = -np.inf, None
for ep in range(14):
    net.train()
    tot = 0.0
    for x, yn, ya in batches(tr, 64, seed=ep):
        opt.zero_grad()
        pn, pa, _ = net(x)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(pn, yn)
                + 0.5 * torch.nn.functional.mse_loss(pa, ya))
        loss.backward()
        opt.step()
        tot += loss.item()
    net.eval()
    with torch.no_grad():
        pv, av = [], []
        for x, yn, ya in batches(va, 128, shuffle=False):
            a, b, _ = net(x)
            pv.append(a.numpy())
            av.append(b.numpy())
    auc = roc_auc_score(y_norm[va], np.concatenate(pv))
    mae = float(np.mean(np.abs(np.concatenate(av) * 20 + 60 - age[va])))
    log(f"    ptbxl ep{ep+1:02d} loss {tot/max(len(tr)//64,1):.4f} val NORM-AUROC {auc:.3f} age-MAE {mae:.2f}y")
    if auc > best:
        best, best_state = auc, {k: v.clone() for k, v in net.state_dict().items()}
net.load_state_dict(best_state)
net.eval()
with torch.no_grad():
    pt, at = [], []
    for x, yn, ya in batches(te, 128, shuffle=False):
        a, b, _ = net(x)
        pt.append(a.numpy())
        at.append(b.numpy())
enc_auc = roc_auc_score(y_norm[te], np.concatenate(pt))
enc_mae = float(np.mean(np.abs(np.concatenate(at) * 20 + 60 - age[te])))
log(f"    PTB-XL encoder held-out (fold 10): NORM AUROC {enc_auc:.3f}, age MAE {enc_mae:.2f} y")
torch.save(net.enc.state_dict(), out(cfg, "ptbxl_encoder.pt"))

# embedding-space domain gap
with torch.no_grad():
    Hr = np.concatenate([net.enc(torch.tensor(Xr_ok[i:i + 128] / scale, dtype=torch.float32)).numpy()
                         for i in range(0, min(len(Xr_ok), n_cmp), 128)])
    Hs = np.concatenate([net.enc(torch.tensor(Xs_ok[i:i + 128] / scale, dtype=torch.float32)).numpy()
                         for i in range(0, min(len(Xs_ok), n_cmp), 128)])
auc_e, _ = PB.domain_discriminability(pd.DataFrame(Hr), pd.DataFrame(Hs))
log(f"    encoder-embedding space: real vs synthetic AUROC = {auc_e:.3f}")
np.savez_compressed(out(cfg, "ptbxl_embeddings.npz"), Hr=Hr.astype(np.float32),
                    Hs=Hs.astype(np.float32))

pd.DataFrame([dict(n_loaded=len(Xr), qc_pass_frac=float(qr["pass"].mean()),
                   qc_pass_frac_synth=float(qs["pass"].mean()),
                   n_compared=n_cmp,
                   median_abs_smd=float(smd.smd.abs().median()),
                   frac_smd_gt_0p5=float((smd.smd.abs() > 0.5).mean()),
                   domain_auroc_features=auc_f, domain_auroc_embedding=auc_e,
                   ptbxl_norm_auroc=enc_auc, ptbxl_age_mae=enc_mae)]) \
    .to_csv(out(cfg, "ptbxl_summary.csv"), index=False)

# ---- transfer probe on the synthetic task
log("\n    transfer probe: PTB-XL-pretrained encoder vs random initialisation")
ctx = E.Ctx(cfg)
res = []
for tag, pretrained in [("pretrained_ptbxl", True), ("random_init", False)]:
    w = M.WaveNet(len(cfg.simulation.leads), cfg)
    if pretrained:
        w.enc.load_state_dict(torch.load(out(cfg, "ptbxl_encoder.pt")))
    w, p, mte = E.run_torch(ctx, w, log=log, epochs=8, seed=5)
    sc = T.score_all(p, mte)
    sc["init"] = tag
    res.append(sc)
    log(f"    {tag}: stageAUC {sc['stage_auroc_macro']:.3f} volR2 {sc['volume_r2']:.3f} "
        f"decAUC {sc['decomp_auroc']:.3f} resR2 {sc['reserve_r2']:.3f}")
pd.DataFrame(res).to_csv(out(cfg, "ptbxl_transfer.csv"), index=False)

log(f"\nPTBXL_COMPLETE  {time.time()-t00:.0f}s")
