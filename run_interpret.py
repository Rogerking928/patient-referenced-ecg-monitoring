"""Stage 4: interpretability.

  I1  SHAP for the handcrafted model (feature and family level)
  I2  integrated gradients for the waveform model, attributed to cardiac phases
  I3  latent-space geometry of the learned embedding
  I4  per-subject prediction trajectories
  I5  sensitivity analysis: sweep one latent at a time through the simulator
  I6  counterfactual simulation on a fixed synthetic patient
"""
import time
import numpy as np
import pandas as pd
import torch
import joblib

from hemo import (config, ecg, experiments as E, features as FE, interpret as I,
                  models as M, physiology as phys, train as T)
from hemo.config import out

torch.set_num_threads(12)
cfg = config.load()
log = E.log_to(out(cfg, "run_interpret.log"))
t00 = time.time()
ctx = E.Ctx(cfg)

b_all = joblib.load(out(cfg, "model_handcrafted.joblib"))
cnn = M.WaveNet(len(cfg.simulation.leads), cfg)
cnn.load_state_dict(torch.load(out(cfg, "model_cnn.pt")))
cnn.eval()
hyb = M.HybridNet(len(cfg.simulation.leads), ctx.A.shape[1], cfg)
hyb.load_state_dict(torch.load(out(cfg, "model_hybrid.pt")))
hyb.eval()

# =================================================================== I1 SHAP
log("=== I1  SHAP, handcrafted model ===")
sv_stage = None
for task in ["stage", "volume", "reserve", "decomp"]:
    df, sv, ix = I.shap_handcrafted(b_all, ctx.A[ctx.te], ctx.names, task=task, seed=1)
    if task == "stage":
        sv_stage = sv
    df.insert(0, "task", task)
    df.to_csv(out(cfg, f"shap_{task}.csv"), index=False)
    fam = df.groupby("family").mean_abs_shap.sum().sort_values(ascending=False)
    fam = (fam / fam.sum()).round(4)
    log(f"    {task:8s} family share: {fam.to_dict()}")
    log(f"    {task:8s} top features: {df.feature.head(8).tolist()}")
np.save(out(cfg, "shap_stage_values.npy"), sv_stage)

# =================================================================== I2 integrated gradients
log("\n=== I2  integrated gradients, waveform model ===")
rng = np.random.default_rng(3)
mte = ctx.meta.iloc[ctx.te].reset_index(drop=True)
rows, attr_store = [], {}
for stage in range(5):
    pool = np.where(mte.stage.to_numpy() == stage)[0]
    if len(pool) == 0:
        continue
    pick = rng.choice(pool, min(30, len(pool)), replace=False)
    for j in pick:
        k = ctx.te[j]
        x = np.asarray(ctx.X[k], np.float32) / ctx.wave_scale
        a = I.integrated_gradients(cnn, x, lambda o: o["stage"][0, stage], steps=24)
        ph = I.saliency_by_phase(a, np.asarray(ctx.X[k]), cfg.simulation.fs)
        if ph:
            rows.append(dict(stage=stage, window=int(k), **ph))
    # keep one example per stage for plotting
    k0 = int(ctx.te[pick[0]])
    x0 = np.asarray(ctx.X[k0], np.float32) / ctx.wave_scale
    attr_store[f"attr_{stage}"] = I.integrated_gradients(cnn, x0, lambda o: o["stage"][0, stage], steps=48)
    attr_store[f"wave_{stage}"] = np.asarray(ctx.X[k0], np.float32)
ig = pd.DataFrame(rows)
ig.to_csv(out(cfg, "ig_phase_attribution.csv"), index=False)
np.savez_compressed(out(cfg, "ig_examples.npz"), **attr_store)
share = ig.groupby("stage")[["TP_baseline", "P_PR", "QRS", "ST", "T"]].mean().round(3)
log(f"    attribution share by cardiac phase\n{share}")

# also do it for volume regression (single scalar head)
rows = []
pick = rng.choice(len(ctx.te), 120, replace=False)
for j in pick:
    k = ctx.te[j]
    x = np.asarray(ctx.X[k], np.float32) / ctx.wave_scale
    a = I.integrated_gradients(cnn, x, lambda o: o["volume"][0], steps=24)
    ph = I.saliency_by_phase(a, np.asarray(ctx.X[k]), cfg.simulation.fs)
    if ph:
        rows.append(dict(window=int(k), **ph))
pd.DataFrame(rows).to_csv(out(cfg, "ig_phase_attribution_volume.csv"), index=False)
log(f"    volume head phase share: "
    f"{pd.DataFrame(rows)[['TP_baseline','P_PR','QRS','ST','T']].mean().round(3).to_dict()}")

# =================================================================== I3 latent space
log("\n=== I3  latent space ===")
ds_te = ctx.ds(ctx.te)
H = I.embed_dataset(cnn, ds_te)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
pc = PCA(8).fit(H)
P = pc.transform(H)
sub = rng.choice(len(H), min(2500, len(H)), replace=False)
tsne = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(H[sub])
np.savez_compressed(out(cfg, "latent_space.npz"), H=H.astype(np.float32), pca=P.astype(np.float32),
                    tsne=tsne.astype(np.float32), tsne_idx=sub,
                    evr=pc.explained_variance_ratio_)
log(f"    embedding {H.shape}, PCA explained variance {pc.explained_variance_ratio_[:4].round(3)}")
# how much of each latent does the embedding linearly encode?
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score
probe = []
for name in ["volume_lost", "preload", "symp", "para", "resp", "lactate", "hr_true",
             "map_frac", "reserve", "sv", "co"]:
    y = mte[name].to_numpy()
    pr = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 20)), H, y,
                           cv=GroupKFold(5), groups=mte.subject.to_numpy())
    probe.append(dict(latent=name, r2=float(r2_score(y, pr))))
pd.DataFrame(probe).to_csv(out(cfg, "latent_probe.csv"), index=False)
log(f"    linear probe R2: {pd.DataFrame(probe).set_index('latent').r2.round(3).to_dict()}")

# =================================================================== I4 prediction trajectories
log("\n=== I4  prediction trajectories ===")
pred_cnn = T.predict_dataset(cnn, ds_te)
pred_hyb = T.predict_dataset(hyb, ctx.ds(ctx.te))
pred_hc = T.predict_boosted(b_all, ctx.A[ctx.te])
traj = mte.copy()
for tag, pr in [("cnn", pred_cnn), ("hyb", pred_hyb), ("hc", pred_hc)]:
    traj[f"{tag}_volume"] = pr["volume"]
    traj[f"{tag}_reserve"] = pr["reserve"]
    traj[f"{tag}_decomp"] = pr["decomp"]
    for k in range(5):
        traj[f"{tag}_p{k}"] = pr["stage"][:, k]
traj.to_parquet(out(cfg, "test_trajectories.parquet"))

# early-warning analysis: risk before the pressure falls
w = np.load(out(cfg, "world.npz"))
lab_all, D_all, t_all = w["lab"], w["D"], w["t"]
subj = pd.read_parquet(out(cfg, "subjects.parquet"))
rows = []
for s, g in traj.groupby("subject"):
    i = int(subj.index[subj.subject == s][0])
    ev = np.where((lab_all[i] == 2) | (lab_all[i] == 3))[0]
    if len(ev) == 0:
        continue
    t_dec = t_all[ev[0]]
    map0 = float(subj.map0.iloc[i])
    below = np.where(D_all[i, :, 2] < 0.9 * map0)[0]
    t_map = t_all[below[0]] if len(below) else np.nan
    g = g.sort_values("t_min")
    for tag in ["cnn", "hyb", "hc"]:
        pre = g[g.t_min < t_dec]
        rows.append(dict(subject=s, model=tag, t_decomp=t_dec, t_map_drop=t_map,
                         lead_time_map=float(t_dec - t_map) if np.isfinite(t_map) else np.nan,
                         risk_10min_before=float(pre[np.abs(pre.t_min - (t_dec - 10)) < 2.0][f"{tag}_decomp"].mean())
                         if len(pre[np.abs(pre.t_min - (t_dec - 10)) < 2.0]) else np.nan,
                         risk_at_baseline=float(pre[pre.t_min < 7.6][f"{tag}_decomp"].mean())
                         if len(pre[pre.t_min < 7.6]) else np.nan))
pd.DataFrame(rows).to_csv(out(cfg, "early_warning.csv"), index=False)
log(f"    early-warning table: {len(rows)} rows")

# =================================================================== I5 sensitivity
log("\n=== I5  sensitivity analysis (one latent swept at a time) ===")
Z, D, t = w["Z"], w["D"], w["t"]
predict_cnn = I.make_torch_predictor(cnn, ctx.mu, ctx.sd, ctx.wave_scale, cfg)
predict_hc = I.make_boosted_predictor(b_all, cfg, ctx.names)

te_sub = np.where(subj.split == "test")[0]
choose = te_sub[rng.choice(len(te_sub), 12, replace=False)]
grids = {"preload": np.linspace(0.30, 1.10, 9),
         "sympathetic_tone": np.linspace(0.05, 0.95, 9),
         "metabolic_stress": np.linspace(0.0, 1.0, 9),
         "respiratory_rate": np.linspace(10, 34, 9)}
sens = []
w_steps = int(cfg.simulation.window_s / cfg.simulation.dt_s)
for i in choose:
    k0 = int(np.argmax(lab_all[i] == 0))
    row = subj.iloc[i]
    tw, Zw, Dw = t[k0:k0 + w_steps] * 60.0, Z[i, k0:k0 + w_steps], D[i, k0:k0 + w_steps]
    for latent, grid in grids.items():
        d = I.counterfactual_sweep(cfg, row, tw, Zw, Dw, latent, grid, predict_cnn,
                                   n_rep=4, seed=int(i))
        d["subject"] = int(row.subject)
        d["model"] = "raw_waveform_cnn"
        sens.append(d)
sens = pd.concat(sens, ignore_index=True)
sens.to_csv(out(cfg, "sensitivity.csv"), index=False)
log(f"    {len(sens)} sensitivity simulations across {len(choose)} subjects")

# =================================================================== I6 counterfactual
log("\n=== I6  counterfactual on a single fixed patient ===")
i = int(choose[0])
row = subj.iloc[i]
k0 = int(np.argmax(lab_all[i] == 0))
tw, Zw, Dw = t[k0:k0 + w_steps] * 60.0, Z[i, k0:k0 + w_steps], D[i, k0:k0 + w_steps]
cf = []
for latent, grid in [("preload", np.linspace(0.28, 1.10, 13)),
                     ("sympathetic_tone", np.linspace(0.05, 0.98, 13))]:
    for tag, pfn in [("raw_waveform_cnn", predict_cnn), ("handcrafted_xgb", predict_hc)]:
        d = I.counterfactual_sweep(cfg, row, tw, Zw, Dw, latent, grid, pfn, n_rep=8, seed=17)
        d["model"] = tag
        cf.append(d)
cf = pd.concat(cf, ignore_index=True)
cf["subject"] = int(row.subject)
cf.to_csv(out(cfg, "counterfactual.csv"), index=False)
log(f"    counterfactual patient: subject {int(row.subject)}, {row.phenotype}, "
    f"age {row.age:.0f}, HR0 {row.hr0:.0f}, beta-blocked {int(row.beta_blocked)}")

log(f"\nINTERPRET_COMPLETE  {time.time()-t00:.0f}s")
