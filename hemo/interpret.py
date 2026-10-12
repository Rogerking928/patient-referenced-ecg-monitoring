"""Interpretability: SHAP, integrated gradients, latent space, sensitivity,
counterfactual simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import ecg, features as FE, physiology as phys

# feature families, for grouped attribution
FAMILIES = {
    "rate": ["hr_mean", "hr_sd", "rr_mean"],
    "rate_variability": ["sdnn", "rmssd", "pnn50", "pnn20", "cvrr", "sd1", "sd2", "sd1_sd2",
                         "rr_skew", "rr_kurt", "sampen", "dfa_a1", "lf", "hf", "lf_hf",
                         "tot_pow", "lf_nu", "rr_kept_frac"],
    "P_wave": ["p_amp", "p_dur", "pr_int", "p_area", "p_r_ratio"],
    "QRS": ["r_amp", "q_amp", "s_amp", "qrs_ptp", "qrs_dur", "r_amp_sd", "r_amp_cv", "r_resp_mod"],
    "ST_T": ["st_level", "st_slope", "t_amp", "t_dur", "t_sym", "qt", "t_area", "qtc",
             "t_r_ratio", "t_amp_sd", "qt_sd", "qtv_index"],
    "respiratory": ["resp_rate_wander", "wander_power", "resp_rate_rsa", "rsa_strength"],
    "quality": ["hf_noise", "snr_db", "n_beats", "flat_frac", "beat_corr"],
}


def family_of(name):
    base = name.split("_", 1)[1] if name.split("_", 1)[0] in ("II", "V5") else name
    for fam, keys in FAMILIES.items():
        if base in keys:
            return fam
    return "other"


# ------------------------------------------------------------------ SHAP
def shap_handcrafted(bundle, A, names, task="stage", max_rows=1500, seed=0):
    """Tree SHAP for the boosted model.  Returns per-feature mean |SHAP|."""
    import shap
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(A), min(max_rows, len(A)), replace=False)
    ex = shap.TreeExplainer(bundle[task])
    sv = ex.shap_values(A[ix])
    sv = np.asarray(sv)
    if sv.ndim == 3:                      # (n, features, classes)
        imp = np.abs(sv).mean(axis=(0, 2))
        per_class = np.abs(sv).mean(axis=0)          # (features, classes)
    else:
        imp = np.abs(sv).mean(axis=0)
        per_class = imp[:, None]
    df = pd.DataFrame({"feature": names, "mean_abs_shap": imp})
    df["family"] = [family_of(n) for n in names]
    for k in range(per_class.shape[1]):
        df[f"shap_class{k}"] = per_class[:, k]
    return df.sort_values("mean_abs_shap", ascending=False), sv, ix


# ------------------------------------------------------------------ integrated gradients
def integrated_gradients(model, x, target, baseline=None, steps=48, feat=None):
    """IG for a waveform input.  target is a callable(out_dict) -> scalar tensor."""
    model.eval()
    x = x if torch.is_tensor(x) else torch.tensor(x)
    x = x.float().unsqueeze(0) if x.ndim == 2 else x.float()
    base = torch.zeros_like(x) if baseline is None else baseline
    f = torch.zeros(x.shape[0], 1) if feat is None else feat
    total = torch.zeros_like(x)
    for a in np.linspace(1.0 / steps, 1.0, steps):
        xi = (base + a * (x - base)).clone().requires_grad_(True)
        out = model(xi, f)
        s = target(out)
        g, = torch.autograd.grad(s, xi)
        total += g
    return ((x - base) * total / steps).detach().numpy()[0]


def saliency_by_phase(attr, wave, fs, refractory=0.22):
    """Split |attribution| into cardiac phases relative to detected R peaks.

    Answers "which ECG component carries the haemodynamic information".
    """
    a = np.abs(attr).sum(0)
    pk = FE.detect_rpeaks(np.asarray(wave[0], float), fs, refractory)
    n = len(a)
    lab = np.zeros(n, np.int8)                 # 0 = other / TP segment
    if len(pk) < 3:
        return None
    rr = np.median(np.diff(pk)) / fs
    for p in pk:
        seg = lambda lo, hi: (max(0, p + int(lo * fs)), min(n, p + int(hi * fs)))
        a0, a1 = seg(-min(0.30, 0.42 * rr), -0.055)
        lab[a0:a1] = 1                          # P wave / PR
        a0, a1 = seg(-0.055, 0.06)
        lab[a0:a1] = 2                          # QRS
        a0, a1 = seg(0.06, 0.14)
        lab[a0:a1] = 3                          # ST segment
        a0, a1 = seg(0.14, min(0.46, 0.62 * rr))
        lab[a0:a1] = 4                          # T wave
    names = ["TP_baseline", "P_PR", "QRS", "ST", "T"]
    tot = a.sum() + 1e-12
    frac = {names[k]: float(a[lab == k].sum() / tot) for k in range(5)}
    dens = {names[k] + "_per_sample": float(a[lab == k].mean()) if (lab == k).any() else np.nan
            for k in range(5)}
    return {**frac, **dens}


# ------------------------------------------------------------------ counterfactual
def counterfactual_sweep(cfg, subj_row, t_win, Zw, Dw, latent, grid, predictor, n_rep=6, seed=0):
    """Hold one synthetic subject fixed, move one latent, watch the prediction.

    `predictor` maps a (n_leads, n_samples) waveform to a dict of predictions.
    """
    rng = np.random.default_rng(seed)
    out = []
    for g in grid:
        for r in range(n_rep):
            sig = ecg.render_window(cfg, subj_row, t_win, Zw, Dw,
                                    np.random.default_rng(seed * 977 + r),
                                    override={latent: g})
            pr = predictor(sig)
            out.append(dict(latent=latent, value=float(g), rep=r, **pr))
    return pd.DataFrame(out)


def make_torch_predictor(model, mu, sd, wave_scale, cfg, feature_fn=None):
    """Wrap a trained network as waveform -> prediction dict."""
    def predict(sig):
        x = torch.tensor(np.asarray(sig, np.float32) / wave_scale).unsqueeze(0)
        if feature_fn is None:
            f = torch.zeros(1, 1)
        else:
            v = feature_fn(sig)
            f = torch.tensor(np.clip((v - mu) / sd, -8, 8).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            o = model(x, f)
        p = torch.softmax(o["stage"], 1).numpy()[0]
        return dict(volume=float(o["volume"]), reserve=float(o["reserve"]),
                    decomp=float(torch.sigmoid(o["decomp"])),
                    **{f"p_{phys.STAGES[k]}": float(p[k]) for k in range(len(p))})
    return predict


def make_boosted_predictor(bundle, cfg, names, mu=None, sd=None):
    def predict(sig):
        f = FE.extract_window(sig, cfg)
        v = np.array([[f.get(n, np.nan) for n in names]], np.float32)
        v = np.nan_to_num(v, nan=0.0)
        p = bundle["stage"].predict_proba(v)[0]
        return dict(volume=float(bundle["volume"].predict(v)[0]),
                    reserve=float(bundle["reserve"].predict(v)[0]),
                    decomp=float(bundle["decomp"].predict_proba(v)[0, 1]),
                    **{f"p_{phys.STAGES[k]}": float(p[k]) for k in range(len(p))})
    return predict


# ------------------------------------------------------------------ latent space
def embed_dataset(model, ds, bs=128):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
    Hs = []
    model.eval()
    with torch.no_grad():
        for b in dl:
            Hs.append(model.embed(b[0], b[1]).numpy())
    return np.concatenate(Hs)
