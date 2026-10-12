"""ECG observation model:  ECG(t) = G(z(t), subject_parameters) + noise.

Beats are synthesised as sums of Gaussian waves (P, Q, R, S, T) laid on a
time axis whose spacing is the RR series.  Every wave parameter is a function
of the *latent* state at that instant multiplied by that subject's own
coupling coefficient -- haemorrhage itself never appears in this file.

Because each subject draws its own couplings (several of which straddle zero),
no single ECG feature moves in a fixed direction with volume loss across the
cohort; only the joint distribution shifts.
"""
from __future__ import annotations

import numpy as np

LEAD_ORDER = ["II", "V5"]

# analysis window around each R peak, in seconds
_PRE, _POST = 0.42, 0.82


def _lin(x, y, xi):
    return np.interp(xi, x, y)


def _rr_series(rng, cfg, subj_row, t_win, hr, para, resp, symp, para0):
    """Beat times over the window, with RSA, Mayer waves and jitter."""
    ec = cfg.ecg
    rsa_g = subj_row["c_rsa_gain"]
    lf_g = subj_row["c_lf_gain"]
    jit = subj_row["c_rr_jitter"]

    phi_r = rng.uniform(0, 2 * np.pi)
    phi_l = rng.uniform(0, 2 * np.pi)
    dur = t_win[-1] - t_win[0]

    times, rrs = [], []
    tk = 0.0
    resp_phase = phi_r
    guard = 0
    while tk < dur and guard < 4000:
        guard += 1
        hr_k = _lin(t_win - t_win[0], hr, tk)
        p_k = _lin(t_win - t_win[0], para, tk)
        s_k = _lin(t_win - t_win[0], symp, tk)
        f_k = _lin(t_win - t_win[0], resp, tk) / 60.0

        vagal = np.clip(p_k / max(para0, 1e-3), 0.0, 2.0)
        a_hf = rsa_g * vagal
        # Mayer-wave power follows an inverted U in sympathetic tone: it rises
        # with early compensation and collapses at maximal recruitment.
        a_lf = lf_g * 4.0 * np.clip(s_k, 0, 1) * (1.0 - np.clip(s_k, 0, 1)) / 0.5
        rr = 60.0 / hr_k * (
            1.0
            + a_hf * np.sin(resp_phase)
            + a_lf * np.sin(2 * np.pi * ec.lf_hz * tk + phi_l)
            + jit * vagal * rng.standard_normal()
        )
        rr = float(np.clip(rr, 0.26, 2.2))
        times.append(tk)
        rrs.append(rr)
        resp_phase += 2 * np.pi * f_k * rr
        tk += rr
    return np.array(times), np.array(rrs), phi_r


def render_window(cfg, subj_row, t_win, Zw, Dw, rng, noise_scale=1.0,
                  override=None, return_truth=False):
    """Render one multi-lead ECG window.

    Parameters
    ----------
    t_win : (n_t,) seconds, absolute time of the latent samples
    Zw    : (n_t, 9) latent states over the window
    Dw    : (n_t, 9) derived variables over the window
    override : optional dict of latent overrides for counterfactual simulation,
               e.g. {"preload": 0.6} or {"sympathetic_tone": 0.9}
    """
    ec, ph = cfg.ecg, cfg.physiology
    fs = cfg.simulation.fs
    n = int(round(cfg.simulation.window_s * fs))
    tt = np.arange(n) / fs

    hr = Dw[:, 0].astype(float).copy()
    preload = np.clip(Dw[:, 7].astype(float), 0.05, 2.0).copy()
    symp = Zw[:, 4].astype(float).copy()
    para = Zw[:, 5].astype(float).copy()
    resp = Zw[:, 6].astype(float).copy()
    metab = np.clip(Zw[:, 8].astype(float) / 8.0, 0, 1).copy()

    if override:
        for k, v in override.items():
            arr = {"heart_rate": hr, "preload": preload, "sympathetic_tone": symp,
                   "parasympathetic_tone": para, "respiratory_rate": resp,
                   "metabolic_stress": metab}[k]
            arr[:] = v if np.isscalar(v) else v

    para0 = subj_row["para_rest"]
    s_rest = ph.symp_rest

    bt, rr, phi_r = _rr_series(cfg, subj_row, t_win, hr, para, resp, symp, para0) \
        if False else _rr_series(rng, cfg, subj_row, t_win, hr, para, resp, symp, para0)
    if len(bt) < 4:
        bt = np.arange(0, cfg.simulation.window_s, 0.8)
        rr = np.full(len(bt), 0.8)

    rel = t_win - t_win[0]
    g = lambda arr, x: np.interp(x, rel, arr)
    S_b, P_b, M_b, PL_b, F_b = (g(symp, bt), g(para, bt), g(metab, bt),
                                g(preload, bt), g(resp, bt) / 60.0)

    dS = S_b - s_rest
    # --------------------------------------------------------------- morphology
    p_amp = subj_row["p_amp"] * PL_b ** subj_row["c_p_amp_preload"] * (1 + subj_row["c_p_amp_symp"] * dS)
    p_dur = subj_row["p_dur"] * (1 + subj_row["c_p_dur_symp"] * dS)
    pr = subj_row["pr0"] * (1 + subj_row["c_pr_symp"] * dS)
    qrs_scale = PL_b ** subj_row["c_qrs_amp_preload"] * (1 + subj_row["c_qrs_amp_symp"] * dS)
    qrs_dur = subj_row["qrs0"] * (1 + subj_row["c_qrs_dur_metab"] * M_b)
    qt = (subj_row["qt0"] * rr ** subj_row["qt_alpha"] * (1 + subj_row["c_qt_metab"] * M_b)
          + rng.standard_normal(len(bt)) * subj_row["c_qtv_symp"] * np.clip(S_b, 0, 1))
    t_amp = (subj_row["t_amp"] * (1 + subj_row["c_t_amp_metab"] * M_b)
             * (1 + subj_row["c_t_amp_symp"] * dS) * PL_b ** 0.15)
    t_dur = subj_row["t_dur"] * (1 + 0.25 * (rr - 0.85))
    st = subj_row["c_st_metab"] * M_b * subj_row["amp_scale"]

    # respiratory amplitude modulation of the QRS: the electrocardiographic
    # analogue of pulse-pressure variation, and it grows as preload falls
    resp_phase = 2 * np.pi * np.cumsum(F_b * np.r_[rr[0], rr[:-1]]) + phi_r
    ramp = 1.0 + subj_row["c_ramp_resp"] * (ec.get("ramp_base", 0.35) + ec.get("ramp_preload_slope", 1.5) * np.clip(1 - PL_b, 0, 1)) * np.sin(resp_phase)
    qrs_scale = qrs_scale * ramp
    p_amp = p_amp * (1.0 + 0.4 * (ramp - 1.0))

    qt = np.clip(qt, 0.20, 0.62)
    p_dur = np.clip(p_dur, 0.05, 0.16)
    qrs_dur = np.clip(qrs_dur, 0.055, 0.19)
    pr = np.clip(pr, 0.085, 0.30)
    t_dur = np.clip(t_dur, 0.09, 0.28)

    # --------------------------------------------------------------- render
    n_lead = len(cfg.simulation.leads)
    out = np.zeros((n_lead, n), np.float32)
    loc = np.arange(-int(_PRE * fs), int(_POST * fs)) / fs
    L = len(loc)

    for li, lead in enumerate(cfg.simulation.leads):
        gp, gq, gt_ = cfg.ecg.lead_gain[lead]
        buf = np.zeros(n + 2 * L)
        off = L
        for k in range(len(bt)):
            centre = int(round(bt[k] * fs))
            w = np.zeros(L)
            # P wave
            w += gp * p_amp[k] * np.exp(-0.5 * ((loc + pr[k] - 0.5 * p_dur[k]) / (p_dur[k] / 3.6)) ** 2)
            # QRS complex
            w += gq * qrs_scale[k] * subj_row["q_amp"] * np.exp(
                -0.5 * ((loc + 0.42 * qrs_dur[k]) / (qrs_dur[k] / 11.0)) ** 2)
            w += gq * qrs_scale[k] * subj_row["r_amp"] * np.exp(
                -0.5 * (loc / (qrs_dur[k] / 6.4)) ** 2)
            w += gq * qrs_scale[k] * subj_row["s_amp"] * np.exp(
                -0.5 * ((loc - 0.44 * qrs_dur[k]) / (qrs_dur[k] / 8.5)) ** 2)
            # ST segment: plateau between S offset and T onset
            j = 0.55 * qrs_dur[k]
            t_on = qt[k] - t_dur[k]
            w += st[k] * 0.5 * (np.tanh((loc - j) / 0.012) - np.tanh((loc - t_on) / 0.012))
            # T wave, asymmetric (slower upstroke than downstroke)
            tc = qt[k] - 0.5 * t_dur[k]
            sl, sr = t_dur[k] / 5.2, t_dur[k] / 3.4
            sig = np.where(loc < tc, sl, sr)
            w += gt_ * t_amp[k] * np.exp(-0.5 * ((loc - tc) / sig) ** 2)
            a, b = off + centre - int(_PRE * fs), off + centre - int(_PRE * fs) + L
            buf[a:b] += w
        out[li] = buf[off:off + n]

    out = _add_noise(cfg, subj_row, out, tt, resp, rel, rng, noise_scale)

    if return_truth:
        truth = dict(
            beat_times=bt, rr=rr, p_amp=p_amp, p_dur=p_dur, pr=pr,
            qrs_scale=qrs_scale, qrs_dur=qrs_dur, qt=qt, t_amp=t_amp, st=st,
        )
        return out, truth
    return out


def _add_noise(cfg, subj_row, sig, tt, resp, rel, rng, scale=1.0):
    ec = cfg.ecg
    fs = cfg.simulation.fs
    n = sig.shape[1]
    f_resp = np.interp(tt, rel, resp) / 60.0
    phase = 2 * np.pi * np.cumsum(f_resp) / fs + rng.uniform(0, 2 * np.pi)

    for li in range(sig.shape[0]):
        # baseline wander tracks respiration (plus a slow drift)
        w = subj_row["wander_mv"] * scale
        sig[li] += (w * np.sin(phase) + 0.45 * w * np.sin(0.5 * phase + 1.1)).astype(np.float32)
        drift = np.cumsum(rng.standard_normal(n)) / fs
        sig[li] += (0.35 * w * drift / (np.abs(drift).max() + 1e-9)).astype(np.float32)
        # electrode / EMG noise, mildly band-limited
        e = rng.standard_normal(n)
        e = e - np.convolve(e, np.ones(5) / 5, mode="same") * 0.5
        sig[li] += (subj_row["emg_mv"] * scale * e).astype(np.float32)
        # mains interference
        if subj_row["powerline_mv"] > 0:
            sig[li] += (subj_row["powerline_mv"] * scale
                        * np.sin(2 * np.pi * ec.powerline_hz * tt + rng.uniform(0, 6.3))).astype(np.float32)
    # motion artefact: a shared burst across leads
    if rng.random() < ec.motion_prob:
        amp = rng.uniform(*ec.motion_mv_range) * scale
        c = rng.integers(0, n)
        w = int(rng.uniform(0.4, 2.5) * fs)
        idx = np.arange(max(0, c - w), min(n, c + w))
        env = 0.5 * (1 + np.cos(np.pi * (idx - c) / max(w, 1)))
        burst = np.cumsum(rng.standard_normal(len(idx)))
        burst = burst / (np.abs(burst).max() + 1e-9)
        for li in range(sig.shape[0]):
            sig[li, idx] += (amp * env * burst * rng.uniform(0.6, 1.4)).astype(np.float32)
    return sig
