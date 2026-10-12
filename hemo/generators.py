"""Three ways of turning the same latent state into a waveform.

A model trained on one generator and tested on another cannot be exploiting that
generator's private syntax.  All three renderers consume an identical per-beat
parameter set produced by `beat_params`, so the only thing that differs between
them is how a beat is drawn:

  A  gaussian    sums of Gaussian waves on a warped time axis (the v1 generator)
  B  dynamical   McSharry-style limit-cycle ODE; morphology emerges from
                 integration, wave widths scale with the angular velocity, and
                 successive beats are smoothly coupled rather than independent
  C  empirical   median beats extracted from real PTB-XL recordings, warped
                 segment-by-segment to the requested intervals and amplitudes

`beat_params` is deliberately the only place the physiology is read, so the
comparison is fair.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal as sps

from .ecg import _add_noise, _rr_series

TEMPLATE_BANK = Path(__file__).resolve().parent.parent / "out" / "ptbxl_templates.npz"


# ------------------------------------------------------------------ shared physiology
def beat_params(cfg, subj_row, t_win, Zw, Dw, rng, override=None):
    """Latent state -> per-beat morphology.  Identical for all three generators."""
    ph = cfg.physiology
    hr = Dw[:, 0].astype(float).copy()
    preload = np.clip(Dw[:, 7].astype(float), 0.05, 2.0).copy()
    symp = Zw[:, 4].astype(float).copy()
    para = Zw[:, 5].astype(float).copy()
    resp = Zw[:, 6].astype(float).copy()
    metab = np.clip(Zw[:, 8].astype(float) / 8.0, 0, 1).copy()
    if override:
        for k, v in override.items():
            {"heart_rate": hr, "preload": preload, "sympathetic_tone": symp,
             "parasympathetic_tone": para, "respiratory_rate": resp,
             "metabolic_stress": metab}[k][:] = v

    para0 = subj_row["para_rest"]
    bt, rr, phi_r = _rr_series(rng, cfg, subj_row, t_win, hr, para, resp, symp, para0)
    if len(bt) < 4:
        bt = np.arange(0, cfg.simulation.window_s, 0.8)
        rr = np.full(len(bt), 0.8)

    rel = t_win - t_win[0]
    g = lambda a, x: np.interp(x, rel, a)
    S_b, P_b, M_b, PL_b, F_b = (g(symp, bt), g(para, bt), g(metab, bt),
                                g(preload, bt), g(resp, bt) / 60.0)
    dS = S_b - ph.symp_rest

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

    resp_phase = 2 * np.pi * np.cumsum(F_b * np.r_[rr[0], rr[:-1]]) + phi_r
    ramp = 1.0 + subj_row["c_ramp_resp"] * (0.35 + 1.5 * np.clip(1 - PL_b, 0, 1)) * np.sin(resp_phase)
    qrs_scale = qrs_scale * ramp
    p_amp = p_amp * (1.0 + 0.4 * (ramp - 1.0))

    return dict(bt=bt, rr=rr, phi_r=phi_r, resp=resp, rel=rel,
                p_amp=p_amp, p_dur=np.clip(p_dur, 0.05, 0.16),
                pr=np.clip(pr, 0.085, 0.30), qrs_scale=qrs_scale,
                qrs_dur=np.clip(qrs_dur, 0.055, 0.19), qt=np.clip(qt, 0.20, 0.62),
                t_amp=t_amp, t_dur=np.clip(t_dur, 0.09, 0.28), st=st)


# ------------------------------------------------------------------ A: gaussian
_PRE, _POST = 0.42, 0.82


def render_gaussian(cfg, subj_row, pp, lead_i, lead):
    fs = cfg.simulation.fs
    n = int(round(cfg.simulation.window_s * fs))
    gp, gq, gt_ = cfg.ecg.lead_gain[lead]
    loc = np.arange(-int(_PRE * fs), int(_POST * fs)) / fs
    L = len(loc)
    buf = np.zeros(n + 2 * L)
    off = L
    for k in range(len(pp["bt"])):
        c = int(round(pp["bt"][k] * fs))
        pd_, qd, qtk, td = pp["p_dur"][k], pp["qrs_dur"][k], pp["qt"][k], pp["t_dur"][k]
        w = gp * pp["p_amp"][k] * np.exp(-0.5 * ((loc + pp["pr"][k] - 0.5 * pd_) / (pd_ / 3.6)) ** 2)
        w += gq * pp["qrs_scale"][k] * subj_row["q_amp"] * np.exp(-0.5 * ((loc + 0.42 * qd) / (qd / 11.0)) ** 2)
        w += gq * pp["qrs_scale"][k] * subj_row["r_amp"] * np.exp(-0.5 * (loc / (qd / 6.4)) ** 2)
        w += gq * pp["qrs_scale"][k] * subj_row["s_amp"] * np.exp(-0.5 * ((loc - 0.44 * qd) / (qd / 8.5)) ** 2)
        j, t_on = 0.55 * qd, qtk - td
        w += pp["st"][k] * 0.5 * (np.tanh((loc - j) / 0.012) - np.tanh((loc - t_on) / 0.012))
        tc = qtk - 0.5 * td
        sig = np.where(loc < tc, td / 5.2, td / 3.4)
        w += gt_ * pp["t_amp"][k] * np.exp(-0.5 * ((loc - tc) / sig) ** 2)
        a = off + c - int(_PRE * fs)
        buf[a:a + L] += w
    return buf[off:off + n]


# ------------------------------------------------------------------ B: dynamical
def render_dynamical(cfg, subj_row, pp, lead_i, lead):
    """McSharry-style limit cycle.

    z is driven by Gaussian terms placed at fixed angles on a circular trajectory
    whose angular velocity is 2*pi/RR, so wave widths follow the cycle length
    automatically and consecutive beats are joined by the ODE rather than summed.
    """
    fs = cfg.simulation.fs
    n = int(round(cfg.simulation.window_s * fs))
    dt = 1.0 / fs
    gp, gq, gt_ = cfg.ecg.lead_gain[lead]
    bt, rr = pp["bt"], pp["rr"]

    # per-beat angular positions of P, Q, R, S, T relative to R at theta = 0
    nb = len(bt)
    rrk = np.clip(np.asarray(rr[:nb], float), 0.25, None)
    ang = 2 * np.pi / rrk
    C = np.stack([-(pp["pr"][:nb] - 0.5 * pp["p_dur"][:nb]) * ang,
                  -0.42 * pp["qrs_dur"][:nb] * ang,
                  np.zeros(nb),
                  0.44 * pp["qrs_dur"][:nb] * ang,
                  (pp["qt"][:nb] - 0.5 * pp["t_dur"][:nb]) * ang], 1)
    W = np.maximum(np.stack([pp["p_dur"][:nb] / 3.6 * ang, pp["qrs_dur"][:nb] / 11.0 * ang,
                             pp["qrs_dur"][:nb] / 6.4 * ang, pp["qrs_dur"][:nb] / 8.5 * ang,
                             pp["t_dur"][:nb] / 4.2 * ang], 1), 1e-3)
    A = np.stack([gp * pp["p_amp"][:nb],
                  gq * pp["qrs_scale"][:nb] * subj_row["q_amp"],
                  gq * pp["qrs_scale"][:nb] * subj_row["r_amp"],
                  gq * pp["qrs_scale"][:nb] * subj_row["s_amp"],
                  gt_ * pp["t_amp"][:nb]], 1)
    st = np.asarray(pp["st"][:nb], float)
    edges = np.asarray(bt[:nb], float)

    # theta advances at the beat's angular velocity, so it can be built in closed
    # form; z then solves a first-order linear ODE driven by the Gaussian terms,
    # which is an IIR filter rather than a per-sample Python loop.
    tgrid = np.arange(n) * dt
    kidx = np.clip(np.searchsorted(edges, tgrid, side="right") - 1, 0, nb - 1)
    theta = -np.pi + np.cumsum(ang[kidx] * dt)
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    d = (theta[:, None] - C[kidx] + np.pi) % (2 * np.pi) - np.pi
    Wk, Ak = W[kidx], A[kidx]
    # the peak of z produced by a term a*dtheta*exp(-dtheta^2/2b^2) is a*b^2/omega,
    # so the drive carries omega/b^2 to make that peak the requested amplitude
    drive = -np.sum(Ak * ang[kidx][:, None] / Wk ** 2 * d * np.exp(-0.5 * (d / Wk) ** 2),
                    axis=1)
    return sps.lfilter([dt], [1.0, -(1.0 - dt)], drive + st[kidx])


# ------------------------------------------------------------------ C: empirical
_BANK = None


def load_bank():
    global _BANK
    if _BANK is None:
        if not TEMPLATE_BANK.exists():
            raise FileNotFoundError(
                f"{TEMPLATE_BANK} not found -- run build_templates.py first")
        d = np.load(TEMPLATE_BANK)
        _BANK = {k: d[k] for k in d.files}
    return _BANK


def render_empirical(cfg, subj_row, pp, lead_i, lead, template_id=None):
    """Real median beats, warped segment-by-segment to the requested intervals.

    The P-wave, QRS and T-wave segments of a real beat are time-scaled and
    amplitude-scaled independently, so the requested PR, QRS duration, QT and
    wave amplitudes are honoured while the *shape* inside each segment stays
    whatever a real heart produced.
    """
    bank = load_bank()
    T_ = bank["templates"]                 # (n_templates, n_leads, L)
    LM = bank["landmarks"]                 # (n_templates, 5): p0 p1 q0 q1 t1, samples
    r0 = int(bank["r_index"])
    fsT = float(bank["fs"])
    fs = cfg.simulation.fs
    n = int(round(cfg.simulation.window_s * fs))
    tid = int(subj_row["subject"]) % len(T_) if template_id is None else template_id
    tpl = T_[tid, min(lead_i, T_.shape[1] - 1)].astype(float)
    p0, p1, q0, q1, t1 = LM[tid]

    # segment the real beat around its own landmarks
    seg_p = tpl[p0:p1]
    seg_q = tpl[q0:q1]
    seg_t = tpl[q1:t1]
    # normalise by the SIGNED peak, so the requested amplitude sets the polarity;
    # dividing by max|.| would silently invert any template whose extreme is negative
    sgn = lambda v: v[int(np.argmax(np.abs(v)))] if len(v) else 1.0
    amp_p = sgn(seg_p) + (1e-6 if abs(sgn(seg_p)) < 1e-6 else 0.0)
    amp_q = abs(sgn(seg_q)) + 1e-6
    amp_t = sgn(seg_t) + (1e-6 if abs(sgn(seg_t)) < 1e-6 else 0.0)
    pr_t = (q0 - p0) / fsT
    qrs_t = (q1 - q0) / fsT
    qt_t = (t1 - q0) / fsT

    gp, gq, gt_ = cfg.ecg.lead_gain[lead]
    buf = np.zeros(n + 4 * fs)
    off = 2 * fs
    for k in range(len(pp["bt"])):
        c = int(round(pp["bt"][k] * fs)) + off
        pr_k, qd, qtk = pp["pr"][k], pp["qrs_dur"][k], pp["qt"][k]
        n_p = max(int(round(pr_k * fs)), 4)
        n_q = max(int(round(qd * fs)), 4)
        n_t = max(int(round((qtk - qd) * fs)), 6)
        wp = sps.resample(seg_p, n_p) / amp_p * gp * pp["p_amp"][k]
        wq = sps.resample(seg_q, n_q) / amp_q * gq * pp["qrs_scale"][k] * subj_row["r_amp"]
        wt = sps.resample(seg_t, n_t) / amp_t * gt_ * pp["t_amp"][k]
        qstart = c - int(round(0.42 * qd * fs))
        pstart = qstart - n_p
        buf[pstart:pstart + n_p] += wp
        buf[qstart:qstart + n_q] += wq
        buf[qstart + n_q:qstart + n_q + n_t] += wt + pp["st"][k]
    return buf[off:off + n]


RENDERERS = {"gaussian": render_gaussian, "dynamical": render_dynamical,
             "empirical": render_empirical}


def render_window(cfg, subj_row, t_win, Zw, Dw, rng, generator="gaussian",
                  noise_scale=1.0, override=None):
    """One window, from any generator, with the same physiology and the same noise."""
    fs = cfg.simulation.fs
    n = int(round(cfg.simulation.window_s * fs))
    tt = np.arange(n) / fs
    pp = beat_params(cfg, subj_row, t_win, Zw, Dw, rng, override=override)
    fn = RENDERERS[generator]
    out = np.zeros((len(cfg.simulation.leads), n), np.float32)
    for li, lead in enumerate(cfg.simulation.leads):
        out[li] = fn(cfg, subj_row, pp, li, lead)
    return _add_noise(cfg, subj_row, out, tt, pp["resp"], pp["rel"], rng, noise_scale)
