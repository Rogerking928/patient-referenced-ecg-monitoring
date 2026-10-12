"""Handcrafted physiological feature extraction from a raw ECG window.

Three families:
  rate/variability  -- HR, SDNN, RMSSD, pNN50, LF, HF, LF/HF, sample entropy,
                       Poincare SD1/SD2, DFA alpha1
  morphology        -- P/QRS/T amplitudes and durations, PR, QT, QTc, ST level,
                       T symmetry, plus their beat-to-beat variability
  respiratory / QC  -- respiratory rate from RSA and from baseline wander,
                       R-wave respiratory modulation depth, SNR, artefact burden

Nothing here knows about haemorrhage; these are the measurements a monitor
could make.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

EPS = 1e-9


# ------------------------------------------------------------------ R peaks
def _bandpass(x, fs, lo, hi, order=3):
    b, a = sps.butter(order, [lo / (fs / 2), min(hi / (fs / 2), 0.99)], btype="band")
    return sps.filtfilt(b, a, x)


def detect_rpeaks(x, fs, refractory=0.22):
    """Pan-Tompkins style detector."""
    f = _bandpass(x, fs, 5.0, 22.0)
    d = np.diff(f, prepend=f[0])
    sq = d ** 2
    win = max(int(0.12 * fs), 1)
    integ = np.convolve(sq, np.ones(win) / win, mode="same")

    thr = 0.35 * np.percentile(integ, 98)
    cand = sps.find_peaks(integ, height=thr, distance=int(refractory * fs))[0]
    # snap to the local extremum of the band-passed trace
    peaks = []
    half = int(0.06 * fs)
    for c in cand:
        a, b = max(0, c - half), min(len(x), c + half)
        if b - a < 3:
            continue
        seg = f[a:b]
        peaks.append(a + int(np.argmax(np.abs(seg))))
    peaks = np.array(sorted(set(peaks)), int)
    if len(peaks) > 2:
        # drop peaks whose amplitude is a small fraction of the median
        amp = np.abs(f[peaks])
        peaks = peaks[amp > 0.3 * np.median(amp)]
    return peaks


# ------------------------------------------------------------------ HRV
def _sampen(x, m=2, r=0.2):
    x = np.asarray(x, float)
    n = len(x)
    if n < m + 4:
        return np.nan
    r = r * (np.std(x) + EPS)

    def phi(mm):
        emb = np.array([x[i:i + mm] for i in range(n - mm)])
        if len(emb) < 2:
            return np.nan
        d = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
        np.fill_diagonal(d, np.inf)
        return (d <= r).sum()

    a, b = phi(m + 1), phi(m)
    if b is None or np.isnan(a) or np.isnan(b) or b == 0:
        return np.nan
    if a == 0:                       # no matches at m+1: report the ceiling
        a = 0.5
    return float(-np.log(a / b))


def _dfa_alpha1(x, scales=(4, 5, 6, 7, 8, 9, 10, 11, 12)):
    x = np.asarray(x, float)
    if len(x) < 16:
        return np.nan
    y = np.cumsum(x - x.mean())
    F = []
    used = []
    for s in scales:
        if len(y) < 2 * s:
            continue
        n_seg = len(y) // s
        seg = y[:n_seg * s].reshape(n_seg, s)
        tt = np.arange(s)
        res = []
        for row in seg:
            c = np.polyfit(tt, row, 1)
            res.append(np.mean((row - np.polyval(c, tt)) ** 2))
        F.append(np.sqrt(np.mean(res)))
        used.append(s)
    if len(F) < 3:
        return np.nan
    return float(np.polyfit(np.log(used), np.log(np.array(F) + EPS), 1)[0])


def clean_rr(rr):
    """Standard ectopy/artefact filter: drop intervals far from the local median."""
    rr = np.asarray(rr, float)
    rr = rr[(rr > 0.25) & (rr < 2.5)]
    if len(rr) < 5:
        return rr, 1.0
    k = min(7, len(rr) if len(rr) % 2 else len(rr) - 1)
    loc = sps.medfilt(rr, kernel_size=max(k, 3))
    ok = np.abs(rr - loc) < 0.22 * loc
    if ok.sum() < 5:
        return rr, 1.0
    return rr[ok], float(ok.mean())


def hrv_features(rr, cfg):
    """rr in seconds."""
    f = {}
    rr, kept = clean_rr(rr)
    if len(rr) < 5:
        return {k: np.nan for k in HRV_KEYS}
    f["rr_kept_frac"] = kept
    d = np.diff(rr)
    f["hr_mean"] = 60.0 / rr.mean()
    f["hr_sd"] = np.std(60.0 / rr)
    f["rr_mean"] = rr.mean()
    f["sdnn"] = rr.std(ddof=1) * 1000
    f["rmssd"] = np.sqrt(np.mean(d ** 2)) * 1000
    f["pnn50"] = 100.0 * np.mean(np.abs(d) > 0.05)
    f["pnn20"] = 100.0 * np.mean(np.abs(d) > 0.02)
    f["cvrr"] = 100.0 * rr.std(ddof=1) / (rr.mean() + EPS)
    sd1 = np.sqrt(0.5) * np.std(d, ddof=1) * 1000
    sd2 = np.sqrt(max(2 * rr.std(ddof=1) ** 2 - 0.5 * np.std(d, ddof=1) ** 2, 0)) * 1000
    f["sd1"], f["sd2"] = sd1, sd2
    f["sd1_sd2"] = sd1 / (sd2 + EPS)
    f["rr_skew"] = float(((rr - rr.mean()) ** 3).mean() / (rr.std() ** 3 + EPS))
    f["rr_kurt"] = float(((rr - rr.mean()) ** 4).mean() / (rr.std() ** 4 + EPS))
    f["sampen"] = _sampen(rr, cfg.features.sampen_m, cfg.features.sampen_r)
    f["dfa_a1"] = _dfa_alpha1(rr)

    # spectral: Lomb-Scargle on the unevenly sampled tachogram
    t = np.cumsum(rr) - rr[0]
    y = rr - rr.mean()
    lo, hi = cfg.features.lf_band[0], cfg.features.hf_band[1]
    fr = np.linspace(max(lo, 1.2 / max(t[-1], 1e-3)), hi, 160)
    try:
        pxx = sps.lombscargle(t, y, 2 * np.pi * fr, normalize=False) / len(t)
        lf = np.trapezoid(pxx[(fr >= cfg.features.lf_band[0]) & (fr < cfg.features.lf_band[1])],
                          fr[(fr >= cfg.features.lf_band[0]) & (fr < cfg.features.lf_band[1])])
        hf = np.trapezoid(pxx[(fr >= cfg.features.hf_band[0]) & (fr <= cfg.features.hf_band[1])],
                          fr[(fr >= cfg.features.hf_band[0]) & (fr <= cfg.features.hf_band[1])])
    except Exception:
        lf = hf = np.nan
    f["lf"] = lf * 1e6
    f["hf"] = hf * 1e6
    f["lf_hf"] = lf / (hf + EPS)
    f["tot_pow"] = (lf + hf) * 1e6
    f["lf_nu"] = 100 * lf / (lf + hf + EPS)
    return f


HRV_KEYS = ["rr_kept_frac", "hr_mean", "hr_sd", "rr_mean", "sdnn", "rmssd", "pnn50", "pnn20", "cvrr",
            "sd1", "sd2", "sd1_sd2", "rr_skew", "rr_kurt", "sampen", "dfa_a1",
            "lf", "hf", "lf_hf", "tot_pow", "lf_nu"]


# ------------------------------------------------------------------ morphology
def _median_beat(x, peaks, fs, pre, post):
    a, b = int(pre * fs), int(post * fs)
    segs = [x[p - a:p + b] for p in peaks if p - a >= 0 and p + b < len(x)]
    if len(segs) < 3:
        return None, None
    S = np.array(segs)
    base = np.median(S[:, : int(0.06 * fs)], axis=1, keepdims=True)
    S = S - base
    return np.median(S, axis=0), S


def morph_features(x, peaks, rr, fs, cfg, prefix=""):
    ft = cfg.features
    rrm = float(np.median(rr)) if len(rr) else 0.85
    rrm = float(np.clip(rrm, 0.28, 2.0))
    # landmark search windows must scale with the cycle length, otherwise at
    # 170 bpm the "P-wave window" contains the previous beat's T wave.
    pre = min(ft.median_beat_pre_s, 0.42 * rrm)
    post = min(ft.median_beat_post_s, 0.62 * rrm)
    med, S = _median_beat(x, peaks, fs, pre, post)
    keys = [prefix + k for k in MORPH_KEYS]
    if med is None:
        return {k: np.nan for k in keys}

    r0 = int(pre * fs)
    tax = (np.arange(len(med)) - r0) / fs
    f = {}

    # --- QRS: search the fiducial window around the R peak
    qwin = (tax > -0.09) & (tax < 0.11)
    r_amp = med[r0]
    q_idx = np.where((tax > -0.07) & (tax < 0.0))[0]
    s_idx = np.where((tax > 0.0) & (tax < 0.10))[0]
    q_amp = med[q_idx].min() if len(q_idx) else np.nan
    s_amp = med[s_idx].min() if len(s_idx) else np.nan
    f["r_amp"] = r_amp
    f["q_amp"] = q_amp
    f["s_amp"] = s_amp
    f["qrs_ptp"] = np.ptp(med[qwin])

    # QRS onset/offset from the smoothed absolute derivative: unlike an
    # amplitude threshold this does not stop at the R-S polarity crossing.
    k = max(int(0.008 * fs), 1)
    env = np.convolve(np.abs(np.gradient(med)), np.ones(k) / k, mode="same")
    qthr = 0.10 * env[qwin].max()
    lo = r0
    while lo > max(1, r0 - int(0.11 * fs)) and env[lo - 1] > qthr:
        lo -= 1
    hi = r0
    while hi < min(len(med) - 2, r0 + int(0.14 * fs)) and env[hi + 1] > qthr:
        hi += 1
    f["qrs_dur"] = (hi - lo) / fs * 1000 if hi > lo else np.nan
    qrs_on = lo

    # --- P wave: largest deflection in the PR search window before QRS onset
    on = r0 - int(0.05 * fs)
    p_lo = r0 - int(min(0.32, 0.40 * rrm) * fs)
    seg = med[max(p_lo, 0):on]
    if len(seg) > 5:
        pi = int(np.argmax(np.abs(seg)))
        f["p_amp"] = seg[pi]
        pa = max(p_lo, 0) + pi
        pthr = 0.30 * abs(seg[pi])
        near = np.where(np.abs(seg) > pthr)[0]
        f["p_dur"] = (near[-1] - near[0]) / fs * 1000 if len(near) > 2 else np.nan
        f["pr_int"] = (r0 - (max(p_lo, 0) + (near[0] if len(near) else pi))) / fs * 1000
        f["p_area"] = float(np.trapezoid(seg, dx=1 / fs)) * 1000
    else:
        f["p_amp"] = f["p_dur"] = f["pr_int"] = f["p_area"] = np.nan

    # --- ST / T
    j = r0 + int(0.055 * fs)
    f["st_level"] = float(np.mean(med[j:j + int(0.03 * fs)]))
    f["st_slope"] = float((med[j + int(0.06 * fs)] - med[j]) / 0.06) if j + int(0.06 * fs) < len(med) else np.nan
    t_lo = r0 + int(0.10 * fs)
    t_hi = min(r0 + int(min(0.50, 0.62 * rrm) * fs), len(med) - 1)
    tseg = med[t_lo:t_hi]
    if len(tseg) > 8:
        ti = int(np.argmax(np.abs(tseg)))
        f["t_amp"] = tseg[ti]
        tthr = 0.25 * abs(tseg[ti])
        near = np.where(np.abs(tseg) > tthr)[0]
        if len(near) > 3:
            f["t_dur"] = (near[-1] - near[0]) / fs * 1000
            left = ti - near[0]
            right = near[-1] - ti
            f["t_sym"] = left / (right + EPS)
            f["qt"] = (t_lo + near[-1] - qrs_on) / fs * 1000
        else:
            f["t_dur"] = f["t_sym"] = f["qt"] = np.nan
        f["t_area"] = float(np.trapezoid(tseg, dx=1 / fs)) * 1000
    else:
        f["t_amp"] = f["t_dur"] = f["t_sym"] = f["qt"] = f["t_area"] = np.nan

    rrm = float(np.mean(rr)) if len(rr) else np.nan
    f["qtc"] = f["qt"] / np.sqrt(rrm) if np.isfinite(f.get("qt", np.nan)) and rrm > 0 else np.nan
    f["t_r_ratio"] = f["t_amp"] / (abs(r_amp) + EPS)
    f["p_r_ratio"] = f["p_amp"] / (abs(r_amp) + EPS)

    # --- beat-to-beat variability of the same landmarks
    r_series = S[:, r0]
    f["r_amp_sd"] = float(np.std(r_series))
    f["r_amp_cv"] = float(np.std(r_series) / (np.abs(np.mean(r_series)) + EPS))
    # respiratory modulation depth of R amplitude: (max-min)/mean over the window
    f["r_resp_mod"] = float((np.percentile(r_series, 95) - np.percentile(r_series, 5))
                            / (np.abs(np.mean(r_series)) + EPS))
    tt_amp = S[:, t_lo:t_hi]
    f["t_amp_sd"] = float(np.std(tt_amp[:, np.argmax(np.abs(np.median(tt_amp, 0)))])) if tt_amp.shape[1] > 2 else np.nan
    # QT variability: per-beat T offset
    qts = []
    for row in S:
        seg = row[t_lo:t_hi]
        if len(seg) < 8:
            continue
        ti = np.argmax(np.abs(seg))
        thr2 = 0.25 * abs(seg[ti])
        nz = np.where(np.abs(seg) > thr2)[0]
        if len(nz) > 2:
            qts.append((t_lo + nz[-1] - r0) / fs)
    f["qt_sd"] = float(np.std(qts) * 1000) if len(qts) > 4 else np.nan
    f["qtv_index"] = (f["qt_sd"] / (np.nanmean(qts) * 1000 + EPS)) if len(qts) > 4 else np.nan
    f["beat_corr"] = float(np.median([np.corrcoef(row, med)[0, 1] for row in S[:50]]))
    return {prefix + k: v for k, v in f.items()}


MORPH_KEYS = ["r_amp", "q_amp", "s_amp", "qrs_ptp", "qrs_dur", "p_amp", "p_dur",
              "pr_int", "p_area", "st_level", "st_slope", "t_amp", "t_dur", "t_sym",
              "qt", "t_area", "qtc", "t_r_ratio", "p_r_ratio", "r_amp_sd", "r_amp_cv",
              "r_resp_mod", "t_amp_sd", "qt_sd", "qtv_index", "beat_corr"]


# ------------------------------------------------------------------ respiration / QC
def resp_qc_features(x, peaks, rr, fs):
    f = {}
    # respiratory rate from the baseline wander band (0.1-0.6 Hz)
    b, a = sps.butter(2, [0.08 / (fs / 2), 0.7 / (fs / 2)], btype="band")
    w = sps.filtfilt(b, a, x)
    fr, px = sps.welch(w, fs, nperseg=min(len(w), int(16 * fs)))
    band = (fr >= 0.08) & (fr <= 0.7)
    f["resp_rate_wander"] = float(fr[band][np.argmax(px[band])] * 60) if band.any() else np.nan
    f["wander_power"] = float(np.trapezoid(px[band], fr[band]))

    # respiratory rate from RSA (tachogram peak in the HF band)
    if len(rr) > 8:
        t = np.cumsum(rr) - rr[0]
        y = rr - rr.mean()
        frs = np.linspace(0.10, 0.60, 90)
        try:
            p = sps.lombscargle(t, y, 2 * np.pi * frs, normalize=True)
            f["resp_rate_rsa"] = float(frs[np.argmax(p)] * 60)
            f["rsa_strength"] = float(p.max())
        except Exception:
            f["resp_rate_rsa"] = f["rsa_strength"] = np.nan
    else:
        f["resp_rate_rsa"] = f["rsa_strength"] = np.nan

    # crude SNR: QRS band power vs high-frequency residual
    hf = sps.filtfilt(*sps.butter(3, 45 / (fs / 2), btype="high"), x)
    f["hf_noise"] = float(np.std(hf))
    f["snr_db"] = float(20 * np.log10((np.std(x) + EPS) / (np.std(hf) + EPS)))
    f["n_beats"] = len(peaks)
    f["flat_frac"] = float(np.mean(np.abs(np.diff(x)) < 1e-6))
    return f


RESP_KEYS = ["resp_rate_wander", "wander_power", "resp_rate_rsa", "rsa_strength",
             "hf_noise", "snr_db", "n_beats", "flat_frac"]


# ------------------------------------------------------------------ driver
def feature_names(cfg):
    names = list(HRV_KEYS)
    for lead in cfg.simulation.leads:
        names += [f"{lead}_{k}" for k in MORPH_KEYS]
    names += RESP_KEYS
    return names


def extract_window(x, cfg):
    """x: (n_leads, n_samples) -> dict of features."""
    fs = cfg.simulation.fs
    lead0 = np.asarray(x[0], float)
    peaks = detect_rpeaks(lead0, fs, cfg.features.rpeak_refractory_s)
    rr = np.diff(peaks) / fs if len(peaks) > 1 else np.array([])

    f = hrv_features(rr, cfg)
    for li, lead in enumerate(cfg.simulation.leads):
        f.update(morph_features(np.asarray(x[li], float), peaks, rr, fs, cfg, prefix=f"{lead}_"))
    f.update(resp_qc_features(lead0, peaks, rr, fs))
    return f


def extract_batch(X, cfg, n_jobs=None):
    from joblib import Parallel, delayed
    import pandas as pd
    n_jobs = n_jobs or cfg.run.n_jobs
    chunks = np.array_split(np.arange(len(X)), max(n_jobs * 4, 1))

    def work(ix):
        return [extract_window(X[k], cfg) for k in ix]

    res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(work)(ix) for ix in chunks if len(ix))
    flat = [r for c in res for r in c]
    return pd.DataFrame(flat, columns=feature_names(cfg))
