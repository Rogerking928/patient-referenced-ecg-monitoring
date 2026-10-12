"""Synthetic subject generation.

Every subject is a bundle of (a) demographic descriptors, (b) baseline
cardiovascular set-points, (c) autonomic / cardiac reserve parameters,
(d) a *personal* set of latent-to-ECG coupling coefficients, and (e) a
scenario (bleed or not, treated or not).

The coupling coefficients are the mechanism by which the requirement
"haemorrhage must not deterministically move any single ECG feature" is
satisfied: each subject draws its own gain for every latent -> feature
arrow, and several of those gains are allowed to change sign.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PHENOTYPES = ["high_responder", "normal_responder", "poor_compensator"]

# Latent->morphology couplings whose sign is allowed to flip between subjects.
# (Physiologically these are the ones with genuinely mixed reported direction.)
SIGN_LABILE = ["p_amp_symp", "qrs_amp_symp", "t_amp_symp", "p_dur_symp"]


def _lognormal(rng, mean, cv, size):
    """Lognormal with given arithmetic mean and coefficient of variation."""
    sigma = np.sqrt(np.log1p(cv ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return rng.lognormal(mu, sigma, size)


def make_cohort(cfg, seed=None) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.run.seed if seed is None else seed)
    n = cfg.cohort.n_subjects
    ph = cfg.physiology
    ec = cfg.ecg

    # ---------------------------------------------------------------- demographics
    age = np.clip(rng.normal(56, 18, n), 18, 92)
    sex = rng.integers(0, 2, n)                     # 1 = male
    weight = np.clip(rng.normal(78 - 12 * (1 - sex), 14, n), 42, 135)
    height = np.clip(rng.normal(174 - 13 * (1 - sex), 8, n), 145, 200)
    bsa = 0.007184 * weight ** 0.425 * height ** 0.725

    # ---------------------------------------------------------------- phenotype
    pw = np.array([cfg.cohort.phenotypes[p] for p in PHENOTYPES], float)
    pw = pw / pw.sum()
    pheno_i = rng.choice(len(PHENOTYPES), n, p=pw)
    pheno = np.array(PHENOTYPES)[pheno_i]

    # autonomic responsiveness: phenotype sets the mean, subject draws around it
    auto_mean = np.select(
        [pheno_i == 0, pheno_i == 1, pheno_i == 2], [1.42, 1.00, 0.60]
    )
    autonomic_gain = np.clip(auto_mean * _lognormal(rng, 1.0, 0.16, n), 0.28, 2.2)
    # age blunts the baroreflex
    autonomic_gain *= np.clip(1.18 - 0.0042 * age, 0.62, 1.05)

    # cardiac reserve (inotropic headroom); poor compensators have least
    res_mean = np.select([pheno_i == 0, pheno_i == 1, pheno_i == 2], [1.15, 1.0, 0.70])
    cardiac_reserve = np.clip(res_mean * _lognormal(rng, 1.0, 0.18, n), 0.25, 1.8)
    cardiac_reserve *= np.clip(1.20 - 0.0045 * age, 0.6, 1.1)

    # early sympathetic withdrawal threshold (fraction of volume lost).
    # Poor compensators decompensate abruptly and early.
    withdraw_at = np.select(
        [pheno_i == 0, pheno_i == 1, pheno_i == 2], [0.415, 0.355, 0.265]
    ) + rng.normal(0, 0.030, n)
    withdraw_at = np.clip(withdraw_at, 0.16, 0.55)

    # ---------------------------------------------------------------- medication
    # beta blockade blunts chronotropic response and lowers resting HR
    p_bb = np.clip(0.06 + 0.0055 * (age - 40), 0.03, 0.42)
    beta_blocked = rng.random(n) < p_bb
    bb_strength = np.where(beta_blocked, rng.uniform(0.35, 0.85, n), 0.0)
    # calcium-channel / other negative chronotropes lumped in
    on_vasodilator = rng.random(n) < 0.16

    # ---------------------------------------------------------------- baselines
    hr0 = np.clip(
        rng.normal(ph.hr_bpm + 0.06 * (age - 56) - 4.0 * sex, 9.5, n)
        * (1 - 0.16 * bb_strength),
        44, 108,
    )
    map0 = np.clip(rng.normal(ph.map_mmhg + 0.11 * (age - 56), 9.0, n), 65, 118)
    map0 *= np.where(on_vasodilator, 0.94, 1.0)
    sv0 = np.clip(rng.normal(ph.stroke_volume_ml, 10.0, n) * (bsa / 1.85), 35, 115)
    blood_volume = np.clip(
        (70.0 * sex + 62.0 * (1 - sex)) * weight / 1000.0 * _lognormal(rng, 1.0, 0.06, n),
        2.9, 7.2,
    )
    hb0 = np.clip(rng.normal(ph.hb_g_dl + 0.9 * sex, 1.4, n), 8.5, 18.0)
    # resting vagal tone falls with age -> baseline HRV
    para_rest = np.clip(ph.para_rest * (1.35 - 0.0060 * age) * _lognormal(rng, 1.0, 0.20, n), 0.12, 0.95)
    rsa_gain = ec.rsa_gain * np.clip(_lognormal(rng, 1.0, 0.35, n), 0.25, 2.6) * (1.30 - 0.0058 * age)
    rsa_gain = np.clip(rsa_gain, 0.004, 0.16)
    resp0 = np.clip(rng.normal(ph.resp_rate_bpm, 2.4, n), 8.5, 22.0)

    # conduction characteristics
    pr0 = np.clip(rng.normal(ec.pr_s, 0.021, n) + 0.00025 * (age - 56), 0.098, 0.245)
    qrs0 = np.clip(rng.normal(ec.qrs_dur_s, 0.011, n), 0.062, 0.145)
    qt0 = np.clip(rng.normal(ec.qt_s + 0.012 * (1 - sex), 0.024, n), 0.30, 0.47)
    qt_alpha = np.clip(rng.normal(ec.coupling.qt_rate_alpha, 0.055, n), 0.25, 0.58)

    # ---------------------------------------------------------------- morphology
    amp_scale = _lognormal(rng, 1.0, 0.22, n)          # body-habitus / electrode factor
    amp_scale *= np.clip(1.25 - 0.0045 * weight, 0.55, 1.35)
    p_amp = ec.p_amp_mv * amp_scale * _lognormal(rng, 1.0, 0.18, n)
    r_amp = ec.r_amp_mv * amp_scale * _lognormal(rng, 1.0, 0.20, n)
    q_amp = ec.q_amp_mv * amp_scale * _lognormal(rng, 1.0, 0.30, n)
    s_amp = ec.s_amp_mv * amp_scale * _lognormal(rng, 1.0, 0.30, n)
    t_amp = ec.t_amp_mv * amp_scale * _lognormal(rng, 1.0, 0.24, n)
    p_dur = np.clip(rng.normal(ec.p_dur_s, 0.010, n), 0.06, 0.135)
    t_dur = np.clip(rng.normal(ec.t_dur_s, 0.016, n), 0.11, 0.23)

    # ---------------------------------------------------------------- noise level
    noise_level = _lognormal(rng, 1.0, 0.55, n)        # per-subject recording quality
    wander = rng.uniform(*ec.wander_mv_range, n) * noise_level
    emg = rng.uniform(*ec.emg_mv_range, n) * noise_level
    powerline = rng.uniform(*ec.powerline_mv_range, n)

    df = pd.DataFrame(
        dict(
            subject=np.arange(n),
            age=age, sex=sex, weight=weight, height=height, bsa=bsa,
            phenotype=pheno,
            autonomic_gain=autonomic_gain, cardiac_reserve=cardiac_reserve,
            withdraw_at=withdraw_at,
            beta_blocked=beta_blocked.astype(int), bb_strength=bb_strength,
            on_vasodilator=on_vasodilator.astype(int),
            hr0=hr0, map0=map0, sv0=sv0, blood_volume=blood_volume, hb0=hb0,
            para_rest=para_rest, rsa_gain=rsa_gain, resp0=resp0,
            pr0=pr0, qrs0=qrs0, qt0=qt0, qt_alpha=qt_alpha,
            p_amp=p_amp, q_amp=q_amp, r_amp=r_amp, s_amp=s_amp, t_amp=t_amp,
            p_dur=p_dur, t_dur=t_dur,
            amp_scale=amp_scale, noise_level=noise_level,
            wander_mv=wander, emg_mv=emg, powerline_mv=powerline,
        )
    )

    # ---------------------------------------------------------------- personal couplings
    # Each latent->feature arrow gets a subject-specific gain. Sign-labile arrows
    # are drawn from a distribution straddling zero so that no single ECG feature
    # moves in a fixed direction with haemorrhage across the cohort.
    for key, mean in cfg.ecg.coupling.items():
        if key == "qt_rate_alpha":
            continue                                    # already drawn above
        if key in SIGN_LABILE:
            df["c_" + key] = rng.normal(mean, abs(mean) * 1.15 + 0.05, n)
        else:
            df["c_" + key] = mean * _lognormal(rng, 1.0, 0.30, n)
    df["c_rsa_gain"] = rsa_gain
    df["c_lf_gain"] = ec.lf_gain * _lognormal(rng, 1.0, 0.40, n)
    df["c_rr_jitter"] = ec.rr_jitter * _lognormal(rng, 1.0, 0.35, n)
    df["c_ramp_resp"] = ec.ramp_resp_gain * _lognormal(rng, 1.0, 0.40, n)

    # ---------------------------------------------------------------- scenario
    df = _assign_scenarios(cfg, df, rng)
    return df


def _assign_scenarios(cfg, df, rng) -> pd.DataFrame:
    n = len(df)
    hem, trt = cfg.hemorrhage, cfg.treatment

    bleeder = rng.random(n) >= cfg.cohort.frac_nonbleeder
    # among non-bleeders, some get a pure sympathetic stressor (no volume loss)
    stressor = (~bleeder) & (rng.random(n) < cfg.cohort.frac_stressor_among_nonbleeders)

    rate = rng.lognormal(hem.rate_lognorm_mu, hem.rate_lognorm_sigma, n)
    rate = np.clip(rate, 0.008, 0.32) * bleeder            # L/min
    onset = rng.uniform(*hem.onset_min_range, n)
    spont_stop = bleeder & (rng.random(n) < hem.frac_spontaneous_stop)
    stop_min = onset + rng.uniform(*hem.stop_after_min_range, n)
    stop_min = np.where(spont_stop, stop_min, 1e9)

    treated = bleeder & (rng.random(n) < trt.frac_treated)
    rx_delay = rng.uniform(*trt.detect_delay_min_range, n)
    fluid_rate = rng.uniform(*trt.fluid_rate_l_per_min_range, n) * treated
    blood_products = treated & (rng.random(n) < trt.frac_blood_products)

    stressor_onset = rng.uniform(10, 45, n)
    stressor_amp = np.where(stressor, rng.uniform(0.25, 1.10, n), 0.0)
    stressor_dur = rng.uniform(8, 40, n)

    # The recording does not start when the illness does.  Drawing an observation
    # offset decouples "how far into the recording" from "how far into the bleed";
    # without it the window index alone predicts the outcome.
    sim_min = cfg.simulation.duration_min
    obs_min = cfg.simulation.get("obs_window_min", sim_min)
    obs_start = rng.uniform(0.0, max(sim_min - obs_min, 0.0), n)

    reb_p = cfg.hemorrhage.get("rebleed_prob", 0.0)
    rebleed = spont_stop & (rng.random(n) < reb_p) if reb_p > 0 else np.zeros(n, bool)
    reb_start = stop_min + rng.uniform(*cfg.hemorrhage.get("rebleed_gap_min_range", [10, 40]), n)
    reb_start = np.where(rebleed, reb_start, 1e9)
    reb_end = reb_start + rng.uniform(*cfg.hemorrhage.get("rebleed_dur_min_range", [10, 60]), n)

    df = df.assign(
        obs_start_min=obs_start,
        rebleed=rebleed.astype(int), rebleed_start_min=reb_start, rebleed_end_min=reb_end,
        bleeder=bleeder.astype(int),
        stressor=stressor.astype(int),
        hemorrhage_rate=rate,
        onset_min=np.where(bleeder, onset, 1e9),
        stop_min=stop_min,
        treated=treated.astype(int),
        rx_delay_min=rx_delay,
        fluid_rate=fluid_rate,
        blood_products=blood_products.astype(int),
        stressor_onset_min=np.where(stressor, stressor_onset, 1e9),
        stressor_amp=stressor_amp,
        stressor_dur_min=stressor_dur,
        haemostasis_delay_min=rng.uniform(*trt.haemostasis_delay_min_range, n),
    )
    return df


def assign_splits(cfg, df, seed=None) -> pd.DataFrame:
    """Patient-level split, stratified on phenotype x bleeder."""
    rng = np.random.default_rng((cfg.run.seed if seed is None else seed) + 7)
    strat = df.phenotype.astype(str) + "_" + df.bleeder.astype(str)
    split = np.empty(len(df), dtype=object)
    ftr, fva = cfg.cohort.frac_train, cfg.cohort.frac_val
    for key in strat.unique():
        idx = np.where(strat.values == key)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr, n_va = int(round(ftr * n)), int(round(fva * n))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "val"
        split[idx[n_tr + n_va:]] = "test"
    return df.assign(split=split)
