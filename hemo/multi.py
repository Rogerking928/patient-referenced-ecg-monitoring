"""Multi-aetiology extension: several systemic illnesses through one latent machine.

The hypothesis under test is not "can an ECG detect haemorrhage" but "is an ECG a
systemic-state sensor".  The sharpest form of that question is whether the ECG can
distinguish aetiologies that produce *similar heart rates through opposite
mechanisms*:

    haemorrhage      preload down,   resistance up,   output down
    sepsis           preload down/normal, resistance DOWN, output up then down
    cardiogenic      preload UP (congestion), resistance up, output down
    hypoxaemia       preload normal, resistance normal, output up
    anaemia          preload normal, resistance slightly down, output up
    hyperkalaemia    haemodynamics near normal; a direct ion-channel effect on the
                     waveform instead -- included precisely because it is the one
                     aetiology that does NOT act through the circulation
    chronic_systemic anaemia + low-grade inflammation + cachexia, static, no acute
                     event.  This is the "cancer-like" phenotype.  It is named after
                     its mechanism, not after a disease, because the coupling
                     strengths from a tumour to any of these mediators are not
                     quantified anywhere -- simulating "cancer" directly would only
                     return whatever was assumed.

Direct (non-haemodynamic) waveform effects are applied by rewriting the subject's
own morphology parameters for that window, so `ecg.render_window` is reused
unchanged.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from . import subjects as subj_mod

AETIOLOGIES = ["none", "haemorrhage", "sepsis", "cardiogenic", "hypoxaemia",
               "anaemia", "hyperkalaemia", "chronic_systemic"]
AET_ID = {a: i for i, a in enumerate(AETIOLOGIES)}

STATE_NAMES = [
    "blood_volume", "venous_return", "stroke_volume", "cardiac_output",
    "sympathetic_tone", "parasympathetic_tone", "respiratory_rate",
    "oxygenation", "metabolic_stress", "sao2", "inflammation", "potassium",
    "haemoglobin",
]
DERIVED_NAMES = ["heart_rate", "svr", "map", "pulse_pressure", "reserve",
                 "volume_lost_frac", "preload", "filling", "contractility", "do2"]


def merge_cfg(base, extra):
    """Deep-merge the multi-arm overrides on top of the default configuration."""
    out = copy.deepcopy(base)

    def rec(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                rec(a[k], v)
            else:
                a[k] = v
    rec(out, extra)
    return out


# ------------------------------------------------------------------ cohort
def make_cohort(cfg, seed=None):
    """Base subjects, then an aetiology, severity and timing for each."""
    rng = np.random.default_rng((cfg.run.seed if seed is None else seed) + 31)
    df = subj_mod.make_cohort(cfg, seed=cfg.run.seed)
    n = len(df)
    ac = cfg.aetiology

    names = list(ac.mix.keys())
    p = np.array([ac.mix[k] for k in names], float)
    p = p / p.sum()
    primary = rng.choice(names, n, p=p)

    a, b = ac.severity_beta
    sev = rng.beta(a, b, n)
    onset = rng.uniform(*ac.onset_min_range, n)
    ramp = rng.uniform(*ac.ramp_min_range, n)
    sev = np.where(primary == "none", 0.0, sev)

    # second, independent insult
    comorbid = np.where(rng.random(n) < ac.frac_comorbid,
                        rng.choice(ac.comorbid_pool, n), "none")
    comorbid = np.where(primary == "none", "none", comorbid)
    comorbid = np.where(comorbid == primary, "none", comorbid)
    sev2 = np.where(comorbid == "none", 0.0, rng.beta(a, b + 0.8, n))
    onset2 = rng.uniform(*ac.onset_min_range, n)
    ramp2 = rng.uniform(*ac.ramp_min_range, n)

    df = df.assign(aetiology=primary, severity_peak=sev,
                   aet_onset_min=onset, aet_ramp_min=ramp,
                   comorbid=comorbid, comorbid_severity=sev2,
                   comorbid_onset_min=onset2, comorbid_ramp_min=ramp2)
    # the haemorrhage arm reuses the existing bleed machinery; everything else
    # drives its own pathway, so those subjects must not also bleed
    df["bleeder"] = (df.aetiology == "haemorrhage").astype(int)
    # scale the bleed rate by the drawn severity so that "severity" means the same
    # thing across aetiologies
    df["hemorrhage_rate"] = np.where(
        df.bleeder == 1, df.hemorrhage_rate * (0.35 + 1.75 * df.severity_peak), 0.0)
    df["onset_min"] = np.where(df.bleeder == 1, df.aet_onset_min, 1e9)
    df["stop_min"] = np.where(df.bleeder == 1, df.stop_min, 1e9)
    df["treated"] = df.treated * df.bleeder
    df["fluid_rate"] = df.fluid_rate * df.bleeder
    return df


def severity_at(cfg, kind, sev_peak, onset, ramp, t_min):
    """Severity ramp for one insult at one time (vectorised over subjects)."""
    if cfg.aetiology.chronic_is_static:
        static = np.asarray(kind) == "chronic_systemic"
    else:
        static = np.zeros(len(np.atleast_1d(kind)), bool)
    frac = np.clip((t_min - onset) / np.maximum(ramp, 1e-6), 0.0, 1.0)
    return np.where(static, sev_peak, sev_peak * frac)


# ------------------------------------------------------------------ simulator
def simulate(cfg, subj, seed=None):
    """Latent trajectories under an arbitrary mixture of systemic insults."""
    ph, pw, trt = cfg.physiology, cfg.pathways, cfg.treatment
    rng = np.random.default_rng((cfg.run.seed if seed is None else seed) + 211)
    dt = float(cfg.simulation.dt_s)
    n_steps = int(cfg.simulation.duration_min * 60 / dt)
    n = len(subj)
    t_min = np.arange(n_steps) * dt / 60.0

    s = {c: subj[c].to_numpy() for c in subj.columns if subj[c].dtype != object}
    kind1 = subj.aetiology.to_numpy()
    kind2 = subj.comorbid.to_numpy()

    V0 = s["blood_volume"]
    V_uns = ph.unstressed_fraction * V0
    Vs0 = V0 - V_uns
    SV0, HR0, MAP0, Hb0 = s["sv0"], s["hr0"], s["map0"], s["hb0"]
    CO0 = SV0 * HR0 / 1000.0
    SVR0 = (MAP0 - ph.cvp_mmhg) / np.maximum(CO0, 0.5)
    SAO2_0 = 0.975
    DO20 = CO0 * Hb0 * SAO2_0
    para0, resp0 = s["para_rest"], s["resp0"]
    S_rest, S_max = ph.symp_rest, ph.symp_max

    V, Hb = V0.copy(), Hb0.copy()
    S = np.full(n, S_rest) + rng.normal(0, 0.012, n)
    P, RR_resp = para0.copy(), resp0.copy()
    M = np.zeros(n)
    K = np.full(n, pw.potassium_baseline)
    SaO2 = np.full(n, SAO2_0)
    baro_int = np.zeros(n)
    env_prev = np.zeros(n)
    ox_prev = np.ones(n)
    rx_started = np.full(n, np.inf)
    map_trigger = trt.trigger_map_frac * MAP0

    bb = s["bb_strength"]
    hr_symp_gain = ph.hr_symp_gain * (1.0 - 0.62 * bb)
    svr_symp_gain = ph.svr_symp_gain * (1.0 - 0.15 * bb)
    inotropy = ph.inotropy_gain * (1.0 - 0.45 * bb) * s["cardiac_reserve"]

    Z = np.empty((n, n_steps, len(STATE_NAMES)), np.float32)
    D = np.empty((n, n_steps, len(DERIVED_NAMES)), np.float32)
    SEV = np.empty((n, n_steps, 2), np.float32)

    for i in range(n_steps):
        tm = t_min[i]
        sev1 = severity_at(cfg, kind1, s["severity_peak"], s["aet_onset_min"],
                           s["aet_ramp_min"], tm)
        sev2 = severity_at(cfg, kind2, s["comorbid_severity"], s["comorbid_onset_min"],
                           s["comorbid_ramp_min"], tm)
        SEV[:, i, 0], SEV[:, i, 1] = sev1, sev2

        def sev_of(name):
            return np.where(kind1 == name, sev1, 0.0) + np.where(kind2 == name, sev2, 0.0)

        s_sep = np.clip(sev_of("sepsis"), 0, 1)
        s_car = np.clip(sev_of("cardiogenic"), 0, 1)
        s_hyp = np.clip(sev_of("hypoxaemia"), 0, 1)
        s_ana = np.clip(sev_of("anaemia"), 0, 1)
        s_kal = np.clip(sev_of("hyperkalaemia"), 0, 1)
        s_chr = np.clip(sev_of("chronic_systemic"), 0, 1)

        # ---------------------------------------------- volume
        haemostasis = rx_started + s["haemostasis_delay_min"]
        bleeding = (tm >= s["onset_min"]) & (tm < s["stop_min"]) & (tm < haemostasis)
        loss = s["hemorrhage_rate"] * bleeding * dt / 60.0
        leak = pw.sepsis_leak_l_per_min * s_sep * dt / 60.0
        lost_frac = np.clip((V0 - V) / V0, 0.0, 0.95)
        refill = ph.refill_gain * lost_frac * V0 * dt
        rx_on = tm >= rx_started
        gain_fluid = s["fluid_rate"] * rx_on * dt / 60.0
        eff_gain = gain_fluid * np.where(s["blood_products"] > 0, 1.0, trt.intravascular_fraction)
        V = np.clip(V + refill + eff_gain - loss - leak + rng.normal(0, ph.noise_volume_l, n),
                    0.35 * V0, 1.25 * V0)

        # ---------------------------------------------- haemoglobin / oxygen carriage
        hb_target = Hb0 * (1.0 - pw.anaemia_hb_loss * s_ana - pw.chronic_hb_loss * s_chr)
        d_hb = (hb_target - Hb) / 900.0 * dt                       # ~15 min time constant
        d_hb -= Hb * (loss + refill) / np.maximum(V, 0.5)
        d_hb -= Hb * (gain_fluid * trt.intravascular_fraction) / np.maximum(V, 0.5) * (s["blood_products"] == 0)
        d_hb += (12.0 - Hb) * gain_fluid / np.maximum(V, 0.5) * (s["blood_products"] > 0)
        Hb = np.clip(Hb + d_hb, 2.5, 19.0)
        SaO2 = np.clip(SAO2_0 - pw.hypox_sao2_drop * s_hyp, 0.45, 1.0)

        # ---------------------------------------------- potassium (direct ECG pathway)
        K = pw.potassium_baseline + pw.kalaemia_rise * s_kal
        k_norm = np.clip((K - pw.potassium_baseline) / pw.kalaemia_rise, 0, 1)

        # ---------------------------------------------- inflammation
        I = np.clip(s_sep + pw.chronic_inflammation * s_chr, 0, 1.2)

        # ---------------------------------------------- pump and vessels
        # Chemoreflex drive from arterial desaturation, from low oxygen delivery
        # (the anaemic route) and from acidosis.  It acts on rate, vagal tone and
        # ventilation DIRECTLY, not through the pressure-regulating loop -- if it
        # went through the baroreflex the integral term would cancel it, and an
        # anaemic or hypoxaemic patient would come out with a normal heart rate.
        chemo = np.clip(pw.chemo_sao2_gain * np.clip(SAO2_0 - SaO2, 0, 1)
                        + pw.chemo_do2_gain * np.clip(1.0 - ox_prev, 0, 1)
                        + pw.chemo_lactate_gain * np.clip(M / 8.0, 0, 1), 0.0, 1.5)
        HR = HR0 * (1.0 + hr_symp_gain * (S - S_rest) + ph.central_hr_gain * env_prev
                    + pw.infl_hr_gain * I + pw.chemo_hr_gain * chemo) \
            / (1.0 + ph.hr_para_gain * (P - para0))
        HR = HR * (1.0 - pw.kalaemia_brady * k_norm)
        HR = np.clip(HR, 30.0, 195.0 * (1 - 0.20 * bb))

        t_dia = np.maximum(60.0 / HR - ph.systole_k * np.sqrt(60.0 / HR), 0.02)
        t_dia0 = 60.0 / HR0 - ph.systole_k * np.sqrt(60.0 / HR0)
        fill = np.clip((t_dia / t_dia0) ** ph.filling_exp, 0.30, 1.08)

        mobilised = ph.mobilisable_l * S * (1.0 - 0.25 * bb)
        Vs = np.maximum(V - V_uns + mobilised, 0.02)
        # cardiogenic congestion raises filling volume even though total volume is normal
        Vs = Vs * (1.0 + pw.cardio_congestion * s_car)
        preload = Vs / (Vs0 + ph.mobilisable_l * S_rest)

        depress = pw.sepsis_depression * np.clip(
            (s_sep - pw.sepsis_depression_delay) / max(1 - pw.sepsis_depression_delay, 1e-6), 0, 1)
        contract = (1.0 + inotropy * (S - S_rest)) * (1.0 - pw.cardio_contractility_loss * s_car) \
            * (1.0 - depress)
        SV = np.clip(SV0 * _starling(preload * fill, ph.starling_hill, ph.starling_k) * contract,
                     6.0, 1.8 * SV0)
        CO = SV * HR / 1000.0

        M_norm = np.clip(M / 8.0, 0, 1)
        SVR = (SVR0 * (1.0 + svr_symp_gain * (S - S_rest)) * (1.0 - ph.svr_acidosis_loss * M_norm)
               * (1.0 - pw.sepsis_svr_loss * s_sep) * (1.0 - pw.anaemia_viscosity_svr * s_ana))
        MAP = ph.cvp_mmhg + CO * SVR
        PP = 0.42 * SV / np.maximum(SVR / SVR0, 0.35) * (MAP0 / 90.0)

        # ---------------------------------------------- metabolism
        DO2 = CO * Hb * SaO2
        ox = DO2 / DO20
        deficit = np.clip(1.0 - ox / ph.do2_crit_frac, 0.0, 1.0)
        # sepsis raises lactate through impaired microcirculatory extraction, not
        # through global oxygen delivery, so it needs its own production term.
        # Written as a target concentration so the config value is interpretable.
        prod = ph.lactate_prod * deficit + ph.lactate_clear * (
            pw.sepsis_lactate * s_sep + pw.chronic_metabolic * s_chr)
        M = np.clip(M + (prod - ph.lactate_clear * M) * dt, 0.0, 18.0)

        # ---------------------------------------------- autonomic control
        in_stress = (tm >= s["stressor_onset_min"]) & (tm < s["stressor_onset_min"] + s["stressor_dur_min"])
        env = s["stressor_amp"] * in_stress * np.sin(
            np.pi * np.clip((tm - s["stressor_onset_min"]) / np.maximum(s["stressor_dur_min"], 1e-6), 0, 1))
        err = (MAP0 - MAP) / MAP0
        baro_int = np.clip(baro_int + err * dt / ph.baro_tau_i, -0.6, ph.baro_int_max)
        # chronic inflammation degrades the reflex itself
        g_auto = s["autonomic_gain"] * (1.0 - pw.chronic_baro_loss * s_chr)
        drive = (ph.baro_gain * g_auto * err + ph.baro_int_gain * g_auto * baro_int
                 + ph.chemo_gain * M_norm)
        gate = 1.0 / (1.0 + np.exp(-(lost_frac - s["withdraw_at"]) / ph.withdraw_width))
        withdraw = ph.withdraw_depth * gate
        S_target = np.clip(S_rest + (S_max - S_rest) * np.clip(drive, 0, None), S_rest, S_max)
        S_target = S_target * (1.0 - withdraw) + S_rest * withdraw
        S = np.clip(S + (S_target - S) / ph.tau_symp_s * dt + rng.normal(0, ph.noise_symp, n), 0, 1)

        P_target = np.clip(para0 * (1.0 - 1.15 * (S - S_rest) - ph.central_vagal_gain * env
                                    - pw.infl_vagal_loss * I
                                    - pw.chemo_vagal_loss * chemo) + 0.30 * withdraw, 0.02, 1.0)
        P = np.clip(P + (P_target - P) / ph.tau_para_s * dt + rng.normal(0, ph.noise_para, n), 0, 1)

        RR_t = (resp0 + ph.resp_metab_gain * M_norm + ph.resp_symp_gain * (S - S_rest)
                + pw.hypox_resp_gain * s_hyp + pw.chemo_resp_gain * chemo)
        RR_resp = np.clip(RR_resp + (RR_t - RR_resp) / 12.0 * dt + rng.normal(0, ph.noise_resp, n),
                          6.0, 48.0)

        newly = (~np.isfinite(rx_started)) & (s["treated"] > 0) & (
            (MAP < map_trigger) | (HR > trt.trigger_hr_frac * HR0))
        rx_started = np.where(newly, tm + s["rx_delay_min"], rx_started)
        env_prev, ox_prev = env, ox

        Z[:, i] = np.stack([V, CO, SV, CO, S, P, RR_resp, ox, M, SaO2, I, K, Hb], 1)
        D[:, i] = np.stack([HR, SVR, MAP, PP,
                            np.clip(1.0 - (S - S_rest) / (S_max - S_rest), 0, 1),
                            lost_frac, preload * fill, fill, contract, ox], 1)
    return Z, D, t_min, SEV


def _starling(x, hill, k):
    x = np.clip(x, 1e-3, 3.0)
    return (x ** hill / (x ** hill + k ** hill)) * (1.0 + k ** hill)


def viability_mask(cfg, D):
    dead = D[:, :, 2] < cfg.staging.map_nonviable
    return ~(np.maximum.accumulate(dead.astype(np.int8), axis=1) > 0)


def compromise_label(cfg, subj, Z, D):
    """Generic 'the circulation is failing' flag, aetiology-agnostic."""
    from .physiology import _trailing_mean
    k = max(int(cfg.staging.stage_average_s / cfg.simulation.dt_s), 1)
    mf = _trailing_mean(D[:, :, 2] / subj.map0.to_numpy()[:, None], k)
    la = _trailing_mean(Z[:, :, 8], k)
    res = _trailing_mean(D[:, :, 4], k)
    # "compromised" means the circulation is failing OR is only being held together
    # by near-maximal autonomic recruitment.  The reserve arm is what makes the
    # label comparable across aetiologies: a compensated cardiogenic patient and a
    # compensated bleeding patient are both precarious even with a normal pressure.
    return ((mf < cfg.multi_tasks.compromise_map_frac)
            | (la > cfg.multi_tasks.compromise_lactate)
            | (res < cfg.multi_tasks.compromise_reserve)).astype(np.int8)


# ------------------------------------------------------------------ direct ECG effects
def window_row(cfg, row, k_norm, sev_chronic, inflammation):
    """Rewrite a subject's morphology parameters for the direct, non-haemodynamic
    effects.  `ecg.render_window` is then used unchanged."""
    pw = cfg.pathways
    r = row.copy()
    if k_norm > 1e-4:
        # hyperkalaemia: tall narrow T, flattening P, lengthening PR, widening QRS
        r["t_amp"] = r["t_amp"] * (1.0 + 1.15 * k_norm)
        r["t_dur"] = r["t_dur"] * (1.0 - 0.38 * k_norm)
        r["p_amp"] = r["p_amp"] * (1.0 - 0.88 * k_norm)
        r["pr0"] = min(r["pr0"] * (1.0 + 0.32 * k_norm), 0.30)
        r["qrs0"] = min(r["qrs0"] * (1.0 + 0.48 * k_norm), 0.19)
    if sev_chronic > 1e-4:
        v = 1.0 - pw.chronic_cachexia_voltage * sev_chronic
        for key in ["p_amp", "q_amp", "r_amp", "s_amp", "t_amp"]:
            r[key] = r[key] * v
    if inflammation > 1e-4:
        f = 1.0 - pw.infl_hrv_loss * min(inflammation, 1.0)
        r["c_rsa_gain"] = r["c_rsa_gain"] * f
        r["c_rr_jitter"] = r["c_rr_jitter"] * f
    return r
