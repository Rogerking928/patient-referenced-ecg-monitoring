"""Latent haemodynamic state-space model.

    z(t+1) = F(z(t), hemorrhage_rate(t), compensation, treatment) + w(t)

State vector (order fixed, see STATE_NAMES):

    0 blood_volume       L
    1 venous_return      L/min
    2 stroke_volume      mL
    3 cardiac_output     L/min
    4 sympathetic_tone   0..1
    5 parasympathetic    0..1
    6 respiratory_rate   breaths/min
    7 oxygenation        DO2 / DO2_baseline  (0..~1.2)
    8 metabolic_stress   lactate-like, mmol/L

Derived (not part of z, computed from it): heart rate, SVR, MAP, pulse
pressure, autonomic reserve.  Disease state never touches the ECG directly:
haemorrhage moves z, and only z reaches the observation model.

Integrated with explicit Euler, dt = 1 s, vectorised across the whole cohort.
"""
from __future__ import annotations

import numpy as np

STATE_NAMES = [
    "blood_volume", "venous_return", "stroke_volume", "cardiac_output",
    "sympathetic_tone", "parasympathetic_tone", "respiratory_rate",
    "oxygenation", "metabolic_stress",
]
DERIVED_NAMES = ["heart_rate", "svr", "map", "pulse_pressure", "reserve", "hb",
                 "volume_lost_frac", "preload", "filling"]

STAGES = ["baseline", "compensated", "early_decomp", "overt_shock", "recovery"]
STAGE_ID = {s: i for i, s in enumerate(STAGES)}


def _starling(rel_preload, hill, k):
    """Frank-Starling curve normalised so f(1) == 1."""
    x = np.clip(rel_preload, 1e-3, 3.0)
    f = x ** hill / (x ** hill + k ** hill)
    f1 = 1.0 / (1.0 + k ** hill)
    return f / f1


def simulate_cohort(cfg, subj, seed=None):
    """Run every subject's trajectory.

    Returns
    -------
    Z    (n_subj, n_steps, 9)  latent states
    D    (n_subj, n_steps, 9)  derived variables
    t    (n_steps,)            time in minutes
    bh   (n_subj, n_steps)     bleeding indicator (after haemostasis it is False)
    """
    ph, hemcfg, trt = cfg.physiology, cfg.hemorrhage, cfg.treatment
    rng = np.random.default_rng((cfg.run.seed if seed is None else seed) + 101)

    dt = float(cfg.simulation.dt_s)
    n_steps = int(cfg.simulation.duration_min * 60 / dt)
    n = len(subj)
    t_min = np.arange(n_steps) * dt / 60.0

    s = {c: subj[c].to_numpy() for c in subj.columns if subj[c].dtype != object}

    # ------------------------------------------------------------ baselines
    V0 = s["blood_volume"]
    V_unstressed = ph.unstressed_fraction * V0
    Vs0 = V0 - V_unstressed                        # baseline stressed volume
    SV0, HR0, MAP0, Hb0 = s["sv0"], s["hr0"], s["map0"], s["hb0"]
    CO0 = SV0 * HR0 / 1000.0
    SVR0 = (MAP0 - ph.cvp_mmhg) / np.maximum(CO0, 0.5)
    DO20 = CO0 * Hb0
    para0 = s["para_rest"]
    resp0 = s["resp0"]
    S_rest, S_max = ph.symp_rest, ph.symp_max

    # ------------------------------------------------------------ init state
    V = V0.copy()
    Hb = Hb0.copy()
    S = np.full(n, S_rest) + rng.normal(0, 0.012, n)
    P = para0.copy()
    RR_resp = resp0.copy()
    M = np.zeros(n)                                 # metabolic stress
    crystalloid_in = np.zeros(n)

    Z = np.empty((n, n_steps, len(STATE_NAMES)), np.float32)
    D = np.empty((n, n_steps, len(DERIVED_NAMES)), np.float32)

    rx_started = np.full(n, np.inf)
    bleed_hist = np.zeros((n, n_steps), bool)
    env_prev = np.zeros(n)
    baro_int = np.zeros(n)
    map_trigger = trt.trigger_map_frac * MAP0

    bb = s["bb_strength"]
    hr_symp_gain = ph.hr_symp_gain * (1.0 - 0.62 * bb)
    svr_symp_gain = ph.svr_symp_gain * (1.0 - 0.15 * bb)
    inotropy = ph.inotropy_gain * (1.0 - 0.45 * bb) * s["cardiac_reserve"]

    for i in range(n_steps):
        tm = t_min[i]

        # ---------------------------------------------- volume in / out
        haemostasis = rx_started + s["haemostasis_delay_min"]
        bleeding = (tm >= s["onset_min"]) & (tm < s["stop_min"]) & (tm < haemostasis)
        if "rebleed_start_min" in s:
            bleeding |= ((tm >= s["rebleed_start_min"]) & (tm < s["rebleed_end_min"])
                         & (tm < haemostasis))
        bleed_hist[:, i] = bleeding
        loss = s["hemorrhage_rate"] * bleeding * dt / 60.0

        lost_frac = np.clip((V0 - V) / V0, 0.0, 0.95)
        refill = ph.refill_gain * lost_frac * V0 * dt          # transcapillary refill

        rx_on = tm >= rx_started
        gain_fluid = s["fluid_rate"] * rx_on * dt / 60.0
        eff_gain = gain_fluid * np.where(
            s["blood_products"] > 0, 1.0, trt.intravascular_fraction
        )
        crystalloid_in += gain_fluid * (s["blood_products"] == 0)

        V = V + refill + eff_gain - loss + rng.normal(0, ph.noise_volume_l, n)
        V = np.clip(V, 0.35 * V0, 1.25 * V0)

        # haemoglobin: lost with whole blood, diluted by crystalloid, restored by product
        d_hb = np.zeros(n)
        d_hb -= Hb * loss / np.maximum(V, 0.5)                  # whole-blood loss is iso-Hb
        d_hb -= Hb * refill / np.maximum(V, 0.5)                # interstitial fluid dilutes
        d_hb -= Hb * (gain_fluid * trt.intravascular_fraction) / np.maximum(V, 0.5) * (s["blood_products"] == 0)
        d_hb += (12.0 - Hb) * gain_fluid / np.maximum(V, 0.5) * (s["blood_products"] > 0)
        Hb = np.clip(Hb + d_hb, 2.5, 19.0)

        # ---------------------------------------------- venous return / preload
        mobilised = ph.mobilisable_l * S * (1.0 - 0.25 * bb)
        Vs = np.maximum(V - V_unstressed + mobilised, 0.02)
        preload = Vs / (Vs0 + ph.mobilisable_l * S_rest)

        # central command acts on rate and vagal tone but NOT on vasomotor tone,
        # which is why a stressor can mimic the tachycardia of hypovolaemia
        # without mimicking its preload signature.
        HR = HR0 * (1.0 + hr_symp_gain * (S - S_rest) + ph.central_hr_gain * env_prev) \
             / (1.0 + ph.hr_para_gain * (P - para0))
        HR = np.clip(HR, 33.0, 195.0 * (1 - 0.20 * bb))

        # diastolic filling time shortens as rate rises, so tachycardia buys
        # progressively less cardiac output -- the reason extreme tachycardia
        # stops compensating, and the reason a stressor does not send MAP to 160.
        t_dia = np.maximum(60.0 / HR - ph.systole_k * np.sqrt(60.0 / HR), 0.02)
        t_dia0 = 60.0 / HR0 - ph.systole_k * np.sqrt(60.0 / HR0)
        fill = np.clip((t_dia / t_dia0) ** ph.filling_exp, 0.30, 1.08)

        SV = SV0 * _starling(preload * fill, ph.starling_hill, ph.starling_k) * (1.0 + inotropy * (S - S_rest))
        SV = np.clip(SV, 6.0, 1.8 * SV0)

        CO = SV * HR / 1000.0
        VR = CO                                                 # steady state
        M_norm = np.clip(M / 8.0, 0, 1)
        SVR = SVR0 * (1.0 + svr_symp_gain * (S - S_rest)) * (1.0 - ph.svr_acidosis_loss * M_norm)
        MAP = ph.cvp_mmhg + CO * SVR
        PP = 0.42 * SV / np.maximum(SVR / SVR0, 0.35) * (MAP0 / 90.0)

        # ---------------------------------------------- oxygen & metabolism
        DO2 = CO * Hb
        ox = DO2 / DO20
        deficit = np.clip(1.0 - ox / ph.do2_crit_frac, 0.0, 1.0)
        M = M + (ph.lactate_prod * deficit - ph.lactate_clear * M) * dt
        M = np.clip(M, 0.0, 18.0)

        # ---------------------------------------------- autonomic control
        # pure sympathetic stressor (pain / anxiety / ambulation), NO volume loss.
        # Modelled the way central command actually works: it resets the
        # baroreflex set-point upward AND feeds forward onto sympathetic outflow,
        # so it is not simply buffered away by the baroreflex.
        in_stress = (tm >= s["stressor_onset_min"]) & (tm < s["stressor_onset_min"] + s["stressor_dur_min"])
        env = s["stressor_amp"] * in_stress * np.sin(
            np.pi * np.clip((tm - s["stressor_onset_min"]) / np.maximum(s["stressor_dur_min"], 1e-6), 0, 1)
        )
        err = (MAP0 - MAP) / MAP0
        # Proportional-plus-integral baroreflex.  A purely proportional reflex has
        # a steady-state error by construction, so sympathetic tone can only rise
        # after pressure has already fallen -- which would make compensated shock
        # unrepresentable.  The integral term is the slow reflex resetting that
        # lets mean pressure be defended while sympathetic recruitment climbs.
        baro_int = np.clip(baro_int + err * dt / ph.baro_tau_i, -0.6, ph.baro_int_max)
        drive = ph.baro_gain * s["autonomic_gain"] * err \
            + ph.baro_int_gain * s["autonomic_gain"] * baro_int + ph.chemo_gain * M_norm

        # decompensation: abrupt sympatho-inhibition once volume loss passes the
        # subject's threshold (Bezold-Jarisch-like); depth scaled by phenotype
        gate = 1.0 / (1.0 + np.exp(-(lost_frac - s["withdraw_at"]) / ph.withdraw_width))
        withdraw = ph.withdraw_depth * gate

        S_target = np.clip(S_rest + (S_max - S_rest) * np.clip(drive, 0, None), S_rest, S_max)
        S_target = S_target * (1.0 - withdraw) + S_rest * withdraw
        S = S + (S_target - S) / ph.tau_symp_s * dt + rng.normal(0, ph.noise_symp, n)
        S = np.clip(S, 0.0, 1.0)

        P_target = np.clip(
            para0 * (1.0 - 1.15 * (S - S_rest) - ph.central_vagal_gain * env) + 0.30 * withdraw,
            0.02, 1.0,
        )
        P = P + (P_target - P) / ph.tau_para_s * dt + rng.normal(0, ph.noise_para, n)
        P = np.clip(P, 0.0, 1.0)

        RR_t = resp0 + ph.resp_metab_gain * M_norm + ph.resp_symp_gain * (S - S_rest)
        RR_resp = RR_resp + (RR_t - RR_resp) / 12.0 * dt + rng.normal(0, ph.noise_resp, n)
        RR_resp = np.clip(RR_resp, 6.0, 45.0)

        # ---------------------------------------------- treatment trigger
        # Clinicians do not wait for the pressure: a sustained tachycardia, or an
        # obvious source of bleeding, triggers resuscitation too.  With a
        # pressure-only trigger, a defended mean pressure means nobody is treated
        # until collapse -- which is the clinical problem this project is about,
        # but it is not what actually happens in a resuscitation bay.
        newly = (~np.isfinite(rx_started)) & (s["treated"] > 0) & (
            (MAP < map_trigger) | (HR > trt.trigger_hr_frac * HR0))
        rx_started = np.where(newly, tm + s["rx_delay_min"], rx_started)

        env_prev = env
        reserve = np.clip(1.0 - (S - S_rest) / (S_max - S_rest), 0.0, 1.0)

        Z[:, i, 0] = V
        Z[:, i, 1] = VR
        Z[:, i, 2] = SV
        Z[:, i, 3] = CO
        Z[:, i, 4] = S
        Z[:, i, 5] = P
        Z[:, i, 6] = RR_resp
        Z[:, i, 7] = ox
        Z[:, i, 8] = M
        D[:, i, 0] = HR
        D[:, i, 1] = SVR
        D[:, i, 2] = MAP
        D[:, i, 3] = PP
        D[:, i, 4] = reserve
        D[:, i, 5] = Hb
        D[:, i, 6] = lost_frac
        D[:, i, 7] = preload * fill        # relative end-diastolic volume
        D[:, i, 8] = fill

    return Z, D, t_min, bleed_hist


def _trailing_mean(A, k):
    """Backward-looking moving average along the time axis."""
    if k <= 1:
        return A
    C = np.cumsum(np.pad(A.astype(np.float64), ((0, 0), (k, 0)), mode="edge"), axis=1)
    return ((C[:, k:] - C[:, :-k]) / k).astype(np.float32)


def stage_labels(cfg, subj, Z, D, t_min, bleed_hist=None):
    """Assign a clinical stage to every time step from the LATENT state only.

    No clock time enters the rule; a subject that never loses volume can never
    leave `baseline` no matter how tachycardic a stressor makes them.
    """
    st = cfg.staging
    MAP0 = subj["map0"].to_numpy()[:, None]
    MAP = D[:, :, 2]
    lost = D[:, :, 6]
    reserve = D[:, :, 4]
    lact = Z[:, :, 8]

    map_frac = MAP / MAP0

    # Stage on a trailing 60-second average, not on an instantaneous sample.
    # Beat-to-beat pressure varies by a few percent, and a clinician does not
    # stage a patient on one heartbeat.  The average is strictly backward-looking,
    # so no future information enters the label.
    k = max(int(st.stage_average_s / cfg.simulation.dt_s), 1)
    map_frac = _trailing_mean(map_frac, k)
    reserve = _trailing_mean(reserve, k)
    lact = _trailing_mean(lact, k)
    lost = _trailing_mean(lost, k)

    shock = (map_frac < st.map_frac_shock) | (lact > st.lactate_shock)
    early = (~shock) & (
        (map_frac < st.map_frac_early) | (reserve < st.reserve_frac_early)
    ) & (lost > 0.03)
    comp = (~shock) & (~early) & (lost >= st.loss_frac_compensated)

    # recovery: has been at least early-decompensated, is no longer bleeding,
    # perfusion pressure has come back, and lactate is being cleared.
    ever_bad = np.maximum.accumulate((shock | early).astype(np.int8), axis=1) > 0
    if bleed_hist is None:
        bleeding = (t_min[None, :] >= subj["onset_min"].to_numpy()[:, None]) & (
            t_min[None, :] < subj["stop_min"].to_numpy()[:, None]
        )
    else:
        bleeding = bleed_hist
    dLact = np.gradient(lact, axis=1)
    recovering = (
        ever_bad & (~bleeding) & (map_frac >= st.recovery_min_map_frac)
        & ((dLact < 0) | (lact < 2.0))
    )

    lab = np.zeros(MAP.shape, np.int8)              # baseline
    lab[comp] = STAGE_ID["compensated"]
    lab[early] = STAGE_ID["early_decomp"]
    lab[shock] = STAGE_ID["overt_shock"]
    lab[recovering] = STAGE_ID["recovery"]
    return lab


def viability_mask(cfg, D):
    """False once mean arterial pressure falls below a non-survivable floor.

    Those time steps are dropped from the dataset rather than being modelled,
    because the state-space model has no cardiac-arrest dynamics and anything
    past that point is extrapolation.
    """
    floor = cfg.staging.map_nonviable
    dead = D[:, :, 2] < floor
    ever_dead = np.maximum.accumulate(dead.astype(np.int8), axis=1) > 0
    return ~ever_dead


def time_to_decomp(cfg, labels, t_min):
    """Minutes until the first `early_decomp`-or-worse step, per time point.

    Returns (tte, event) where event==0 marks right-censoring (never happens
    within the simulated horizon).
    """
    bad = (labels == STAGE_ID["early_decomp"]) | (labels == STAGE_ID["overt_shock"])
    n, T = bad.shape
    tte = np.full((n, T), np.inf, np.float32)
    event = np.zeros((n, T), np.int8)
    for i in range(n):
        idx = np.where(bad[i])[0]
        if len(idx) == 0:
            continue
        # for each t, the next bad index at or after t
        nxt = np.searchsorted(idx, np.arange(T), side="left")
        ok = nxt < len(idx)
        tte[i, ok] = t_min[idx[nxt[ok]]] - t_min[np.arange(T)[ok]]
        event[i, ok] = 1
    tte[~np.isfinite(tte)] = t_min[-1] - t_min[None, :].repeat(n, 0)[~np.isfinite(tte)]
    return tte, event
