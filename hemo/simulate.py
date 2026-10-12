"""Cohort -> dataset.  Renders every ECG window and assembles the label table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import ecg, physiology as phys
from .config import out as out_path


def window_times(cfg, offset=0.0):
    """Window times inside one subject's observation window.

    `offset` shifts the whole observation window along the simulated trajectory,
    so that the position of a window in the recording carries no information about
    the position of the recording in the illness.
    """
    return offset + np.linspace(cfg.simulation.window_first_min,
                                cfg.simulation.window_last_min,
                                cfg.simulation.n_windows)


def _render_subject(cfg, row, t_s, Zi, Di, starts, n_samp, seed, noise_scale):
    rng = np.random.default_rng(seed)
    w = int(cfg.simulation.window_s / cfg.simulation.dt_s)
    xs = np.empty((len(starts), len(cfg.simulation.leads), n_samp), np.float32)
    for j, k0 in enumerate(starts):
        xs[j] = ecg.render_window(cfg, row, t_s[k0:k0 + w], Zi[k0:k0 + w], Di[k0:k0 + w],
                                  rng, noise_scale=noise_scale)
    return xs


def build_dataset(cfg, subj, Z, D, t_min, lab, viab, tag="main",
                  noise_scale=1.0, seed_offset=0, subject_idx=None, save=True):
    """Render windows and build the aligned metadata table.

    A window is kept only if every one of its time steps is viable (see
    `physiology.viability_mask`).  Labels are taken at the window mid-point.
    """
    fs = cfg.simulation.fs
    n_samp = int(round(cfg.simulation.window_s * fs))
    w = int(cfg.simulation.window_s / cfg.simulation.dt_s)
    t_s = t_min * 60.0
    idx = np.arange(len(subj)) if subject_idx is None else np.asarray(subject_idx)
    has_offset = "obs_start_min" in subj.columns

    tte, event = phys.time_to_decomp(cfg, lab, t_min)
    hz = cfg.tasks.decomp_horizon_min

    rows, keep_starts = [], []
    for i in idx:
        off = float(subj.obs_start_min.iloc[i]) if has_offset else 0.0
        wt = window_times(cfg, off)
        starts = np.searchsorted(t_min, wt)
        mids = np.clip(starts + w // 2, 0, len(t_min) - 1)
        for j, (k0, km) in enumerate(zip(starts, mids)):
            if not viab[i, k0:k0 + w].all():
                continue
            cur = int(lab[i, km])
            rows.append(dict(
                subject=int(subj.subject.iloc[i]), row_i=int(i), win=j,
                t_min=float(wt[j] - off), t_sim_min=float(wt[j]), obs_start_min=off,
                stage=cur,
                volume_lost=float(D[i, km, 6]),
                reserve=float(D[i, km, 4]),
                map=float(D[i, km, 2]), map_frac=float(D[i, km, 2] / subj.map0.iloc[i]),
                hr_true=float(D[i, km, 0]), preload=float(D[i, km, 7]),
                symp=float(Z[i, km, 4]), para=float(Z[i, km, 5]),
                resp=float(Z[i, km, 6]), lactate=float(Z[i, km, 8]),
                sv=float(Z[i, km, 2]), co=float(Z[i, km, 3]), ox=float(Z[i, km, 7]),
                tte=float(tte[i, km]), tte_event=int(event[i, km]),
                decomp_soon=int((cur < phys.STAGE_ID["early_decomp"] or cur == phys.STAGE_ID["recovery"])
                                and event[i, km] == 1 and tte[i, km] <= hz),
                at_risk=int(cur < phys.STAGE_ID["early_decomp"] or cur == phys.STAGE_ID["recovery"]),
            ))
            keep_starts.append((i, k0))

    meta = pd.DataFrame(rows)
    keep = pd.DataFrame(keep_starts, columns=["row_i", "k0"])

    seeds = cfg.run.seed + seed_offset + 1000 + np.asarray(idx)
    jobs = []
    for i in idx:
        st = keep.loc[keep.row_i == i, "k0"].to_numpy()
        jobs.append(delayed(_render_subject)(cfg, subj.iloc[i], t_s, Z[i], D[i], st,
                                             n_samp, int(seeds[list(idx).index(i)]), noise_scale))
    chunks = Parallel(n_jobs=cfg.run.n_jobs, verbose=0)(jobs)

    X = np.concatenate([c for c in chunks if len(c)], axis=0).astype(np.float32)
    assert len(X) == len(meta), (len(X), len(meta))

    meta = meta.merge(
        subj[["subject", "split", "phenotype", "age", "sex", "bleeder", "stressor",
              "treated", "beta_blocked", "bb_strength", "autonomic_gain",
              "cardiac_reserve", "noise_level", "hr0", "map0", "hemorrhage_rate"]],
        on="subject", how="left",
    )
    if save:
        np.save(out_path(cfg, f"X_{tag}.npy"), X)
        meta.to_parquet(out_path(cfg, f"meta_{tag}.parquet"))
    return X, meta


def load(cfg, tag="main", mmap=True):
    X = np.load(out_path(cfg, f"X_{tag}.npy"), mmap_mode="r" if mmap else None)
    meta = pd.read_parquet(out_path(cfg, f"meta_{tag}.parquet"))
    return X, meta


def run_world(cfg, subj=None, seed=None):
    """Subjects -> latent trajectories -> labels.  Returns everything needed."""
    from . import subjects as subj_mod
    if subj is None:
        subj = subj_mod.assign_splits(cfg, subj_mod.make_cohort(cfg, seed=seed))
    Z, D, t, bh = phys.simulate_cohort(cfg, subj, seed=seed)
    lab = phys.stage_labels(cfg, subj, Z, D, t, bh)
    viab = phys.viability_mask(cfg, D)
    return subj, Z, D, t, lab, viab
