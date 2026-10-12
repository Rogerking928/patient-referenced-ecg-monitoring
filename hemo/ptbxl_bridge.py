"""Public-data arm: PTB-XL preprocessing, quality control, encoder, and an
honest audit of how far the synthetic ECGs sit from real ones.

Nothing here claims a haemorrhage result.  The purpose is (a) to build the
real-data preprocessing and QC path that stage 2 will reuse, and (b) to measure
the domain gap so the simulation's limits are stated as a number rather than a
disclaimer.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

# PTB-XL 1.0.3 from physionet.org; set PTBXL_DIR or place it under data/PTB-XL
PTBXL = Path(os.environ.get("PTBXL_DIR", Path(__file__).resolve().parents[1] / "data" / "PTB-XL"))
LEAD_INDEX = {"I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
              "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11}


# ------------------------------------------------------------------ loading
def load_ptbxl(cfg, n=1800, seed=0, leads=("II", "V5"), only_normal=False):
    import wfdb
    db = pd.read_csv(PTBXL / "ptbxl_database.csv", index_col="ecg_id")
    db["scp"] = db.scp_codes.apply(eval)
    agg = pd.read_csv(PTBXL / "scp_statements.csv", index_col=0)
    agg = agg[agg.diagnostic == 1]

    def superclass(d):
        return sorted({agg.loc[k].diagnostic_class for k in d if k in agg.index})
    db["super"] = db.scp.apply(superclass)
    db["is_norm"] = db["super"].apply(lambda s: s == ["NORM"])

    pool = db[db.is_norm] if only_normal else db
    rng = np.random.default_rng(seed)
    take = pool.sample(min(n, len(pool)), random_state=int(rng.integers(1e6)))

    sig, keep = [], []
    for eid, row in take.iterrows():
        p = PTBXL / row.filename_lr
        try:
            rec = wfdb.rdrecord(str(p))
        except Exception:
            continue
        arr = rec.p_signal[:, [LEAD_INDEX[l] for l in leads]].T.astype(np.float32)
        if not np.isfinite(arr).all():
            continue
        sig.append(arr)
        keep.append(eid)
    X = np.stack(sig)
    meta = take.loc[keep, ["age", "sex", "height", "weight", "device", "site",
                           "strat_fold", "is_norm", "baseline_drift", "static_noise"]].copy()
    meta["super"] = take.loc[keep, "super"].apply(lambda s: "|".join(s) if s else "NONE")
    return X, meta.reset_index()


# ------------------------------------------------------------------ preprocessing / QC
def preprocess(X, fs, hp=0.5, lp=40.0, notch=50.0):
    """Baseline removal, band limiting, notch -- the same path stage 2 will use."""
    b, a = sps.butter(3, [hp / (fs / 2), min(lp / (fs / 2), 0.99)], btype="band")
    Y = sps.filtfilt(b, a, X, axis=-1)
    if notch and notch < fs / 2:
        bn, an = sps.iirnotch(notch / (fs / 2), Q=30)
        Y = sps.filtfilt(bn, an, Y, axis=-1)
    return Y.astype(np.float32)


def quality_control(X, fs, min_beats=6, max_beats=45, max_flat=0.20,
                    min_beat_corr=0.80, max_amp_mv=8.0):
    """Per-record QC.  Returns the pass mask and the per-check reason table.

    Note on the noise check: a high-frequency residual is useless as a quality
    metric on the 100 Hz PTB-XL release, because that release is already
    anti-alias filtered and has essentially no content above 40 Hz -- every
    record scores >60 dB.  Beat-to-beat template consistency is used instead;
    it is sampling-rate agnostic and it is what actually fails on a bad
    recording.  The spectral figure is still reported, but it does not gate.
    """
    from . import features as FE
    rows = []
    for x in X:
        lead0 = np.asarray(x[0], float)
        pk = FE.detect_rpeaks(lead0, fs)
        hf = sps.filtfilt(*sps.butter(3, min(45 / (fs / 2), 0.98), btype="high"), lead0)
        snr = 20 * np.log10((np.std(lead0) + 1e-9) / (np.std(hf) + 1e-9))
        flat = float(np.mean(np.abs(np.diff(lead0)) < 1e-6))
        med, S = FE._median_beat(lead0, pk, fs, min(0.25, 0.35 * (len(lead0) / fs) / max(len(pk), 1)), 0.35)
        if med is None or S is None or len(S) < 3:
            bc = np.nan
        else:
            bc = float(np.median([np.corrcoef(r, med)[0, 1] for r in S[:60]]))
        rows.append(dict(n_beats=len(pk), snr_db=float(snr), flat_frac=flat,
                         beat_corr=bc, max_abs_mv=float(np.max(np.abs(x))),
                         sat_frac=float(np.mean(np.abs(x) > max_amp_mv))))
    q = pd.DataFrame(rows)
    q["pass_beats"] = q.n_beats.between(min_beats, max_beats)
    q["pass_beat_corr"] = q.beat_corr.fillna(0) >= min_beat_corr
    q["pass_flat"] = q.flat_frac <= max_flat
    q["pass_amp"] = q.max_abs_mv <= max_amp_mv
    q["pass"] = q[["pass_beats", "pass_beat_corr", "pass_flat", "pass_amp"]].all(axis=1)
    return q


# ------------------------------------------------------------------ matched synthetic view
def synthetic_to_ptbxl_view(cfg, X, fs_out=100, seconds=10.0, offset_s=2.0):
    """Resample synthetic windows to the PTB-XL sampling rate and duration."""
    fs_in = cfg.simulation.fs
    a = int(offset_s * fs_in)
    b = a + int(seconds * fs_in)
    n_out = int(seconds * fs_out)
    n_lead = np.asarray(X[0]).shape[0]
    out = np.empty((len(X), n_lead, n_out), np.float32)
    for i in range(len(X)):
        seg = np.asarray(X[i][:, a:b], float)
        out[i] = sps.resample(seg, n_out, axis=1).astype(np.float32)
    return out


def matched_cfg(cfg, fs=100, window_s=10.0):
    c = copy.deepcopy(cfg)
    c["simulation"]["fs"] = fs
    c["simulation"]["window_s"] = window_s
    return c


# ------------------------------------------------------------------ domain gap
def domain_discriminability(Fr, Fs, n_splits=5, seed=0):
    """AUROC of a classifier asked to tell real from synthetic.

    0.5 = indistinguishable, 1.0 = trivially separable.  Reported as-is.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    A = np.vstack([Fr.to_numpy(np.float32), Fs.to_numpy(np.float32)])
    y = np.r_[np.zeros(len(Fr)), np.ones(len(Fs))]
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=250, random_state=seed),
                          A, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p)), p


def standardised_mean_difference(Fr, Fs):
    a, b = Fr.to_numpy(np.float64), Fs.to_numpy(np.float64)
    ma, mb = np.nanmean(a, 0), np.nanmean(b, 0)
    sa, sb = np.nanstd(a, 0), np.nanstd(b, 0)
    pooled = np.sqrt((sa ** 2 + sb ** 2) / 2) + 1e-9
    return pd.DataFrame({"feature": Fr.columns, "real_mean": ma, "synth_mean": mb,
                         "real_sd": sa, "synth_sd": sb, "smd": (mb - ma) / pooled})
