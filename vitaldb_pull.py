"""External real-data arm: intraoperative ECG and arterial pressure from VitalDB.

VitalDB is openly accessible (no credentialing).  It gives 500 Hz lead-II ECG,
continuous arterial pressure, drug infusion rates and a recorded estimated blood
loss for each operation.  That makes it the closest openly available analogue of
the question this project is about: a within-patient, longitudinal, acute change
in circulating volume with a haemodynamic ground truth.

Design
------
  bleeding arm : estimated blood loss >= 500 mL
  control arm  : estimated blood loss <= 50 mL
  baseline     : a window ending at surgical incision, before any blood is lost
  follow-up    : windows at +30, +60 and +90 minutes of surgery

Each ECG track is ~64 MB, so tracks are downloaded one at a time, the windows are
cut out, and the track is deleted before the next one starts.
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import signal as sps

OUT = Path(__file__).resolve().parent / os.environ.get("VDB_OUT", "out_vitaldb")
OUT.mkdir(exist_ok=True)
# cases.csv and trks.csv from https://api.vitaldb.net (set VITALDB_DIR or use data/VitalDB)
TMP = Path(os.environ.get("VITALDB_DIR", Path(__file__).resolve().parent / "data" / "VitalDB"))
API = "https://api.vitaldb.net"
FS_IN, FS_OUT = 500, 250
WIN_S = 30.0
OFFSETS_MIN = [int(v) for v in os.environ.get("VDB_OFFSETS", "-5,30,60,90").split(",")]
N_PER_ARM = int(os.environ.get("VDB_N", 45))

NUMERIC = ["Solar8000/ART_MBP", "Solar8000/ART_SBP", "Solar8000/ART_DBP",
           "Solar8000/HR", "Solar8000/PLETH_SPO2", "Solar8000/BT",
           "Orchestra/PHEN_RATE", "Orchestra/NEPI_RATE", "Orchestra/PPF20_CE",
           "Primus/MAC"]


def get_csv(url, tries=3):
    for k in range(tries):
        try:
            r = requests.get(url, timeout=900)
            r.raise_for_status()
            return pd.read_csv(io.BytesIO(r.content))
        except Exception as e:
            if k == tries - 1:
                print("   ! failed", url[-12:], e, flush=True)
                return None
            time.sleep(4)


def select_cases():
    cases = pd.read_csv(TMP / "cases.csv")
    trks = pd.read_csv(TMP / "trks.csv")
    has_ecg = set(trks[trks.tname == "SNUADC/ECG_II"].caseid)
    has_map = set(trks[trks.tname == "Solar8000/ART_MBP"].caseid)
    c = cases[cases.caseid.isin(has_ecg & has_map)].copy()
    c = c[(c.age >= 18) & (c.ane_type == "General") & c.intraop_ebl.notna()]
    c = c[(c.opend - c.opstart) >= 100 * 60]          # at least 100 min of surgery
    bleed = c[c.intraop_ebl >= 500].copy()
    ctrl = c[c.intraop_ebl <= 50].copy()
    rng = np.random.default_rng(7)

    # coarse frequency matching on age and sex so the arms are not trivially different
    picked = []
    pool = ctrl.copy()
    bleed = bleed.sample(min(N_PER_ARM, len(bleed)), random_state=7)
    for _, b in bleed.iterrows():
        cand = pool[(pool.sex == b.sex) & (abs(pool.age - b.age) <= 8)]
        if not len(cand):
            cand = pool[pool.sex == b.sex]
        if not len(cand):
            continue
        pick = cand.sample(1, random_state=int(rng.integers(1e6)))
        picked.append(pick)
        pool = pool.drop(pick.index)
    ctrl = pd.concat(picked) if picked else ctrl.head(0)
    bleed["arm"] = "bleeding"
    ctrl["arm"] = "control"
    sel = pd.concat([bleed, ctrl], ignore_index=True)
    return sel, trks


def track_time(df):
    """VitalDB writes Time only on the first two and last rows of a fixed-rate track;
    the samples in between are uniformly spaced. Treating that column as a per-row
    timestamp silently yields empty windows, so reconstruct it from the row index."""
    t = df["Time"].to_numpy(float)
    if np.isfinite(t).mean() > 0.5:
        return t                                    # irregular track, timestamps are real
    fin = np.where(np.isfinite(t))[0]
    if len(fin) >= 2:
        dt = (t[fin[-1]] - t[fin[0]]) / max(fin[-1] - fin[0], 1)
    else:
        dt = 1.0 / FS_IN
    return t[fin[0]] + (np.arange(len(t)) - fin[0]) * dt


def cut_windows(sig_df, t0_list, col):
    """sig_df has Time (s from case start) and one signal column."""
    t = track_time(sig_df)
    v = sig_df[col].to_numpy(np.float32)
    out = {}
    n_out = int(WIN_S * FS_OUT)
    for name, t0 in t0_list:
        a, b = np.searchsorted(t, [t0, t0 + WIN_S])
        seg = v[a:b]
        if len(seg) < 0.9 * WIN_S * FS_IN:
            continue
        seg = seg[np.isfinite(seg)] if np.isfinite(seg).mean() > 0.98 else seg
        if len(seg) < 0.9 * WIN_S * FS_IN or not np.isfinite(seg).all():
            continue
        out[name] = sps.resample(seg.astype(float), n_out).astype(np.float32)
    return out


def main():
    sel, trks = select_cases()
    print(f"selected {len(sel)} cases: "
          f"{(sel.arm=='bleeding').sum()} bleeding, {(sel.arm=='control').sum()} control",
          flush=True)
    sel.to_csv(OUT / "selected_cases.csv", index=False)

    tid_ecg = trks[trks.tname == "SNUADC/ECG_II"].set_index("caseid").tid
    tid_num = {n: trks[trks.tname == n].set_index("caseid").tid for n in NUMERIC}

    waves, rows = [], []
    done = 0
    for _, cs in sel.iterrows():
        cid = int(cs.caseid)
        if cid not in tid_ecg.index:
            continue
        t_inc = float(cs.opstart)
        want = [(f"{o:+d}min", t_inc + o * 60) for o in OFFSETS_MIN]
        want = [(n, t) for n, t in want if t >= 0 and t + WIN_S <= float(cs.caseend)]
        if len(want) < 2:
            continue
        t0 = time.time()
        ecg = get_csv(f"{API}/{tid_ecg.loc[cid]}")
        if ecg is None or ecg.shape[1] < 2:
            continue
        segs = cut_windows(ecg, want, ecg.columns[1])
        del ecg
        if not segs:
            continue
        # numeric context, tiny files
        ctx = {}
        for n, s in tid_num.items():
            if cid not in s.index:
                continue
            d = get_csv(f"{API}/{s.loc[cid]}")
            if d is None or d.shape[1] < 2:
                continue
            tt, vv = track_time(d), d[d.columns[1]].to_numpy(float)
            for name, tw in want:
                m = (tt >= tw) & (tt <= tw + WIN_S)
                ctx[(name, n.split("/")[-1])] = (float(np.nanmean(vv[m]))
                                                 if m.sum() and np.isfinite(vv[m]).any()
                                                 else np.nan)
            del d
        for name, w in segs.items():
            waves.append(w)
            r = dict(caseid=cid, arm=cs.arm, window=name,
                     t_rel_min=dict(want)[name] - t_inc,
                     age=cs.age, sex=cs.sex, weight=cs.weight, height=cs.height,
                     ebl=cs.intraop_ebl, crystalloid=cs.intraop_crystalloid,
                     colloid=cs.intraop_colloid, rbc=cs.intraop_rbc,
                     uo=cs.intraop_uo, optype=cs.optype, asa=cs.asa,
                     preop_hb=cs.preop_hb, op_min=(cs.opend - cs.opstart) / 60.0)
            for (wn, k), v in ctx.items():
                if wn == name:
                    r[k.lower()] = v
            rows.append(r)
        done += 1
        print(f"   [{done}] case {cid} {cs.arm:8s} ebl={cs.intraop_ebl:6.0f} "
              f"windows={len(segs)} ({time.time()-t0:.0f}s)", flush=True)
        if done % 10 == 0:
            np.save(OUT / "X_vitaldb.npy", np.stack(waves))
            pd.DataFrame(rows).to_parquet(OUT / "meta_vitaldb.parquet")

    if not waves:
        print("VITALDB_PULL_EMPTY: no windows extracted", flush=True)
        return
    X = np.stack(waves)
    meta = pd.DataFrame(rows)
    np.save(OUT / "X_vitaldb.npy", X)
    meta.to_parquet(OUT / "meta_vitaldb.parquet")
    print(f"\nVITALDB_PULL_COMPLETE  {X.shape} windows from {meta.caseid.nunique()} cases",
          flush=True)


if __name__ == "__main__":
    main()
