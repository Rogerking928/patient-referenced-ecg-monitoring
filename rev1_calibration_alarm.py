"""Reviewer 2, major comment 9 and question 5: calibration, and what an alarm built on
the trajectory model would do.

The five-minute decompensation model on the trajectory representation (absolute
window + slope + curvature over ten minutes, the 0.953 of the manuscript) is refitted
exactly as in run_longitudinal.py.  Its probabilities are recalibrated on the
validation subjects by logistic (Platt) recalibration; calibration is then measured
once on the held-out subjects.

Alarms: a window raises an alarm when the recalibrated risk crosses a threshold, with
or without a persistence requirement (two consecutive windows).  Thresholds are fixed
on the validation subjects at two operating points -- 80% window-level sensitivity,
and 0.5 false alarms per monitored hour -- and applied unchanged to the test subjects.
Windows are three minutes apart, so warning time has three-minute resolution.

    event          the first decompensation of a subject inside the recording
    detected       at least one alarm in the 30 minutes before the event
    warning time   event time minus the first alarm in that interval
    false alarm    an alarm with no event in the following 15 minutes
    burden         false alarms per hour of monitored at-risk time
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

import rev1_common as RC
from hemo import config

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/out_rev1"
cfg = config.load()
t00 = time.time()
STEP = RC.step_minutes(cfg)
LOG = open(f"{OUT}/calibration_alarm.log", "w")


def log(m):
    print(m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


meta = pd.read_parquet(f"{HERE}/out/meta_main.parquet").reset_index(drop=True)
F = pd.read_parquet(f"{HERE}/out/features_main.parquet")
A, _, _ = RC.standardise(F, meta)
meta, A = RC.sort_by_subject(meta, A)
ref = RC.first_window_reference(A, meta, max_t=7.6)
reps = RC.representations(A, meta, ref, STEP)
tr, va, te = (np.where(meta.split == s)[0] for s in ["train", "val", "test"])
yy, usable = RC.horizon_label(meta, 5.0)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cal_stats(y, p):
    lp = logit(p)
    slope = LogisticRegression(C=1e6, max_iter=2000).fit(lp[:, None], y).coef_[0, 0]
    # calibration-in-the-large: intercept with the logit as an offset
    import statsmodels.api as sm
    itc = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                 offset=lp).fit().params[0]
    bins = np.quantile(p, np.linspace(0, 1, 11))
    bins[0], bins[-1] = -1, 2
    k = np.digitize(p, bins[1:-1])
    ece = sum(abs(y[k == j].mean() - p[k == j].mean()) * (k == j).mean()
              for j in range(10) if (k == j).any())
    return dict(auroc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)),
                prevalence=float(y.mean()), cal_slope=float(slope), cal_intercept=float(itc),
                ece=float(ece))


def deciles(y, p, tag):
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    d = pd.DataFrame(dict(y=y, p=p, q=q)).groupby("q").agg(mean_predicted=("p", "mean"),
                                                            observed=("y", "mean"),
                                                            n=("y", "size")).reset_index()
    d["model"] = tag
    return d


rows, curves, dca, alarms = [], [], [], []
for name in ["trajectory", "reference+trajectory"]:
    r, pr = RC.score(reps[name], meta, tr, va, te, want=("h",))
    c, p_te, model, b, p_va = pr["h5"]
    log(f"{RC.stamp(t00)} {name}: five-minute AUROC {r['decomp_auroc_5min']:.3f} "
        f"on {len(c)} at-risk test windows, {yy[c].sum()} positive")
    pl = LogisticRegression(C=1e6, max_iter=2000).fit(logit(p_va)[:, None], yy[b])
    q_te = pl.predict_proba(logit(p_te)[:, None])[:, 1]
    q_va = pl.predict_proba(logit(p_va)[:, None])[:, 1]
    for tag, p in [("as fitted (class-weighted)", p_te), ("recalibrated on validation", q_te)]:
        s = cal_stats(yy[c], p)
        rows.append(dict(representation=name, probabilities=tag, n_windows=len(c),
                         n_positive=int(yy[c].sum()), **s))
        log(f"    {tag:28s} Brier {s['brier']:.3f}  slope {s['cal_slope']:.2f}  "
            f"intercept {s['cal_intercept']:+.2f}  ECE {s['ece']:.3f}")
        curves.append(deciles(yy[c], p, f"{name} | {tag}"))

    # decision curve on the recalibrated risk
    prev = yy[c].mean()
    for th in np.arange(0.02, 0.51, 0.02):
        tp = ((q_te >= th) & (yy[c] == 1)).mean()
        fp = ((q_te >= th) & (yy[c] == 0)).mean()
        dca.append(dict(representation=name, threshold=float(th),
                        net_benefit_model=float(tp - fp * th / (1 - th)),
                        net_benefit_all=float(prev - (1 - prev) * th / (1 - th)),
                        net_benefit_none=0.0))

    # ------------------------------------------------------------ alarms
    def episodes(idx, q):
        """Per-subject sequences of (t_min, risk, at_risk, event_time)."""
        d = meta.iloc[idx][["subject", "t_min", "tte", "tte_event"]].copy()
        d["q"] = q
        out = {}
        for s, g in d.groupby("subject"):
            g = g.sort_values("t_min")
            ev = (g.t_min + g.tte)[g.tte_event == 1]
            out[s] = (g.t_min.to_numpy(), g.q.to_numpy(),
                      float(ev.iloc[0]) if len(ev) else np.inf)
        return out

    def alarm_stats(ep, th, persist):
        n_ev = det = fa = 0
        hours = 0.0
        warn, n_alarm, tp_alarm = [], 0, 0
        for s, (t, q, E) in ep.items():
            a = q >= th
            if persist > 1:
                a = np.convolve(a.astype(int), np.ones(persist, int), "full")[:len(a)] >= persist
            hours += len(t) * STEP / 60.0
            for tk, ak in zip(t, a):
                if not ak:
                    continue
                n_alarm += 1
                if E - 15.0 <= tk < E:
                    tp_alarm += 1
                elif tk < E - 15.0 or not np.isfinite(E):
                    fa += 1
            if np.isfinite(E):
                n_ev += 1
                pre = (t >= E - 30.0) & (t < E) & a
                if pre.any():
                    det += 1
                    warn.append(E - t[pre][0])
        return dict(n_events=n_ev, events_detected=det,
                    event_sensitivity=det / n_ev if n_ev else np.nan,
                    median_warning_min=float(np.median(warn)) if warn else np.nan,
                    warning_iqr=(f"{np.percentile(warn, 25):.1f}-{np.percentile(warn, 75):.1f}"
                                 if warn else ""),
                    false_alarms=fa, monitored_hours=hours,
                    false_alarms_per_hour=fa / hours if hours else np.nan,
                    alarms=n_alarm, ppv=tp_alarm / n_alarm if n_alarm else np.nan)

    ep_va, ep_te = episodes(b, q_va), episodes(c, q_te)
    grid = np.unique(np.quantile(q_va, np.linspace(0.5, 0.999, 400)))
    for persist in [1, 2]:
        stats_va = [(th, alarm_stats(ep_va, th, persist)) for th in grid]
        # operating point 1: 80% window-level sensitivity on validation
        pos_va = q_va[yy[b] == 1]
        th80 = float(np.quantile(pos_va, 0.20))
        # operating point 2: the lowest threshold with <= 0.5 false alarms per hour
        ok = [th for th, s in stats_va if s["false_alarms_per_hour"] <= 0.5]
        th05 = float(min(ok)) if ok else float(grid[-1])
        for op, th in [("80% window sensitivity (validation)", th80),
                       ("0.5 false alarms per hour (validation)", th05)]:
            s = alarm_stats(ep_te, th, persist)
            # subject bootstrap for the two headline rates
            subs = list(ep_te)
            rng = np.random.default_rng(1)
            bs_sens, bs_fah = [], []
            for _ in range(400):
                pick = rng.choice(len(subs), len(subs), replace=True)
                e2 = {i: ep_te[subs[k]] for i, k in enumerate(pick)}
                x = alarm_stats(e2, th, persist)
                bs_sens.append(x["event_sensitivity"])
                bs_fah.append(x["false_alarms_per_hour"])
            s.update(representation=name, operating_point=op, threshold=th,
                     persistence_windows=persist,
                     event_sensitivity_lo=float(np.nanpercentile(bs_sens, 2.5)),
                     event_sensitivity_hi=float(np.nanpercentile(bs_sens, 97.5)),
                     fah_lo=float(np.nanpercentile(bs_fah, 2.5)),
                     fah_hi=float(np.nanpercentile(bs_fah, 97.5)))
            alarms.append(s)
            log(f"    alarm | {op:40s} persist {persist}: events {s['events_detected']}/"
                f"{s['n_events']} ({s['event_sensitivity']*100:.0f}%, "
                f"{s['event_sensitivity_lo']*100:.0f}-{s['event_sensitivity_hi']*100:.0f}), "
                f"warning {s['median_warning_min']:.1f} min (IQR {s['warning_iqr']}), "
                f"{s['false_alarms_per_hour']:.2f} false alarms/h "
                f"({s['fah_lo']:.2f}-{s['fah_hi']:.2f}), PPV {s['ppv']:.2f}")

pd.DataFrame(rows).to_csv(f"{OUT}/calibration.csv", index=False)
pd.concat(curves).to_csv(f"{OUT}/calibration_deciles.csv", index=False)
pd.DataFrame(dca).to_csv(f"{OUT}/decision_curve.csv", index=False)
pd.DataFrame(alarms).to_csv(f"{OUT}/alarm_burden.csv", index=False)
log(f"CALIBRATION_ALARM_COMPLETE {RC.stamp(t00)}")
