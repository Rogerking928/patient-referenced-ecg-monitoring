"""The external cohort's exclusion cascade, and a sensitivity check on the two
ultra-short heart-rate-variability features.

Two things a reviewer will ask about the VitalDB arm are answered here rather
than in prose, and written to a CSV so the manuscript can read them:

  1. how the cohort goes from the operations retrieved to the operations
     analysed, one exclusion at a time.  The manuscript previously quoted the
     case count of the bleeding-identification analysis in the same sentence as
     the window count of the hypotension analysis.  They are different analyses
     over different window schedules and their counts are not interchangeable.

  2. whether the hypotension result depends on sample entropy and the
     short-scale detrended fluctuation exponent, which are estimated here from
     roughly three dozen RR intervals and are not reliable at that length.  The
     check simply drops them and refits.

Run:  python3 run_vitaldb_cohort_audit.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

OUT = "out_vitaldb2/"
UNSTABLE = ["sampen", "dfa_a1"]          # unreliable over a 30 s window


def load():
    meta = pd.read_parquet(OUT + "meta_vitaldb.parquet").reset_index(drop=True)
    feat = pd.read_parquet(OUT + "features_vitaldb.parquet").reset_index(drop=True)
    return pd.concat([meta, feat.add_prefix("f_")], axis=1)


def main():
    d = load()
    rows = [dict(step="operations retrieved with continuous arterial pressure",
                 windows=len(d), cases=d.caseid.nunique(), dropped_cases=0)]

    bc = "f_II_beat_corr" if "f_II_beat_corr" in d.columns else "f_beat_corr"
    ok = (d.f_n_beats.between(10, 70) & (d[bc] > 0.75)
          & d.f_hr_mean.between(30, 180) & d.f_rr_kept_frac.gt(0.6)
          & d.art_mbp.between(30, 160))
    q = d[ok].reset_index(drop=True)
    rows.append(dict(step="after signal quality control on the windows",
                     windows=len(q), cases=q.caseid.nunique(),
                     dropped_cases=d.caseid.nunique() - q.caseid.nunique()))

    feats = [c for c in q.columns if c.startswith("f_")]
    base = q[q.window == "-5min"].groupby("caseid")[feats].median()
    a = q[q.caseid.isin(base.index)].reset_index(drop=True)
    rows.append(dict(step="after requiring a usable pre-incision reference window",
                     windows=len(a), cases=a.caseid.nunique(),
                     dropped_cases=q.caseid.nunique() - a.caseid.nunique()))
    cascade = pd.DataFrame(rows)

    for c in feats:
        a["d_" + c] = a[c].to_numpy() - base.loc[a.caseid, c].to_numpy()
    a["hypotension"] = (a.art_mbp < 65).astype(int)
    y, grp = a.hypotension.to_numpy(), a.caseid.to_numpy()

    def prep(cols):
        A = a[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        A = np.nan_to_num(A)
        return (A - A.mean(0)) / (A.std(0) + 1e-9)

    def auc(cols):
        p = cross_val_predict(
            LogisticRegression(max_iter=5000, C=0.05), prep(cols), y,
            cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
            groups=grp, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, p))

    unst = {"f_" + u for u in UNSTABLE}
    keep_abs = [c for c in feats if c not in unst]
    keep_del = ["d_" + c for c in keep_abs]
    sens = pd.DataFrame([
        dict(representation="all ECG features, absolute",
             full=auc(feats), without_entropy_dfa=auc(keep_abs)),
        dict(representation="all ECG features, change from own baseline",
             full=auc(["d_" + c for c in feats]), without_entropy_dfa=auc(keep_del)),
    ])

    summary = pd.DataFrame([dict(
        cases_retrieved=int(cascade.cases.iloc[0]),
        windows_dropped_qc=int(cascade.windows.iloc[0] - cascade.windows.iloc[1]),
        cases_after_qc=int(cascade.cases.iloc[1]),
        cases_lost_no_baseline=int(cascade.dropped_cases.iloc[2]),
        cases_analysed=int(cascade.cases.iloc[2]),
        windows_analysed=int(cascade.windows.iloc[2]),
        cases_bleeding=int((a.groupby("caseid").arm.first() == "bleeding").sum()),
        cases_control=int((a.groupby("caseid").arm.first() == "control").sum()),
        pos_windows=int(y.sum()),
        pos_cases=int(a[a.hypotension == 1].caseid.nunique()),
        beats_median=float(a.f_n_beats.median()),
        beats_q1=float(a.f_n_beats.quantile(.25)),
        beats_q3=float(a.f_n_beats.quantile(.75)),
        auroc_delta_full=float(sens.full.iloc[1]),
        auroc_delta_noent=float(sens.without_entropy_dfa.iloc[1]),
        auroc_abs_full=float(sens.full.iloc[0]),
        auroc_abs_noent=float(sens.without_entropy_dfa.iloc[0]),
    )])

    cascade.to_csv("out/vitaldb_cohort_cascade.csv", index=False)
    sens.to_csv("out/vitaldb_ultrashort_sensitivity.csv", index=False)
    summary.to_csv("out/vitaldb_cohort_audit.csv", index=False)
    print(cascade.to_string(index=False), "\n")
    print(sens.to_string(index=False), "\n")
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
