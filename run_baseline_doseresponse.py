"""How bad does the personal reference have to get before it stops being worth having?

`run_baseline_robustness.py` degrades the reference once, at one severity, and finds the
advantage almost intact. That is one point on a curve, and one point cannot say whether
the degradation was realistic or merely small. Here the same degradation is applied at a
range of severities, so the question becomes a break-even point:

    at what reference quality does the personal model stop beating the population model?

Severity multiplies every component of the degradation at once — electrode gain and
placement, device resampling, posture and breathing, ectopy, additive noise — and the
similarity between the degraded reference and the clean one is measured directly, so the
x-axis is an observable rather than a knob.
"""
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score

from hemo import config, experiments as E, features as FE, train as T
from hemo.config import out

import run_baseline_robustness as BR      # reuse degrade(), the split and the cohort

cfg = config.load()
log = E.log_to(out(cfg, "run_baseline_doseresponse.log"))
O = lambda p: out(cfg, p)
t00 = time.time()

X, meta, A, names = BR.X, BR.meta, BR.A, BR.names   # A is unstandardised
is_base, tr, va, te = BR.is_base, BR.tr, BR.va, BR.te
XGBP = BR.XGBP
LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
base_ix = np.where(is_base)[0]


def evaluate(Xf):
    y = meta.stage.to_numpy()
    cnt = np.bincount(y[tr], minlength=5).astype(float)
    wg = (cnt.sum() / (5 * np.maximum(cnt, 1))) ** 0.5
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=5, eval_metric="mlogloss",
                            early_stopping_rounds=40, **XGBP)
    clf.fit(Xf[tr], y[tr], sample_weight=wg[y[tr]], eval_set=[(Xf[va], y[va])], verbose=False)
    P = clf.predict_proba(Xf[te])
    r = dict(stage_auroc_macro=float(np.mean(
        [roc_auc_score((y[te] == k).astype(int), P[:, k])
         for k in range(5) if (y[te] == k).sum() > 3])),
        stage_bacc=float(balanced_accuracy_score(y[te], P.argmax(1))))
    for task, col in [("volume", "volume_lost"), ("reserve", "reserve")]:
        reg = xgb.XGBRegressor(objective="reg:squarederror", early_stopping_rounds=40, **XGBP)
        reg.fit(Xf[tr], meta[col].to_numpy()[tr],
                eval_set=[(Xf[va], meta[col].to_numpy()[va])], verbose=False)
        r[f"{task}_r2"] = float(r2_score(meta[col].to_numpy()[te], reg.predict(Xf[te])))
    at = meta.at_risk.to_numpy().astype(bool)
    yd = meta.decomp_soon.to_numpy()
    itr, iva, ite = tr[at[tr]], va[at[va]], te[at[te]]
    c2 = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                           early_stopping_rounds=40,
                           scale_pos_weight=float((1 - yd[itr]).sum() / max(yd[itr].sum(), 1)),
                           **XGBP)
    c2.fit(Xf[itr], yd[itr], eval_set=[(Xf[iva], yd[iva])], verbose=False)
    r["decomp_auroc"] = float(roc_auc_score(yd[ite], c2.predict_proba(Xf[ite])[:, 1]))
    return r


rows = []
pop = evaluate(A)
pop.update(level=np.nan, strategy="population model (no personal reference)",
           ref_similarity=np.nan, n_ref=int(is_base.sum()))
rows.append(pop)
log(f"[{time.time()-t00:5.0f}s] population model            stageAUC "
    f"{pop['stage_auroc_macro']:.3f}  volR2 {pop['volume_r2']:+.3f}")

for lvl in LEVELS:
    rng = np.random.default_rng(1000 + int(lvl * 10))
    Ad = A.copy()
    sims = []
    if lvl > 0:
        feats = []
        for k in base_ix:
            xd = BR.degrade(X[k], rng, level=lvl)
            feats.append(FE.extract_window(xd, cfg))
            sims.append(float(np.corrcoef(np.asarray(X[k], float).ravel(),
                                          np.asarray(xd, float).ravel())[0, 1]))
        # inserted raw, exactly as run_baseline_robustness.py does: the trees take
        # the unstandardised matrix, and missing cells fall back to the column median
        Fd = pd.DataFrame(feats).reindex(columns=names)
        Ad[base_ix] = np.nan_to_num(Fd.to_numpy(np.float32), nan=np.nanmedian(A, axis=0))
    else:
        sims = [1.0]
    Rm = BR.reference_matrix(Ad, "one")
    r = evaluate(A - Rm)
    r.update(level=lvl, strategy="one prior recording",
             ref_similarity=float(np.mean(sims)), n_ref=int(is_base.sum()))
    rows.append(r)
    pd.DataFrame(rows).to_csv(O("baseline_doseresponse.csv"), index=False)
    log(f"[{time.time()-t00:5.0f}s] level {lvl:>3}  waveform corr {np.mean(sims):.3f}   "
        f"stageAUC {r['stage_auroc_macro']:.3f}  volR2 {r['volume_r2']:+.3f}  "
        f"gain vs population {r['stage_auroc_macro'] - pop['stage_auroc_macro']:+.3f} / "
        f"{r['volume_r2'] - pop['volume_r2']:+.3f}")

d = pd.DataFrame(rows)
p = d[d.strategy == "one prior recording"].sort_values("level")
for key in ["stage_auroc_macro", "volume_r2"]:
    gain = p[key].to_numpy() - float(pop[key])
    neg = np.where(gain <= 0)[0]
    log(f"break-even for {key}: "
        + (f"level {p.level.to_numpy()[neg[0]]:.1f} (waveform correlation "
           f"{p.ref_similarity.to_numpy()[neg[0]]:.3f})" if len(neg)
           else f"not reached within the tested range (worst gain {gain.min():+.3f} "
                f"at level {p.level.to_numpy()[gain.argmin()]:.1f}, waveform correlation "
                f"{p.ref_similarity.to_numpy()[gain.argmin()]:.3f})"))
log(f"DOSERESPONSE_COMPLETE [{time.time()-t00:.0f}s]")
