"""Why the block-permutation label control did not reach chance.

Predicts the stage from the window's TIME POSITION alone -- no ECG at all.  If the
outcome is predictable from time, then any label permutation that preserves each
subject's label trajectory also preserves that predictability, and the "negative"
control is not negative.  This is a property of the study design, not of the model,
and it is the single most transferable warning for the real-data stage.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from hemo import config
from hemo.config import out

cfg = config.load()
m = pd.read_parquet(out(cfg, "meta_main.parquet"))
tr, va, te = [(m.split == s).to_numpy() for s in ("train", "val", "test")]
rng = np.random.default_rng(cfg.run.seed + 17)
# same classifier, weighting and early stopping as the metadata-only baselines in
# run_leakage_and_matching.py, so the true-label value equals "absolute time only" there
XGBC = dict(n_estimators=350, max_depth=4, learning_rate=0.07, subsample=0.85,
            colsample_bytree=0.8, n_jobs=cfg.run.n_jobs, tree_method="hist", random_state=0)
A = m[["t_min"]].to_numpy(np.float32)


def macro_auc(y):
    clf = xgb.XGBClassifier(objective="multi:softprob", num_class=5,
                            eval_metric="mlogloss", early_stopping_rounds=30, **XGBC)
    cnt = np.bincount(y[tr], minlength=5).astype(float)
    wgt = (cnt.sum() / (5 * np.maximum(cnt, 1))) ** 0.5
    clf.fit(A[tr], y[tr], sample_weight=wgt[y[tr]], eval_set=[(A[va], y[va])], verbose=False)
    P, y_te = clf.predict_proba(A[te]), y[te]
    return float(np.mean([roc_auc_score((y_te == k).astype(int), P[:, k]) for k in range(5)
                          if (y_te == k).sum() > 3]))


rows = []
y = m.stage.to_numpy()
rows.append(dict(label_variant="true stage",
                 macro_auroc_from_time_alone=macro_auc(y),
                 comment="the outcome itself is strongly time-dependent by construction"))

subs = m.subject.unique()
mp = dict(zip(subs, rng.permutation(subs)))
src = m.set_index(["subject", "win"]).stage
blk = np.array([src.get((mp[s], w), src.get((s, w))) for s, w in zip(m.subject, m.win)], int)
rows.append(dict(label_variant="labels permuted in whole blocks between subjects",
                 macro_auroc_from_time_alone=macro_auc(blk),
                 comment="the permutation preserves the label's time position, so the "
                         "control is not negative"))

full = rng.permutation(y)
rows.append(dict(label_variant="labels permuted completely",
                 macro_auroc_from_time_alone=macro_auc(full),
                 comment="time alignment destroyed; this is what a negative control should look like"))

df = pd.DataFrame(rows)
df.to_csv(out(cfg, "control_diagnostic.csv"), index=False)
print(df.to_string(index=False))
print("DIAGNOSTIC_COMPLETE")
