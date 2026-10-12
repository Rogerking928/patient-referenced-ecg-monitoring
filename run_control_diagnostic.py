"""Why the block-permutation label control did not reach chance.

Predicts the stage from the window's TIME POSITION alone -- no ECG at all.  If the
outcome is predictable from time, then any label permutation that preserves each
subject's label trajectory also preserves that predictability, and the "negative"
control is not negative.  This is a property of the study design, not of the model,
and it is the single most transferable warning for the real-data stage.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from hemo import config
from hemo.config import out

cfg = config.load()
m = pd.read_parquet(out(cfg, "meta_main.parquet"))
tr, te = (m.split == "train").to_numpy(), (m.split == "test").to_numpy()
rng = np.random.default_rng(cfg.run.seed + 17)


def macro_auc(y_tr, y_te):
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    clf.fit(m.loc[tr, ["t_min"]], y_tr)
    P = clf.predict_proba(m.loc[te, ["t_min"]])
    ks = [k for k in np.unique(y_tr) if (y_te == k).sum() > 3 and (y_te != k).sum() > 3]
    return float(np.mean([roc_auc_score((y_te == k).astype(int), P[:, list(clf.classes_).index(k)])
                          for k in ks]))


rows = []
y = m.stage.to_numpy()
rows.append(dict(label_variant="true stage",
                 macro_auroc_from_time_alone=macro_auc(y[tr], y[te]),
                 comment="the outcome itself is strongly time-dependent by construction"))

subs = m.subject.unique()
mp = dict(zip(subs, rng.permutation(subs)))
src = m.set_index(["subject", "win"]).stage
blk = np.array([src.get((mp[s], w), src.get((s, w))) for s, w in zip(m.subject, m.win)], int)
rows.append(dict(label_variant="labels permuted in whole blocks between subjects",
                 macro_auroc_from_time_alone=macro_auc(blk[tr], blk[te]),
                 comment="the permutation preserves the label's time position, so the "
                         "control is not negative"))

full = rng.permutation(y)
rows.append(dict(label_variant="labels permuted completely",
                 macro_auroc_from_time_alone=macro_auc(full[tr], full[te]),
                 comment="time alignment destroyed; this is what a negative control should look like"))

df = pd.DataFrame(rows)
df.to_csv(out(cfg, "control_diagnostic.csv"), index=False)
print(df.to_string(index=False))
print("DIAGNOSTIC_COMPLETE")
