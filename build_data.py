"""Stage 1: build the synthetic world and the ECG dataset."""
import sys, time
import numpy as np
import pandas as pd

from hemo import config, simulate, features
from hemo.config import out

cfg = config.load()
t0 = time.time()

subj, Z, D, t, lab, viab = simulate.run_world(cfg)
subj.to_parquet(out(cfg, "subjects.parquet"))
np.savez_compressed(out(cfg, "world.npz"), Z=Z.astype(np.float32), D=D.astype(np.float32),
                    t=t, lab=lab, viab=viab)
print(f"[{time.time()-t0:6.1f}s] world: {len(subj)} subjects, "
      f"{viab.mean()*100:.1f}% viable steps", flush=True)

X, meta = simulate.build_dataset(cfg, subj, Z, D, t, lab, viab, tag="main")
print(f"[{time.time()-t0:6.1f}s] rendered {X.shape} windows", flush=True)

F = features.extract_batch(X, cfg)
F.to_parquet(out(cfg, "features_main.parquet"))
print(f"[{time.time()-t0:6.1f}s] features {F.shape}, NaN {F.isna().mean().mean()*100:.2f}%", flush=True)

print(meta.groupby("split").size())
print(meta.stage.value_counts(normalize=True).round(3).sort_index())
print("DATA_COMPLETE", flush=True)
