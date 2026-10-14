"""The three main-text composite figures.

The reviewer's layout: everything the reader needs to follow the argument lives in
three figures and two tables; every other panel moves to the supplement.

    Main figure 1  design      A causal diagram | B latent trajectories (2 phenotypes)
                              C ECG across states | D study workflow
    Main figure 2  the claim   A model comparison | B population vs personal baseline
                              C temporal extent of the input | D gain by outcome
    Main figure 3  the defence A heart-rate matching | B non-ECG shortcut baselines
                              C prediction horizon | D simulation-to-reality gap

Panels are re-plotted from the CSVs in out/, not stitched from the standalone figures,
because the layout subsets their content (two phenotypes rather than three, one metric
per panel rather than two axes).

Figure 2C/D and Figure 3B replaced earlier panels after review: three hand-picked example
trajectories are not evidence, and a preload sweep through the author's own observation
model is an internal consistency check rather than a falsification test.  What can be
predicted with no electrocardiogram at all is the experiment that can actually break the
result, so it takes the panel instead.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.lines import Line2D

from hemo import config, ecg, features as FE, figures as G, physiology as phys
from hemo.config import fig as figpath, out

G.style()
cfg = config.load()
O = lambda *p: out(cfg, *p)
have = lambda *p: O(*p).exists()
HERE = O("..").resolve()
O_X = lambda *p: HERE / "out_xgen" / Path(*p)      # cross-generator arm
O_V = lambda *p: HERE / "out_vitaldb" / Path(*p)   # real intraoperative arm
O_V2 = lambda *p: HERE / "out_vitaldb2" / Path(*p)

subj = pd.read_parquet(O("subjects.parquet"))
w = np.load(O("world.npz"))
Z, D, t, lab, viab = w["Z"], w["D"], w["t"], w["lab"], w["viab"]

TASK_LABEL = ["State classification\n(macro AUROC)", "Blood-volume loss\n(R²)",
              "Imminent decompensation\n(AUROC)", "Physiological reserve\n(R²)"]
TASK_KEY = ["stage_auroc_macro", "volume_r2", "decomp_auroc", "reserve_r2"]
TASK_REF = [0.5, 0.0, 0.5, 0.0]
# narrow multi-panel strips (Figure 2A/B): four-line labels so neighbours never touch
TASK_LABEL_NARROW = ["State\nclassification\n(macro AUROC)", "Blood-volume\nloss\n(R²)",
                     "Imminent\ndecompensation\n(AUROC)", "Physiological\nreserve\n(R²)"]


def head(fig, axes, letter, text, pad=0.026):
    """Panel letter and its headline on one baseline, in the gutter above the panel.

    Positioned from the realised axes boxes rather than guessed figure coordinates,
    so nested gridspecs of different depths still line up.
    """
    axes = axes if isinstance(axes, (list, tuple)) else [axes]
    bbs = [a.get_position() for a in axes]
    x0 = min(b.x0 for b in bbs)
    y1 = max(b.y1 for b in bbs)
    x = max(0.006, x0 - 0.042)
    fig.text(x, y1 + pad, letter, fontsize=14, fontweight="bold", color=G.INK, va="bottom")
    fig.text(x + 0.021, y1 + pad, text, fontsize=10.2, fontweight="bold", color=G.INK,
             va="bottom")


def smooth_stage(st, win=61):
    st = np.asarray(st)
    o = st.copy()
    h = win // 2
    for i in range(len(st)):
        o[i] = np.bincount(st[max(0, i - h):i + h + 1], minlength=5).argmax()
    return o


# ==================================================================== MAIN FIGURE 1
fig = plt.figure(figsize=(16.4, 11.9))
outer = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.40, wspace=0.17,
                         left=0.060, right=0.985, top=0.93, bottom=0.04)

# ---------------------------------------------------------------- 1A causal diagram
NODES = {
    "haemorrhage": (0.090, 0.50, "Haemor-\nrhage", "exposure"),
    "volume":      (0.285, 0.50, "Blood\nvolume", "latent"),
    "preload":     (0.475, 0.72, "Preload", "latent"),
    "reserve":     (0.475, 0.28, "Autonomic\nreserve", "latent"),
    "metab":       (0.665, 0.50, "Metabolic\nstress", "latent"),
    "ecg":         (0.885, 0.72, "ECG\nwaveform", "observed"),
    "state":       (0.885, 0.28, "Clinical\nstate", "derived"),
}
EDGES = [("haemorrhage", "volume"), ("volume", "preload"), ("volume", "reserve"),
         ("preload", "metab"), ("reserve", "metab"),
         ("preload", "ecg"), ("reserve", "ecg"), ("metab", "ecg"),
         ("preload", "state"), ("reserve", "state"), ("metab", "state")]
KIND = {"exposure": "#4a3aa7", "latent": G.CAT[0], "derived": "#5598e7",
        "observed": G.CAT[1], "modifier": G.MUTED}

axA = fig.add_subplot(outer[0, 0])
for a_, b_ in EDGES:
    xa, ya = NODES[a_][:2]
    xb, yb = NODES[b_][:2]
    to_ecg = b_ == "ecg"
    axA.add_patch(FancyArrowPatch((xa, ya), (xb, yb), connectionstyle="arc3,rad=0.06",
                                  arrowstyle="-|>", mutation_scale=16,
                                  lw=2.6 if to_ecg else 1.6,
                                  color=G.CAT[1] if to_ecg else G.AXIS,
                                  alpha=0.95 if to_ecg else 0.85,
                                  shrinkA=22, shrinkB=22, zorder=1))
# the arrow that does not exist is the point of the panel, so it is drawn as absent
axA.add_patch(FancyArrowPatch((0.090, 0.50), (0.885, 0.72),
                              connectionstyle="arc3,rad=-0.40", arrowstyle="-|>",
                              mutation_scale=14, lw=1.8, color=G.STATUS["critical"],
                              linestyle=(0, (5, 4)), alpha=0.65, shrinkA=34, shrinkB=34,
                              zorder=1))
axA.text(0.47, 0.955, "no direct pathway", color=G.STATUS["critical"], fontsize=9.5,
         ha="center", va="center", fontweight="bold")
axA.plot([0.44, 0.50], [0.905, 0.868], color=G.STATUS["critical"], lw=2.2)
axA.plot([0.44, 0.50], [0.868, 0.905], color=G.STATUS["critical"], lw=2.2)
for k, (x, y, lbl, kind) in NODES.items():
    c = KIND[kind]
    axA.add_patch(FancyBboxPatch((x - 0.082, y - 0.082), 0.164, 0.164,
                                 boxstyle="round,pad=0.008,rounding_size=0.022",
                                 linewidth=2.4, edgecolor=c,
                                 facecolor="#ffffff" if kind != "observed" else c, zorder=2))
    axA.text(x, y, lbl, ha="center", va="center", fontsize=8.4, zorder=3,
             color="#ffffff" if kind == "observed" else G.INK)
# the notes that used to sit here restate the caption and are not repeated on the page
axA.set_xlim(-0.005, 1.005)
axA.set_ylim(-0.02, 1.02)
axA.axis("off")

# ---------------------------------------------------------------- 1B trajectories
gsB = outer[0, 1].subgridspec(2, 3, hspace=0.30, wspace=0.32)
rowsB = [("normal_responder", "Normal responder"), ("poor_compensator", "Poor compensator")]
SERIES = [("Volume lost (%)", lambda i, v: [("Volume lost", D[i, v, 6] * 100, G.CAT[0], "-")]),
          ("Rate / pressure", lambda i, v: [("HR (bpm)", D[i, v, 0], G.CAT[0], "-"),
                                            ("MAP (mmHg)", D[i, v, 2], G.CAT[1], "--")]),
          ("Autonomic reserve", lambda i, v: [("Reserve", D[i, v, 4], G.CAT[0], "-")])]
axB0, axB_all = None, []
_presentB = set()
for r, (ph_name, ph_lab) in enumerate(rowsB):
    m = (subj.phenotype == ph_name) & (subj.bleeder == 1)
    ok = np.where(m.to_numpy() & (lab == 3).any(1))[0]
    i = int(ok[0]) if len(ok) else int(np.where(m.to_numpy())[0][0])
    v = viab[i]
    tt, st = t[v], smooth_stage(lab[i][v])
    _presentB |= {int(k) for k in np.unique(st)}
    for c, (name, fn) in enumerate(SERIES):
        ax = fig.add_subplot(gsB[r, c])
        axB_all.append(ax)
        if axB0 is None:
            axB0 = ax
        G.stage_bands(ax, tt, st)
        for lbl_, y, col, ls in fn(i, v):
            ax.plot(tt, y, color=col, lw=1.6, ls=ls, zorder=3, label=lbl_ if r == 0 else None)
        if r == 0:
            ax.set_title(name, fontsize=8.8)
            if len(fn(i, v)) > 1:      # a one-line panel is direct-labelled by its title
                lo_, hi_ = ax.get_ylim()
                ax.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))   # headroom: legend clear of the lines
                ax.legend(loc="upper left", fontsize=6.6, handlelength=2.8, frameon=False)
        if r == 1:
            ax.set_xlabel("Time (min)", fontsize=8)
        if c == 0:
            ax.set_ylabel(ph_lab, fontsize=8.8, color=G.INK, fontweight="bold")
        ax.tick_params(labelsize=7.5)
G.stage_legend(axB_all[-2], present=sorted(_presentB), loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=5,
               fontsize=7.4, columnspacing=0.8, handlelength=1.0, handletextpad=0.4)

# ---------------------------------------------------------------- 1C ECG across states
gsC = outer[1, 0].subgridspec(4, 2, width_ratios=[2.1, 1.0], hspace=0.62, wspace=0.26)
clean = np.where((subj.noise_level.values < 0.6) & ((lab == 3).sum(1) > 120)
                 & ((lab == 1).sum(1) > 120))[0]
i = int(clean[0])
row = subj.iloc[i]
ts = t * 60.0
stages_present = [k for k in range(5) if (lab[i] == k).any()][:4]
axC_beat = fig.add_subplot(gsC[:, 1])
axC0 = None
for j, k in enumerate(stages_present):
    k0 = int(np.argmax(lab[i] == k))
    sig = ecg.render_window(cfg, row, ts[k0:k0 + 30], Z[i, k0:k0 + 30], D[i, k0:k0 + 30],
                            np.random.default_rng(100 + j))
    tw = np.arange(sig.shape[1]) / cfg.simulation.fs
    axw = fig.add_subplot(gsC[j, 0])
    if axC0 is None:
        axC0 = axw
    axw.plot(tw, sig[0], color=G.STAGE_COLOURS[k], lw=0.75,
             label=f"_series:{G.STAGE_LABELS[k]} (lead II)")
    axw.set_xlim(0, 6)
    axw.tick_params(labelsize=7)
    axw.set_title(f"{G.STAGE_LABELS[k]}:  HR {D[i,k0,0]:.0f},  preload {D[i,k0,7]:.2f},  "
                  f"MAP {D[i,k0,2]:.0f}", fontsize=7.6, fontweight="normal",
                  color=G.STAGE_COLOURS[k] if k >= 2 else G.INK)
    if j == len(stages_present) - 1:
        axw.set_xlabel("Time (s), lead II, 6 s of a 30 s window", fontsize=8)
    else:
        axw.set_xticklabels([])

    fs_ = cfg.simulation.fs
    pk = FE.detect_rpeaks(np.asarray(sig[0], float), fs_)
    rr_ = float(np.median(np.diff(pk)) / fs_) if len(pk) > 2 else 0.85
    pre_, post_ = min(0.30, 0.42 * rr_), min(0.50, 0.62 * rr_)
    med, _ = FE._median_beat(np.asarray(sig[0], float), pk, fs_, pre_, post_)
    if med is not None:
        axC_beat.plot((np.arange(len(med)) - pre_ * fs_) / fs_ * 1000, med,
                      color=G.STAGE_COLOURS[k], lw=1.7, label=G.STAGE_LABELS[k])
axC_beat.axhline(0, color=G.AXIS, lw=0.8)
axC_beat.set_xlim(-250, 460)
axC_beat.set_ylim(-0.22, 1.12)        # headroom so the legend sits above the R wave
axC_beat.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6])
axC_beat.set_title("Median beat, lead II", fontsize=8.8)
axC_beat.set_xlabel("Time from R (ms)", fontsize=8)
axC_beat.legend(fontsize=6.8, loc="upper right", frameon=False, handlelength=1.2,
                borderaxespad=0.1)
axC_beat.tick_params(labelsize=7.5)

# ---------------------------------------------------------------- 1D workflow
axD = fig.add_subplot(outer[1, 1])
n_sub = len(subj)
meta_all = pd.read_parquet(O("meta_main.parquet"))
n_win = len(meta_all)
splits = {s: (meta_all.split == s).sum() for s in ["train", "val", "test"]}
subj_split = {s: meta_all[meta_all.split == s].subject.nunique() for s in ["train", "val", "test"]}
STEPS = [
    (0.86, "Physiological state-space model", f"{n_sub} synthetic subjects, 90 min at 1 s\n"
     f"{int((subj.bleeder==0).sum())} never bleed, {int(subj.stressor.sum())} given a pure "
     "sympathetic stressor", "exposure"),
    (0.665, "Observation model", "Gaussian-wave beats, leads II and V5, 250 Hz\n"
     "subject-specific latent-to-ECG couplings", "latent"),
    (0.47, "Windowing and quality control", f"{n_win} analysable 30 s windows\n"
     "81 handcrafted features, no access to the latent state", "derived"),
    (0.275, "Subject-level split", f"{subj_split['train']} / {subj_split['val']} / "
     f"{subj_split['test']} subjects\n{splits['train']} / {splits['val']} / {splits['test']} windows",
     "modifier"),
    (0.08, "Three representations x four outcomes",
     "absolute | personal delta | absolute + delta\n"
     "state, volume lost, imminent decompensation, reserve", "observed"),
]
for y, title_, body, kind in STEPS:
    c = KIND[kind]
    axD.add_patch(FancyBboxPatch((0.06, y - 0.055), 0.88, 0.125,
                                 boxstyle="round,pad=0.008,rounding_size=0.02",
                                 linewidth=1.6, edgecolor=c, facecolor="#ffffff", zorder=2))
    axD.text(0.5, y + 0.045, title_, ha="center", va="center", fontsize=9,
             fontweight="bold", color=G.INK, zorder=3)
    axD.text(0.5, y - 0.015, body, ha="center", va="center", fontsize=7.4,
             color=G.INK2, zorder=3)
for y0, y1 in zip([s[0] for s in STEPS[:-1]], [s[0] for s in STEPS[1:]]):
    axD.add_patch(FancyArrowPatch((0.5, y0 - 0.058), (0.5, y1 + 0.072),
                                  arrowstyle="-|>", mutation_scale=13,
                                  color=G.AXIS, lw=1.3, zorder=1))
axD.set_xlim(0, 1)
axD.set_ylim(0.0, 1.02)
axD.axis("off")

fig.suptitle("Figure 1  Generative model, latent trajectories, the observed ECG, and the study design",
             x=0.008, y=0.985, ha="left", fontsize=12.5, fontweight="bold", color=G.INK)
head(fig, axA, "A", "No arrow runs from haemorrhage to the ECG", pad=0.045)
head(fig, axB_all, "B", "Pressure is defended while autonomic reserve is consumed", pad=0.045)
head(fig, [axC0, axC_beat], "C", "QRS amplitude falls with preload; QT shortens with rate")
head(fig, axD, "D", "Study workflow")
G.finish(fig, figpath(cfg, "MAIN1_design.png"), tight=False, dpi=700, vector=True)
print("MAIN1 written")


# ==================================================================== MAIN FIGURE 2
R = pd.read_csv(O("results_main.csv"))
Cc = pd.read_csv(O("results_controls.csv"))
PB = Cc[Cc.experiment == "C9b_personal_baseline_reference_excluded"]

REPS = [("population model (absolute features)", "Population\n(absolute)", G.MUTED),
        ("personal baseline (delta features)", "Personal\n(delta)", G.CAT[0]),
        ("absolute + delta", "Absolute\n+ delta", G.CAT[2])]
orderA = [m for m in ["hybrid", "raw_waveform_cnn", "handcrafted_xgb", "baseline_heart_rate_only"]
          if (R.model == m).any()]
labA = [G.MODEL_LABEL.get(m, m) for m in orderA]
colA = [G.MODEL_COLOUR.get(m, G.MUTED) for m in orderA]

fig = plt.figure(figsize=(16.4, 12.0))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.40,
                         left=0.150, right=0.985, top=0.93, bottom=0.075)
gsTop = outer[0].subgridspec(1, 2, wspace=0.22)
gsA = gsTop[0].subgridspec(1, 4, wspace=0.10)
gsB = gsTop[1].subgridspec(1, 4, wspace=0.10)

axA0, axA_all = None, []
for c, (key, name, ref) in enumerate(zip(TASK_KEY, TASK_LABEL_NARROW, TASK_REF)):
    ax = fig.add_subplot(gsA[0, c])
    axA_all.append(ax)
    if axA0 is None:
        axA0 = ax
    est = np.array([R.loc[R.model == m, key].values[0] for m in orderA], float)
    lo = np.array([R.loc[R.model == m, key + "_lo"].values[0] if key + "_lo" in R else np.nan
                   for m in orderA], float)
    hi = np.array([R.loc[R.model == m, key + "_hi"].values[0] if key + "_hi" in R else np.nan
                   for m in orderA], float)
    G.dot_ci(ax, labA, est, lo, hi, colours=colA, ref=ref, ref_label=None,
             xlabel=name, xlim=(0.45, 1.0) if ref == 0.5 else (-0.1, 1.0), fmt="{:.2f}")
    ax.tick_params(labelsize=7.2)
    ax.set_xlabel(name, fontsize=7.6)
    ax.set_xticks([0.6, 0.8] if ref == 0.5 else [0.0, 0.5])   # no tick on a shared edge
    if c:
        ax.set_yticklabels([])

axB0, axB_all = None, []
for c, (key, name, ref) in enumerate(zip(TASK_KEY, TASK_LABEL_NARROW, TASK_REF)):
    ax = fig.add_subplot(gsB[0, c])
    axB_all.append(ax)
    if axB0 is None:
        axB0 = ax
    est = np.array([PB.loc[PB.model == m, key].values[0] for m, _, _ in REPS], float)
    G.dot_ci(ax, [l for _, l, _ in REPS], est, None, None,
             colours=[col for _, _, col in REPS], ref=ref, ref_label=None,
             xlabel=name, xlim=(0.45, 1.0) if ref == 0.5 else (-0.1, 1.0), fmt="{:.2f}")
    ax.tick_params(labelsize=7.2)
    ax.set_xlabel(name, fontsize=7.6)
    ax.set_xticks([0.6, 0.8] if ref == 0.5 else [0.0, 0.5])
    if c:
        ax.set_yticklabels([])

# ---- 2C temporal extent of the input / 2D where each representation helps
LONG = pd.read_csv(O("longitudinal.csv"))
LROWS = [("single window", "A single 30 s window", G.MUTED),
         ("5 minute trajectory", "The preceding 5 min", G.SEQ[4]),
         ("10 minute trajectory", "The preceding 10 min", G.SEQ[7]),
         ("change from the subject's first window", "Change from first\n(pre-onset) window", G.CAT[0]),
         ("value + slope + curvature", "Value + slope + curvature", G.CAT[2]),
         ("recurrent model over the 10 minute trajectory", "Recurrent model, 10 min", G.CAT[1])]
LROWS = [r for r in LROWS if (LONG.representation == r[0]).any()]

gsBot = outer[1].subgridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.36)
gsC = gsBot[0].subgridspec(1, 3, wspace=0.10)
CTASK = [("stage_auroc_macro", "State classification\n(macro AUROC)", 0.5, (0.45, 1.0)),
         ("volume_r2", "Blood-volume loss\n(R²)", 0.0, (-0.1, 1.0)),
         ("decomp_auroc", "Imminent decompensation\n(AUROC)", 0.5, (0.45, 1.0))]
axC0, axC_all = None, []
for c, (key, name, ref, xlim) in enumerate(CTASK):
    ax = fig.add_subplot(gsC[0, c])
    axC_all.append(ax)
    if axC0 is None:
        axC0 = ax
    est = np.array([float(LONG.loc[LONG.representation == r[0], key].values[0])
                    for r in LROWS], float)
    G.dot_ci(ax, [r[1] for r in LROWS], est, None, None, colours=[r[2] for r in LROWS],
             ref=ref, ref_label=None, xlabel=name, xlim=xlim, fmt="{:.2f}")
    ax.tick_params(labelsize=7.2)
    ax.set_xticks([0.6, 0.8] if ref == 0.5 else [0.0, 0.5])   # no tick on a shared edge
    ax.set_xlabel(name, fontsize=7.6)
    if c:
        ax.set_yticklabels([])

axD = fig.add_subplot(gsBot[1])
base = LONG[LONG.representation == "single window"].iloc[0]
per_ = LONG[LONG.representation == "change from the subject's first window"].iloc[0]
tra_ = LONG[LONG.representation == "value + slope + curvature"].iloc[0]
short = ["State\nclassification", "Blood-volume\nloss", "Imminent\ndecompensation",
         "Physiological\nreserve"]
g_per = [float(per_[k]) - float(base[k]) for k in TASK_KEY]
g_tra = [float(tra_[k]) - float(base[k]) for k in TASK_KEY]
y = np.arange(len(short))[::-1]
h = 0.34
axD.barh(y + h / 2, g_per, height=h, color=G.CAT[0], zorder=2,
         label="Change from first (pre-onset) window")
axD.barh(y - h / 2, g_tra, height=h, color=G.CAT[2], zorder=2,
         label="Value + slope + curvature")
axD.axvline(0, color=G.AXIS, lw=1.0)
for yy, a, b in zip(y, g_per, g_tra):
    axD.text(a + 0.004, yy + h / 2, f"{a:+.3f}", va="center", fontsize=7.4, color=G.INK2)
    axD.text(b + 0.004, yy - h / 2, f"{b:+.3f}", va="center", fontsize=7.4, color=G.INK2)
axD.set_yticks(y)
axD.set_yticklabels(short, fontsize=8)
axD.set_xlim(min(0, min(g_per + g_tra) * 1.2), max(g_per + g_tra) * 1.4)
axD.set_xlabel("Gain over a single 30 s window", fontsize=8.4)
axD.grid(axis="y", visible=False)
axD.legend(fontsize=7.4, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2,
           columnspacing=1.0, handlelength=1.2)
# the one-line reading of this panel is in the caption, not repeated on the page

fig.suptitle("Figure 2  A personal reference estimates where the patient is; the recent "
             "trajectory estimates where the patient is going",
             x=0.008, y=0.985, ha="left", fontsize=12.5, fontweight="bold", color=G.INK)
head(fig, axA_all, "A", "A single 30 s ECG, cross-sectional models")
head(fig, axB_all, "B", "The same ECG read against the subject’s own baseline")
head(fig, axC_all, "C", "How much of the subject's recent history the model sees")
head(fig, axD, "D", "Complementary, not competing")
G.finish(fig, figpath(cfg, "MAIN2_personalisation.png"), tight=False, dpi=700, vector=True)
print("MAIN2 written")


# ==================================================================== MAIN FIGURE 3
# A heart-rate matching | B non-ECG shortcuts | C cross-generator | D real recordings.
# The prediction-horizon panel moved to the supplement: it describes the result, it does
# not test it.  These four are the experiments that could have ended the paper.
fig = plt.figure(figsize=(16.4, 11.6))
outer = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.50,
                         left=0.10, right=0.985, top=0.93, bottom=0.10)

# ---- 3A heart-rate matching
axA = fig.add_subplot(outer[0, 0])
M = pd.read_csv(O("matched_hr_main.csv"))
M["band"] = M.hr_lo.astype(int).astype(str) + "–" + M.hr_hi.astype(int).astype(str)
bands = M.drop_duplicates("band").sort_values("hr_lo").band.tolist()
xx = np.arange(len(bands))
for m in ["baseline_heart_rate_only", "handcrafted_xgb", "raw_waveform_cnn", "hybrid"]:
    d = M[M.model == m].set_index("band").reindex(bands)
    if d.stage_auroc_macro.isna().all():
        continue
    col = G.STATUS["critical"] if m == "baseline_heart_rate_only" else G.MODEL_COLOUR.get(m, G.MUTED)
    axA.plot(xx, d.stage_auroc_macro.to_numpy(), "-o", color=col,
             lw=2.4 if m == "baseline_heart_rate_only" else 1.9,
             label=G.MODEL_LABEL.get(m, m), markeredgecolor=G.SURFACE, markeredgewidth=1.2)
axA.axhline(0.5, color=G.STATUS["critical"], lw=1.2, ls=(0, (4, 3)))
axA.text(0.0, 0.505, "chance", color=G.STATUS["critical"], fontsize=7.5, ha="left", va="bottom")
axA.set_xticks(xx)
axA.set_xticklabels([b + "\n(n=" + str(int(M[M.band == b].n_band.iloc[0])) + ")"
                     for b in bands], fontsize=8)
axA.set_ylim(0.30, 0.90)
axA.set_ylabel("State classification (macro AUROC)", fontsize=8.6)
axA.set_xlabel("Heart-rate band of the held-out windows (bpm)", fontsize=8.6)
axA.legend(fontsize=7.6, loc="lower right", ncol=2, borderaxespad=0.2)

# ---- 3B what is predictable with no ECG at all
axB = fig.add_subplot(outer[0, 1])
LK = pd.read_csv(O("leakage_baselines.csv"))
MT = pd.read_csv(O("matched_time.csv"))
# Rows that contain when the subject's trajectory ends (fraction completed, duration,
# censoring) are oracle -- a monitor cannot know them -- and are drawn in grey and
# labelled as such; the prospective counterpart comes from run_incremental.py.
SHOW = [("window index only", "Window index only", G.STATUS["critical"]),
        ("fraction of trajectory completed", "Trajectory position (oracle)", G.MUTED),
        ("measured heart rate only", "Measured heart rate only", G.STATUS["critical"]),
        ("heart rate + time", "Heart rate + time", G.STATUS["critical"])]
SHOW = [s for s in SHOW if (LK.features == s[0]).any()]
labs = [s[1] for s in SHOW]
vals = [float(LK.loc[LK.features == s[0], "stage_auroc_macro"].values[0]) for s in SHOW]
cols = [s[2] for s in SHOW]
IVp = pd.read_csv(O("incremental_value.csv"))
_p = IVp[(IVp.setting == "all held-out windows") & (IVp.task == "stage_auroc_macro")].iloc[0]
labs += ["Elapsed time + characteristics\n+ heart rate (prospective)",
         "  + trajectory end (oracle)", "ECG: handcrafted features"]
vals += [float(_p.metadata_hr),
         float(LK.loc[LK.features == "all metadata + heart rate", "stage_auroc_macro"].values[0]),
         float(LK.loc[LK.features == "ECG: handcrafted_xgb", "stage_auroc_macro"].values[0])]
cols += [G.STATUS["critical"], G.MUTED, G.CAT[0]]
mt_h = MT[(MT.subset == "time+rate-matched") & (MT.model == "handcrafted_xgb")]
mt_r = MT[(MT.subset == "time+rate-matched") & (MT.model == "baseline_heart_rate_only")]
if len(mt_h) and len(mt_r):
    labs += ["time + rate matched:\nheart rate only", "time + rate matched:\nhandcrafted features"]
    vals += [float(mt_r.stage_auroc_macro.values[0]), float(mt_h.stage_auroc_macro.values[0])]
    cols += [G.STATUS["critical"], G.CAT[0]]
y = np.arange(len(vals))[::-1]
axB.barh(y, vals, color=cols, height=0.62, zorder=2)
axB.axvline(0.5, color=G.INK2, lw=1.2, ls=(0, (4, 3)))
for yy, v in zip(y, vals):
    axB.text(v + 0.006, yy, f"{v:.3f}", va="center", fontsize=7.6, color=G.INK2)
axB.set_yticks(y)
axB.set_yticklabels(labs, fontsize=7.6)
axB.set_xlim(0.45, 1.02)
axB.set_xlabel("State classification (macro AUROC)", fontsize=8.6)
axB.grid(axis="y", visible=False)
axB.axhline(1.5, color=G.AXIS, lw=1.2, ls=":")

# ---- 3C cross-generator transfer
gsC = outer[1, 0].subgridspec(1, 2, wspace=0.08)
XG = pd.read_csv(O_X("cross_generator.csv"))
GAP = pd.read_csv(O_X("generator_gaps.csv"))
TRAIN = ["gaussian", "dynamical", "empirical", "gaussian+dynamical"]
TEST = ["gaussian", "dynamical", "empirical"]
GL = {"gaussian": "Gaussian", "dynamical": "Dynamical",
      "empirical": "Real beat", "gaussian+dynamical": "Gauss. + dyn."}
H = XG[XG.family == "handcrafted"]
axC_all = []
for k, (key, name, vmin, vmax, fmt) in enumerate(
        [("stage_auroc_macro", "State (macro AUROC)", 0.5, 1.0, "{:.3f}"),
         ("volume_r2", "Blood volume (R²)", 0.4, 1.0, "{:.3f}")]):
    ax = fig.add_subplot(gsC[0, k])
    axC_all.append(ax)
    Mx = np.full((len(TRAIN), len(TEST)), np.nan)
    for r_, tr in enumerate(TRAIN):
        for c_, te in enumerate(TEST):
            d = H[(H.train_on == tr) & (H.test_on == te)]
            if len(d):
                Mx[r_, c_] = float(d[key].values[0])
    G.heat(ax, Mx, [GL[t] for t in TEST], [GL[t] for t in TRAIN],
           vmin=vmin, vmax=vmax, fmt=fmt)
    for r_, tr in enumerate(TRAIN):                    # ring every in-distribution test
        for te in TEST:
            if te in tr.split("+"):
                ax.add_patch(plt.Rectangle((TEST.index(te) - 0.5, r_ - 0.5), 1, 1, fill=False,
                                           edgecolor=G.INK, lw=2.0, zorder=5))
    if ax.images:                                      # colour scale for each heat map
        cb = fig.colorbar(ax.images[0], ax=ax, orientation="horizontal", fraction=0.045,
                          pad=0.42, aspect=18)
        cb.set_ticks([0.6, 0.8])            # interior ticks only: end labels collide
        cb.ax.tick_params(labelsize=6.5)
        cb.outline.set_visible(False)
    ax.set_title(name, fontsize=8.6)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Tested on", fontsize=8.4)
    if k == 0:
        ax.set_ylabel("Trained on", fontsize=8.4)
    else:
        ax.set_yticklabels([])
# the outlined-cell key and the generator discriminability are stated in the caption

# ---- 3D how bad can the personal reference get
axD = fig.add_subplot(outer[1, 1])
DR = pd.read_csv(O("baseline_doseresponse.csv"))
pop = DR[DR.strategy.str.startswith("population")].iloc[0]
pr = DR[DR.strategy == "one prior recording"].sort_values("ref_similarity", ascending=False)
xx = pr.ref_similarity.to_numpy()
for key, lab, col in [("stage_auroc_macro", "State classification (macro AUROC)", G.CAT[0]),
                      ("volume_r2", "Blood-volume loss (R²)", G.CAT[2])]:
    axD.plot(xx, pr[key].to_numpy() - float(pop[key]), "-o", color=col, lw=2.2, label=lab,
             markeredgecolor=G.SURFACE, markeredgewidth=1.2)
axD.axhline(0, color=G.STATUS["critical"], lw=1.4, ls=(0, (4, 3)))
axD.text(0.98, 0.004, "break-even", transform=axD.get_yaxis_transform(),
         color=G.STATUS["critical"], fontsize=7.4, va="bottom", ha="right")
axD.invert_xaxis()
axD.set_xlabel("Similarity of the degraded reference to a clean one\n"
               "(waveform correlation; degradation increases to the right)", fontsize=8.6)
axD.set_ylabel("Gain over the population model", fontsize=8.6)
axD.set_ylim(-0.045, 0.215)
axD.legend(fontsize=8, loc="upper right")
for j_, (x_, lv) in enumerate(zip(xx, pr.level.to_numpy())):
    # adjacent mild levels sit close together on the x axis: stagger them vertically
    axD.text(x_, -0.024 if j_ % 2 == 0 else -0.040, f"{lv:g}x", ha="center", fontsize=6.8,
             color=G.MUTED)
# the severity multiplier is defined in the caption

fig.suptitle("Figure 3  Stress tests of the hypothesis: a heart-rate shortcut, a metadata "
             "shortcut, a change of generator, and a degraded reference",
             x=0.008, y=0.985, ha="left", fontsize=12.5, fontweight="bold", color=G.INK)
head(fig, axA, "A", "Rate-only model collapses inside narrow rate bands")
head(fig, axB, "B", "Baselines without any ECG, and time–rate matching")
head(fig, axC_all, "C", "Trained on one generator, tested on another")
head(fig, axD, "D", "A degraded personal reference: gain against reference quality")
G.finish(fig, figpath(cfg, "MAIN3_defence.png"), tight=False, dpi=700, vector=True)
print("MAIN3 written")


# ==================================================================== MAIN FIGURE 4
# The real-world arm, promoted out of a corner of Figure 3.  It is the only part of the
# study performed on recordings from patients undergoing a real haemodynamic insult, and
# most of what it says is negative, which is why it needs the room.
# Explicit axes rectangles, in fractions of the printed 9.0 x 6.3 in page area: the
# long row labels of A, B and C need fixed gutters, which a gridspec does not give.
fig = plt.figure(figsize=(16.4, 16.4 * 6.3 / 9.0))
RECT = {"A": [0.215, 0.665, 0.235, 0.26], "B": [0.715, 0.665, 0.255, 0.26],
        "C": [0.265, 0.16, 0.315, 0.36], "D1": [0.765, 0.42, 0.205, 0.10],
        "D2": [0.765, 0.16, 0.205, 0.15]}


class _Outer:
    def __getitem__(self, k):
        return {(0, 0): "A", (0, 1): "B", (1, 0): "C"}[k]


outer = _Outer()
_add = fig.add_subplot
fig.add_subplot = lambda key, *a, **k: fig.add_axes(RECT[key]) if isinstance(key, str) \
    else _add(key, *a, **k)

SD = pd.read_csv(O_V2("state_detection.csv"))
DS = pd.read_csv(O_V("discrimination.csv"))
CT = pd.read_csv(O_V("continuous_targets.csv"))
CH = pd.read_csv(O_V("cohort_summary.csv")).set_index("quantity")
hyp = SD[SD.target.str.contains("hypotension")]

# ---- 4A concurrent hypotension in real patients
axA = fig.add_subplot(outer[0, 0])
ORDER = [("heart rate only", "Heart rate only", G.STATUS["critical"]),
         ("all ECG features, absolute", "ECG, absolute", G.MUTED),
         ("recorded confounders only", "Anaesthetic and vasopressor state", G.CAT[3]),
         ("all ECG features, absolute + change", "ECG, absolute + change", G.SEQ[5]),
         ("ECG (absolute + change) + confounders", "ECG (absolute + change)\n+ anaesthetic state", G.CAT[2]),
         ("all ECG features, change from own baseline", "ECG, change from own baseline",
          G.CAT[0])]
rows_ = [(l, float(hyp.loc[hyp.predictors == k, "auroc"].values[0]), c)
         for k, l, c in ORDER if (hyp.predictors == k).any()]
rows_.sort(key=lambda r: r[1])
y = np.arange(len(rows_))
axA.barh(y, [r[1] for r in rows_], color=[r[2] for r in rows_], height=0.62, zorder=2)
axA.axvline(0.5, color=G.INK2, lw=1.3, ls=(0, (4, 3)))
for yy, r in zip(y, rows_):
    axA.text(r[1] + 0.006, yy, f"{r[1]:.3f}", va="center", fontsize=7.8, color=G.INK2,
             bbox=dict(fc="white", ec="none", pad=0.4), zorder=4)
axA.set_yticks(y)
axA.set_yticklabels([r[0] for r in rows_], fontsize=7.8)
axA.set_xlim(0.3, 0.78)
axA.set_xlabel("AUROC, concurrent hypotension (MAP < 65 mmHg)", fontsize=8.6)
axA.grid(axis="y", visible=False)
# window and event counts are stated in the caption

# ---- 4B what does not work
axB = fig.add_subplot(outer[0, 1])
lab, val, col = [], [], []
for k, l in [("ECG change only", "ECG change"),
             ("ECG change + confounders", "ECG change + anaesthetic state"),
             ("recorded confounders only", "Anaesthetic state only"),
             ("monitor haemodynamics only (MAP and HR change)", "Monitor: pressure and rate")]:
    d = DS[DS.predictors == k]
    if len(d):
        lab.append(l); val.append(float(d.window_auroc.values[0]))
        col.append(G.CAT[0] if "ECG" in l else G.MUTED)
y = np.arange(len(val))[::-1]
axB.barh(y, val, color=col, height=0.6, zorder=2)
axB.axvline(0.5, color=G.INK2, lw=1.3, ls=(0, (4, 3)))
for yy, v in zip(y, val):
    axB.text(v + 0.006, yy, f"{v:.3f}", va="center", fontsize=7.8, color=G.INK2,
             bbox=dict(fc="white", ec="none", pad=0.4), zorder=4)
axB.set_yticks(y)
axB.set_yticklabels(lab, fontsize=7.8)
axB.set_xlim(0.3, 0.90)
axB.set_xlabel("AUROC: is this a bleeding operation?", fontsize=8.6)
axB.grid(axis="y", visible=False)
# Two colours covering four rows, so the colour needs naming; panel A is not given
# a legend because there each bar has its own colour and its own axis label.
axB.legend(handles=[Patch(facecolor=G.CAT[0], label="From the ECG"),
                    Patch(facecolor=G.MUTED, label="Already monitored\nor recorded")],
           loc="center right", bbox_to_anchor=(1.0, 0.56), fontsize=7.2, frameon=False, handlelength=1.1,
           handleheight=1.1, borderpad=0.2, labelspacing=0.45)
mapr = CT[(CT.target.str.contains("mean arterial")) & (CT.predictors == "ECG change")]
# the cohort split and the MAP-change R² are stated in the caption

# ---- 4C what the ECG adds to what is already known
axC = fig.add_subplot(outer[1, 0])
IV = pd.read_csv(O("incremental_value.csv"))
IVV = pd.read_csv(O("incremental_vitaldb.csv")) if have("incremental_vitaldb.csv") else None
TL = {"stage_auroc_macro": "State", "volume_r2": "Blood-volume loss",
      "reserve_r2": "Reserve", "decomp_auroc": "Imminent decompensation"}
items = []
for setting, tag, c in [("all held-out windows", "sim.", G.MUTED),
                        ("time + rate matched", "sim., matched", G.CAT[0])]:
    for k, l in TL.items():
        d = IV[(IV.setting == setting) & (IV.task == k)]
        if len(d) and k == "decomp_auroc" and int(d.n_decomp_events.values[0]) < 20:
            continue                     # too few events in the matched subset (see Table S13)
        if len(d):
            items.append((f"{l}  ({tag})", float(d.delta.values[0]),
                          float(d.delta_lo.values[0]), float(d.delta_hi.values[0]), c))
if IVV is not None:
    for _, r in IVV.iterrows():
        items.append((f"Hypotension  (real, "
                      f"{r.level.split('(')[0].strip().replace(' level', '')})", float(r.delta), float(r.delta_lo),
                      float(r.delta_hi), G.CAT[1]))
y = np.arange(len(items))[::-1]
for yy, (_, d_, lo, hi, c) in zip(y, items):
    if np.isfinite(lo):
        axC.plot([lo, hi], [yy, yy], color=c, lw=2.0, alpha=0.55, solid_capstyle="round")
    axC.plot([d_], [yy], "o", color=c, markersize=7, markeredgecolor=G.SURFACE,
             markeredgewidth=1.5)
axC.axvline(0, color=G.STATUS["critical"], lw=1.4, ls=(0, (4, 3)))
axC.set_yticks(y)
axC.set_yticklabels([it[0] for it in items], fontsize=8.6)
axC.set_xlabel("Change in performance when the ECG is added to elapsed time,\n"
               "heart rate and recorded patient state", fontsize=8.6)
axC.grid(axis="y", visible=False)
# Colour is the only thing that groups these nine rows into the three settings they
# come from, so it is named here as well as in each row's own suffix.
axC.legend(handles=[
    Line2D([], [], color=G.MUTED, marker="o", lw=2.0, markersize=7,
           markeredgecolor=G.SURFACE, markeredgewidth=1.5,
           label="Simulation, all held-out windows"),
    Line2D([], [], color=G.CAT[0], marker="o", lw=2.0, markersize=7,
           markeredgecolor=G.SURFACE, markeredgewidth=1.5,
           label="Simulation, time and rate matched"),
    Line2D([], [], color=G.CAT[1], marker="o", lw=2.0, markersize=7,
           markeredgecolor=G.SURFACE, markeredgewidth=1.5,
           label="Real intraoperative cohort")],
    loc="upper left", bbox_to_anchor=(-0.55, -0.24), ncol=3, fontsize=7.2, frameon=False,
    handlelength=2.8, borderpad=0.2, columnspacing=1.2)
# bootstrap and interval conventions are stated in the caption

# ---- 4D how far the simulation is from a real recording
PSx = pd.read_csv(O("ptbxl_summary.csv")).iloc[0]
axD1 = fig.add_subplot("D1")
vals2 = [float(PSx.domain_auroc_features), float(PSx.domain_auroc_embedding)]
y = np.arange(2)[::-1]
axD1.barh(y, vals2, color=[G.STATUS["critical"], G.STATUS["serious"]], height=0.5, zorder=2)
axD1.axvline(0.5, color=G.INK2, lw=1.3, ls=(0, (4, 3)))
for yy, v in zip(y, vals2):
    axD1.text(v - 0.02, yy, f"{v:.3f}", va="center", ha="right", fontsize=8.4, color="#ffffff")
axD1.set_yticks(y)
axD1.set_yticklabels(["Handcrafted\nfeature space", "Encoder\nembedding space"], fontsize=8)
axD1.set_xlim(0.4, 1.05)
axD1.set_xlabel("Real vs synthetic AUROC (0.5 = same)", fontsize=8.4)
axD1.grid(axis="y", visible=False)

axD2 = fig.add_subplot("D2")
SMx = pd.read_csv(O("ptbxl_smd.csv"))
sc = "smd" if "smd" in SMx.columns else [c for c in SMx.columns if "smd" in c.lower()][0]
fc = "feature" if "feature" in SMx.columns else SMx.columns[0]
top = SMx.reindex(SMx[sc].abs().sort_values(ascending=False).index).head(6)
y = np.arange(len(top))[::-1]
axD2.barh(y, top[sc].to_numpy(), height=0.6, zorder=2,
          color=[G.CAT[0] if v < 0 else G.CAT[1] for v in top[sc]])
axD2.axvline(0, color=G.AXIS, lw=1.0)
axD2.set_yticks(y)
axD2.set_yticklabels([G.feature_label(str(f)) for f in top[fc]], fontsize=7.6)
axD2.set_xlabel("SMD (synthetic − real)", fontsize=8.4)
axD2.grid(axis="y", visible=False)
# One line rather than two: the second line used to run into the note below it.
axD2.legend(handles=[Patch(facecolor=G.CAT[1], label="Synthetic higher"),
                     Patch(facecolor=G.CAT[0], label="Synthetic lower")],
            loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=7.2,
            frameon=False, handlelength=1.1, handleheight=1.1, borderpad=0.2,
            columnspacing=1.2)
axD2.set_title(f"Median |SMD| {float(PSx.median_abs_smd):.2f}; "
               f"{float(PSx.frac_smd_gt_0p5)*100:.0f}% of features > 0.5 SD",
               fontsize=7.4, fontweight="normal", color=G.INK2)

fig.suptitle("Figure 4  Real recordings: not a bleeding detector, but a personalised "
             "physiological monitor",
             x=0.008, y=0.985, ha="left", fontsize=12.5, fontweight="bold", color=G.INK)
head(fig, axA, "A", "Monitoring: the ordering reproduces")
head(fig, axB, "B", "Diagnosis: the bleeding operation is not identified")
head(fig, axC, "C", "What the ECG adds to what is already known")
head(fig, [axD1, axD2], "D", "Simulation versus real recordings")
fig.add_subplot = _add
G.finish(fig, figpath(cfg, "MAIN4_realworld.png"), tight=False, dpi=700, vector=True)
print("MAIN4 written")
