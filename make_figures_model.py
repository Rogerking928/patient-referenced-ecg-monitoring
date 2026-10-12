"""Figures 1-4: causal diagram, state-space schematic, latent trajectories,
example synthetic ECGs.  These depend only on the simulator, not on any model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from hemo import config, ecg, figures as G, physiology as phys, simulate
from hemo.config import fig as figpath, out

G.style()
cfg = config.load()
subj = pd.read_parquet(out(cfg, "subjects.parquet"))
w = np.load(out(cfg, "world.npz"))
Z, D, t, lab, viab = w["Z"], w["D"], w["t"], w["lab"], w["viab"]

# =================================================================== F1 causal diagram
NODES = {
    "haemorrhage":   (0.055, 0.72, "Haemorrhage\nrate", "exposure"),
    "treatment":     (0.055, 0.46, "Fluid /\ntransfusion", "exposure"),
    "volume":        (0.205, 0.59, "Blood\nvolume", "latent"),
    "preload":       (0.360, 0.72, "Venous return\n/ preload", "latent"),
    "symp":          (0.360, 0.42, "Sympathetic\ntone", "latent"),
    "para":          (0.360, 0.20, "Parasympathetic\ntone", "latent"),
    "sv":            (0.520, 0.85, "Stroke\nvolume", "latent"),
    "hr":            (0.520, 0.31, "Heart rate", "derived"),
    "resp":          (0.520, 0.09, "Respiratory\nrate", "latent"),
    "co":            (0.675, 0.72, "Cardiac\noutput", "latent"),
    "map":           (0.675, 0.60, "Mean arterial\npressure", "derived"),
    "ox":            (0.675, 0.92, "Oxygen\ndelivery", "latent"),
    "metab":         (0.830, 0.83, "Metabolic\nstress", "latent"),
    "ecg":           (0.945, 0.30, "ECG\nwaveform", "observed"),
    "subject":       (0.205, 0.10, "Subject\nparameters", "modifier"),
    "noise":         (0.830, 0.10, "Recording\nnoise", "modifier"),
}
FEEDBACK = {("map", "symp"), ("map", "para"), ("metab", "symp"), ("hr", "sv")}
EDGES = [
    ("haemorrhage", "volume"), ("treatment", "volume"),
    ("volume", "preload"), ("preload", "sv"), ("sv", "co"), ("hr", "co"),
    ("co", "map"), ("symp", "map"), ("map", "symp"), ("map", "para"),
    ("symp", "hr"), ("para", "hr"), ("symp", "sv"), ("symp", "preload"),
    ("co", "ox"), ("ox", "metab"), ("metab", "symp"), ("metab", "resp"),
    ("symp", "resp"), ("hr", "sv"),
    ("preload", "ecg"), ("symp", "ecg"), ("para", "ecg"), ("resp", "ecg"),
    ("metab", "ecg"), ("hr", "ecg"), ("subject", "ecg"), ("noise", "ecg"),
    ("subject", "symp"), ("subject", "preload"),
]
KIND = {"exposure": "#4a3aa7", "latent": G.CAT[0], "derived": "#5598e7",
        "observed": G.CAT[1], "modifier": G.MUTED}

fig, ax = plt.subplots(figsize=(12.6, 7.4))
for a_, b_ in EDGES:
    xa, ya = NODES[a_][:2]
    xb, yb = NODES[b_][:2]
    to_ecg = b_ == "ecg"
    fb = (a_, b_) in FEEDBACK
    # routed so that no arrow into the waveform passes behind a box it does not belong to
    # (curvatures checked against every box by geometry: no arrow into the waveform is hidden)
    rad = {("symp", "ecg"): -0.22, ("preload", "ecg"): 0.05, ("subject", "ecg"): 0.05,
           ("para", "ecg"): 0.10, ("hr", "ecg"): 0.0, ("metab", "ecg"): 0.0,
           ("resp", "ecg"): 0.0, ("noise", "ecg"): 0.0, ("subject", "symp"): -0.15,
           ("subject", "preload"): -0.10}.get(
        (a_, b_), -0.28 if fb else 0.10)
    ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb),
                                 connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=13,
                                 lw=1.7 if to_ecg else 1.1,
                                 color=G.CAT[1] if to_ecg else (G.MUTED if fb else G.AXIS),
                                 alpha=0.9 if to_ecg else 0.8,
                                 linestyle="--" if fb else "-",
                                 shrinkA=17, shrinkB=17, zorder=1))
for k, (x, y, lbl, kind) in NODES.items():
    c = KIND[kind]
    ax.add_patch(FancyBboxPatch((x - 0.049, y - 0.041), 0.098, 0.082,
                                boxstyle="round,pad=0.006,rounding_size=0.014",
                                linewidth=1.7, edgecolor=c,
                                facecolor="#ffffff" if kind != "observed" else c,
                                zorder=2))
    ax.text(x, y, lbl, ha="center", va="center", fontsize=7.4, zorder=3,
            color="#ffffff" if kind == "observed" else G.INK)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [Patch(facecolor="#ffffff", edgecolor=KIND[k], label=v) for k, v in
           [("exposure", "Exposure / intervention"), ("latent", "Latent physiological state"),
            ("derived", "Derived haemodynamic variable"),
            ("modifier", "Subject / measurement modifier")]]
handles += [Patch(facecolor=KIND["observed"], edgecolor=KIND["observed"], label="Observation (ECG)"),
            Line2D([0], [0], color=G.MUTED, ls="--", lw=1.3, label="Negative feedback")]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.11), ncol=6, fontsize=8)
ax.set_xlim(-0.005, 1.005)
ax.set_ylim(0.0, 1.0)
ax.axis("off")
fig.tight_layout(rect=[0, 0.03, 1, 0.90])
G.finish(fig, figpath(cfg, "F01_causal_diagram.png"), tight=False)

# =================================================================== F2 trajectories
def smooth_stage(st, win=61):
    """Majority filter: process noise flips the label at boundaries, and a
    stripe-per-flip shading is unreadable."""
    st = np.asarray(st)
    out = st.copy()
    h = win // 2
    for i in range(len(st)):
        seg = st[max(0, i - h):i + h + 1]
        out[i] = np.bincount(seg, minlength=5).argmax()
    return out


def pick(pheno, want_shock=True):
    m = (subj.phenotype == pheno) & (subj.bleeder == 1)
    ok = np.where(m.to_numpy() & ((lab == 3).any(1) == want_shock))[0]
    return int(ok[0]) if len(ok) else int(np.where(m.to_numpy())[0][0])


rows = [("high_responder", "High responder"), ("normal_responder", "Normal responder"),
        ("poor_compensator", "Poor compensator")]
COLS = ["Volume state (% of baseline)", "Rate and pressure (bpm / mmHg)",
        "Autonomic reserve (0-1)", "Metabolic stress (mmol/L)"]
fig, axes = plt.subplots(3, 4, figsize=(15.0, 8.6), sharex=True)
for r, (ph_name, ph_lab) in enumerate(rows):
    i = pick(ph_name)
    v = viab[i]
    tt, st = t[v], smooth_stage(lab[i][v])
    panels = [
        [("Volume lost", D[i, v, 6] * 100, G.CAT[0], "-"),
         ("Preload (% of baseline)", D[i, v, 7] * 100, G.CAT[1], "--")],
        [("Heart rate", D[i, v, 0], G.CAT[0], "-"),
         ("Mean arterial pressure", D[i, v, 2], G.CAT[1], "--")],
        [("Autonomic reserve", D[i, v, 4], G.CAT[0], "-")],
        [("Lactate", Z[i, v, 8], G.CAT[0], "-")],
    ]
    for c, series in enumerate(panels):
        ax = axes[r, c]
        G.stage_bands(ax, tt, st)
        for name, y, col, ls in series:
            ax.plot(tt, y, color=col, lw=1.9, ls=ls, zorder=3,
                    label=name if r == 0 else None)
        if v.sum() < len(v):
            ax.axvline(tt[-1], color=G.STATUS["critical"], lw=1.2, ls=":")
            # the marker is explained in the caption; text here would sit on the curves
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.30 * (hi_ - lo_))     # headroom for the legend
        if r == 0:
            ax.set_title(COLS[c], fontsize=9.5)
            if len(series) > 1:
                ax.legend(loc="upper left", fontsize=7.2, framealpha=0.0)
        if r == 2:
            ax.set_xlabel("Time (min)")
        if c == 0:
            ax.set_ylabel(ph_lab, fontsize=9.5, color=G.INK, fontweight="bold")
_present = sorted({int(k) for i_ in [pick(p_) for p_, _ in rows] for k in np.unique(lab[i_][viab[i_]])})
G.stage_legend(axes[2, 0], present=_present, loc="upper center", bbox_to_anchor=(2.35, -0.22),
               ncol=5, fontsize=8.5)
fig.suptitle("Figure 2  Latent trajectories of one representative bleeding subject per "
             "compensation phenotype (background shading = clinical state)",
             x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
fig.tight_layout(rect=[0, 0.055, 1, 0.965])
G.finish(fig, figpath(cfg, "F02_trajectories.png"), tight=False)

# =================================================================== F3 example ECGs
rng = np.random.default_rng(7)
clean = np.where((subj.noise_level.values < 0.6) & ((lab == 3).sum(1) > 120)
                 & ((lab == 1).sum(1) > 120))[0]
i = int(clean[0])
row = subj.iloc[i]
ts = t * 60.0
fig = plt.figure(figsize=(14.5, 7.6))
gs = fig.add_gridspec(4, 3, width_ratios=[2.6, 1.0, 1.0], hspace=0.55, wspace=0.28)
stages_present = [k for k in range(5) if (lab[i] == k).any()]
for j, k in enumerate(stages_present[:4]):
    k0 = int(np.argmax(lab[i] == k))
    sig = ecg.render_window(cfg, row, ts[k0:k0 + 30], Z[i, k0:k0 + 30], D[i, k0:k0 + 30],
                            np.random.default_rng(100 + j))
    tt = np.arange(sig.shape[1]) / cfg.simulation.fs
    axw = fig.add_subplot(gs[j, 0])
    axw.plot(tt, sig[0], color=G.STAGE_COLOURS[k], lw=0.9)
    axw.set_xlim(0, 8)
    axw.set_ylabel(G.STAGE_LABELS[k], fontsize=8.5, color=G.INK)
    axw.text(0.995, 0.04, f"HR {D[i,k0,0]:.0f} bpm   preload {D[i,k0,7]:.2f}   "
                          f"MAP {D[i,k0,2]:.0f} mmHg   lactate {Z[i,k0,8]:.1f} mmol/L",
             transform=axw.transAxes, ha="right", va="bottom", fontsize=7.5, color=G.INK2)
    if j == 3:
        axw.set_xlabel("Time (s)")
    if j == 0:
        axw.set_title("Lead II, 8 s of a 30 s window", fontsize=9.5)

    # median beat, both leads
    from hemo import features as FE
    for c, leadname in enumerate(cfg.simulation.leads):
        fs_ = cfg.simulation.fs
        pk = FE.detect_rpeaks(np.asarray(sig[0], float), fs_)
        rr_ = float(np.median(np.diff(pk)) / fs_) if len(pk) > 2 else 0.85
        # same rate-adaptive window the feature extractor uses, so the panel shows
        # exactly what the model measures rather than an over-long template
        pre_, post_ = min(0.30, 0.42 * rr_), min(0.50, 0.62 * rr_)
        med, _ = FE._median_beat(np.asarray(sig[c], float), pk, fs_, pre_, post_)
        axb = fig.add_subplot(gs[j, 1 + c])
        if med is not None:
            axb.plot((np.arange(len(med)) - pre_ * fs_) / fs_ * 1000,
                     med, color=G.STAGE_COLOURS[k], lw=1.8)
        axb.axhline(0, color=G.AXIS, lw=0.8)
        axb.set_xlim(-250, 460)
        if j == 0:
            axb.set_title(f"Median beat, lead {leadname}", fontsize=9.5)
        if j == 3:
            axb.set_xlabel("Time from R (ms)")
fig.suptitle("Figure 3  Synthetic ECG of one subject across the haemodynamic states",
             x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
fig.tight_layout(rect=[0, 0, 1, 0.965])
G.finish(fig, figpath(cfg, "F03_example_ecg.png"), tight=False)

# =================================================================== F4 no single-feature rule
X, meta = simulate.load(cfg)
F = pd.read_parquet(out(cfg, "features_main.parquet"))
keys = ["hr_mean", "rmssd", "lf_hf", "II_r_amp", "II_p_amp", "II_qtc",
        "II_r_resp_mod", "II_st_level", "resp_rate_wander"]
fig, axes = plt.subplots(3, 3, figsize=(12.5, 8.0))
for ax, k in zip(axes.ravel(), keys):
    data = [F[k].to_numpy()[meta.stage.to_numpy() == s] for s in range(5)]
    data = [d[np.isfinite(d)] for d in data]
    allv = np.concatenate(data)
    lo_c, hi_c = np.percentile(allv, [0.5, 99.5])
    data = [np.clip(d, lo_c, hi_c) for d in data]
    parts = ax.violinplot(data, showextrema=False, widths=0.85)
    for pc, c in zip(parts["bodies"], G.STAGE_COLOURS):
        pc.set_facecolor(c)
        pc.set_alpha(0.75)
        pc.set_edgecolor(G.SURFACE)
    for s in range(5):
        ax.plot([s + 1], [np.median(data[s])], "o", color=G.INK, ms=3.5, zorder=4)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["Base", "Comp", "Early", "Shock", "Recov"], fontsize=7.5)
    ax.set_title(k, fontsize=9)
fig.suptitle("Figure 4  Feature distributions by state (black dot = median, display clipped to the "
             "0.5-99.5 percentile): every marginal overlaps heavily, by construction",
             x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
fig.tight_layout(rect=[0, 0, 1, 0.955])
G.finish(fig, figpath(cfg, "F04_feature_overlap.png"), tight=False)

print("model figures written")
