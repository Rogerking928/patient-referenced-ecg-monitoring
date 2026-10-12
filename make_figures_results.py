"""Figures 5-16: results, controls, interpretability and the public-data arm.
Each block is skipped if its inputs are not on disk yet."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hemo import config, figures as G, physiology as phys
from hemo.config import fig as figpath, out

G.style()
cfg = config.load()
O = lambda *p: out(cfg, *p)
have = lambda *p: O(*p).exists()

TASKS = [("stage_auroc_macro", "State classification\n(macro AUROC, 5 classes)", 0.5, (0.45, 1.0)),
         ("volume_r2", "Blood-volume loss\n(R²)", 0.0, (-0.35, 1.0)),
         ("decomp_auroc", "Imminent decompensation\n(AUROC, 10 min horizon)", 0.5, (0.45, 1.0)),
         ("reserve_r2", "Physiological reserve\n(R²)", 0.0, (-0.35, 1.0))]


def panelled_dotci(df, order, labels, colours, title, fname, note=None, figsize=(14.6, 5.2)):
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    for ax, (key, name, ref, xlim) in zip(axes, TASKS):
        est = [df.loc[df.model == m, key].values[0] if (df.model == m).any() else np.nan for m in order]
        lo = [df.loc[df.model == m, key + "_lo"].values[0]
              if (df.model == m).any() and key + "_lo" in df else np.nan for m in order]
        hi = [df.loc[df.model == m, key + "_hi"].values[0]
              if (df.model == m).any() and key + "_hi" in df else np.nan for m in order]
        G.dot_ci(ax, labels, np.array(est, float), np.array(lo, float), np.array(hi, float),
                 colours=colours, ref=ref,
                 ref_label="chance" if ref == 0.5 else "no skill",
                 xlabel=name, xlim=xlim)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.suptitle(title, x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
    if note:
        fig.text(0.008, 0.005, note, fontsize=8, color=G.INK2, va="bottom")
    fig.tight_layout(rect=[0, 0.05 if note else 0, 1, 0.93])
    return G.finish(fig, figpath(cfg, fname), tight=False)


# =================================================================== F5/F6 main comparison
if have("results_main.csv"):
    R = pd.read_csv(O("results_main.csv"))
    order = ["hybrid", "raw_waveform_cnn", "handcrafted_xgb", "handcrafted_mlp",
             "morphology_only_no_rr", "no_rate_features", "baseline_rr_series_only",
             "baseline_heart_rate_only"]
    order = [m for m in order if (R.model == m).any()]
    labels = [G.MODEL_LABEL.get(m, m) for m in order]
    cols = [G.MODEL_COLOUR.get(m, G.MUTED) for m in order]
    panelled_dotci(R, order, labels, cols,
                   "Figure 5  Three-model comparison and reference baselines on held-out subjects",
                   "F05_model_comparison.png",
                   note="Point = held-out estimate; bar = 95% subject-cluster bootstrap interval "
                        f"({cfg.controls.bootstrap_n} resamples of the 118 test subjects). "
                        "Colour marks the three headline models; reference baselines are shown in muted ink.")

if have("confusions.npz"):
    C = np.load(O("confusions.npz"))
    keys = [k for k in ["handcrafted_xgb", "raw_waveform_cnn", "hybrid"] if k in C]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.3))
    axes = np.atleast_1d(axes)
    short = ["Baseline", "Compensated", "Early decomp.", "Overt shock", "Recovery"]
    for ax, k in zip(axes, keys):
        G.heat(ax, C[k], short, short, vmin=0, vmax=1, fmt="{:.2f}")
        ax.set_title(G.MODEL_LABEL.get(k, k), fontsize=10)
        ax.set_xlabel("Predicted")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        if ax is axes[0]:
            ax.set_ylabel("True state")
    fig.suptitle("Figure 6  Row-normalised confusion matrices (fraction of each true state)",
                 x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    G.finish(fig, figpath(cfg, "F06_confusion.png"), tight=False)

# =================================================================== F7-F10 controls
if have("results_controls.csv"):
    Cc = pd.read_csv(O("results_controls.csv"))
    Rm = pd.read_csv(O("results_main.csv")) if have("results_main.csv") else None

    # ---- F7 negative controls and rate ablation
    sel = Cc[Cc.experiment.isin(["C2_shuffled_ecg", "C2c_moment_matched_noise",
                                 "C3_random_labels", "C3b_fully_random_labels",
                                 "C4_rate_ablation", "C5_rr_vs_morphology"])]
    ref_rows = []
    if Rm is not None:
        for m in ["handcrafted_xgb", "raw_waveform_cnn"]:
            r = Rm[Rm.model == m].copy()
            r["experiment"] = "reference (intact data)"
            ref_rows.append(r)
    sel = pd.concat(ref_rows + [sel], ignore_index=True)
    EXP = {"reference (intact data)": "intact data",
           "C2_shuffled_ecg": "within-window permutation",
           "C2c_moment_matched_noise": "moment-matched noise",
           "C3_random_labels": "block label permutation",
           "C3b_fully_random_labels": "full label permutation",
           "C4_rate_ablation": "rate ablation", "C5_rr_vs_morphology": "rate ablation"}

    MNAME = {"handcrafted_xgb": "Handcrafted", "raw_waveform_cnn": "Raw waveform",
             "handcrafted_xgb_shuffled": "Handcrafted", "raw_waveform_cnn_shuffled": "Raw waveform",
             "handcrafted_xgb_randomlabel": "Handcrafted",
             "raw_waveform_cnn_randomlabel": "Raw waveform",
             "raw_waveform_cnn_rate_removed": "Raw waveform, rate-normalised beats",
             "handcrafted_morphology_only_no_rr": "Handcrafted, morphology only",
             "handcrafted_rr_series_only": "Handcrafted, RR series only",
             "handcrafted_no_rate_features": "Handcrafted, rate features removed",
             "handcrafted_xgb_moment_noise": "Handcrafted",
             "handcrafted_xgb_fully_random_labels": "Handcrafted",
             "raw_waveform_cnn_fully_random_labels": "Raw waveform"}

    def _mname(m):
        return MNAME.get(m, m.replace("_", " "))
    lab = [f"{_mname(r.model)}\n({EXP.get(r.experiment, r.experiment)})"
           for _, r in sel.iterrows()]
    first = sel.experiment.isin(["C2_shuffled_ecg", "C3_random_labels"]).to_numpy()
    redesigned = sel.experiment.isin(["C2c_moment_matched_noise",
                                      "C3b_fully_random_labels"]).to_numpy()
    cols = [G.STATUS["critical"] if f else (G.INK if r else G.CAT[0])
            for f, r in zip(first, redesigned)]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 6.4))
    for ax, (key, name, ref, xlim) in zip(axes, TASKS):
        G.dot_ci(ax, lab, sel[key].to_numpy(float),
                 sel.get(key + "_lo", pd.Series(np.nan, index=sel.index)).to_numpy(float),
                 sel.get(key + "_hi", pd.Series(np.nan, index=sel.index)).to_numpy(float),
                 colours=cols, ref=ref, ref_label="chance" if ref == 0.5 else "no skill",
                 xlabel=name, xlim=(min(xlim[0], 0.38), xlim[1]) if ref == 0.5 else xlim)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    from matplotlib.patches import Patch as _P
    fig.legend(handles=[_P(color=G.STATUS["critical"], label="negative control as first designed"),
                        _P(color=G.INK, label="redesigned negative control"),
                        _P(color=G.CAT[0], label="intact data and rate ablations")],
               loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Figure 7  Negative controls and rate ablations "
                 "(red = a control that must land at chance)",
                 x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    G.finish(fig, figpath(cfg, "F07_controls.png"), tight=False)

    # ---- F8 noise + medication
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.4))
    nz = Cc[Cc.experiment == "C6_noise"]
    for ax, (key, name, ref, xlim) in zip(axes, TASKS):
        for m in ["handcrafted_xgb", "raw_waveform_cnn", "hybrid"]:
            d = nz[nz.model == m].sort_values("noise_multiplier")
            if not len(d):
                continue
            ax.plot(d.noise_multiplier, d[key], "-o", color=G.MODEL_COLOUR[m],
                    label=G.MODEL_LABEL[m], markeredgecolor=G.SURFACE, markeredgewidth=1.2)
        ax.axhline(ref, color=G.STATUS["critical"], lw=1.2, ls=(0, (4, 3)))
        ax.set_xlabel("Noise amplitude multiplier")
        ax.set_ylabel(name.replace("\n", " "), fontsize=8)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3,
                   fontsize=7.5, handlelength=2.8, columnspacing=1.2)
    fig.suptitle("Figure 8  Noise robustness: models fixed, test recordings re-rendered "
                 "at higher artefact amplitude",
                 x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    G.finish(fig, figpath(cfg, "F08_noise.png"), tight=False)

    med = Cc[Cc.experiment == "C7_medication"]
    if len(med):
        fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.6))
        pert = med.perturbation.dropna().unique().tolist()
        xs = np.arange(len(pert))
        for ax, (key, name, ref, xlim) in zip(axes, TASKS):
            for j, m in enumerate(["handcrafted_xgb", "raw_waveform_cnn", "hybrid"]):
                d = med[med.model == m].set_index("perturbation").reindex(pert)
                ax.bar(xs + (j - 1) * 0.26, d[key].to_numpy(float), width=0.24,
                       color=G.MODEL_COLOUR[m], label=G.MODEL_LABEL[m], zorder=2)
            ax.axhline(ref, color=G.STATUS["critical"], lw=1.2, ls=(0, (4, 3)))
            ax.set_xticks(xs)
            PL = {"all_beta_blocked": "all\nbeta-blocked", "none_beta_blocked": "none\nbeta-blocked",
                  "all_vasodilated": "resting\npressure -6%"}
            ax.set_xticklabels([PL.get(p, p.replace("_", " ")) for p in pert], fontsize=7.5)
            ax.set_ylabel(name.replace("\n", " "), fontsize=8)
        axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3,
                   fontsize=7.5, handlelength=2.8, columnspacing=1.2)
        fig.suptitle("Figure 9  Medication and confounder perturbation: the whole cohort "
                     "re-simulated under a different drug regimen",
                     x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        G.finish(fig, figpath(cfg, "F09_medication.png"), tight=False)

    # ---- F10 leave-one-phenotype-out
    lo = Cc[Cc.experiment == "C8_leave_phenotype_out"]
    if len(lo):
        fig, axes = plt.subplots(1, 4, figsize=(15.4, 4.6))
        phs = lo.held_out_phenotype.dropna().unique().tolist()
        xs = np.arange(len(phs))
        for ax, (key, name, ref, xlim) in zip(axes, TASKS):
            for j, base in enumerate(["handcrafted_xgb", "raw_waveform_cnn"]):
                v = [lo[(lo.held_out_phenotype == p) & (lo.model.str.startswith(base))][key].mean()
                     for p in phs]
                ax.bar(xs + (j - 0.5) * 0.32, v, width=0.30, color=G.MODEL_COLOUR[base],
                       label=G.MODEL_LABEL[base], zorder=2)
            if Rm is not None:
                for j, base in enumerate(["handcrafted_xgb", "raw_waveform_cnn"]):
                    ax.axhline(Rm[Rm.model == base][key].values[0], color=G.MODEL_COLOUR[base],
                               lw=1.1, ls=":")
            ax.axhline(ref, color=G.STATUS["critical"], lw=1.2, ls=(0, (4, 3)))
            ax.set_xticks(xs)
            ax.set_xticklabels([p.replace("_", "\n") for p in phs], fontsize=8)
            ax.set_ylabel(name.replace("\n", " "), fontsize=8)
        axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3,
                   fontsize=7.5, handlelength=2.8, columnspacing=1.2)
        fig.suptitle("Figure 10  Leave-one-phenotype-out (dotted line = same model trained on "
                     "all phenotypes)", x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        G.finish(fig, figpath(cfg, "F10_phenotype.png"), tight=False)

    # ---- F11 personal baseline
    pb = Cc[Cc.experiment == "C9_personal_baseline"]
    if len(pb) and Rm is not None:
        rows = pd.concat([Rm[Rm.model == "handcrafted_xgb"].assign(
            model="population model (absolute features)"), pb], ignore_index=True)
        lab = [m.replace("handcrafted_xgb_", "").replace("_", " ") for m in rows.model]
        fig, axes = plt.subplots(1, 4, figsize=(14.6, 3.6))
        for ax, (key, name, ref, xlim) in zip(axes, TASKS):
            G.dot_ci(ax, lab, rows[key].to_numpy(float),
                     rows.get(key + "_lo", pd.Series(np.nan, index=rows.index)).to_numpy(float),
                     rows.get(key + "_hi", pd.Series(np.nan, index=rows.index)).to_numpy(float),
                     colours=[G.CAT[0], G.CAT[1], G.CAT[2]][:len(rows)],
                     ref=ref, xlabel=name, xlim=xlim)
            if ax is not axes[0]:
                ax.set_yticklabels([])
        fig.suptitle("Figure 11  Subject-specific baseline normalisation versus the population model",
                     x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        G.finish(fig, figpath(cfg, "F11_personal_baseline.png"), tight=False)

# =================================================================== F12 SHAP
if have("shap_stage.csv"):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.6),
                             gridspec_kw={"width_ratios": [1.25, 1.25, 1.0]})
    for ax, task in zip(axes[:2], ["stage", "volume"]):
        d = pd.read_csv(O(f"shap_{task}.csv")).head(18)
        G.hbar(ax, d.feature.tolist(), d.mean_abs_shap.to_numpy(), colour=G.CAT[0],
               xlabel=f"mean |SHAP| ({task} model)", value_fmt="{:.3f}", n_label=6)
        ax.set_title(f"Top features, {task} head", fontsize=10)
    fams = []
    for task in ["stage", "volume", "reserve", "decomp"]:
        if have(f"shap_{task}.csv"):
            d = pd.read_csv(O(f"shap_{task}.csv"))
            s = d.groupby("family").mean_abs_shap.sum()
            fams.append((s / s.sum()).rename(task))
    Fm = pd.concat(fams, axis=1).fillna(0)
    order = Fm.mean(1).sort_values(ascending=False).index
    Fm = Fm.loc[order]
    G.heat(axes[2], Fm.to_numpy(), Fm.columns.tolist(), [i.replace("_", " ") for i in Fm.index],
           vmin=0, vmax=float(Fm.to_numpy().max()), fmt="{:.2f}")
    axes[2].set_title("Share of total attribution by feature family", fontsize=10)
    fig.suptitle("Figure 12  SHAP attribution for the handcrafted model",
                 x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    G.finish(fig, figpath(cfg, "F12_shap.png"), tight=False)

# =================================================================== F13 integrated gradients
if have("ig_phase_attribution.csv") and have("ig_examples.npz"):
    ig = pd.read_csv(O("ig_phase_attribution.csv"))
    ex = np.load(O("ig_examples.npz"))
    phases = ["TP_baseline", "P_PR", "QRS", "ST", "T"]
    fig = plt.figure(figsize=(15.0, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.0, 1.0], hspace=0.45, wspace=0.28)
    for r, st in enumerate([0, 3]):
        if f"attr_{st}" not in ex:
            continue
        a = np.abs(ex[f"attr_{st}"]).sum(0)
        wv = ex[f"wave_{st}"][0]
        tt = np.arange(len(wv)) / cfg.simulation.fs
        ax = fig.add_subplot(gs[r, 0])
        lo, hi = float(np.min(wv)), float(np.max(wv))
        span = hi - lo
        # attribution drawn as a band anchored below the trace on the SAME axis --
        # no second value scale, only a normalised annotation layer
        band = lo - 0.30 * span + (a / (a.max() + 1e-12)) * 0.26 * span
        ax.fill_between(tt, lo - 0.30 * span, band, color=G.STAGE_COLOURS[st], alpha=0.55,
                        lw=0, label="|attribution| (normalised, arbitrary scale)")
        ax.plot(tt, wv, color=G.INK, lw=0.8, label="Lead II (mV)")
        ax.set_ylim(lo - 0.34 * span, hi + 0.16 * span)
        ax.set_xlim(0, 6)
        ax.set_title(f"{G.STAGE_LABELS[st]}: waveform and attribution", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    axb = fig.add_subplot(gs[:, 1])
    sh = ig.groupby("stage")[phases].mean()
    G.heat(axb, sh.to_numpy(), phases, [G.STAGE_LABELS[int(i)] for i in sh.index],
           vmin=0, vmax=float(sh.to_numpy().max()), fmt="{:.2f}")
    axb.set_title("Share of total |attribution| by cardiac phase", fontsize=10)
    axc = fig.add_subplot(gs[:, 2])
    dens = ig.groupby("stage")[[p + "_per_sample" for p in phases]].mean()
    dens = dens.div(dens.sum(1), axis=0)
    G.heat(axc, dens.to_numpy(), phases, ["" for _ in dens.index],
           vmin=0, vmax=float(dens.to_numpy().max()), fmt="{:.2f}")
    axc.set_title("Attribution density per sample\n(corrects for segment length)", fontsize=10)
    fig.suptitle("Figure 13  Integrated gradients for the raw-waveform model",
                 x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    G.finish(fig, figpath(cfg, "F13_integrated_gradients.png"), tight=False)

# =================================================================== F14 latent space
if have("latent_space.npz") and have("meta_test.parquet"):
    L = np.load(O("latent_space.npz"))
    mte = pd.read_parquet(O("meta_test.parquet")).reset_index(drop=True)
    probe = pd.read_csv(O("latent_probe.csv")) if have("latent_probe.csv") else None
    fig, axes = plt.subplots(1, 4, figsize=(16.2, 4.3))
    idx = L["tsne_idx"]
    ts = L["tsne"]
    for k in range(5):
        m = mte.stage.to_numpy()[idx] == k
        axes[0].scatter(ts[m, 0], ts[m, 1], s=4, color=G.STAGE_COLOURS[k], alpha=0.65,
                        label=G.STAGE_LABELS[k], linewidths=0)
    axes[0].legend(fontsize=7, markerscale=2.2, loc="best")
    axes[0].set_title("t-SNE of the waveform embedding, by state", fontsize=10)
    sc = axes[1].scatter(ts[:, 0], ts[:, 1], s=4, c=mte.volume_lost.to_numpy()[idx] * 100,
                         cmap=G.CMAP_SEQ, linewidths=0)
    cb = fig.colorbar(sc, ax=axes[1], fraction=0.04)
    cb.set_label("Volume lost (%)", fontsize=8)
    cb.outline.set_visible(False)
    axes[1].set_title("Same embedding, by volume lost", fontsize=10)
    P = L["pca"]
    for k in range(5):
        m = mte.stage.to_numpy() == k
        axes[2].scatter(P[m, 0], P[m, 1], s=3, color=G.STAGE_COLOURS[k], alpha=0.5, linewidths=0)
    axes[2].set_xlabel(f"PC1 ({L['evr'][0]*100:.0f}% var)")
    axes[2].set_ylabel(f"PC2 ({L['evr'][1]*100:.0f}% var)")
    axes[2].set_title("Principal components", fontsize=10)
    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    if probe is not None:
        d = probe.sort_values("r2", ascending=False)
        G.hbar(axes[3], d.latent.tolist(), d.r2.to_numpy(), colour=G.CAT[0],
               xlabel="cross-validated R²", n_label=len(d))
        axes[3].set_title("Linear probe: what the embedding encodes", fontsize=10)
    fig.suptitle("Figure 14  Geometry of the learned latent space (held-out subjects)",
                 x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    G.finish(fig, figpath(cfg, "F14_latent_space.png"), tight=False)

# =================================================================== F15 trajectories + early warning
if have("test_trajectories.parquet"):
    tj = pd.read_parquet(O("test_trajectories.parquet"))
    cand = tj.groupby("subject").apply(
        lambda g: ((g.stage >= 2).sum() > 3) and ((g.stage == 0).sum() > 3), include_groups=False)
    picks = cand[cand].index.tolist()[:3]
    fig, axes = plt.subplots(3, 3, figsize=(14.6, 8.0), sharex=True)
    for r, s in enumerate(picks):
        g = tj[tj.subject == s].sort_values("t_min")
        G.stage_bands(axes[r, 0], g.t_min.to_numpy(), g.stage.to_numpy())
        axes[r, 0].plot(g.t_min, g.volume_lost * 100, color=G.INK, lw=2.0, label="True")
        for tag, c in [("hc", G.CAT[0]), ("cnn", G.CAT[1]), ("hyb", G.CAT[2])]:
            axes[r, 0].plot(g.t_min, g[f"{tag}_volume"] * 100, color=c, lw=1.6, alpha=0.9,
                            label={"hc": "Handcrafted", "cnn": "Raw waveform", "hyb": "Hybrid"}[tag])
        axes[r, 0].set_ylabel(f"subject {int(s)}", fontsize=9, fontweight="bold", color=G.INK)

        G.stage_bands(axes[r, 1], g.t_min.to_numpy(), g.stage.to_numpy())
        axes[r, 1].plot(g.t_min, g.reserve, color=G.INK, lw=2.0, label="True")
        for tag, c in [("hc", G.CAT[0]), ("cnn", G.CAT[1]), ("hyb", G.CAT[2])]:
            axes[r, 1].plot(g.t_min, g[f"{tag}_reserve"], color=c, lw=1.6, alpha=0.9)

        G.stage_bands(axes[r, 2], g.t_min.to_numpy(), g.stage.to_numpy())
        for tag, c in [("hc", G.CAT[0]), ("cnn", G.CAT[1]), ("hyb", G.CAT[2])]:
            axes[r, 2].plot(g.t_min, g[f"{tag}_decomp"], color=c, lw=1.6, alpha=0.9)
        axes[r, 2].set_ylim(0, 1)
        first_bad = g[g.stage >= 2].t_min.min()
        if np.isfinite(first_bad):
            for c in range(3):
                axes[r, c].axvline(first_bad, color=G.STATUS["critical"], lw=1.3, ls="--")
    for c, t_ in enumerate(["Volume lost (%)", "Physiological reserve",
                            "P(decompensation within 10 min)"]):
        axes[0, c].set_title(t_, fontsize=10)
        axes[2, c].set_xlabel("Time (min)")
    axes[0, 0].legend(fontsize=7, loc="upper left", ncol=2)
    fig.text(0.008, 0.012, "Dashed red line marks the first early-decompensation time step. "
             "Background shading = true clinical state.", fontsize=8, color=G.INK2)
    fig.suptitle("Figure 15  Predicted trajectories for three held-out subjects",
                 x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    G.finish(fig, figpath(cfg, "F15_trajectories_pred.png"), tight=False)

# =================================================================== F16 sensitivity + counterfactual
if have("sensitivity.csv") and have("counterfactual.csv"):
    se = pd.read_csv(O("sensitivity.csv"))
    cf = pd.read_csv(O("counterfactual.csv"))
    lats = se.latent.unique().tolist()
    fig, axes = plt.subplots(2, max(len(lats), 2), figsize=(4.0 * len(lats), 7.2))
    for c, lat in enumerate(lats):
        d = se[se.latent == lat]
        ax = axes[0, c]
        for s, g in d.groupby("subject"):
            m = g.groupby("value")["volume"].mean()
            ax.plot(m.index, m.values * 100, color=G.CAT[0], alpha=0.35, lw=1.2)
        m = d.groupby("value")["volume"].mean()
        ax.plot(m.index, m.values * 100, color=G.CAT[0], lw=2.6, label="cohort mean")
        ax.set_title(lat.replace("_", " "), fontsize=10)
        ax.set_ylabel("Predicted volume lost (%)" if c == 0 else "", fontsize=8)
        if c == 0:
            ax.legend(fontsize=7.5)
        ax2 = axes[1, c]
        for s, g in d.groupby("subject"):
            m = g.groupby("value")["decomp"].mean()
            ax2.plot(m.index, m.values, color=G.CAT[1], alpha=0.35, lw=1.2)
        m = d.groupby("value")["decomp"].mean()
        ax2.plot(m.index, m.values, color=G.CAT[1], lw=2.6)
        ax2.set_ylabel("P(decompensation)" if c == 0 else "", fontsize=8)
        ax2.set_xlabel("latent value")
    fig.suptitle("Figure 16  Sensitivity analysis: one latent swept through the observation "
                 "model, everything else held fixed (thin lines = individual subjects)",
                 x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    G.finish(fig, figpath(cfg, "F16_sensitivity.png"), tight=False)

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.2))
    for ax, lat in zip(axes[:2], cf.latent.unique()):
        for m, c in [("raw_waveform_cnn", G.CAT[1]), ("handcrafted_xgb", G.CAT[0])]:
            d = cf[(cf.latent == lat) & (cf.model == m)]
            if not len(d):
                continue
            gmean = d.groupby("value")["volume"].mean() * 100
            gsd = d.groupby("value")["volume"].std() * 100
            ax.plot(gmean.index, gmean.values, color=c, lw=2.2, label=G.MODEL_LABEL[m])
            ax.fill_between(gmean.index, gmean - gsd, gmean + gsd, color=c, alpha=0.18, lw=0)
        ax.set_xlabel(lat.replace("_", " "))
        ax.set_ylabel("Predicted volume lost (%)", fontsize=8)
        ax.set_title(f"Counterfactual: move {lat.replace('_',' ')} only", fontsize=10)
    axes[0].legend(fontsize=7.5)
    d = cf[(cf.latent == "preload") & (cf.model == "raw_waveform_cnn")]
    for k in range(5):
        col = f"p_{phys.STAGES[k]}"
        if col in d:
            m = d.groupby("value")[col].mean()
            axes[2].plot(m.index, m.values, color=G.STAGE_COLOURS[k], lw=2.0,
                         label=G.STAGE_LABELS[k])
    axes[2].set_xlabel("preload")
    axes[2].set_ylabel("Predicted state probability", fontsize=8)
    axes[2].set_title("Same sweep, full state posterior", fontsize=10)
    axes[2].legend(fontsize=7)
    fig.suptitle(f"Figure 17  Counterfactual simulation on one fixed synthetic patient "
                 f"(subject {int(cf.subject.iloc[0])}); shaded band = 1 SD across repeat renders",
                 x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    G.finish(fig, figpath(cfg, "F17_counterfactual.png"), tight=False)

# =================================================================== F18 PTB-XL
if have("ptbxl_summary.csv"):
    ps = pd.read_csv(O("ptbxl_summary.csv")).iloc[0]
    smd = pd.read_csv(O("ptbxl_smd.csv"))
    qc = pd.read_csv(O("ptbxl_qc.csv"))
    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.6))
    steps = ["Loaded", "Beats in range", "Beat consistency", "Not flat", "Amplitude pass",
             "All checks"]
    vals = [len(qc), int(qc.pass_beats.sum()), int(qc.pass_beat_corr.sum()),
            int(qc.pass_flat.sum()), int(qc.pass_amp.sum()), int(qc["pass"].sum())]
    axes[0].bar(range(len(steps)), vals, color=G.CAT[0], zorder=2)
    for i, v in enumerate(vals):
        axes[0].text(i, v, f"{v}\n{100*v/len(qc):.0f}%", ha="center", va="bottom", fontsize=7.5,
                     color=G.INK2)
    axes[0].set_xticks(range(len(steps)))
    axes[0].set_xticklabels(steps, rotation=40, ha="right", rotation_mode="anchor", fontsize=7.5)
    axes[0].set_title("Quality-control funnel, PTB-XL", fontsize=10)
    axes[0].set_ylim(0, max(vals) * 1.16)

    d = smd.reindex(smd.smd.abs().sort_values(ascending=False).index).head(16)[::-1]
    cols = [G.DIVERGING[2] if v > 0 else G.DIVERGING[0] for v in d.smd]
    axes[1].barh(range(len(d)), d.smd, color=cols, zorder=2)
    axes[1].set_yticks(range(len(d)))
    axes[1].set_yticklabels(d.feature, fontsize=7.5)
    axes[1].axvline(0, color=G.AXIS, lw=1.0)
    axes[1].set_xlabel("standardised mean difference (synthetic - real)")
    axes[1].set_title("Largest distribution gaps", fontsize=10)
    axes[1].grid(axis="y", visible=False)

    axes[2].hist(smd.smd, bins=26, color=G.CAT[0], zorder=2)
    axes[2].axvline(0, color=G.AXIS, lw=1.0)
    axes[2].set_xlabel("standardised mean difference")
    axes[2].set_title(f"All {int(np.isfinite(smd.smd).sum())} features with a finite SMD\n"
                      f"median |SMD| = {ps.median_abs_smd:.2f}", fontsize=10)

    bars = [("Handcrafted\nfeature space", ps.domain_auroc_features),
            ("Encoder\nembedding space", ps.domain_auroc_embedding)]
    axes[3].bar([b[0] for b in bars], [b[1] for b in bars], color=G.STATUS["critical"], zorder=2)
    axes[3].axhline(0.5, color=G.STATUS["good"], lw=1.6, ls=(0, (4, 3)))
    axes[3].text(0.02, 0.52, "0.5 = indistinguishable", transform=axes[3].transAxes,
                 fontsize=7.5, color=G.STATUS["good"])
    for i, b in enumerate(bars):
        axes[3].text(i, b[1], f"{b[1]:.3f}", ha="center", va="bottom", fontsize=8, color=G.INK2)
    axes[3].set_ylim(0.4, 1.05)
    axes[3].set_ylabel("real-vs-synthetic AUROC", fontsize=8)
    axes[3].set_title("Domain discriminability", fontsize=10)
    fig.suptitle("Figure 18  Public-data arm: real-ECG quality control and how far the "
                 "simulation sits from PTB-XL",
                 x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    G.finish(fig, figpath(cfg, "F18_ptbxl.png"), tight=False)

# =================================================================== F19 time-to-event
if have("tte_results.csv"):
    tt = pd.read_csv(O("tte_results.csv"))
    cal = pd.read_csv(O("tte_calibration.csv")) if have("tte_calibration.csv") else None
    lt = pd.read_csv(O("lead_time.csv")) if have("lead_time.csv") else None
    main3 = [m for m in ["handcrafted_xgb", "raw_waveform_cnn", "hybrid"] if (tt.model == m).any()]
    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.3))

    d = tt[tt.model.isin(main3)].set_index("model").reindex(main3)
    axes[0].bar(range(len(d)), d.c_index, color=[G.MODEL_COLOUR[m] for m in main3], zorder=2)
    for i, v in enumerate(d.c_index):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=G.INK2)
    axes[0].axhline(0.5, color=G.STATUS["critical"], lw=1.3, ls=(0, (4, 3)))
    axes[0].set_xticks(range(len(d)))
    axes[0].set_xticklabels([G.MODEL_LABEL[m].replace(" ", "\n") for m in main3], fontsize=8)
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_ylabel("Harrell's C", fontsize=8)
    axes[0].set_title("Concordance with time to decompensation", fontsize=10)

    hz = [c for c in tt.columns if c.startswith("auroc_")]
    hzv = [float(c.split("_")[1].replace("min", "")) for c in hz]
    for m in main3:
        r = tt[tt.model == m].iloc[0]
        axes[1].plot(hzv, [r[c] for c in hz], "-o", color=G.MODEL_COLOUR[m],
                     label=G.MODEL_LABEL[m], markeredgecolor=G.SURFACE, markeredgewidth=1.2)
    axes[1].axhline(0.5, color=G.STATUS["critical"], lw=1.2, ls=(0, (4, 3)))
    axes[1].set_xlabel("Prediction horizon (min)")
    axes[1].set_ylabel("AUROC", fontsize=8)
    axes[1].set_title("Time-dependent discrimination", fontsize=10)
    axes[1].legend(fontsize=7.5, loc="upper right")

    if cal is not None:
        axes[2].plot([0, 1], [0, 1], color=G.MUTED, lw=1.2, ls="--", label="perfect calibration")
        axes[2].plot(cal.mean_predicted, cal.observed_rate, "-o", color=G.CAT[0],
                     markeredgecolor=G.SURFACE, markeredgewidth=1.2,
                     label=G.MODEL_LABEL.get(cal.model.iloc[0], cal.model.iloc[0]))
        axes[2].set_xlabel("Mean predicted risk (decile)")
        axes[2].set_ylabel("Observed rate", fontsize=8)
        axes[2].set_title("Calibration by risk decile", fontsize=10)
        axes[2].legend(fontsize=7.5)

    if lt is not None and lt.lead_over_map.notna().any():
        axes[3].hist(lt.lead_over_map.dropna(), bins=22, color=G.CAT[0], zorder=2)
        med = lt.lead_over_map.median()
        axes[3].axvline(med, color=G.STATUS["critical"], lw=1.6)
        axes[3].text(med, axes[3].get_ylim()[1] * 0.95, f"  median {med:.1f} min",
                     color=G.STATUS["critical"], fontsize=8, va="top")
        axes[3].set_xlabel("Minutes from reserve exhaustion to a 10% fall in mean pressure")
        axes[3].set_ylabel("Subjects", fontsize=8)
        axes[3].set_title("Size of the window being predicted", fontsize=10)
    fig.suptitle("Figure 19  Time-to-event evaluation of the decompensation head",
                 x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    G.finish(fig, figpath(cfg, "F19_time_to_event.png"), tight=False)

# =================================================================== F20 matched heart rate
if have("matched_hr_main.csv"):
    mh = pd.read_csv(O("matched_hr_main.csv"))
    mte = pd.read_parquet(O("meta_test.parquet"))
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    data = [mte[mte.stage == k].hr_true.to_numpy() for k in range(5)]
    ypos = np.arange(5)[::-1]
    bp = axes[0].boxplot(data, vert=False, positions=ypos, widths=0.62, showfliers=False,
                         patch_artist=True, medianprops=dict(color=G.SURFACE, lw=1.8),
                         whiskerprops=dict(color=G.MUTED, lw=1.2),
                         capprops=dict(color=G.MUTED, lw=1.2))
    for b, c in zip(bp["boxes"], G.STAGE_COLOURS):
        b.set_facecolor(c)
        b.set_alpha(0.85)
        b.set_edgecolor(G.SURFACE)
    for lo, hi in sorted({(r.hr_lo, r.hr_hi) for _, r in mh.iterrows()}):
        for v in (lo, hi):
            axes[0].axvline(v, color=G.INK2, lw=1.0, ls=":", zorder=1)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(G.STAGE_LABELS, fontsize=8)
    axes[0].set_xlabel("Heart rate (bpm)")
    axes[0].set_title("Heart rate by state, and the matched bands", fontsize=10)
    axes[0].grid(False)                       # band edges must not hide behind grid lines

    show = [m for m in ["handcrafted_xgb", "raw_waveform_cnn", "hybrid",
                        "baseline_heart_rate_only"] if (mh.model == m).any()]
    for j, (key, name, ref) in enumerate([("stage_auroc_macro", "State macro AUROC", 0.5),
                                          ("volume_r2", "Blood-volume loss R²", 0.0)]):
        ax = axes[1 + j]
        for m_ in show:
            d = mh[mh.model == m_].sort_values("hr_lo")
            x = (d.hr_lo + d.hr_hi) / 2
            hr_only = m_ == "baseline_heart_rate_only"      # red, as in Figure 3A
            ax.plot(x, d[key], marker="o", ls="-",
                    color=G.STATUS["critical"] if hr_only else G.MODEL_COLOUR.get(m_, G.MUTED),
                    label=G.MODEL_LABEL.get(m_, m_), markeredgecolor=G.SURFACE,
                    markeredgewidth=1.2)
        ax.axhline(ref, color=G.STATUS["critical"], lw=1.3, ls=(0, (4, 3)))
        ax.set_xlabel("Heart-rate band centre (bpm)")
        ax.set_ylabel(name, fontsize=8)
        ax.set_title(name + " within a band", fontsize=10)
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Figure 20  Matching on heart rate: a rate-only model collapses, the full "
                 "models do not",
                 x=0.008, ha="left", fontsize=11, fontweight="bold", color=G.INK)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    G.finish(fig, figpath(cfg, "F20_matched_heart_rate.png"), tight=False)

print("result figures written")
