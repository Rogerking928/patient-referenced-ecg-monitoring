"""Figure construction.

Colour assignment follows the reference data-viz palette without modification:
  * three-model comparisons use categorical slots 1-3 (validated all-pairs),
  * clinical stage is ordinal, so it uses the single-hue blue ramp starting at
    step 250 (the lightest step that still clears 2:1 on a light surface),
    with recovery -- which is not on the severity axis -- given a categorical
    hue and always direct-labelled,
  * magnitude heatmaps use the sequential blue ramp,
  * signed quantities use the diverging blue-red pair with a neutral midpoint,
  * "should be at chance" reference marks use the fixed status palette.
Identity is never carried by colour alone: every panel is legended or direct-labelled.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------ palette
SURFACE = "#ffffff"   # pure white: the journal asked for figures without a background
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
STATUS = dict(good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b")
DIVERGING = ("#2a78d6", "#f0efec", "#d03b3b")

# ordinal severity ramp (steps 250/350/450/650) + off-axis hue for recovery
STAGE_COLOURS = ["#86b6ef", "#5598e7", "#2a78d6", "#104281", "#1baf7a"]
STAGE_LABELS = ["Baseline", "Compensated", "Early decompensation", "Overt shock", "Recovery"]

CMAP_SEQ = LinearSegmentedColormap.from_list("seqblue", SEQ)
CMAP_DIV = LinearSegmentedColormap.from_list("divbr", [DIVERGING[0], DIVERGING[1], DIVERGING[2]])

MODEL_COLOUR = {
    "handcrafted_xgb": CAT[0], "handcrafted": CAT[0],
    "raw_waveform_cnn": CAT[1], "raw_waveform": CAT[1],
    "hybrid": CAT[2],
}
MODEL_LABEL = {
    "handcrafted_xgb": "Handcrafted features",
    "raw_waveform_cnn": "Raw waveform",
    "hybrid": "Hybrid",
    "handcrafted_mlp": "Handcrafted (MLP)",
    "baseline_heart_rate_only": "Heart rate only",
    "baseline_rr_series_only": "RR series only",
    "morphology_only_no_rr": "Morphology only",
    "no_rate_features": "Rate features removed",
}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2.0, "lines.markersize": 6,
    })


#: Width at which the main figures are printed, in inches.  The main figures were
#: drawn on a 16.4 in canvas and placed at about 7 in, which left 7.5 pt labels at
#: roughly 3 pt on the page.  They are now resized to the printed width and every
#: label is rescaled so that nothing on the page is smaller than MIN_PT.
PRINT_W = 9.0
FONT_GAIN = 1.55
MIN_PT = 6.5
MAX_H = 6.9


def _to_print_size(fig):
    import matplotlib.text as mtext
    w, h = fig.get_size_inches()
    s = PRINT_W / w
    if fig._suptitle is not None:                 # the caption carries the title
        fig._suptitle.set_visible(False)
    for t in fig.findobj(mtext.Text):
        t.set_fontsize(max(t.get_fontsize() * s * FONT_GAIN, MIN_PT))
    for ln in fig.findobj(plt.Line2D):
        ln.set_markersize(ln.get_markersize() * max(s * 1.6, 0.6))
        ln.set_linewidth(max(ln.get_linewidth() * s * 1.6, 0.6))
    fig.set_size_inches(PRINT_W, h * s)
    # Labels that hang outside the canvas would widen the tight bounding box past
    # PRINT_W.  The canvas is narrowed by the overhang (fonts stay as set) until
    # the saved figure, labels included, is PRINT_W wide and at most MAX_H tall.
    for _ in range(4):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        cw, chh = fig.get_size_inches()
        nw = cw - (bb.width - (PRINT_W - 0.08))      # 0.08 = savefig pad on both sides
        nh = chh - max(bb.height - MAX_H, 0)
        if abs(nw - cw) < 0.01 and abs(nh - chh) < 0.01:
            break
        fig.set_size_inches(nw, nh)
    report_overlaps(fig)


def report_overlaps(fig, tol_pt=0.6):
    """Print every pair of visible labels whose boxes intersect on the page."""
    import matplotlib.text as mtext
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = fig.dpi / 72.0
    # Tick labels: matplotlib keeps Text objects for ticks that fall outside the
    # view interval and never draws them, but they still report a window extent.
    # Only the ticks the axis actually draws (Axis._update_ticks) are checked;
    # every other tick label is excluded, everything else is checked as found.
    drawn, all_tick = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                all_tick.update(map(id, (tk.label1, tk.label2)))
            if not ax.get_visible() or not axis.get_visible():
                continue
            for tk in axis._update_ticks():
                for lb in (tk.label1, tk.label2):
                    if lb.get_visible():
                        drawn.add(id(lb))
    ts = [t for t in fig.findobj(mtext.Text)
          if t.get_visible() and t.get_text().strip() and t.figure is fig
          and (id(t) not in all_tick or id(t) in drawn)]
    bbs = []
    for t in ts:
        try:
            bbs.append(t.get_window_extent(r))
        except Exception:                                      # noqa: BLE001
            bbs.append(None)
    n = 0
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            a, b = bbs[i], bbs[j]
            if a is None or b is None:
                continue
            if ts[i].get_rotation() % 90 and ts[j].get_rotation() % 90:
                continue          # axis-aligned boxes of slanted labels overlap; glyphs do not
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > tol_pt * k and oy > tol_pt * k:
                n += 1
                print(f"   OVERLAP {ts[i].get_text()[:30]!r} x {ts[j].get_text()[:30]!r}"
                      f"  at ({a.x0/fig.dpi:.2f},{a.y0/fig.dpi:.2f}) in")
    print(f"   overlap check: {n} overlapping label pairs")


def finish(fig, path, tight=True, dpi=700, vector=False):
    if fig._suptitle is not None:              # the caption carries the title and number
        fig._suptitle.set_visible(False)
    """Write the figure.

    dpi is deliberately modest by default: these figures are embedded by the
    dozen into the supplementary document, and a higher setting there buys
    nothing but file size.  The four main figures pass dpi and vector=True,
    because a journal sets a resolution floor for bitmapped line drawings and
    accepts a vector file instead of meeting it.
    """
    if tight:
        fig.tight_layout()
    if vector:
        _to_print_size(fig)
    if not vector:
        report_overlaps(fig)
    kw = dict(bbox_inches="tight", pad_inches=0.04)   # nothing drawn outside the canvas is lost
    fig.savefig(path, dpi=dpi, **kw)
    if vector:
        from pathlib import Path as _P
        fig.savefig(str(_P(path).with_suffix(".pdf")), **kw)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ primitives
def dot_ci(ax, labels, est, lo=None, hi=None, colours=None, ref=None, ref_label=None,
           xlabel="", xlim=None, annotate=True, fmt="{:.3f}"):
    """Estimate-with-interval plot: the right form when uncertainty is the point."""
    y = np.arange(len(labels))[::-1]
    colours = colours or [CAT[0]] * len(labels)
    if ref is not None:
        ax.axvline(ref, color=STATUS["critical"], lw=1.4, ls=(0, (4, 3)), zorder=1)
        if ref_label:
            ax.text(ref, len(labels) - 0.35, "  " + ref_label, color=STATUS["critical"],
                    fontsize=7.5, va="top", ha="left")
    for i, yy in enumerate(y):
        if lo is not None and np.isfinite(lo[i]) and np.isfinite(hi[i]):
            ax.plot([lo[i], hi[i]], [yy, yy], color=colours[i], lw=2.0, solid_capstyle="round",
                    alpha=0.55, zorder=2)
        ax.plot([est[i]], [yy], "o", color=colours[i], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
        if annotate:
            ax.text(est[i], yy + 0.30, fmt.format(est[i]), color=INK2, fontsize=7.5,
                    ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_ylim(-0.8, len(labels) - 0.2)
    return ax


def heat(ax, Mtx, xlabels, ylabels, cmap=CMAP_SEQ, vmin=0, vmax=1, fmt="{:.2f}",
         cbar_label=None, fig=None):
    im = ax.imshow(Mtx, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.grid(False)
    thr = vmin + 0.55 * (vmax - vmin)
    for i in range(Mtx.shape[0]):
        for j in range(Mtx.shape[1]):
            if np.isfinite(Mtx[i, j]):
                ax.text(j, i, fmt.format(Mtx[i, j]), ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if Mtx[i, j] > thr else INK)
    if fig is not None and cbar_label:
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(cbar_label, fontsize=8, color=INK2)
        cb.outline.set_visible(False)
    return im


def hbar(ax, labels, values, colour=CAT[0], xlabel="", value_fmt="{:.3f}", n_label=8):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colour, height=0.62, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    span = max(values) - min(0, min(values)) + 1e-12
    for i, (yy, v) in enumerate(zip(y, values)):
        if i < n_label:
            # negative values are labelled to the left of the bar end, clear of the axis
            ax.text(v + (0.015 if v >= 0 else -0.015) * span, yy, value_fmt.format(v),
                    va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=INK2)
    lo = min(values)
    ax.set_xlim(min(0, lo - 0.30 * span) if lo < 0 else 0, max(values) * 1.22)
    return ax


def stage_bands(ax, t, stage, alpha=0.24, ymin=0, ymax=1):
    """Shade a time axis by clinical stage; labelled via the caller's legend."""
    t = np.asarray(t)
    s = np.asarray(stage)
    edges = np.where(np.diff(s) != 0)[0]
    starts = np.r_[0, edges + 1]
    ends = np.r_[edges, len(s) - 1]
    for a, b in zip(starts, ends):
        ax.axvspan(t[a], t[b], color=STAGE_COLOURS[int(s[a])], alpha=alpha, lw=0, zorder=0)


def stage_legend(ax, present=None, **kw):
    from matplotlib.patches import Patch
    ks = range(len(STAGE_LABELS)) if present is None else present
    # same transparency as the shading it explains, so the key matches what is drawn
    h = [Patch(facecolor=STAGE_COLOURS[k], alpha=0.24,
               label=STAGE_LABELS[k]) for k in ks]
    ax.legend(handles=h, **kw)
