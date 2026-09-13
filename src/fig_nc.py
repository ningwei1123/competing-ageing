"""
================================================================================
 LFP-ETGNet  |  Module 8c: Nature-Communications composite figures
================================================================================
 Collapses the 19 single-panel figures + 6 tables into 7 main figures and 3
 tables (10 display items), the Nature Communications limit.  Three of the
 seven main figures are genuinely new composites (Fig 1, Fig 5, Fig 6); the
 other four reuse the already-validated law figures and the graph-abstraction
 figure unchanged.  Everything else is copied verbatim into the Supplementary
 figure set, so no information is discarded.

   Fig 1   two-channel degradation mechanism (RVE + two-phase + decomposition
           + coupling chain)            [new composite: f01b + f02c + f03c + f03d]
   Fig 2   degradation channels & partition law   [reuse fig21]
   Fig 3   scaling laws                   [reuse fig22]
   Fig 4   percolation law                [reuse fig23]
   Fig 5   predictive performance & shift [new: f11a + f12R2 + f14a + f19c]
   Fig 6   interpretability & validation  [new: f15b + f16a + f17c + sev2]
   Fig 7   graph abstraction of the electrode  [reuse fig04]

   Supplementary Figs 1-10: equivalent models, framework, architecture,
   multi-task scheme, pseudocode, efficiency, case study, data efficiency,
   Severson distribution gap, Severson early fade.
================================================================================
"""
from __future__ import annotations
import os, sys, json, shutil, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Patch
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_science as FS          # reconstruct_electrode / build_graph / ...
import fig_results as FR          # PAL, ORDER, TASKS, TLAB, _load
import fig_severson as SV         # load_split, qd_series

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figs")
DATA = os.path.join(ROOT, "data")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.9, "savefig.dpi": 600, "figure.dpi": 110,
    "axes.titlesize": 8.0, "axes.labelsize": 8.0,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "mathtext.fontset": "dejavusans",
})

# --- palette (mirrors fig_science / fig_laws) -----------------------------
C_AM   = FS.C_AM
C_CBD  = FS.C_CBD
C_RED  = FS.C_RED
C_BLUE = FS.C_BLUE
C_GREY = FS.C_GREY
C_FP   = FS.C_FP

TITLE = dict(loc="left", fontweight="bold", fontsize=7.8)


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", name, flush=True)
    return p


# ============================================================================
#  Fig. 1  two-channel degradation mechanism  (composite)
# ============================================================================
def fig1_mechanisms(seed=11):
    rng = np.random.default_rng(seed)
    micro = FS.reconstruct_electrode(rng)
    graph = FS.build_graph(micro, rng=rng)
    proto = FS.ref_proto(c_rate=2.0, T_celsius=35.0, dod=1.0, n_eol=1500)
    sim = FS.simulate_aging(graph, proto, rng, n_check=25)
    rec = sim["record"]

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)

    # ---- (a) reconstructed electrode RVE ---------------------------------
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("(a) Electrode RVE (zoom)", **TITLE)
    frame = Rectangle((0.2, 0.2), 9.6, 9.6, fc=FS.C_PORE, ec="#95A5A6", lw=0.7)
    ax.add_patch(frame)
    c = micro["centres"][:, :2]; r = micro["radii"]; box = micro["box"]
    f = 0.36
    lo = box * (0.5 - f / 2); hi = box * (0.5 + f / 2); span = hi - lo
    s = 9.6 / span
    cb_c = micro["cbd_centres"]; cb_r = micro["cbd_radii"]
    if len(cb_c):
        m = ((cb_c[:, 0] > lo - 0.2) & (cb_c[:, 0] < hi + 0.2) &
             (cb_c[:, 1] > lo - 0.2) & (cb_c[:, 1] < hi + 0.2))
        if m.any():
            sc = ax.scatter((cb_c[m, 0] - lo) * s + 0.2, (cb_c[m, 1] - lo) * s + 0.2,
                            s=cb_r[m] * s * 6, c=FS.C_CBD, alpha=0.85, lw=0)
            sc.set_clip_path(frame)
    o = np.argsort(-r)
    for i in o:
        circ = Circle(((c[i, 0] - lo) * s + 0.2, (c[i, 1] - lo) * s + 0.2), r[i] * s,
                      fc=FS.C_AM, ec="#24455F", lw=0.45, alpha=0.95)
        ax.add_patch(circ); circ.set_clip_path(frame)
    ax.text(0.5, 9.45, r"$\phi_{AM}$ = 0.60   $\phi_{pore}$ = 0.30",
            fontsize=6.0, va="top", color="#2C3E50")
    ax.text(0.5, 0.45, "carbon-black / binder network", fontsize=5.8, color="#2C3E50")
    iax = ax.inset_axes([0.755, 0.735, 0.225, 0.225])
    iax.add_patch(Rectangle((0, 0), box, box, fc=FS.C_PORE, ec="#95A5A6", lw=0.4))
    iax.scatter(c[:, 0], c[:, 1], s=0.35, c=FS.C_AM, lw=0)
    iax.add_patch(Rectangle((lo, lo), span, span, fc="none", ec=C_RED, lw=0.9))
    iax.set_xlim(0, box); iax.set_ylim(0, box); iax.set_xticks([]); iax.set_yticks([])

    # ---- (b) two-phase reaction ------------------------------------------
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.set_title("(b) Two-phase reaction", **TITLE)
    ax.add_patch(Rectangle((1.0, 3.4), 3.0, 2.6, fc=C_AM, ec="#24455F", lw=0.8))
    ax.add_patch(Rectangle((4.6, 3.6), 2.7, 2.2, fc=C_FP, ec="#8A4B24", lw=0.8))
    ax.text(2.5, 4.7, "LiFePO$_4$", ha="center", fontsize=6.8, color="white", fontweight="bold")
    ax.text(5.95, 4.7, "FePO$_4$", ha="center", fontsize=6.8, color="white", fontweight="bold")
    ax.plot([4.0, 4.6], [4.7, 4.7], color="#2C3E50", lw=1.2)
    ax.annotate("charge", xy=(4.3, 5.3), xytext=(3.0, 7.0), fontsize=6.4, color=C_RED,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=C_RED))
    ax.annotate("discharge", xy=(4.3, 4.1), xytext=(3.0, 1.7), fontsize=6.4,
                color=C_BLUE, arrowprops=dict(arrowstyle="->", lw=0.9, color=C_BLUE))
    ax.text(4.3, 2.85, "misfit strain\n$\\Delta V/V$ = 6.8 %", ha="center", fontsize=6.2)
    ax.set_xlim(0.4, 8.2); ax.set_ylim(1.0, 7.9)

    # ---- (c) capacity-fade decomposition ---------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("(c) Fade decomposition", **TITLE)
    q_lli = 100 * (proto["k_lli_ref"] * np.sqrt(rec["n"]) + proto["k_lin_ref"] * rec["n"])
    lam = 100 * rec["LAM"]
    ax.stackplot(rec["n"], q_lli, lam, labels=["LLI", "LAM"],
                 colors=[C_BLUE, C_RED], alpha=0.90)
    ax.axhline(20, color="k", ls="--", lw=1.0)
    ax.text(int(0.97 * rec["n"][-1]), 20.7, "EOL threshold (80 % SOH)",
            fontsize=5.6, ha="right")
    ax.set_xlabel("cycle number"); ax.set_ylabel("capacity loss (%)")
    ax.set_xlim(0, int(rec["n"][-1])); ax.set_ylim(0, 23)
    ax3 = ax.twinx()
    ax3.spines["right"].set_visible(True)
    ax3.plot(rec["n"], 100 * rec["CL"], color=C_CBD, lw=1.4)
    ax3.set_ylabel("contact loss (%)", color=C_CBD, fontsize=6.4)
    ax3.tick_params(axis="y", colors=C_CBD, labelsize=6.4)
    ax3.set_ylim(0, 100); ax3.set_xlim(0, int(rec["n"][-1]))
    ax.legend(handles=[Patch(fc=C_BLUE, alpha=0.9, label="LLI"),
                       Patch(fc=C_RED, alpha=0.9, label="LAM"),
                       Line2D([], [], color=C_CBD, lw=1.4, label="CL (resistance)")],
              frameon=False, loc="upper left", fontsize=5.8)

    # ---- (d) mechanism coupling chain ------------------------------------
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    ax.set_title("(d) Coupling chain", **TITLE)
    boxes = [("Breathing strain\n$\\Delta V/V$ = 6.8 %", 0.04, 0.74, C_AM),
             ("Contact-neck fatigue\n& carbon delamination", 0.04, 0.46, C_CBD),
             ("Loss of electronic\npercolation", 0.04, 0.18, C_RED)]
    for txt, x, y, cc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.92, 0.20, boxstyle="round,pad=0.015",
                                    fc=cc, ec="none", alpha=0.90, transform=ax.transAxes))
        ax.text(x + 0.46, y + 0.10, txt, ha="center", va="center", fontsize=5.8,
                color="white", transform=ax.transAxes)
    for y1, y2 in [(0.74, 0.66), (0.46, 0.38)]:
        ax.annotate("", xy=(0.50, y2), xytext=(0.50, y1), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#34495E"))
    ax.add_patch(FancyBboxPatch((0.04, 0.00), 0.92, 0.145, boxstyle="round,pad=0.015",
                                fc="#8E44AD", ec="none", alpha=0.90, transform=ax.transAxes))
    ax.text(0.50, 0.072, "Isolated particles → LAM", ha="center", va="center",
            fontsize=5.8, color="white", transform=ax.transAxes)
    ax.annotate("", xy=(0.50, 0.00), xytext=(0.50, 0.175), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#34495E",
                                connectionstyle="angle3,angleA=0,angleB=-90"))
    ax.text(0.50, -0.09, "+ SEI growth consumes cyclable Li (LLI)",
            ha="center", fontsize=5.6, color="#1A5276", transform=ax.transAxes)

    _save(fig, "fig01_mechanisms.png")


# ============================================================================
#  Fig. 5  predictive performance & distribution shift  (composite)
# ============================================================================
def fig5_performance():
    M = FR._load()
    bundle = pickle.load(open(os.path.join(RES, "bundle.pkl"), "rb"))
    present = [n for n in FR.ORDER if n in M]

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.32)

    # ---- (a) predicted vs measured (RUL) ---------------------------------
    ax = fig.add_subplot(gs[0, 0])
    pr, tr = bundle["pred"], bundle["true"]
    y, p = tr[:, 0], pr[:, 0]
    ax.scatter(y, p, s=13, c="#2C7FB8", alpha=0.65, edgecolor="none")
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9)
    ss = 1 - np.sum((y - p) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
    ax.text(0.05, 0.90, f"$R^2$ = {ss:.3f}", transform=ax.transAxes, fontsize=6.6, va="top")
    ax.set_xlabel("measured RUL (cycles)"); ax.set_ylabel("predicted RUL")
    ax.set_title("(a) Remaining useful life", **TITLE)

    # ---- (b) macro R2 across models --------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    vals = [max(M[n]["metrics"]["_macro"]["R2"], 0) for n in present]
    errs = [M[n]["metrics"]["_macro_std"]["R2"] for n in present]
    bars = ax.bar(range(len(present)), vals, yerr=errs, capsize=2.0,
                  color=[FR.PAL[n] for n in present], edgecolor="white", lw=0.5)
    bars[0].set_edgecolor("#7B241C"); bars[0].set_linewidth(1.4)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=38, ha="right", fontsize=6.2)
    ax.set_ylabel(r"macro $R^2$"); ax.set_ylim(0, 1.12)
    ax.set_title("(b) In-distribution accuracy", **TITLE)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=5.6)

    # ---- (c) branch ablation ---------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    names = [n for n in ["LFP-ETGNet", "GNN-only", "Transformer-only"] if n in M]
    r2 = [max(M[n]["metrics"]["_macro"]["R2"], 0) for n in names]
    b = ax.bar(range(len(names)), r2, color=[FR.PAL[n] for n in names],
               edgecolor="white", lw=0.6)
    b[0].set_edgecolor("#7B241C"); b[0].set_linewidth(1.4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=28, ha="right", fontsize=6.2)
    ax.set_ylabel(r"macro $R^2$"); ax.set_ylim(0, 1.06)
    ax.set_title("(c) Branch ablation", **TITLE)
    for i, v in enumerate(r2):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=6.0)

    # ---- (d) generalisation gap ------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    O = FR._load_ood()
    keys = [k for k in ("temp", "life") if k in O]
    names_present = [n for n in present if all(n in O[k] for k in keys)]
    w = 0.38; x = np.arange(len(names_present))
    for j, k in enumerate(keys):
        g = [M[n]["metrics"]["_macro"]["R2"] - O[k][n]["metrics"]["_macro"]["R2"]
             for n in names_present]
        ax.bar(x + (j - 0.5 * (len(keys) - 1)) * w, g, w,
               color=["#C0392B", "#2E86AB"][j], edgecolor="white", lw=0.5,
               label={"temp": "temperature shift", "life": "life extrapolation"}[k])
    ax.axhline(0, color="#34495E", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names_present, rotation=38, ha="right", fontsize=6.2)
    ax.set_ylabel(r"$\Delta R^2$ (random $-$ shifted)")
    ax.set_title("(d) Generalisation gap", **TITLE)
    ax.legend(fontsize=6.0, loc="upper left")

    _save(fig, "fig05_performance.png")


# ============================================================================
#  Fig. 6  interpretability & external validation  (composite)
# ============================================================================
def fig6_interpret():
    I = np.load(os.path.join(RES, "interp.npz"), allow_pickle=True)
    cal = json.load(open(os.path.join(RES, "calibration.json")))
    bundle = pickle.load(open(os.path.join(RES, "bundle.pkl"), "rb"))
    cells = pickle.load(open(os.path.join(DATA, "lfp_cells.pkl"), "rb"))

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.32)

    # ---- (a) permutation importance of node descriptors ------------------
    ax = fig.add_subplot(gs[0, 0])
    node_imp = I["node_imp"]
    NODE = ["particle diameter", "inter-particle spacing", "SEI thickness",
            "Li-conc. gradient", "coordination", "contact quality",
            "electrode depth", "local SOC"]
    o = np.argsort(node_imp)
    ax.barh(np.arange(8), node_imp[o], color="#2E86AB", edgecolor="white", lw=0.5)
    ax.set_yticks(np.arange(8)); ax.set_yticklabels([NODE[i] for i in o], fontsize=6.2)
    ax.set_xlabel("relative permutation importance")
    ax.set_title("(a) Node descriptors", **TITLE)

    # ---- (b) uncertainty calibration (RUL) -------------------------------
    ax = fig.add_subplot(gs[0, 1])
    pr, tr = bundle["pred"], bundle["true"]
    y, p = tr[:, 0], pr[:, 0]
    o = np.argsort(y)
    ax.plot(y[o], y[o], "k--", lw=0.8, label="ideal")
    ax.plot(y[o], p[o], color="#C0392B", lw=1.1, label="prediction")
    s = cal["RUL"]["sigma_mean"]
    ax.fill_between(y[o], p[o] - 2 * s, p[o] + 2 * s, color="#C0392B",
                    alpha=0.18, label=r"$\pm 2\sigma$")
    cov = cal["RUL"]["coverage_2s"] * 100
    ax.set_xlabel("measured RUL (cycles)"); ax.set_ylabel("predicted RUL")
    ax.set_title(f"(b) Uncertainty ($\\pm2\\sigma$: {cov:.0f}%)", **TITLE)
    ax.legend(fontsize=5.8, loc="upper left")

    # ---- (c) physical attribution at EOL --------------------------------
    ax = fig.add_subplot(gs[1, 0])
    cr = np.array([c["proto"]["c_rate"] for c in cells])
    bins = [cr < 1.6, (cr >= 1.6) & (cr < 2.6), cr >= 2.6]
    labs = ["<1.6C", "1.6–2.6C", ">2.6C"]
    lli, lam, cl = [], [], []
    for b in bins:
        cs = [c for c, bb in zip(cells, b) if bb]
        lli.append(np.mean([c["curve"]["LLI"][-1] for c in cs]) * 100)
        lam.append(np.mean([c["curve"]["LAM"][-1] for c in cs]) * 100)
        cl.append(np.mean([c["curve"]["CL"][-1] for c in cs]) * 100)
    x = np.arange(3)
    ax.bar(x, lli, 0.55, color="#2C7FB8", label="LLI", edgecolor="white", lw=0.5)
    ax.bar(x, lam, 0.55, bottom=lli, color="#C0392B", label="LAM", edgecolor="white", lw=0.5)
    ax.axhline(20, color="k", ls="--", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.4)
    ax.set_ylabel("capacity loss at EOL (%)")
    ax.set_ylim(0, max(23.0, 1.20 * (np.array(lli) + np.array(lam)).max()))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, cl, "D--", ms=3.8, lw=1.0, color="#2E2E2E")
    ax2.set_ylabel("contact loss (%)", fontsize=6.2, color="#2E2E2E")
    ax2.tick_params(axis="y", labelsize=6.0, colors="#2E2E2E")
    ax2.set_ylim(0, max(30.0, 1.25 * max(cl)))
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color="#2E2E2E", marker="D", ls="--", ms=3.8))
    l.append("CL (resistance)")
    ax.legend(h, l, fontsize=5.6, loc="upper left")
    ax.set_title("(c) Attribution vs. C-rate", **TITLE)

    # ---- (d) Severson reproduction ---------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    Xtr, ytr = SV.load_split("train")
    Xt1, yt1 = SV.load_split("test1"); keep = np.ones(len(yt1), bool); keep[21] = False
    Xt1, yt1 = Xt1[keep], yt1[keep]
    def feat(X):
        dq = X[:, :, 98] - X[:, :, 8]
        return np.log10(np.var(dq, axis=1))
    ftr, ft1 = feat(Xtr), feat(Xt1)
    A = np.stack([np.ones_like(ftr), ftr], 1)
    w, *_ = np.linalg.lstsq(A, np.log10(ytr), rcond=None)
    def pred(f): return 10 ** (w[0] + w[1] * f)
    p1, y1 = pred(ft1), yt1
    ax.scatter(p1, y1, s=12, color="#1f77b4", alpha=0.7, edgecolor="none")
    ax.plot([100, 2400], [100, 2400], "k--", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(100, 2400); ax.set_ylim(100, 2400)
    rmse = float(np.sqrt(np.mean((y1 - p1) ** 2)))
    ax.set_xlabel("predicted cycle life"); ax.set_ylabel("measured cycle life")
    ax.set_title(f"(d) Severson 2019 test1 (RMSE {rmse:.0f} cyc)", **TITLE)

    _save(fig, "fig06_interpret.png")


# ============================================================================
def _copy(src, dst):
    s = os.path.join(FIG, src); d = os.path.join(FIG, dst)
    if os.path.exists(s):
        shutil.copyfile(s, d)
        print("  copy", src, "->", dst, flush=True)
    else:
        print("  !! MISSING", src, flush=True)


def main():
    print("generating Nature-Communications composite figures ...")
    fig1_mechanisms()
    fig5_performance()
    fig6_interpret()

    # --- reuse the validated law figures and the graph abstraction --------
    _copy("fig21_law_channels.png", "fig02_channels.png")
    _copy("fig22_law_scaling.png", "fig03_scaling.png")
    _copy("fig23_law_percolation.png", "fig04_percolation.png")
    _copy("fig04_graph_abstraction.png", "fig07_methods.png")

    # --- supplementary figures --------------------------------------------
    _copy("fig05_equivalent_models.png", "sup_fig01_equivalent.png")
    _copy("fig06_framework.png", "sup_fig02_framework.png")
    _copy("fig07_architecture.png", "sup_fig03_architecture.png")
    _copy("fig08_multitask.png", "sup_fig04_multitask.png")
    _copy("fig09_pseudocode.png", "sup_fig05_pseudocode.png")
    _copy("fig13_efficiency.png", "sup_fig06_efficiency.png")
    _copy("fig18_case_study.png", "sup_fig07_case.png")
    _copy("fig20_data_efficiency.png", "sup_fig08_dataeff.png")
    _copy("fig_sev1_distribution_gap.png", "sup_fig09_sev1.png")
    _copy("fig_sev3_early_fade.png", "sup_fig10_sev3.png")
    print("done.")


if __name__ == "__main__":
    main()
