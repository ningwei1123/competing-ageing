"""
================================================================================
 LFP-ETGNet  |  Module 5a: Scientific illustrations
      multiscale microstructure, crystal structure, degradation mechanisms,
      graph abstraction and equivalent surrogate models
================================================================================
"""
from __future__ import annotations
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyArrowPatch, Rectangle,
                                FancyBboxPatch, Polygon, Patch)
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfp_micro import P, reconstruct_electrode, build_graph, simulate_aging, _percolating_nodes
from lfp_dataset import (K_LLI_REF, K_LIN_REF, K_CL_BASE, EA_EFF, R_GAS,
                         T_REF)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figs")
os.makedirs(FIG, exist_ok=True)


def ref_proto(c_rate, T_celsius, dod, n_eol):
    """A protocol whose rate coefficients are the calibrated reference values
    scaled by the same Arrhenius and power-law factors that
    :func:`lfp_dataset._protocol` applies.

    The illustration scripts used to carry their own hand-written coefficient
    values, which is how Figure 3c came to be drawn from a structural-loss
    parameter that the aging model no longer contains.  Deriving the
    coefficients from the calibrated constants keeps the figures and the
    simulator in step.
    """
    arr = float(np.exp(-EA_EFF / R_GAS
                       * (1.0 / (T_celsius + 273.15) - 1.0 / T_REF)))
    s = arr * c_rate ** 0.60
    return dict(cell_id=0, c_rate=c_rate, T_celsius=T_celsius, soc_lo=0.0,
                soc_hi=dod, dod=dod,
                k_lli_ref=K_LLI_REF * s * dod ** 0.85,
                k_lin_ref=K_LIN_REF * s * dod ** 0.85,
                k_cl_ref=K_CL_BASE * s * dod ** 1.45,
                Q0=1.0, dt_rise=0.9, n_eol=n_eol)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.9, "savefig.dpi": 600, "figure.dpi": 110,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "mathtext.fontset": "dejavusans",
})

# --- journal palette --------------------------------------------------------
C_AM   = "#3A6EA5"     # active material
C_AM2  = "#5F9BD6"
C_FP   = "#C97B4A"     # FePO4 (delithiated)
C_CBD  = "#2E2E2E"     # carbon black
C_SEI  = "#E8B84B"     # SEI
C_PORE = "#EDF1F5"     # pore/electrolyte
C_LI   = "#7FB069"     # lithium
C_FE   = "#8C4A2F"     # iron
C_PO4  = "#7E57C2"     # phosphate
C_RED  = "#C0392B"
C_BLUE = "#2C7FB8"
C_GREY = "#7F8C8D"


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", name, flush=True)
    return p


# ============================================================================
#  Fig. 1  Multiscale architecture of an automotive LFP cathode
# ============================================================================
def fig_multiscale(seed=7):
    rng = np.random.default_rng(seed)
    micro = reconstruct_electrode(rng)
    fig = plt.figure(figsize=(7.2, 5.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.30)
    TITLE = dict(loc="left", fontweight="bold", fontsize=7.8)

    # ---- (a) cell / electrode stack schematic ---------------------------
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("(a) Prismatic LFP cell", **TITLE)
    ax.add_patch(FancyBboxPatch((0.3, 0.7), 8.0, 8.6, boxstyle="round,pad=0.12",
                                fc="#F7FAFC", ec="#2C3E50", lw=1.1))
    layers = [("#3A6EA5", "LFP cathode"), ("#D5DBDB", "Separator"),
              ("#7F8C8D", "Graphite"), ("#C0392B", "Al foil"),
              ("#B7950B", "Cu foil")]
    y = 8.5
    for c, lab in layers:
        h = 0.70 if lab != "Separator" else 0.34
        ax.add_patch(Rectangle((1.05, y - h), 4.35, h, fc=c, ec="#2C3E50", lw=0.5))
        ax.text(5.60, y - h / 2, lab, va="center", ha="left", fontsize=5.9)
        y -= h + 0.26
    ax.annotate("", xy=(0.72, 8.85), xytext=(0.72, 1.15),
                arrowprops=dict(arrowstyle="<->", color="#34495E", lw=0.8))
    ax.text(0.52, 5.0, "thickness\n~150 µm", rotation=90, va="center", ha="center",
            fontsize=5.9, color="#34495E")

    # ---- (b) electrode cross-section from the reconstruction -------------
    ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("(b) Electrode RVE (zoom)", **TITLE)
    frame = Rectangle((0.2, 0.2), 9.6, 9.6, fc=C_PORE, ec="#95A5A6", lw=0.7)
    ax.add_patch(frame)
    c = micro["centres"][:, :2]; r = micro["radii"]
    box = micro["box"]
    f = 0.36                                    # zoomed fraction of the RVE
    lo = box * (0.5 - f / 2); hi = box * (0.5 + f / 2); span = hi - lo
    s = 9.6 / span
    cb_c = micro["cbd_centres"]; cb_r = micro["cbd_radii"]
    if len(cb_c):
        m = ((cb_c[:, 0] > lo - 0.2) & (cb_c[:, 0] < hi + 0.2) &
             (cb_c[:, 1] > lo - 0.2) & (cb_c[:, 1] < hi + 0.2))
        if m.any():
            sc = ax.scatter((cb_c[m, 0] - lo) * s + 0.2, (cb_c[m, 1] - lo) * s + 0.2,
                            s=cb_r[m] * s * 6, c=C_CBD, alpha=0.85, lw=0)
            sc.set_clip_path(frame)
    o = np.argsort(-r)
    for i in o:
        circ = Circle(((c[i, 0] - lo) * s + 0.2, (c[i, 1] - lo) * s + 0.2), r[i] * s,
                      fc=C_AM, ec="#24455F", lw=0.45, alpha=0.95)
        ax.add_patch(circ)
        circ.set_clip_path(frame)
    ax.text(0.5, 9.45, f"$\\phi_{{AM}}$ = 0.60   $\\phi_{{pore}}$ = 0.30",
            fontsize=6.0, va="top", color="#2C3E50")
    ax.text(0.5, 0.45, "carbon-black / binder network", fontsize=5.8, color="#2C3E50")
    # -- locator inset: full RVE with the zoom window marked -----------------
    iax = ax.inset_axes([0.755, 0.735, 0.225, 0.225])
    iax.add_patch(Rectangle((0, 0), box, box, fc=C_PORE, ec="#95A5A6", lw=0.4))
    iax.scatter(c[:, 0], c[:, 1], s=0.35, c=C_AM, lw=0)
    iax.add_patch(Rectangle((lo, lo), span, span, fc="none", ec=C_RED, lw=0.9))
    iax.set_xlim(0, box); iax.set_ylim(0, box)
    iax.set_xticks([]); iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_linewidth(0.4)

    # ---- (c) SEM-like particle packing ----------------------------------
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    ax.set_title("(c) SEM view", **TITLE)
    _sem_render(ax, micro, n_show=130, seed=seed)

    # ---- (d) single particle: carbon coating + SEI (TEM-like) ------------
    ax = fig.add_subplot(gs[1, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("(d) Carbon coating & SEI", **TITLE)
    ax.add_patch(Circle((4.2, 4.6), 3.62, fc=C_SEI, ec="none"))
    ax.add_patch(Circle((4.2, 4.6), 3.28, fc=C_CBD, ec="none"))
    ax.add_patch(Circle((4.2, 4.6), 3.00, fc=C_AM, ec="#24455F", lw=0.6))
    ax.annotate("carbon coating\n2–5 nm", xy=(4.2 + 3.14, 4.6 + 0.6), xytext=(7.9, 8.9),
                fontsize=5.9, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.annotate("SEI / CEI\n5 → 60 nm", xy=(4.2 + 3.5, 4.6 - 0.9), xytext=(7.9, 2.0),
                fontsize=5.9, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.text(4.2, 4.6, "d$_{50}$\n200 nm", ha="center", va="center",
            fontsize=6.4, color="white")

    # ---- (e) particle-size distribution ----------------------------------
    ax = fig.add_subplot(gs[1, 1])
    d = micro["dia"]
    ax.hist(d, bins=60, color=C_AM, alpha=0.85, edgecolor="white", lw=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("particle diameter (µm)"); ax.set_ylabel("count")
    ax.set_title("(e) Bimodal particle size", **TITLE)
    yl = ax.get_ylim()[1] * 0.94
    ax.axvline(0.20, color=C_RED, ls="--", lw=0.9)
    ax.axvline(0.90, color=C_RED, ls="--", lw=0.9)
    ax.text(0.215, yl, "primary\n200 nm", fontsize=5.8, color=C_RED, va="top")
    ax.text(0.98, yl, "secondary\n0.9 µm", fontsize=5.8, color=C_RED, va="top")

    # ---- (f) key material properties ------------------------------------
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_title("(f) Olivine LFP properties", **TITLE)
    rows = [("Crystal system", "orthorhombic (Pnma)"),
            ("Lattice (Å)", "6.008/10.334/4.693"),
            ("Two-phase", "LiFePO$_4$ ⇄ FePO$_4$"),
            ("Volume change", "6.8 % (2.27 % lin.)"),
            ("$\\sigma_e$", "$\\sim 10^{-9}$ S cm$^{-1}$"),
            ("$D_{Li}$", "$\\sim 10^{-14}$ cm$^{2}$ s$^{-1}$"),
            ("P–O bond", "585 kJ mol$^{-1}$"),
            ("Thermal limit", "$\\gtrsim$ 270 °C"),
            ("Fade @25 °C, 1C", "0.25 % / 100 cyc"),
            ("To 80 % SOH", "~4500 (0.3C: 10000)")]
    y = 9.05
    for k, v in rows:
        ax.text(0.30, y, k, fontsize=5.2, color="#7F8C8D")
        ax.text(9.80, y, v, fontsize=5.3, fontweight="bold",
                color="#1A5276", ha="right")
        y -= 0.955
    return _save(fig, "fig01_multiscale_microstructure.png")


def _sem_render(ax, micro, n_show=140, seed=0):
    """Render a pseudo-SEM view of the packing with greyscale shading."""
    rng = np.random.default_rng(seed)
    box = micro["box"]; r = micro["radii"]; c = micro["centres"]
    o = np.argsort(-r)[:n_show]
    W = 400
    img = np.ones((W, W, 3)) * 0.92
    yy, xx = np.mgrid[0:W, 0:W] / W * box
    # z-order: draw big-to-small with radial shading -> pseudo 3-D relief
    for i in o:
        d2 = (xx - c[i, 0]) ** 2 + (yy - c[i, 1]) ** 2
        rr = r[i] * (1.0 + rng.uniform(-0.02, 0.02))
        m = d2 <= rr ** 2
        if not m.any():
            continue
        dist = np.sqrt(d2) / rr
        light = 0.42 + 0.52 * np.sqrt(np.clip(1 - dist ** 2, 0, 1)) \
                + 0.10 * (xx - c[i, 0]) / rr
        g = np.clip(light, 0, 1)
        img[m] = np.stack([g[m] * 0.94, g[m] * 0.96, g[m]], 1)
    # metallic/textural grain
    img += rng.normal(0, 0.018, img.shape)
    img = np.clip(img, 0, 1)
    ax.imshow(img, extent=[0, 10, 0, 10], origin="lower")
    ax.axis("off")
    # carbon-black speckle
    cb = micro["cbd_centres"]
    if len(cb):
        k = rng.choice(len(cb), size=min(140, len(cb)), replace=False)
        ax.scatter(cb[k, 0] / box * 10, cb[k, 1] / box * 10, s=1.1, c="#1a1a1a", lw=0)
    ax.text(0.3, 9.55, "1 µm", fontsize=6.0, color="#2C3E50")
    ax.plot([0.3, 2.55], [9.25, 9.25], color="#2C3E50", lw=1.0)


# ============================================================================
#  Fig. 2  Olivine crystal structure and the two-phase reaction
# ============================================================================
def fig_crystal(seed=3):
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 4, wspace=0.30)
    T = dict(loc="left", fontweight="bold", fontsize=7.0)

    # ---- (a) 2-D projection of the olivine framework --------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("(a) Olivine framework", **T)
    # FeO6 octahedra on a distorted hexagonal net + PO4 tetrahedra
    fc, pc = [], []
    for i in range(3):
        for j in range(3):
            x = i * 3.1 + (j % 2) * 1.55
            y = j * 2.6
            fc.append(Polygon(_hex(x, y, 1.05), closed=True))
            a = 2 * np.pi * rng.uniform(0, 1)
            pc.append(Polygon(_tetra(x + 1.5, y + 1.3, 0.42, a), closed=True))
    ax.add_collection(PatchCollection(fc, fc=C_FE, ec="#4A2418", lw=0.6, alpha=0.85))
    ax.add_collection(PatchCollection(pc, fc=C_PO4, ec="#4527A0", lw=0.6, alpha=0.90))
    ax.scatter(rng.uniform(0, 10, 26), rng.uniform(0.4, 7.9, 26), s=13, c=C_LI,
               edgecolor="#3E6B32", lw=0.4, zorder=5)
    ax.set_xlim(-0.7, 10.3); ax.set_ylim(-1.0, 8.6); ax.axis("off")
    h = [Line2D([], [], marker="h", ls="", mfc=C_FE, mec="#4A2418", ms=7, label="FeO$_6$"),
         Line2D([], [], marker="^", ls="", mfc=C_PO4, mec="#4527A0", ms=7, label="PO$_4$"),
         Line2D([], [], marker="o", ls="", mfc=C_LI, mec="#3E6B32", ms=6, label="Li")]
    ax.legend(handles=h, loc="lower center", frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.92, fontsize=6.2, ncol=3,
              handletextpad=0.2, columnspacing=0.7)

    # ---- (b) unit-cell parameters --------------------------------------
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.set_title("(b) Unit cell (Pnma)", **T)
    ax.annotate("", xy=(1.5, 1.4), xytext=(6.6, 1.4),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#1A5276"))
    ax.annotate("", xy=(1.5, 1.4), xytext=(1.5, 6.2),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#1A5276"))
    ax.text(4.05, 1.0, "a = 6.008 Å", ha="center", fontsize=6.6, color="#1A5276")
    ax.text(1.75, 3.45, "b = 10.334 Å", va="center", fontsize=6.6, color="#1A5276")
    ax.text(7.4, 4.0, "c = 4.693 Å\n(out of plane)", fontsize=6.2, color=C_GREY)
    ax.add_patch(Rectangle((1.5, 1.4), 5.1, 4.8, fc="none", ec="#7F8C8D",
                           ls="--", lw=0.7))
    ax.annotate("1-D Li$^+$ channel ∥ b", xy=(3.0, 5.7), xytext=(2.2, 7.0),
                fontsize=6.2, color=C_LI,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=C_LI))
    ax.set_xlim(0, 9.4); ax.set_ylim(0.4, 7.8)

    # ---- (c) two-phase reaction -----------------------------------------
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    ax.set_title("(c) Two-phase reaction", **T)
    ax.add_patch(Rectangle((1.0, 3.4), 3.0, 2.6, fc=C_AM, ec="#24455F", lw=0.8))
    ax.add_patch(Rectangle((4.6, 3.6), 2.7, 2.2, fc=C_FP, ec="#8A4B24", lw=0.8))
    ax.text(2.5, 4.7, "LiFePO$_4$", ha="center", fontsize=6.8, color="white",
            fontweight="bold")
    ax.text(5.95, 4.7, "FePO$_4$", ha="center", fontsize=6.8, color="white",
            fontweight="bold")
    ax.plot([4.0, 4.6], [4.7, 4.7], color="#2C3E50", lw=1.2)
    ax.annotate("charge", xy=(4.3, 5.3), xytext=(3.0, 7.0), fontsize=6.4, color=C_RED,
                arrowprops=dict(arrowstyle="->", lw=0.9, color=C_RED))
    ax.annotate("discharge", xy=(4.3, 4.1), xytext=(3.0, 1.7), fontsize=6.4,
                color=C_BLUE, arrowprops=dict(arrowstyle="->", lw=0.9, color=C_BLUE))
    ax.text(4.3, 2.85, "misfit strain\n$\\Delta V/V$ = 6.8 %", ha="center", fontsize=6.2)
    ax.set_xlim(0.4, 8.2); ax.set_ylim(1.0, 7.9)

    # ---- (d) voltage / current profile -----------------------------------
    ax = fig.add_subplot(gs[0, 3])
    ax.set_title("(d) Voltage hysteresis", **T)
    q = np.linspace(0, 1, 400)
    vc = 3.28 + 0.42 * (q ** 0.55) + 0.06 * np.tanh((q - 0.5) * 8)
    vd = 3.30 - 0.40 * (q ** 0.60) + 0.05 * np.tanh((0.5 - q) * 8) + 0.055
    ax.plot(q, vc, color=C_BLUE, lw=1.4, label="charge")
    ax.plot(q, vd, color=C_RED, lw=1.4, label="discharge")
    ax.fill_between(q, vd, vc, color=C_GREY, alpha=0.20)
    ax.annotate("hysteresis\n$\\lesssim$ 60 mV", xy=(0.5, 3.42), xytext=(0.10, 3.62),
                fontsize=6.0, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("normalised capacity"); ax.set_ylabel("voltage vs Li/Li$^+$ (V)")
    ax.set_ylim(2.95, 3.72); ax.legend(frameon=False, loc="lower right")
    return _save(fig, "fig02_crystal_and_two_phase.png")


def _hex(x, y, r):
    return [(x + r * np.cos(a), y + r * np.sin(a))
            for a in np.linspace(0, 2 * np.pi, 7)]


def _tetra(x, y, r, ang):
    return [(x + r * np.cos(ang + 2 * np.pi * k / 3),
             y + r * np.sin(ang + 2 * np.pi * k / 3)) for k in range(3)]


# ============================================================================
#  Fig. 3  Coupled degradation mechanisms
# ============================================================================
def fig_mechanisms(seed=11):
    rng = np.random.default_rng(seed)
    micro = reconstruct_electrode(rng)
    graph = build_graph(micro, rng=rng)
    proto = ref_proto(c_rate=2.0, T_celsius=35.0, dod=1.0, n_eol=1500)
    sim = simulate_aging(graph, proto, rng, n_check=25)
    rec = sim["record"]
    T = dict(loc="left", fontweight="bold", fontsize=7.2)

    fig = plt.figure(figsize=(7.2, 4.9))
    outer = fig.add_gridspec(2, 1, hspace=0.46, height_ratios=[1.0, 1.10])
    top = outer[0].subgridspec(1, 3, wspace=0.62)
    bot = outer[1].subgridspec(1, 4, wspace=0.24)

    # ---- (a) LLI: SEI growth --------------------------------------------
    ax = fig.add_subplot(top[0, 0])
    ax.set_title("(a) LLI — SEI growth ($\\sqrt{t}$)", **T)
    n = np.arange(1, 2001)
    for Tc, c in [(25, C_BLUE), (35, "#F39C12"), (45, C_RED)]:
        arr = np.exp(-EA_EFF / R_GAS * (1 / (Tc + 273.15) - 1 / 298.15))
        ax.plot(n, 5 + 1.35 * np.sqrt(n) * arr, color=c, lw=1.3, label=f"{Tc} °C")
    ax.set_xlabel("cycle number"); ax.set_ylabel("SEI thickness (nm)")
    ax.set_ylim(0, 70); ax.set_xlim(0, 2000)
    ax.legend(frameon=False, loc="lower right", fontsize=6.2)
    ax.axhline(50, color=C_GREY, ls=":", lw=0.8)
    ax.text(1020, 51.5, "aged commercial LFP", fontsize=5.6, color=C_GREY)

    # ---- (b) LAM: isolation by percolation loss -------------------------
    ax = fig.add_subplot(top[0, 1])
    ax.set_title("(b) LAM — percolation loss", **T)
    ax.plot(rec["n"], 100 * rec["LAM"], color=C_RED, lw=1.5)
    ax.set_xlabel("cycle number")
    ax.set_ylabel("isolated AM (%)", color=C_RED)
    ax.tick_params(axis="y", colors=C_RED)
    ax.set_ylim(0, max(8.0, 1.25 * 100 * rec["LAM"].max()))
    ax.set_xlim(0, int(rec["n"][-1]))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(rec["n"], 100 * rec["edge_ret"], color=C_BLUE, lw=1.5)
    ax2.set_ylabel("intact contacts (%)", color=C_BLUE)
    ax2.tick_params(axis="y", colors=C_BLUE)
    ax2.set_ylim(0, 105)
    ax2.set_xlim(0, int(rec["n"][-1]))
    h1 = Line2D([], [], color=C_RED, lw=1.5, label="isolated AM")
    h2 = Line2D([], [], color=C_BLUE, lw=1.5, label="intact contacts")
    ax.legend(handles=[h1, h2], frameon=False, loc="center right", fontsize=6.0)

    # ---- (c) capacity-fade decomposition --------------------------------
    #   Only the interphase channel and the isolated active material remove
    #   capacity; a lost electronic contact does not, it raises the series
    #   resistance.  CL is therefore drawn on its own axis instead of being
    #   stacked into the capacity ledger.
    ax = fig.add_subplot(top[0, 2])
    ax.set_title("(c) Fade decomposition", **T)
    q_lli = 100 * (proto["k_lli_ref"] * np.sqrt(rec["n"])
                   + proto["k_lin_ref"] * rec["n"])
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
    ax3.set_ylabel("electronic contact loss (%)", color=C_CBD, fontsize=6.4)
    ax3.tick_params(axis="y", colors=C_CBD, labelsize=6.4)
    ax3.set_ylim(0, 100); ax3.set_xlim(0, int(rec["n"][-1]))
    ax.legend(handles=[Patch(fc=C_BLUE, alpha=0.9, label="LLI"),
                       Patch(fc=C_RED, alpha=0.9, label="LAM"),
                       Line2D([], [], color=C_CBD, lw=1.4,
                              label="CL (resistance side)")],
              frameon=False, loc="upper left", fontsize=5.8)

    # ---- (d) mechanism chain schematic ----------------------------------
    ax = fig.add_subplot(bot[0, 0]); ax.axis("off")
    ax.set_title("(d) Coupling chain", **T)
    boxes = [("Breathing strain\n$\\Delta V/V$ = 6.8 %", 0.04, 0.74, C_AM),
             ("Contact-neck fatigue\n& carbon delamination", 0.04, 0.46, C_CBD),
             ("Loss of electronic\npercolation", 0.04, 0.18, C_RED)]
    for txt, x, y, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.92, 0.20, boxstyle="round,pad=0.015",
                                    fc=c, ec="none", alpha=0.90,
                                    transform=ax.transAxes))
        ax.text(x + 0.46, y + 0.10, txt, ha="center", va="center", fontsize=5.8,
                color="white", transform=ax.transAxes)
    for y1, y2 in [(0.74, 0.66), (0.46, 0.38)]:
        ax.annotate("", xy=(0.50, y2), xytext=(0.50, y1), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#34495E"))
    ax.add_patch(FancyBboxPatch((0.04, 0.00), 0.92, 0.145, boxstyle="round,pad=0.015",
                                fc="#8E44AD", ec="none", alpha=0.90,
                                transform=ax.transAxes))
    ax.text(0.50, 0.072, "Isolated particles → LAM", ha="center", va="center",
            fontsize=5.8, color="white", transform=ax.transAxes)
    ax.annotate("", xy=(0.50, 0.00), xytext=(0.50, 0.175), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#34495E",
                                connectionstyle="angle3,angleA=0,angleB=-90"))
    ax.text(0.50, -0.09, "+ SEI growth consumes cyclable Li (LLI)",
            ha="center", fontsize=5.6, color="#1A5276", transform=ax.transAxes)

    # ---- (e) topology evolution of one and the same cell ----------------
    #   The simulator stops when SOH reaches 80 %, so the realised end cycle is
    #   reported instead of the requested sweep limit.
    for k, n_age in enumerate([900, 1200, 1500]):
        ax = fig.add_subplot(bot[0, 1 + k])
        tp = dict(proto); tp["n_eol"] = n_age
        s2 = simulate_aging(graph, tp, np.random.default_rng(seed + k), n_check=25)
        _draw_graph(ax, graph, s2, k / 2.0)
        n_real = int(s2["record"]["n"][-1])
        ax.set_title(f"({chr(101+k)}) n = {n_real}\n" +
                     f"intact {100*s2['alive'].mean():.0f}% · "
                     f"isolated {100*(1-s2['percolating'][:graph['n_am']].mean()):.0f}%",
                     loc="center", fontsize=5.8, fontweight="bold")
    return _save(fig, "fig03_degradation_mechanisms.png")


def _draw_graph(ax, graph, sim, frac, title=None, w=140.0):
    c = graph["centres"][:, :2]; box = graph["box"]
    e = graph["edges"]; alive = sim["alive"]
    seg = np.stack([c[e[:, 0]], c[e[:, 1]]], 1)
    ax.add_collection(LineCollection(seg[alive], colors="#95A5A6", lw=0.20,
                                     alpha=0.75))
    if (~alive).any():
        ax.add_collection(LineCollection(seg[~alive], colors=C_RED, lw=0.22,
                                         alpha=0.22, linestyles="dashed"))
    perc = sim["percolating"]
    n_am = graph["n_am"]
    am = np.arange(n_am)
    dead = am[~perc[:n_am]]
    ax.scatter(c[am, 0], c[am, 1], s=_s(graph["radii"][am], box, w), c="#4C6E91",
               lw=0, alpha=0.9)
    if dead.size:
        ax.scatter(c[dead, 0], c[dead, 1], s=_s(graph["radii"][dead], box, w),
                   c=C_RED, lw=0, alpha=0.95)
    ax.set_xlim(0, box); ax.set_ylim(0, box); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title or f"edge retention {100*alive.mean():.0f}% · "
                          f"isolated {100*(1-perc[:n_am].mean()):.0f}%",
                 loc="center", fontsize=6.4, fontweight="bold")


def _s(radii, box, w=140.0):
    """Scatter marker area (pt^2) for a sphere of radius ``radii`` (µm) that
    should look physically proportional on an axes ``w`` points wide."""
    return (2.0 * np.asarray(radii) / float(box) * w) ** 2


# ============================================================================
#  Fig. 4  Graph abstraction of the electrode
# ============================================================================
def fig_graph_abstraction(seed=5):
    rng = np.random.default_rng(seed)
    micro = reconstruct_electrode(rng)
    graph = build_graph(micro, rng=rng)
    proto = ref_proto(c_rate=3.0, T_celsius=35.0, dod=1.0, n_eol=1200)
    sim = simulate_aging(graph, proto, rng, n_check=25)

    fig = plt.figure(figsize=(7.2, 4.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.42, 1.0], hspace=0.30, wspace=0.34)

    # ---- (a) image -> graph ---------------------------------------------
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("(a) Image → graph  G = (V, E)",
                 loc="left", fontweight="bold", fontsize=7.8)
    _sem_inset(ax, micro, [0.005, 0.02, 0.230, 0.94])
    ax.annotate("", xy=(0.355, 0.5), xytext=(0.255, 0.5), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#34495E"))
    ax.text(0.305, 0.62, "segmentation\n+ graph build", ha="center", fontsize=6.0,
            transform=ax.transAxes, color="#34495E")
    gax = ax.inset_axes([0.395, 0.02, 0.595, 0.94])
    _draw_graph(gax, graph, sim, 0.2, w=250.0,
                title=f"|V| = {graph['n_node']}   |E| = {int(sim['alive'].sum())}   "
                      f"mean degree = {2*sim['alive'].sum()/graph['n_node']:.2f}")

    # ---- (b) node attribute maps ---------------------------------------
    defs = [("particle diameter", graph["dia"]), ("coordination number", None),
            ("Li-concentration gradient", sim["grad_li"])]
    for k, (name, val) in enumerate(defs):
        ax = fig.add_subplot(gs[1, k]); ax.set_aspect("equal"); ax.axis("off")
        n_am = graph["n_am"]
        if val is None:
            deg = np.zeros(graph["n_node"])
            np.add.at(deg, graph["edges"][sim["alive"]][:, 0], 1)
            np.add.at(deg, graph["edges"][sim["alive"]][:, 1], 1)
            val = deg[:n_am]
        else:
            val = np.asarray(val)[:n_am] / 1e7
        short = {"particle diameter": "particle diameter",
                 "coordination number": "coordination",
                 "Li-concentration gradient": "Li gradient ($\\times10^{7}$)"}[name]
        ax.set_title(f"({'bcd'[k]}) {short}", loc="left", fontsize=7.4,
                     fontweight="bold")
        sc = ax.scatter(graph["centres"][:n_am, 0], graph["centres"][:n_am, 1],
                        c=val, s=_s(graph["radii"][:n_am], graph["box"], 118.0),
                        cmap="magma_r" if k == 0 else "YlGnBu", lw=0)
        cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=5.6)
        ax.set_xlim(0, graph["box"]); ax.set_ylim(0, graph["box"])
    return _save(fig, "fig04_graph_abstraction.png")


def _sem_inset(fig_or_ax, micro, rect):
    ax = fig_or_ax.inset_axes(rect)
    _sem_render(ax, micro, n_show=110, seed=2)


# ============================================================================
#  Fig. 5  Equivalent surrogate models
# ============================================================================
def fig_equivalent_models():
    fig = plt.figure(figsize=(7.2, 3.1))
    gs = fig.add_gridspec(1, 3, wspace=0.34)
    T = dict(loc="left", fontweight="bold", fontsize=7.2)

    # ---- (a) 1st-order RC + hysteresis ECM ------------------------------
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("(a) Equivalent circuit (ECM)", **T)
    wt = 0.46                                   # main wire height
    ax.plot([0.05, 0.35], [wt, wt], color="#2C3E50", lw=1.2)
    ax.plot([0.70, 0.95], [wt, wt], color="#2C3E50", lw=1.2)
    ax.add_patch(Rectangle((0.14, wt - 0.08), 0.11, 0.16, fc="white",
                           ec="#2C3E50", lw=1.1))
    ax.text(0.195, wt + 0.11, "$R_0$", ha="center", fontsize=6.4)
    # parallel R1 || C1
    for xv in (0.35, 0.70):
        ax.plot([xv, xv], [0.20, 0.74], color="#2C3E50", lw=1.1)
    ax.plot([0.35, 0.47], [0.74, 0.74], color="#2C3E50", lw=1.1)
    ax.plot([0.59, 0.70], [0.74, 0.74], color="#2C3E50", lw=1.1)
    ax.add_patch(Rectangle((0.47, 0.695), 0.12, 0.09, fc="white",
                           ec="#2C3E50", lw=1.1))
    ax.text(0.53, 0.84, "$R_1$", ha="center", fontsize=6.4)
    ax.plot([0.35, 0.475], [0.20, 0.20], color="#2C3E50", lw=1.1)
    ax.plot([0.545, 0.70], [0.20, 0.20], color="#2C3E50", lw=1.1)
    for xp in (0.475, 0.545):
        ax.plot([xp, xp], [0.11, 0.29], color="#2C3E50", lw=1.5)
    ax.text(0.575, 0.20, "$C_1$", ha="left", va="center", fontsize=6.4,
            bbox=dict(fc="white", ec="none", pad=0.6))
    ax.add_patch(Circle((0.82, wt), 0.058, fc="white", ec="#2C3E50", lw=1.1))
    ax.text(0.82, wt - 0.135, "OCV", ha="center", va="top", fontsize=5.6)
    ax.text(0.82, wt + 0.11, "$h$", ha="center", fontsize=6.4, color=C_BLUE)
    ax.text(0.50, 0.00, "µs–ms per step · real-time BMS\nno internal states",
            ha="center", va="bottom", fontsize=5.5, color=C_GREY)

    # ---- (b) single-particle model --------------------------------------
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("(b) Single-particle model", **T)
    for x, lab in [(0.27, "cathode"), (0.73, "anode")]:
        ax.add_patch(Circle((x, 0.58), 0.135, fc=C_AM if x < .5 else "#7F8C8D",
                            ec="#24455F", lw=0.8, alpha=0.92))
        for rr in (0.045, 0.090):
            ax.add_patch(Circle((x, 0.58), rr, fc="none", ec="white", lw=0.6,
                                ls=":"))
        ax.text(x, 0.40, lab, ha="center", fontsize=6.2)
    ax.annotate("", xy=(0.60, 0.58), xytext=(0.40, 0.58),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=C_LI))
    ax.text(0.50, 0.65, "Li$^+$", ha="center", fontsize=6.4, color=C_LI)
    ax.text(0.50, 0.90, "$\\partial c/\\partial t = \\frac{D_s}{r^2}"
                        "\\frac{\\partial}{\\partial r}(r^2\\frac{\\partial c}"
                        "{\\partial r})$", ha="center", fontsize=6.6)
    ax.text(0.50, 0.00, "s–min per step · 1-D radial diffusion\n"
                        "SPM / SPMe · accurate to ~2C",
            ha="center", va="bottom", fontsize=5.5, color=C_GREY)

    # ---- (c) DFN / P2D --------------------------------------------------
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("(c) P2D / DFN model", **T)
    ax.add_patch(Rectangle((0.04, 0.40), 0.30, 0.30, fc=C_AM, alpha=0.88))
    ax.add_patch(Rectangle((0.36, 0.40), 0.12, 0.30, fc="#D5DBDB", ec="#7F8C8D",
                           lw=0.6))
    ax.add_patch(Rectangle((0.50, 0.40), 0.30, 0.30, fc="#7F8C8D", alpha=0.88))
    ax.text(0.19, 0.35, "cathode", ha="center", fontsize=6.0)
    ax.text(0.42, 0.35, "sep.", ha="center", fontsize=6.0)
    ax.text(0.65, 0.35, "anode", ha="center", fontsize=6.0)
    for x in (0.09, 0.17, 0.25, 0.55, 0.63, 0.71):
        ax.add_patch(Circle((x, 0.55), 0.021, fc="white", ec="#24455F", lw=0.5))
    ax.annotate("", xy=(0.80, 0.20), xytext=(0.04, 0.20),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="#2C3E50"))
    ax.text(0.42, 0.225, "$x$ (through-thickness)", ha="center", fontsize=6.0)
    ax.text(0.50, 0.00, "min–h per step · coupled PDEs, ~30–40 params\n"
                        "captures through-thickness heterogeneity",
            ha="center", va="bottom", fontsize=5.5, color=C_GREY)
    return _save(fig, "fig05_equivalent_models.png")


if __name__ == "__main__":
    print("generating scientific illustrations ...")
    fig_multiscale()
    fig_crystal()
    fig_mechanisms()
    fig_graph_abstraction()
    fig_equivalent_models()
    print("done.")
