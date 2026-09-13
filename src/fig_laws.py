"""
================================================================================
 LFP-ETGNet  |  Module 8b: figures for the discovered physical laws
================================================================================
 Three composite figures, each organised so that a reader can read a *law* off
 the panel rather than a comparison of models:

   fig21  the two degradation channels, their partition, the network-damage
          master curve, and the regime map that carries the failure-mode law
   fig22  the scaling laws, each panel annotated with its fitted exponent
   fig23  the percolation law and why the electronic subnetwork governs it

 Everything is rendered from results/laws.json and from the single protocol
 helper in :mod:`laws`, so a figure cannot silently drift away from the model
 that produced the table.
================================================================================
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfp_micro import P, reconstruct_electrode, build_graph, _percolating_nodes
from laws import protocol, run, end_of_life

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figs")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.9, "savefig.dpi": 600, "figure.dpi": 110,
    "axes.titlesize": 8.0, "axes.labelsize": 8.0,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "mathtext.fontset": "dejavusans",
})

C_BLUE = "#2C7FB8"      # interphase / LLI
C_RED = "#C0392B"       # network / LAM
C_GREY = "#7F8C8D"      # contact loss / CL
C_GREEN = "#2E7D32"
C_ORANGE = "#E67E22"
C_PURPLE = "#8E44AD"
PALETTE = [C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE, "#16A085"]

TITLE = dict(loc="left", fontweight="bold", fontsize=8.0)


def _save(fig, name):
    """Save at fixed pixel size.  The DPI metadata is normalised separately by
    ``fix_fig_dpi.py`` so that Word does not crop the figure."""
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", name, flush=True)
    return p


def load_laws():
    with open(os.path.join(RES, "laws.json"), encoding="utf-8") as f:
        return json.load(f)


def _rve(seed=1000, n_target=1200):
    rng = np.random.default_rng(seed)
    m = reconstruct_electrode(rng, n_target=n_target)
    return build_graph(m, rng=rng), m


def _s(radii, box, w=140.0):
    return (2.0 * np.asarray(radii) / float(box) * w) ** 2


def _draw_network(ax, graph, alive, percolating, title):
    c = graph["centres"][:, :2]
    box = graph["box"]
    e = graph["edges"]
    seg = np.stack([c[e[:, 0]], c[e[:, 1]]], 1)
    ax.add_collection(LineCollection(seg[alive], colors="#95A5A6", lw=0.20,
                                     alpha=0.75))
    if (~alive).any():
        ax.add_collection(LineCollection(seg[~alive], colors=C_RED, lw=0.22,
                                         alpha=0.20, linestyles="dashed"))
    n_am = graph["n_am"]
    am = np.arange(n_am)
    dead = am[~percolating[:n_am]]
    ax.scatter(c[am, 0], c[am, 1], s=_s(graph["radii"][am], box),
               c="#4C6E91", lw=0, alpha=0.9)
    if dead.size:
        ax.scatter(c[dead, 0], c[dead, 1], s=_s(graph["radii"][dead], box),
                   c=C_RED, lw=0, alpha=0.95)
    ax.set_xlim(0, box); ax.set_ylim(0, box)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, loc="center", fontsize=6.2, fontweight="bold")


# ===========================================================================
#  Fig. 21 -- the two channels, their partition and the regime map
# ===========================================================================
def fig21_channels(L):
    pc, ms = L["protocol"], L["master"]
    rows = [r for r in pc["rows"] if r["kind"] == "CT"]

    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.30)

    # ---- (a) the two-channel capacity ledger -----------------------------
    ax = fig.add_subplot(gs[0, 0])
    cases = [(1.0, 25.0, 1.00, "1C / 25 $^\\circ$C / 100 % DoD"),
             (3.0, 35.0, 0.70, "3C / 35 $^\\circ$C / 70 % DoD"),
             (6.0, 25.0, 0.55, "6C / 25 $^\\circ$C / 55 % DoD")]
    for k, (C, T, d, lab) in enumerate(cases):
        rec = run(_rve(1000)[0], protocol(C, T, d), 5)["record"]
        n = rec["n"]
        ax.stackplot(n, 100 * rec["LLI"], 100 * rec["LAM"],
                     colors=[C_BLUE, C_RED], alpha=0.85 - 0.28 * k)
        ax.plot(n, 100 * rec["CL"], color=C_GREY, lw=1.0,
                ls=["-", "--", ":"][k])
        ax.plot([], [], color=C_GREY, lw=1.0, ls=["-", "--", ":"][k], label=lab)
    ax.axhline(20, color="k", ls="--", lw=0.9)
    ax.text(0.98, 20.6, "80 % SOH", transform=ax.get_yaxis_transform(),
            fontsize=6.0, ha="right")
    h = [Line2D([], [], color=C_BLUE, lw=5, alpha=0.85),
         Line2D([], [], color=C_RED, lw=5, alpha=0.85),
         Line2D([], [], color=C_GREY, lw=1.2)]
    ax.legend(h, ["LLI (interphase, capacity side)",
                  "LAM (isolated particles, capacity side)",
                  "CL (lost contacts, resistance side)"],
              frameon=False, fontsize=6.0, loc="upper left")
    ax.set_xlabel("cycle number"); ax.set_ylabel("capacity loss (%)")
    ax.set_title("(a) Two channels on one capacity ledger", **TITLE)

    # ---- (b) the partition law ------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    Ts = sorted({r["T"] for r in rows})
    for k, T in enumerate(Ts):
        sub = sorted([r for r in rows if r["T"] == T], key=lambda r: r["C"])
        C = [r["C"] for r in sub]
        s = [100 * r["lam"] / (r["lam"] + r["q_lli"]) for r in sub]
        ax.plot(C, s, "o-", ms=3.0, lw=1.1, color=PALETTE[k % 6],
                label=f"{T:.0f} $^\\circ$C")
    ax.set_xlabel("charge rate (C)"); ax.set_ylabel("network share of loss (%)")
    ax.legend(frameon=False, ncol=2)
    ax.set_title("(b) Failure mode vs. rate and temperature", **TITLE)
    ax.text(0.97, 0.94, "$b_C=%.2f\\pm%.2f$   $b_T=%.2f\\pm%.2f$"
            % (pc["share_C"]["a"], pc["share_C"].get("a_sd", 0),
               pc["share_A"]["a"], pc["share_A"].get("a_sd", 0)),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
            bbox=dict(fc="white", ec="#CCCCCC", lw=0.5, pad=2.0))

    # ---- (c) the master curve -------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    th = np.asarray(ms["theta"]); lam = np.asarray(ms["lam"])
    cyc = np.asarray(ms["cycle"]); grp = np.asarray(ms["group"])
    for k in np.unique(grp):
        m = grp == k
        if m.sum() < 2:
            continue
        o = np.argsort(th[m])
        ax.plot(th[m][o], 100 * lam[m][o], "-", lw=0.9,
                color=PALETTE[k % 6], alpha=0.9)
    o = np.argsort(th)
    ax.plot(th[o], 100 * 10 ** np.polyval(
        np.polyfit(np.log(th), np.log(lam), 1), np.log(th))[o],
        "k--", lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$\\Theta = k_{\\rm CL}\\,n$  (network-damage time)")
    ax.set_ylabel("LAM (%)")
    ax.set_title("(c) Loss of active material collapses in $\\Theta$", **TITLE)
    ax.text(0.04, 0.95, "$\\mathcal{H}\\propto\\Theta^{%.2f}$\n"
            "residual %.1f %% vs %.1f %% for $n$"
            % (ms["exponent"], 100 * ms["rel_resid_theta"],
               100 * ms["rel_resid_cycle"]),
            transform=ax.transAxes, va="top", fontsize=6.2,
            bbox=dict(fc="white", ec="#CCCCCC", lw=0.5, pad=2.0))
    axi = ax.inset_axes([0.60, 0.16, 0.37, 0.36])
    for k in np.unique(grp):
        m = grp == k
        if m.sum() < 2:
            continue
        o = np.argsort(cyc[m])
        axi.plot(cyc[m][o], 100 * lam[m][o], "-", lw=0.7,
                 color=PALETTE[k % 6])
    axi.set_xscale("log")
    axi.tick_params(labelsize=5.0); axi.set_title("vs. cycle $n$",
                                                  fontsize=5.4, pad=1.2)
    axi.set_ylabel("%", fontsize=5.0)
    for s in ("top", "right"):
        axi.spines[s].set_visible(True)

    # ---- (d) the regime map ---------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    Cs = np.array(sorted({r["C"] for r in rows}))
    Ts = np.array(sorted({r["T"] for r in rows}))
    Z = np.full((len(Ts), len(Cs)), np.nan)
    for r in rows:
        i = int(np.argmin(np.abs(Ts - r["T"])))
        j = int(np.argmin(np.abs(Cs - r["C"])))
        Z[i, j] = 100 * r["lam"] / (r["lam"] + r["q_lli"])
    cs = ax.contourf(Cs, Ts, Z, levels=10, cmap="RdYlBu_r", alpha=0.92)
    cl = ax.contour(Cs, Ts, Z, levels=[25, 35, 45], colors="k",
                    linewidths=0.55)
    ax.clabel(cl, fmt="%d%%", fontsize=5.4)
    cb = fig.colorbar(cs, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("network share of capacity loss (%)", fontsize=6.4)
    cb.ax.tick_params(labelsize=6.0)
    ax.set_xlabel("charge rate (C)"); ax.set_ylabel("temperature ($^\\circ$C)")
    ax.set_title("(d) Regime map: chemistry vs. mechanics", **TITLE)

    return _save(fig, "fig21_law_channels.png")


# ===========================================================================
#  Fig. 22 -- the scaling laws, each panel annotated with its exponent
# ===========================================================================
def fig22_scaling(L):
    pc, sv, dz = L["protocol"], L["survival"], L["design"]
    rows = pc["rows"]

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.36)

    def _fitline(ax, x, y, color, label=None, lw=1.0):
        b, a = np.polyfit(np.log(x), np.log(y), 1)
        xs = np.linspace(np.log(min(x)), np.log(max(x)), 20)
        ax.plot(np.exp(xs), np.exp(a + b * xs), "-", color=color, lw=lw,
                alpha=0.9, label=label)
        return b

    # (a) EOL vs C-rate
    ax = fig.add_subplot(gs[0, 0])
    for k, T in enumerate(sorted({r["T"] for r in rows if r["kind"] == "CT"})):
        sub = sorted([r for r in rows if r["kind"] == "CT" and r["T"] == T],
                     key=lambda r: r["C"])
        x = np.array([r["C"] for r in sub]); y = np.array([r["n_eol"] for r in sub])
        ax.plot(x, y, "o", ms=2.8, color=PALETTE[k % 6])
        b = _fitline(ax, x, y, PALETTE[k % 6])
        ax.plot([], [], "o-", ms=2.8, lw=1.0, color=PALETTE[k % 6],
                label=f"{T:.0f} $^\\circ$C, $a$={b:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("charge rate (C)"); ax.set_ylabel("end of life (cycles)")
    ax.legend(frameon=False, ncol=2, fontsize=6.0)
    ax.set_title("(a) Lifetime vs. rate", **TITLE)

    # (b) EOL vs Arrhenius factor
    ax = fig.add_subplot(gs[0, 1])
    for k, C in enumerate(sorted({r["C"] for r in rows if r["kind"] == "CT"})):
        sub = sorted([r for r in rows if r["kind"] == "CT" and r["C"] == C],
                     key=lambda r: r["A"])
        x = np.array([r["A"] for r in sub]); y = np.array([r["n_eol"] for r in sub])
        ax.plot(x, y, "o", ms=2.8, color=PALETTE[k % 6])
        _fitline(ax, x, y, PALETTE[k % 6])
    x = np.linspace(1.0, 3.2, 2)
    b = pc["eol_A"]["a"]
    x0 = 1.0; y0 = np.mean([r["n_eol"] for r in rows
                            if r["kind"] == "CT" and abs(r["A"] - 1.0) < 1e-9])
    ax.plot(x, y0 * x ** b, "k--", lw=1.0,
            label=f"$a_T={b:.2f}\\!\\pm\\!{pc['eol_A'].get('a_sd', 0):.2f}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arrhenius factor $A(T)$"); ax.set_ylabel("end of life (cycles)")
    ax.legend(frameon=False, fontsize=6.2)
    ax.set_title("(b) Lifetime vs. temperature", **TITLE)

    # (c) EOL vs depth of discharge
    ax = fig.add_subplot(gs[0, 2])
    for k, C in enumerate(sorted({r["C"] for r in rows if r["kind"] == "CD"})):
        sub = sorted([r for r in rows if r["kind"] == "CD" and r["C"] == C],
                     key=lambda r: r["dod"])
        x = np.array([r["dod"] for r in sub]); y = np.array([r["n_eol"] for r in sub])
        ax.plot(x, y, "o", ms=2.8, color=PALETTE[k % 6])
        _fitline(ax, x, y, PALETTE[k % 6])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("depth of discharge"); ax.set_ylabel("end of life (cycles)")
    ax.text(0.04, 0.06, "$a_D=%.2f\\!\\pm\\!%.2f$"
            % (pc["eol_D"]["a"], pc["eol_D"].get("a_sd", 0)),
            transform=ax.transAxes, fontsize=6.4,
            bbox=dict(fc="white", ec="#CCCCCC", lw=0.5, pad=2.0))
    ax.set_title("(c) Lifetime vs. depth of discharge", **TITLE)

    # (d) the survival law
    ax = fig.add_subplot(gs[1, 0])
    g = _rve(1000)[0]
    pr = protocol()
    rec = run(g, pr, 5)["record"]
    n = rec["n"]
    n_am, r_am = g["n_am"], g["r_am"]
    e, direct = g["edges"], g["direct"]
    i0 = np.minimum(e[:, 0], n_am - 1); i1 = np.minimum(e[:, 1], n_am - 1)
    eps = np.clip((r_am[i0] + r_am[i1]) / P["r_neck_ref"], 1e-3, None)[direct]
    nbe = 1.0 / (pr["k_cl_ref"] * eps ** P["m_fatigue"])
    s = P["s_fatigue"]
    pred = np.array([np.mean(norm.cdf((np.log(nbe) - np.log(nn)) / s))
                     for nn in n])
    ax.plot(n, 100 * rec["edge_ret"], color=C_RED, lw=1.4,
            label="simulated network")
    ax.plot(n, 100 * pred, "k--", lw=1.0,
            label="lognormal law  $R^2$=%.4f" % sv["r2"])
    ax.set_xlabel("cycle number"); ax.set_ylabel("intact contacts (%)")
    ax.legend(frameon=False)
    ax.set_title("(d) Contact-survival law", **TITLE)
    axi = ax.inset_axes([0.12, 0.13, 0.40, 0.32])
    axi.hist(np.log10(nbe), bins=28, color=C_RED, alpha=0.75)
    axi.axvline(np.log10(sv["n_be_p10"]), color="k", lw=0.7)
    axi.axvline(np.log10(sv["n_be_p90"]), color="k", lw=0.7)
    axi.set_title("neck rupture cycle", fontsize=5.2, pad=1.0)
    axi.tick_params(labelsize=4.8)

    # (e) the size law: the two channels pull in opposite directions
    ax = fig.add_subplot(gs[1, 1])
    x = np.array([r["d_mean"] for r in dz["size"]])
    y = np.array([r["n_eol"] for r in dz["size"]])
    ax.errorbar(x, y, yerr=[r["n_eol_sd"] for r in dz["size"]], fmt="o-",
                ms=3.2, lw=1.1, capsize=2.0, color=C_GREEN, zorder=4)
    #   the two one-channel predictions, each anchored on its own end of the
    #   measured curve; both diverge monotonically, so the observed turnover
    #   is the fingerprint of the channel that the single exponent misses
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, y[0] * (xx / x[0]) ** dz["predicted_a_d_interphase"], "k--",
            lw=0.8, label="interphase only ($+%.1f$)"
            % dz["predicted_a_d_interphase"])
    ax.plot(xx, y[-1] * (xx / x[-1]) ** dz["predicted_a_d_network"], "k:",
            lw=0.9, label="network only ($%+.1f$)" % dz["predicted_a_d_network"])
    if dz.get("d_star"):
        ax.axvline(dz["d_star"], color=C_RED, lw=0.8, ls="-")
        ax.text(dz["d_star"] * 1.04, y.max() * 0.60, "$d^\\star$", color=C_RED,
                fontsize=6.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(y.min() * 0.7, y.max() * 1.5)
    ax.set_xlabel("mean particle diameter ($\\mu$m)")
    ax.set_ylabel("end of life (cycles)")
    loc = dz["size_local"]
    ax.text(0.04, 0.12, "$\\frac{d\\ln\\mathrm{EOL}}{d\\ln d}$: %+.1f$\\to$%+.1f"
            % (loc[0]["a"], loc[-1]["a"]), transform=ax.transAxes, fontsize=6.2,
            bbox=dict(fc="white", ec="#CCCCCC", lw=0.5, pad=2.0))
    ax.legend(fontsize=5.0, loc="lower right", frameon=False, handlelength=1.4)
    ax.set_title("(e) Design law: particle size", **TITLE)

    # (f) the connectivity law
    ax = fig.add_subplot(gs[1, 2])
    x = np.array([r["z"] for r in dz["cal"]])
    y = np.array([r["n_eol"] for r in dz["cal"]])
    ax.plot(x, y, "o-", ms=3.2, lw=1.1, color=C_ORANGE)
    b = _fitline(ax, x, y, "k")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean coordination number $z$")
    ax.set_ylabel("end of life (cycles)")
    ax.text(0.04, 0.07, "$a_z=%.2f$  ($R^2$=%.3f)"
            % (dz["a_z"]["a"], dz["a_z"]["r2"]), transform=ax.transAxes,
            fontsize=6.4, bbox=dict(fc="white", ec="#CCCCCC", lw=0.5, pad=2.0))
    ax.set_title("(f) Design law: connectivity", **TITLE)

    return _save(fig, "fig22_law_scaling.png")


# ===========================================================================
#  Fig. 23 -- the percolation law and the fragility of the electronic network
# ===========================================================================
def fig23_percolation(L):
    pg = L["percolation"]
    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(2, 4, hspace=0.46, wspace=0.34)

    # (a) the two percolation curves
    ax = fig.add_subplot(gs[0, :3])
    for tag, col, lab in (("electronic", C_RED, "electronic (solid--solid)"),
                          ("composite", C_BLUE, "composite (incl. wetted)")):
        cv = pg["curves"][tag]
        f = np.array(cv["f"])
        m = np.array(cv["iso"])
        sd = np.array(cv["iso_sd"])
        ax.plot(100 * f, 100 * m, "-", color=col, lw=1.3, label=lab)
        ax.fill_between(100 * f, 100 * (m - sd), 100 * (m + sd),
                        color=col, alpha=0.20, lw=0)
        for lev, ls in ((pg[tag]["f_10pct"], ":"), (pg[tag]["f_25pct"], "--")):
            if np.isfinite(lev):
                ax.axvline(100 * lev, color=col, ls=ls, lw=0.8, alpha=0.8)
    ax.set_xlabel("solid--solid contacts removed (%)")
    ax.set_ylabel("active particles isolated (%)")
    ax.legend(frameon=False)
    ax.set_title("(a) Percolation of the two candidate networks", **TITLE)
    ax.text(0.03, 0.95,
            "10 %% isolation at $f_c$=%.0f %% (electronic) vs %.0f %% (composite)"
            % (100 * pg["electronic"]["f_10pct"],
               100 * pg["composite"]["f_10pct"]),
            transform=ax.transAxes, va="top", fontsize=6.2,
            bbox=dict(fc="white", ec="#CCCCCC", lw=0.5, pad=2.0))

    # (b) the critical exponent
    ax = fig.add_subplot(gs[0, 3])
    cv = pg["curves"]["electronic"]
    f = np.array(cv["f"]); iso = np.array(cv["iso"])
    fc = pg["electronic"]["f_10pct"]
    m = (f > fc) & (iso > 0.02) & (iso < 0.60)
    if m.sum() >= 4:
        ax.plot(f[m] - fc, iso[m], "o", ms=3.0, color=C_PURPLE)
        b, a = np.polyfit(np.log(f[m] - fc), np.log(iso[m]), 1)
        xs = np.linspace(np.log((f[m] - fc).min()),
                         np.log((f[m] - fc).max()), 20)
        ax.plot(np.exp(xs), np.exp(a + b * xs), "k--", lw=1.0)
        ax.text(0.05, 0.93, "$\\beta=%.2f$\n$R^2$=%.3f" % (b, pg["beta_r2"]),
                transform=ax.transAxes, va="top", fontsize=6.2,
                bbox=dict(fc="white", ec="#CCCCCC", lw=0.5, pad=2.0))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$f-f_c$"); ax.set_ylabel("LAM")
    ax.set_title("(b) Critical exponent", **TITLE)

    # (c) fragility bar chart
    ax = fig.add_subplot(gs[1, 0])
    levs = [("f_5pct", "5 %"), ("f_10pct", "10 %"), ("f_25pct", "25 %")]
    x = np.arange(len(levs)); w = 0.36
    e = [100 * pg["electronic"][k] for k, _ in levs]
    c = [100 * pg["composite"][k] for k, _ in levs]
    ax.bar(x - w / 2, e, w, color=C_RED, label="electronic")
    ax.bar(x + w / 2, c, w, color=C_BLUE, label="composite")
    for xi, (a_, b_) in enumerate(zip(e, c)):
        ax.text(xi - w / 2, a_ + 1.5, "%.0f" % a_, ha="center", fontsize=5.6)
        ax.text(xi + w / 2, b_ + 1.5, "%.0f" % b_, ha="center", fontsize=5.6)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in levs])
    ax.set_xlabel("isolated fraction reached")
    ax.set_ylabel("contacts removed (%)")
    ax.legend(frameon=False)
    ax.set_title("(c) Fragility gap: %.2f$\\times$" % pg["fragility_ratio"],
                 **TITLE)

    # (d-f) network snapshots as contacts are removed
    g = _rve(1000)[0]
    n_node, edges, kind, depth = (g["n_node"], g["edges"], g["kind"],
                                  g["depth"])
    direct = g["direct"]; n_am = g["n_am"]
    pool = np.where(direct)[0]
    rng = np.random.default_rng(31)
    for k, f in enumerate((0.0, 0.20, 0.40)):
        ax = fig.add_subplot(gs[1, 1 + k])
        alive = np.zeros(len(edges), bool)
        keep = rng.choice(pool, size=int(round((1.0 - f) * pool.size)),
                          replace=False)
        alive[keep] = True
        perc = _percolating_nodes(n_node, edges, alive, kind, depth,
                                  subnet=direct)
        iso = 1.0 - perc[:n_am].mean()
        _draw_network(ax, g, alive, perc,
                      f"({chr(100+k)}) {100*f:.0f} % contacts removed\n"
                      f"isolated {100*iso:.1f} %")
    return _save(fig, "fig23_law_percolation.png")


def main():
    print("=" * 78)
    print(" LFP-ETGNet | law figures")
    print("=" * 78, flush=True)
    L = load_laws()
    print("generating ...", flush=True)
    fig21_channels(L)
    fig22_scaling(L)
    fig23_percolation(L)
    print("done", flush=True)


if __name__ == "__main__":
    main()
