"""
================================================================================
 LFP-ETGNet  |  Module 5c: Result figures
================================================================================
"""
from __future__ import annotations
import os, sys, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figs")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.4,
                     "axes.linewidth": 0.9, "savefig.dpi": 600, "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "mathtext.fontset": "dejavusans"})

ORDER = ["LFP-ETGNet", "GAT-TST", "ST-GCN", "GMAN", "MTL-ResNet", "CNN-LSTM",
         "GNN-only", "Transformer-only"]
PAL = {"LFP-ETGNet": "#C0392B", "GAT-TST": "#2E86AB", "ST-GCN": "#5DA271",
       "GMAN": "#8E44AD", "MTL-ResNet": "#C97B4A", "CNN-LSTM": "#7F8C8D",
       "GNN-only": "#16A085", "Transformer-only": "#D68910"}
TASKS = ["RUL", "SOH", "dR_int"]
TLAB = {"RUL": "RUL (cycles)", "SOH": "SOH (–)",
        "dR_int": r"$\Delta R_{int}$ (–)"}


def _load():
    m = json.load(open(os.path.join(OUT, "metrics.json")))
    return m


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", name, flush=True)


# ============================================================================
def fig_convergence(M):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    keys = [("test_MSE", "MSE (normalised)", "(a)"),
            ("test_RMSE", "RMSE (normalised)", "(b)"),
            ("test_R2", r"$R^2$", "(c)")]
    for ax, (k, lab, tag) in zip(axes, keys):
        for name in ORDER:
            if name not in M:
                continue
            h = M[name]["hist"]
            ep = h["epoch"]
            s = 9 if name == "LFP-ETGNet" else 4
            lw = 1.9 if name == "LFP-ETGNet" else 1.1
            ax.plot(ep, h[k], color=PAL[name], lw=lw, marker="o", ms=s * 0.35,
                    markevery=max(1, len(ep) // 12), label=name,
                    zorder=5 if name == "LFP-ETGNet" else 2,
                    alpha=1.0 if name == "LFP-ETGNet" else 0.85)
        ax.set_xlabel("epoch"); ax.set_ylabel(lab)
        ax.set_title(tag, loc="left", fontweight="bold")
        if k == "test_R2":
            ax.set_ylim(0, 1.0)
    axes[0].legend(ncol=2, fontsize=6.0, loc="upper right", handlelength=1.2,
                   columnspacing=0.8, labelspacing=0.28)
    fig.tight_layout()
    _save(fig, "fig10_convergence.png")


# ============================================================================
def fig_scatter(bundle):
    pr, tr = bundle["pred"], bundle["true"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for k, ax in enumerate(axes):
        y, p = tr[:, k], pr[:, k]
        ax.scatter(y, p, s=13, c="#2C7FB8", alpha=0.65, edgecolor="none")
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.9)
        ss = 1 - np.sum((y - p) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
        rm = np.sqrt(np.mean((y - p) ** 2))
        ax.text(0.05, 0.90, f"$R^2$ = {ss:.3f}\nRMSE = {rm:.3g}",
                transform=ax.transAxes, fontsize=6.6, va="top")
        ax.set_xlabel("measured " + TLAB[TASKS[k]])
        ax.set_ylabel("predicted")
        ax.set_title(f"({'abc'[k]})", loc="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig11_prediction_scatter.png")


# ============================================================================
def fig_accuracy(M):
    present = [n for n in ORDER if n in M]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, metric, lab in zip(axes, ["MSE", "RMSE", r"R2"],
                               ["MSE (normalised)", "RMSE (normalised)",
                                r"coefficient of determination $R^2$"]):
        vals = [M[n]["metrics"]["_macro"][metric] for n in present]
        errs = [M[n]["metrics"]["_macro_std"][metric] for n in present]
        if metric == "R2":
            vals = [max(v, 0) for v in vals]
            ax.axhline(1.0, color="#BDC3C7", lw=0.7, ls=":")
        bars = ax.bar(range(len(present)), vals, yerr=errs, capsize=2.0,
                      color=[PAL[n] for n in present], edgecolor="white", lw=0.5)
        bars[0].set_edgecolor("#7B241C")
        bars[0].set_linewidth(1.4)
        if metric != "R2":
            ax.set_yscale("log")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=38, ha="right", fontsize=6.2)
        ax.set_ylabel(lab)
        ax.set_title(f"({'abc'[['MSE','RMSE','R2'].index(metric)]})",
                     loc="left", fontweight="bold")
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i, v * (1.25 if metric != "R2" else 1.02), f"{v:.3f}",
                    ha="center", fontsize=5.6, rotation=90 if metric != "R2" else 0)
        if metric == "R2":
            ax.set_ylim(0, 1.12)
    fig.tight_layout()
    _save(fig, "fig12_accuracy_comparison.png")


# ============================================================================
def fig_efficiency(M, bundle=None):
    present = [n for n in ORDER if n in M]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    lat = [M[n]["latency_s"] * 1000 for n in present]
    b = ax.barh(range(len(present)), lat, color=[PAL[n] for n in present],
                edgecolor="white", lw=0.5, height=0.66)
    b[0].set_edgecolor("#7B241C"); b[0].set_linewidth(1.4)
    ax.set_yticks(range(len(present))); ax.set_yticklabels(present, fontsize=6.4)
    ax.invert_yaxis(); ax.set_xlabel("single-sample inference latency (ms)")
    ax.set_title("(a) Inference latency", loc="left", fontweight="bold")
    for i, v in enumerate(lat):
        ax.text(v * 1.02, i, f"{v:.1f}", va="center", fontsize=5.8)
    ax.set_xlim(0, max(lat) * 1.22)

    ax = axes[1]
    par = [M[n]["n_param"] / 1e3 for n in present]
    r2 = [max(M[n]["metrics"]["_macro"]["R2"], 0) for n in present]
    for i, n in enumerate(present):
        ax.scatter(par[i], r2[i], s=70, c=PAL[n], edgecolor="white", lw=0.8,
                   zorder=4 if n == "LFP-ETGNet" else 3)
        ax.annotate(n, (par[i], r2[i]), textcoords="offset points",
                    xytext=(4, 4.5), fontsize=5.8, color="#34495E")
    ax.set_xlabel("parameters (×10$^3$)"); ax.set_ylabel("macro $R^2$")
    ax.set_title("(b) Accuracy vs. model size", loc="left", fontweight="bold")
    ax.set_ylim(0, 1.05); ax.set_xlim(0, max(par) * 1.28)
    fig.tight_layout()
    _save(fig, "fig13_efficiency.png")


# ============================================================================
def fig_ablation(M, bundle=None):
    """Ablation: full model vs GNN-only, Transformer-only, and no-physics-graph."""
    names = [n for n in ["LFP-ETGNet", "GNN-only", "Transformer-only",
                         "MTL-ResNet", "CNN-LSTM"] if n in M]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    r2 = [max(M[n]["metrics"]["_macro"]["R2"], 0) for n in names]
    b = ax.bar(range(len(names)), r2, color=[PAL[n] for n in names],
               edgecolor="white", lw=0.6)
    b[0].set_edgecolor("#7B241C"); b[0].set_linewidth(1.4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=32, ha="right", fontsize=6.2)
    ax.set_ylabel(r"macro $R^2$"); ax.set_ylim(0, 1.06)
    ax.set_title("(a) Branch ablation", loc="left", fontweight="bold")
    for i, v in enumerate(r2):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=6.0)

    ax = axes[1]
    w = 0.26
    x = np.arange(len(TASKS))
    for j, n in enumerate(names[:3]):
        vals = [max(M[n]["metrics"][t]["R2"], 0) for t in TASKS]
        ax.bar(x + (j - 1) * w, vals, w, color=PAL[n], edgecolor="white",
               lw=0.5, label=n)
    ax.set_xticks(x); ax.set_xticklabels([TLAB[t] for t in TASKS], fontsize=6.4)
    ax.set_ylabel(r"$R^2$"); ax.set_ylim(0, 1.12)
    ax.legend(fontsize=6.0, ncol=1, loc="lower right")
    ax.set_title("(b) Per-task $R^2$", loc="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig14_ablation.png")


# ============================================================================
def fig_interpretability():
    A = np.load(os.path.join(OUT, "attention.npz"))
    I = np.load(os.path.join(OUT, "interp.npz"), allow_pickle=True)
    prof = A["profile"]
    node_imp, seq_imp = I["node_imp"], I["seq_imp"]
    NODE = ["particle diameter", "inter-particle spacing", "SEI thickness",
            "Li-conc. gradient", "coordination", "contact quality",
            "electrode depth", "local SOC"]
    SEQ = ["SOH(n)", r"$R_{int}$(n)", r"$T_{max}$(n)", "dQ/dV(n)",
           r"$\Delta V_{pol}$(n)", "CE(n)", r"$Q_{chg}$(n)"]

    fig = plt.figure(figsize=(7.2, 4.3))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(np.arange(1, len(prof) + 1), prof, color="#C0392B", lw=1.5)
    ax.fill_between(np.arange(1, len(prof) + 1), prof, color="#C0392B", alpha=0.18)
    ax.set_xlabel("cycle index (early-life window)")
    ax.set_ylabel("attention weight")
    ax.set_title("(a) Temporal attention", loc="left", fontweight="bold")
    k = int(np.argmax(prof[5:]) + 5)
    ax.annotate(f"peak @ cycle {k+1}", xy=(k + 1, prof[k]),
                xytext=(k * 0.35 + 5, prof.max() * 0.92), fontsize=6.2,
                arrowprops=dict(arrowstyle="->", lw=0.7))

    ax = fig.add_subplot(gs[0, 1])
    o = np.argsort(node_imp)
    ax.barh(np.arange(8), node_imp[o], color="#2E86AB", edgecolor="white", lw=0.5)
    ax.set_yticks(np.arange(8)); ax.set_yticklabels([NODE[i] for i in o], fontsize=6.2)
    ax.set_xlabel("relative permutation importance")
    ax.set_title("(b) Microstructure node descriptors", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[0, 2])
    o = np.argsort(seq_imp)
    ax.barh(np.arange(7), seq_imp[o], color="#5DA271", edgecolor="white", lw=0.5)
    ax.set_yticks(np.arange(7)); ax.set_yticklabels([SEQ[i] for i in o], fontsize=6.2)
    ax.set_xlabel("relative permutation importance")
    ax.set_title("(c) Trajectory channels", loc="left", fontweight="bold")

    # --- attention matrices (heads) ---
    for j in range(2):
        ax = fig.add_subplot(gs[1, j])
        M = A["matrix"][j] if A["matrix"].ndim == 3 else np.eye(3)
        im = ax.imshow(M, cmap="magma", aspect="auto")
        ax.set_title(f"(d{j+1}) head-{j+1} attention map", loc="left",
                     fontweight="bold")
        ax.set_xlabel("key cycle"); ax.set_ylabel("query cycle")
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02).ax.tick_params(labelsize=5.4)

    # --- descriptor-target correlation ---
    ax = fig.add_subplot(gs[1, 2])
    corr = I["corr"]
    names = [str(x) for x in I["desc_names"]]
    keep = ["edge_retention", "isolated_frac", "mean_degree", "n_edge"]
    kk = [names.index(k) for k in keep if k in names]
    lab = [keep[list(keep).index(names[k])] if names[k] in keep else names[k] for k in kk]
    C = corr[kk]
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["RUL", "SOH", r"$\Delta R_{int}$"],
                                                fontsize=6.4)
    ax.set_yticks(range(len(kk)))
    ax.set_yticklabels([lab[i].replace("_", " ") for i in range(len(kk))], fontsize=6.0)
    ax.set_title("(e) Graph descriptors vs. targets", loc="left", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02).ax.tick_params(labelsize=5.4)
    _save(fig, "fig15_interpretability.png")


# ============================================================================
def fig_uncertainty(bundle):
    try:
        cal = json.load(open(os.path.join(OUT, "calibration.json")))
    except FileNotFoundError:
        print("  (calibration.json missing, skip)"); return
    pr, tr = bundle["pred"], bundle["true"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for k, ax in enumerate(axes):
        t = TASKS[k]
        y, p = tr[:, k], pr[:, k]
        o = np.argsort(y)
        ax.plot(y[o], y[o], "k--", lw=0.8, label="ideal")
        ax.plot(y[o], p[o], color="#C0392B", lw=1.1, label="prediction")
        s = cal[t]["sigma_mean"]
        ax.fill_between(y[o], p[o] - 2 * s, p[o] + 2 * s, color="#C0392B",
                        alpha=0.18, label=r"$\pm 2\sigma$")
        ax.set_xlabel("measured " + TLAB[t]); ax.set_ylabel("predicted")
        cov = cal[t]["coverage_2s"] * 100
        ax.set_title(f"({'abc'[k]}) {t}  ($\\pm2\\sigma$: {cov:.0f}%)",
                     loc="left", fontweight="bold")
        if k == 0:
            ax.legend(fontsize=5.8, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig16_uncertainty.png")


# ============================================================================
def fig_attribution():
    cells = pickle.load(open(os.path.join(HERE, "..", "data", "lfp_cells.pkl"), "rb"))
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.34)

    # (a) capacity-fade curve family coloured by operating temperature
    ax = fig.add_subplot(gs[0, :2])
    Ts = np.array([c["proto"]["T_celsius"] for c in cells])
    cm = plt.cm.coolwarm
    for c in cells[::4]:
        cv = c["curve"]
        ax.plot(cv["n"], 100 * (1 - cv["SOH"]), color=cm((c["proto"]["T_celsius"] - 25) / 25),
                lw=0.5, alpha=0.55)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(25, 50))
    cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.015)
    cb.set_label("temperature (°C)", fontsize=6.4)
    cb.ax.tick_params(labelsize=5.6)
    ax.axhline(20, color="k", ls="--", lw=0.9)
    ax.set_xlabel("cycle number"); ax.set_ylabel("capacity loss (%)")
    ax.set_title("(a) Simulated ageing family — two-channel coupled model",
                 loc="left", fontweight="bold")

    # (b) EOL vs. C-rate & T
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter([c["proto"]["c_rate"] for c in cells],
               [c["n_eol"] for c in cells], c=Ts, cmap="coolwarm", s=11, lw=0)
    ax.set_xlabel("charge C-rate"); ax.set_ylabel("cycles to 80 % SOH")
    ax.set_yscale("log")
    ax.set_title("(b) End-of-life map", loc="left", fontweight="bold")

    # (c) attribution at EOL, grouped by C-rate ----------------------------
    #   Only LLI and LAM remove capacity, so only those two are stacked into
    #   the capacity-loss bar.  CL is the fraction of electronic contacts lost:
    #   it raises the series resistance and it is the cause of LAM, but it
    #   does not subtract capacity.  Stacking it in the same bar (as an earlier
    #   version did) overstated the capacity loss by more than a factor of two
    #   and conflated a capacity quantity with an impedance quantity.
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
    ax.bar(x, lam, 0.55, bottom=lli, color="#C0392B", label="LAM",
           edgecolor="white", lw=0.5)
    ax.axhline(20, color="k", ls="--", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.4)
    ax.set_ylabel("capacity loss at EOL (%)")
    ax.set_ylim(0, max(23.0, 1.20 * (np.array(lli) + np.array(lam)).max()))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, cl, "D--", ms=3.8, lw=1.0, color="#2E2E2E")
    ax2.set_ylabel("electronic contact loss (%)", fontsize=6.2, color="#2E2E2E")
    ax2.tick_params(axis="y", labelsize=6.0, colors="#2E2E2E")
    ax2.set_ylim(0, max(30.0, 1.25 * max(cl)))
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color="#2E2E2E", marker="D", ls="--", ms=3.8))
    l.append("CL (resistance side)")
    ax.legend(h, l, fontsize=5.6, loc="upper left")
    ax.set_title("(c) Attribution vs. C-rate", loc="left", fontweight="bold")

    # (d) edge retention vs. cycle for representative protocols
    ax = fig.add_subplot(gs[1, 1])
    sel = np.argsort([c["proto"]["c_rate"] for c in cells])
    for rank, c in [(0, cells[sel[0]]), (1, cells[sel[len(sel) // 2]]),
                    (2, cells[sel[-1]])]:
        cv = c["curve"]
        ax.plot(cv["n"], 100 * cv["edge_ret"], lw=1.3,
                label=f"{c['proto']['c_rate']:.1f}C, {c['proto']['T_celsius']:.0f}°C, "
                      f"DOD {100*c['proto']['dod']:.0f}%")
    ax.set_xlabel("cycle number"); ax.set_ylabel("conductive-edge retention (%)")
    ax.legend(fontsize=5.6, loc="lower left")
    ax.set_title("(d) Conductive-network degradation", loc="left", fontweight="bold")

    # (e) isolated-particle fraction
    ax = fig.add_subplot(gs[1, 2])
    for rank, c in [(0, cells[sel[0]]), (1, cells[sel[len(sel) // 2]]),
                    (2, cells[sel[-1]])]:
        cv = c["curve"]
        ax.plot(cv["n"], 100 * cv["LAM"], lw=1.3)
    ax.set_xlabel("cycle number"); ax.set_ylabel("isolated active material (%)")
    ax.set_title("(e) LAM from percolation loss", loc="left", fontweight="bold")
    _save(fig, "fig17_attribution.png")


# ============================================================================
def fig_case_study(bundle):
    cells = pickle.load(open(os.path.join(HERE, "..", "data", "lfp_cells.pkl"), "rb"))
    te, pr, tr = bundle["te_idx"], bundle["pred"], bundle["true"]
    o = np.argsort(tr[:, 0])
    picks = [te[o[len(o) // 5]], te[o[len(o) // 2]], te[o[-len(o) // 6]]]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, i in zip(axes, picks):
        c = cells[i]
        cv = c["curve"]
        ax.plot(cv["n"], 100 * cv["SOH"], color="#1F4E79", lw=1.5, label="ground truth")
        ax.axhline(80, color="k", ls="--", lw=0.9)
        ax.axvline(300, color="#7F8C8D", ls=":", lw=0.9)
        k = int(np.where(te == i)[0][0])
        ax.scatter([300], [100 * pr[k, 1]], marker="*", s=90, c="#C0392B", zorder=6,
                   label="prediction @300")
        ax.annotate(f"true RUL = {tr[k,0]:.0f}\npred = {pr[k,0]:.0f}",
                    xy=(0.05, 0.14), xycoords="axes fraction", fontsize=6.0)
        ax.set_xlabel("cycle number"); ax.set_ylabel("SOH (%)")
        p = c["proto"]
        ax.set_title(f"({p['c_rate']:.1f}C, {p['T_celsius']:.0f}°C, "
                     f"DOD {100*p['dod']:.0f}%)", fontsize=7.0, loc="left",
                     fontweight="bold")
        if ax is axes[0]:
            ax.legend(fontsize=5.8, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig18_case_study.png")


# ============================================================================
def _load_ood():
    """Load whichever distribution-shift protocols have been run."""
    out = {}
    for fn, key in (("metrics_ood_temp.json", "temp"),
                    ("metrics_ood_life.json", "life")):
        p = os.path.join(OUT, fn)
        if os.path.exists(p):
            out[key] = json.load(open(p))
    return out


def fig_generalisation(M):
    """Accuracy under distribution shift, and the resulting generalisation gap.

    Panel (a,b) repeat the model comparison on the two out-of-distribution
    protocols; panel (c) shows how much macro R^2 each model loses relative to
    the in-distribution random split, i.e. how well its inductive bias survives
    a shift of the operating regime or of the life range.
    """
    O = _load_ood()
    if not O:
        print("  skip generalisation: no out-of-distribution results")
        return
    keys = [k for k in ("temp", "life") if k in O]
    present = [n for n in ORDER if all(n in O[k] for k in keys)]
    if not present:
        print("  skip generalisation: no common models")
        return
    SUB = {"temp": "(a) Unseen temperature regime",
           "life": "(b) Unseen life range",
           "rand": "(c) Generalisation gap"}
    ncol = len(keys) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(2.4 * ncol, 2.6))
    if ncol == 1:
        axes = [axes]

    for ax, k in zip(axes[:len(keys)], keys):
        vals = [max(O[k][n]["metrics"]["_macro"]["R2"], 0) for n in present]
        errs = [O[k][n]["metrics"]["_macro_std"]["R2"] for n in present]
        b = ax.bar(range(len(present)), vals, yerr=errs, capsize=2.0,
                   color=[PAL[n] for n in present], edgecolor="white", lw=0.5)
        b[0].set_edgecolor("#7B241C"); b[0].set_linewidth(1.4)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=38, ha="right", fontsize=6.2)
        ax.set_ylabel(r"macro $R^2$")
        ax.set_ylim(0, 1.12)
        ax.set_title(SUB[k], loc="left", fontweight="bold")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=5.6)

    ax = axes[-1]
    w = 0.38
    x = np.arange(len(present))
    for j, k in enumerate(keys):
        g = [M[n]["metrics"]["_macro"]["R2"] - O[k][n]["metrics"]["_macro"]["R2"]
             for n in present]
        ax.bar(x + (j - 0.5 * (len(keys) - 1)) * w, g, w,
               color=["#C0392B", "#2E86AB"][j], edgecolor="white", lw=0.5,
               label={"temp": "temperature shift", "life": "life extrapolation"}[k])
    ax.axhline(0, color="#34495E", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(present, rotation=38, ha="right",
                                         fontsize=6.2)
    ax.set_ylabel(r"$\Delta R^2$ (random $-$ shifted)")
    ax.set_title(SUB["rand"], loc="left", fontweight="bold")
    ax.legend(fontsize=6.0, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig19_generalisation.png")


# ============================================================================
def fig_data_efficiency(M):
    """Macro R^2 as a function of the number of training cells.

    Uses the reduced-population runs (``metrics_n*.json``) together with the
    full-population random-split run.  Only drawn when at least one reduced
    run exists.
    """
    sizes = []
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("metrics_n") and fn.endswith(".json"):
            try:
                sizes.append(int(fn[len("metrics_n"):-len(".json")]))
            except ValueError:
                pass
    if not sizes:
        print("  skip data efficiency: no reduced-population runs")
        return
    sizes = sorted(sizes)
    bundle = {n: json.load(open(os.path.join(OUT, f"metrics_n{n}.json")))
              for n in sizes}
    # the full-population run closes the curve
    n_full = 311
    try:
        cells = pickle.load(open(os.path.join(HERE, "..", "data",
                                              "lfp_cells.pkl"), "rb"))
        n_full = len(cells)
    except Exception:
        pass
    xs = sizes + [n_full]

    names = [n for n in ORDER
             if all(n in bundle[s] for s in sizes) and n in M]
    if not names:
        print("  skip data efficiency: no common models")
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for n in names:
        ys = [bundle[s][n]["metrics"]["_macro"]["R2"] for s in sizes]
        ys.append(M[n]["metrics"]["_macro"]["R2"])
        ax.plot(xs, ys, marker="o", ms=3.4, lw=1.3, color=PAL[n], label=n,
                zorder=4 if n == "LFP-ETGNet" else 3)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in xs], fontsize=6.6)
    ax.set_xlim(xs[0] * 0.82, xs[-1] * 1.25)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("number of training cells")
    ax.set_ylabel(r"macro $R^2$")
    ax.legend(fontsize=5.8, ncol=2, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig20_data_efficiency.png")


if __name__ == "__main__":
    print("generating result figures ...")
    M = _load()
    print("models:", list(M.keys()))
    bundle = pickle.load(open(os.path.join(OUT, "bundle.pkl"), "rb"))
    fig_convergence(M)
    fig_scatter(bundle)
    fig_accuracy(M)
    fig_efficiency(M)
    fig_ablation(M)
    try:
        fig_interpretability()
    except FileNotFoundError as e:
        print("  skip interpretability:", e)
    fig_uncertainty(bundle)
    fig_attribution()
    fig_case_study(bundle)
    fig_generalisation(M)
    try:
        fig_data_efficiency(M)
    except FileNotFoundError as e:
        print("  skip data efficiency:", e)
    print("done.")
