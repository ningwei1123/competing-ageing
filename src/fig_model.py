"""
================================================================================
 LFP-ETGNet  |  Module 5b: Framework and architecture schematics
================================================================================
"""
from __future__ import annotations
import os, sys, re, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Ellipse
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figs")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.4,
                     "savefig.dpi": 600, "figure.dpi": 110})

C_D1 = "#1F4E79"; C_D2 = "#2E86AB"; C_D3 = "#5DA271"; C_D4 = "#C97B4A"
C_D5 = "#8E44AD"; C_BG = "#F4F7FA"; C_TXT = "#1B2631"
C_BLUE = "#2C7FB8"; C_RED = "#C0392B"


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", name, flush=True)


def _box(ax, x, y, w, h, txt, fc, fs=7.0, tc="white", ec="none", lw=0.0,
         bold=True, radius=0.018):
    """Rounded box with a label that is automatically wrapped and, if
    necessary, shrunk so that it always stays inside the box."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.004,rounding_size={radius}",
                                fc=fc, ec=ec, lw=lw, transform=ax.transAxes))
    fw, fh = ax.figure.get_size_inches()
    pos = ax.get_position()
    box_w = max(w * pos.width * fw - 0.055, 0.15)      # usable width (inch)
    box_h = max(h * pos.height * fh - 0.030, 0.08)     # usable height (inch)

    def layout(size):
        cw = (0.70 if bold else 0.63) * size / 72.0
        ncol = max(int(box_w / max(cw, 1e-6)), 5)
        lines = []
        for seg in txt.split("\n"):
            # tokenise so that inline mathtext ($...$) is never split
            toks = re.findall(r"\$[^$]*\$|\S+", seg)
            cur = ""
            for tk in toks:
                cand = tk if not cur else cur + " " + tk
                if len(cand) <= ncol or not cur:
                    cur = cand
                else:
                    lines.append(cur)
                    cur = tk
            lines.append(cur)
        return lines, cw

    for _ in range(16):
        lines, cw = layout(fs)
        if (max(len(s) for s in lines) * cw <= box_w and
                len(lines) * 1.34 * fs / 72.0 <= box_h):
            break
        fs *= 0.94
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center",
            fontsize=fs, color=tc, transform=ax.transAxes, linespacing=1.3,
            fontweight="bold" if bold else "normal")


def _arrow(ax, p1, p2, c="#34495E", lw=1.1, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, transform=ax.transAxes,
                                 arrowstyle=style, mutation_scale=9, lw=lw,
                                 color=c, connectionstyle=f"arc3,rad={rad}"))


# ============================================================================
#  Fig. 6  Overall research framework
# ============================================================================
def fig_framework():
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    bands = [(0.812, 0.180, "Data layer", C_D1, "#EAF2F8"),
             (0.596, 0.196, "Feature-engineering layer", C_D3, "#EAF7EE"),
             (0.268, 0.308, "Model layer", C_D4, "#FDF2E9"),
             (0.020, 0.222, "Validation & interpretation layer", C_D5, "#F4ECF7")]
    for y, h, name, c, bg in bands:
        ax.add_patch(FancyBboxPatch((0.02, y), 0.96, h,
                                    boxstyle="round,pad=0.006,rounding_size=0.02",
                                    fc=bg, ec=c, lw=1.0, transform=ax.transAxes))
        ax.text(0.035, y + h - 0.024, name, fontsize=7.6, color=c,
                fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0.965, 0.968, "physics-guided  \"grey-box\"", fontsize=6.8,
            color="#7B241C", va="top", ha="right", transform=ax.transAxes,
            style="italic")

    # ---- data layer ------------------------------------------------------
    _box(ax, 0.055, 0.828, 0.200, 0.110, "Automotive LFP cell\nmicrostructure\n"
         "(SEM / FIB-SEM / TEM)", C_D1, 6.6)
    _box(ax, 0.290, 0.828, 0.200, 0.110, "Fast-charge protocols\n(1–6 C, 25–50 °C,\n"
         "DOD windows)", C_D1, 6.6)
    _box(ax, 0.525, 0.828, 0.200, 0.110, "Early-life cycling data\n"
         "(first 100 cycles)", C_D1, 6.6)
    _box(ax, 0.760, 0.828, 0.185, 0.110, "Literature-calibrated\nphysical parameters",
         C_D1, 6.6)

    # ---- feature engineering --------------------------------------------
    _box(ax, 0.055, 0.634, 0.395, 0.100,
         "Microstructure → graph  $G=(V,E)$\n"
         "nodes: AM particles / carbon domains\n"
         "edges: solid contacts + wetted paths", C_D3, 6.6, tc=C_TXT)
    _box(ax, 0.510, 0.634, 0.435, 0.100,
         "Electrochemical trajectory tensor  $\\mathbf{S}\\in\\mathbb{R}^{T\\times 7}$\n"
         "capacity · resistance · temperature · dQ/dV · polarisation · CE",
         C_D3, 6.6, tc=C_TXT)
    _box(ax, 0.055, 0.598, 0.890, 0.030,
         "node attributes: particle diameter · inter-particle spacing · SEI thickness · "
         "Li-concentration gradient · coordination · contact quality · depth · local SOC",
         "#D5EEE0", 6.0, tc=C_TXT, bold=False)

    # ---- model layer -----------------------------------------------------
    _box(ax, 0.055, 0.430, 0.395, 0.090,
         "Edge-gated GraphSAGE encoder\n(3 layers, 64-d) + attention readout",
         C_D4, 6.6)
    _box(ax, 0.510, 0.430, 0.435, 0.090,
         "Transformer encoder\n(2 layers, 4 heads, 128-d) + sinusoidal PE",
         C_D4, 6.6)
    _box(ax, 0.290, 0.345, 0.420, 0.067,
         "Multi-scale fusion  $\\mathbf{h}=$ ReLU$(\\mathbf{W}[\\mathbf{h}_G;\\mathbf{h}_L])$ "
         "+ LayerNorm$(\\cdot)$", C_D4, 6.6)
    _box(ax, 0.055, 0.268, 0.395, 0.062,
         "Shared encoder (2-layer MLP)", C_D4, 6.6)
    _box(ax, 0.510, 0.268, 0.435, 0.062,
         "Multi-task head:  $\\hat y_{RUL}$, $\\hat y_{SOH}$, $\\hat y_{\\Delta R_{int}}$",
         C_D4, 6.6)

    # ---- validation ------------------------------------------------------
    _box(ax, 0.055, 0.075, 0.215, 0.075, "Accuracy\nMSE / RMSE / MAE / $R^2$",
         C_D5, 6.2)
    _box(ax, 0.285, 0.075, 0.215, 0.075, "Efficiency\nlatency / parameters", C_D5, 6.2)
    _box(ax, 0.515, 0.075, 0.195, 0.075, "Uncertainty\ncalibration", C_D5, 6.2)
    _box(ax, 0.725, 0.075, 0.220, 0.075, "Physical attribution\n(LLI · CL · LAM)",
         C_D5, 6.2)

    # ---- data flow -------------------------------------------------------
    for x in (0.155, 0.390, 0.625, 0.8525):
        _arrow(ax, (x, 0.826), (x, 0.756))
    for x in (0.2525, 0.7275):
        _arrow(ax, (x, 0.632), (x, 0.630))
        _arrow(ax, (x, 0.596), (x, 0.524))
    _arrow(ax, (0.2525, 0.428), (0.40, 0.415))
    _arrow(ax, (0.7275, 0.428), (0.60, 0.415))
    _arrow(ax, (0.50, 0.343), (0.30, 0.333))
    _arrow(ax, (0.452, 0.299), (0.508, 0.299))
    for xb in (0.1625, 0.3925, 0.6125, 0.8350):
        _arrow(ax, (0.50, 0.262), (xb, 0.155))
    _save(fig, "fig06_framework.png")


# ============================================================================
#  Fig. 7  GNN-Transformer architecture
# ============================================================================
def fig_architecture():
    fig = plt.figure(figsize=(7.2, 4.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")

    # ---------- upper branch: graph ----------------------------------------
    ax.text(0.02, 0.965, "(a) Microscale branch — physics-guided GNN",
            fontsize=7.8, fontweight="bold", color=C_D1, transform=ax.transAxes)
    _box(ax, 0.03, 0.700, 0.255, 0.130,
         "microstructure graph\n$G=(V,E)$\n$\\mathbf{X}\\in\\mathbb{R}^{N\\times8}$, "
         "$\\mathbf{Z}\\in\\mathbb{R}^{E\\times3}$", "#D6E4F0", 6.6, tc=C_TXT)
    for k in range(3):
        _box(ax, 0.325 + 0.135 * k, 0.700, 0.115, 0.130,
             f"Edge-gated\nGraphSAGE\nlayer {k+1}",
             C_D2, 6.2)
        if k < 2:
            _arrow(ax, (0.440 + 0.135 * k, 0.765), (0.455 + 0.135 * k, 0.765))
    _box(ax, 0.745, 0.700, 0.235, 0.130,
         "readout\nmean ⊕ attention pooling\n$\\mathbf{h}_G\\in\\mathbb{R}^{128}$",
         "#AED6F1", 6.4, tc=C_TXT)
    _arrow(ax, (0.285, 0.765), (0.325, 0.765))
    _arrow(ax, (0.727, 0.765), (0.745, 0.765))
    ax.text(0.045, 0.660,
            r"$\mathbf{h}_i^{\ell+1}=\sigma\!\left(\mathbf{W}_s\mathbf{h}_i^{\ell}"
            r"+\frac{1}{d_i}\sum_{j\in\mathcal{N}(i)}"
            r"\mathbf{W}_n\mathbf{h}_j^{\ell}\odot\sigma(\mathbf{W}_e\mathbf{z}_{ij})\right)$",
            fontsize=7.0, color=C_TXT, transform=ax.transAxes)

    # ---------- lower branch: transformer ----------------------------------
    ax.text(0.02, 0.565, "(b) Macroscale branch — temporal Transformer",
            fontsize=7.8, fontweight="bold", color=C_D3, transform=ax.transAxes)
    _box(ax, 0.03, 0.300, 0.255, 0.130,
         "electrochemical trajectory\n$\\mathbf{S}\\in\\mathbb{R}^{100\\times7}$\n"
         "(cycle-indexed series)", "#DDEEDD", 6.6, tc=C_TXT)
    _box(ax, 0.325, 0.300, 0.115, 0.130, "linear\nprojection\n$d_{model}$=128",
         C_D3, 6.2)
    _box(ax, 0.460, 0.300, 0.135, 0.130, "sinusoidal\npositional\nencoding", C_D3, 6.2)
    _box(ax, 0.615, 0.300, 0.175, 0.130,
         "2 × Transformer\nencoder layers\n4 heads, FFN 256", C_D3, 6.2)
    _box(ax, 0.810, 0.300, 0.170, 0.130,
         "pooling\n$\\mathbf{h}_L\\in\\mathbb{R}^{256}$", "#C8E6C9", 6.4, tc=C_TXT)
    for x1, x2 in [(0.285, 0.325), (0.440, 0.460), (0.595, 0.615), (0.790, 0.810)]:
        _arrow(ax, (x1, 0.365), (x2, 0.365))
    ax.text(0.045, 0.262,
            r"$A=\mathrm{softmax}\!\left(\mathbf{QK}^{\top}/\sqrt{d_k}\right)\mathbf{V}$"
            r"$\qquad$   $PE_{(t,2i)}=\sin\!\left(t/10000^{2i/d}\right)$",
            fontsize=7.0, color=C_TXT, transform=ax.transAxes)

    # ---------- fusion + heads --------------------------------------------
    ax.add_patch(FancyBboxPatch((0.03, 0.055), 0.95, 0.165,
                                boxstyle="round,pad=0.006,rounding_size=0.02",
                                fc="#FDF2E9", ec=C_D4, lw=1.0,
                                transform=ax.transAxes))
    _box(ax, 0.055, 0.105, 0.185, 0.085,
         "fusion\n$\\mathbf{h}=\\mathbf{W}_f[\\mathbf{h}_G;\\mathbf{h}_L]+\\mathbf{b}_f$",
         C_D4, 6.4)
    _box(ax, 0.265, 0.105, 0.180, 0.085,
         "shared encoder\n2-layer MLP\n$\\mathbf{z}\\in\\mathbb{R}^{128}$", C_D4, 6.4)
    _box(ax, 0.470, 0.155, 0.155, 0.070, "RUL head", "#AF7AC5", 6.4)
    _box(ax, 0.470, 0.075, 0.155, 0.070, "SOH head", "#AF7AC5", 6.4)
    _box(ax, 0.650, 0.115, 0.155, 0.070, "$\\Delta R_{int}$ head", "#AF7AC5", 6.4)
    _box(ax, 0.830, 0.085, 0.145, 0.120,
         "uncertainty\nweighting\n$\\log\\sigma_k$", "#7D3C98", 6.2)
    _arrow(ax, (0.240, 0.148), (0.265, 0.148))
    _arrow(ax, (0.445, 0.148), (0.470, 0.190))
    _arrow(ax, (0.445, 0.148), (0.470, 0.110))
    _arrow(ax, (0.445, 0.148), (0.650, 0.150))
    _arrow(ax, (0.625, 0.190), (0.830, 0.170), rad=-0.15, c="#7D3C98")
    _arrow(ax, (0.625, 0.110), (0.830, 0.125), rad=0.15, c="#7D3C98")
    _arrow(ax, (0.805, 0.150), (0.830, 0.150), c="#7D3C98")
    ax.text(0.055, 0.062,
            r"$\mathcal{L}=\sum_k\left[\frac{1}{2\sigma_k^2}\|\hat y_k-y_k\|^2+\log\sigma_k\right]$",
            fontsize=7.0, color="#7B241C", transform=ax.transAxes)
    _save(fig, "fig07_architecture.png")


# ============================================================================
#  Fig. 8  Multi-task learning scheme
# ============================================================================
def fig_multitask():
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 3, wspace=0.30, width_ratios=[1.0, 1.15, 1.0])

    # ---- (a) shared representation ---------------------------------------
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_title("(a) Shared representation", loc="left",
                 fontweight="bold", fontsize=7.2)
    _box(ax, 0.10, 0.72, 0.80, 0.16, "electrode microstructure topology",
         C_D1, 6.6)
    _box(ax, 0.10, 0.50, 0.80, 0.16, "early-life ageing trajectory", C_D3, 6.6)
    _box(ax, 0.10, 0.24, 0.80, 0.16, "shared latent $\\mathbf{z}$", C_D4, 6.8)
    _arrow(ax, (0.50, 0.72), (0.50, 0.66))
    _arrow(ax, (0.50, 0.50), (0.50, 0.40))
    ax.text(0.5, 0.10,
            "one representation drives three\ncoupled degradation targets",
            ha="center", fontsize=6.4, color="#5D6D7E", transform=ax.transAxes)

    # ---- (b) task isomorphism -------------------------------------------
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.set_title("(b) Structural isomorphism", loc="left", fontweight="bold",
                 fontsize=7.2)
    pairs = [("concrete track slab", "life · damage · strength", "#7F8C8D"),
             ("LFP power battery", "cycle life · capacity · resistance",
              "#C0392B")]
    for i, (a, b, c) in enumerate(pairs):
        y = 0.62 - i * 0.36
        _box(ax, 0.05, y, 0.40, 0.22, a, c, 6.6)
        _box(ax, 0.50, y, 0.45, 0.22, b, "#DDE3E8", 6.2, tc=C_TXT)
        _arrow(ax, (0.45, y + 0.11), (0.50, y + 0.11), style="<|-|>", c=c, lw=1.0)
    ax.text(0.5, 0.06, "one latent state → three coupled targets",
            ha="center", fontsize=6.2, color="#5D6D7E", transform=ax.transAxes)

    # ---- (c) uncertainty weighting value --------------------------------
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("(c) Why uncertainty weighting matters", loc="left",
                 fontweight="bold", fontsize=7.2)
    n = np.linspace(1, 300, 300)
    rng = np.random.default_rng(4)
    ax.plot(n, 1 - 0.0018 * np.sqrt(n) - 2e-6 * n, color=C_BLUE, lw=1.3,
            label="SOH trajectory")
    ax.fill_between(n, 1 - 0.0018 * np.sqrt(n) - 2e-6 * n - 0.004,
                    1 - 0.0018 * np.sqrt(n) - 2e-6 * n + 0.004,
                    color=C_BLUE, alpha=0.22, label="measurement noise")
    ax.set_xlabel("cycle number"); ax.set_ylabel("SOH")
    ax.legend(frameon=False, loc="upper right", fontsize=6.4)
    ax.text(0.03, 0.04, "scarce samples · long tests (months)\n"
                        "adaptive weights $\\succ$ manual tuning",
            transform=ax.transAxes, fontsize=6.2, color="#7B241C")
    return _save(fig, "fig08_multitask.png")


# ============================================================================
#  Fig. 9  Pseudocode
# ============================================================================
def fig_pseudocode():
    fig = plt.figure(figsize=(7.2, 4.3))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    lines = [
        ("Algorithm 1", "Physics-guided GNN-Transformer for LFP multi-task prognosis", "h"),
        ("", "", ""),
        ("Input:", "microstructure graphs $\\{G_j\\}$; early-life series $\\{S_j\\}$; "
                   "targets $\\{y_j=(RUL,SOH,\\Delta R_{int})\\}$", ""),
        ("Input:", "epochs $E$, batch size $B$, learning rate $\\eta$", ""),
        ("Output:", "trained parameters $\\Theta$, log-variances $\\{\\log\\sigma_k\\}$", ""),
        ("", "", ""),
        ("1:", "for epoch $=1$ to $E$ do", "b"),
        ("2:", "   for each mini-batch $\\mathcal{B}\\subset\\mathcal{D}_{train}$ do", ""),
        ("3:", "      $\\mathbf{X}, \\mathbf{E}\\leftarrow$ "
               "BatchGraphs($\\mathcal{B}$)  $\\triangleright$ block-diagonal batching", ""),
        ("4:", "      $\\mathbf{h}\\leftarrow\\mathbf{X}$", ""),
        ("5:", "      for $\\ell=1$ to $L_G$ do  "
               "$\\triangleright$ edge-gated message passing", "k"),
        ("6:", "         $\\mathbf{g}_{ij}\\leftarrow\\sigma(\\mathbf{W}_e\\mathbf{z}_{ij})$"
               "   $\\triangleright$ contact-quality gate", ""),
        ("7:", "         $\\mathbf{h}_i\\leftarrow$ ReLU$(\\mathbf{W}_s\\mathbf{h}_i+"
               "\\frac{1}{d_i}\\sum_j\\mathbf{W}_n\\mathbf{h}_j\\odot\\mathbf{g}_{ij})$", ""),
        ("8:", "      $\\mathbf{h}_G\\leftarrow$ Readout$(\\mathbf{h})$"
               "  $\\triangleright$ mean $\\oplus$ attention pooling", "k"),
        ("9:", "      $\\mathbf{h}_L\\leftarrow$ TransformerEnc($S_{\\mathcal{B}}$)"
               "  $\\triangleright$ self-attention over cycle index", "k"),
        ("10:", "      $\\mathbf{z}\\leftarrow$ SharedMLP(ReLU$(\\mathbf{W}_f"
                "[\\mathbf{h}_G;\\mathbf{h}_L]+\\mathbf{b}_f)$)", ""),
        ("11:", "      $\\hat{\\mathbf{y}}\\leftarrow[\\hat y_{RUL},\\hat y_{SOH},"
                "\\hat y_{\\Delta R_{int}}]$", ""),
        ("12:", "      $\\mathcal{L}\\leftarrow\\sum_{k}[\\frac{1}{2\\sigma_k^2}"
                "\\|\\hat y_k-y_k\\|^2+\\log\\sigma_k]$", "r"),
        ("13:", "      $\\Theta,\\{\\log\\sigma_k\\}\\leftarrow$ Adam"
                "$(\\nabla\\mathcal{L},\\eta)$", ""),
        ("14:", "   end for", ""),
        ("15:", "   $\\eta\\leftarrow$ CosineAnnealing$(\\eta,\\,$epoch$)$", ""),
        ("16:", "end for", "b"),
        ("17:", "return $\\Theta^\\star=\\arg\\min$ validation loss, "
                "$\\{\\log\\sigma_k^\\star\\}$", "r"),
    ]
    y = 0.965
    for tag, txt, style in lines:
        if tag:
            ax.text(0.035, y, tag, fontsize=7.0, fontweight="bold",
                    color="#1F4E79", transform=ax.transAxes, family="monospace")
        col = {"h": "#1A5276", "b": "#7B241C", "k": "#1E8449",
               "r": "#B9770E", "": C_TXT}[style]
        ax.text(0.135, y, txt, fontsize=6.9, color=col, transform=ax.transAxes,
                family="monospace", fontweight="bold" if style in "hbr" else "normal")
        y -= 0.0505
    _save(fig, "fig09_pseudocode.png")


if __name__ == "__main__":
    print("generating framework figures ...")
    fig_framework()
    fig_architecture()
    fig_multitask()
    fig_pseudocode()
    print("done.")
