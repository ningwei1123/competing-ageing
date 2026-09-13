# -*- coding: utf-8 -*-
"""Generate Supplementary Fig. 1: temporal attention profile of the first
encoder layer, from results/attention.npz.

The profile is the attention received by each cycle position from the final
query position (see src/analysis.py::extract_attention).  It shows a dominant
recency peak near the end of the 100-cycle observation window and a secondary
concentration in the first ~10 cycles (the initial transient).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figs")
RES = os.path.join(HERE, "..", "results")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.4,
                     "axes.linewidth": 0.9, "savefig.dpi": 600,
                     "figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False,
                     "mathtext.fontset": "dejavusans"})

A = np.load(os.path.join(RES, "attention.npz"))
prof = A["profile"]            # [100] attention received by each cycle
heads = A["heads"]             # [4, 100, 100]
M = heads.mean(0)              # mean attention matrix over heads

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1),
                         gridspec_kw={"width_ratios": [1.0, 1.25]})

# --- (a) temporal attention profile ----------------------------------------
ax = axes[0]
x = np.arange(1, len(prof) + 1)
ax.plot(x, prof, color="#C0392B", lw=1.5)
ax.fill_between(x, prof, color="#C0392B", alpha=0.18)
# dominant recency peak
k = int(prof.argmax()) + 1
ax.annotate(f"recency peak\ncycle {k}", xy=(k, prof[k - 1]),
            xytext=(k - 22, prof.max() * 1.02), fontsize=7.2,
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#7F8C8D"))
# secondary early transient
ke = int(prof[:15].argmax()) + 1
ax.annotate("early transient", xy=(ke, prof[ke - 1]),
            xytext=(ke + 4, prof[ke - 1] * 0.72), fontsize=7.2,
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#7F8C8D"))
ax.set_xlabel("cycle index (early-life window)")
ax.set_ylabel("attention weight")
ax.set_title("(a) Temporal attention", loc="left", fontweight="bold")
ax.set_xlim(1, 100)

# --- (b) mean attention matrix ---------------------------------------------
ax = axes[1]
im = ax.imshow(M, cmap="magma", aspect="auto", origin="lower",
               extent=[1, 100, 1, 100])
ax.set_xlabel("key cycle")
ax.set_ylabel("query cycle")
ax.set_title("(b) Mean self-attention map", loc="left", fontweight="bold")
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=6.0)

fig.tight_layout()
out = os.path.join(FIG, "sup_fig01_attention.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved", out)
