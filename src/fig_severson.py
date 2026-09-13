"""
Severson 外部验证图: fig_sev1(寿命分布 gap) + fig_sev2(B2 复现) + fig_sev3(A' 早期衰减).
"""
import numpy as np
import glob, pickle, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "data/severson/revisit-severson-et-al-main/data"
FIG = "figs"

C_SIM = "#1f77b4"     # 蓝 = 仿真
C_REAL = "#d62728"    # 红 = 真实
C_GREY = "#888888"


def load_split(split):
    files = sorted(glob.glob(f"{BASE}/{split}/cell*.csv"),
                   key=lambda p: int(p.rsplit("cell", 1)[1][:-4]))
    X = np.array([np.loadtxt(f, delimiter=",") for f in files])
    y = np.loadtxt(f"{BASE}/cycle_lives/{split}_cycle_lives.csv")
    return X, y


def qd_series(a):
    return a.max(axis=0)


def main():
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5,
                         "axes.labelsize": 7, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})

    Xtr, ytr = load_split("train")
    Xt1, yt1 = load_split("test1"); keep = np.ones(len(yt1), bool); keep[21] = False
    Xt1, yt1 = Xt1[keep], yt1[keep]
    Xt2, yt2 = load_split("test2")
    y_all = np.concatenate([ytr, yt1, yt2])

    cells = pickle.load(open("data/lfp_cells.pkl", "rb"))
    sim_eol = np.array([c["n_eol"] for c in cells])

    # ---------------- fig_sev1: 寿命分布 gap ----------------
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    bins = np.logspace(np.log10(100), np.log10(10000), 30)
    ax.hist(y_all, bins=bins, alpha=0.55, color=C_REAL, label="Severson real (n=124)")
    ax.hist(sim_eol, bins=bins, alpha=0.55, color=C_SIM, label="this work (sim, n=%d)" % len(sim_eol))
    ax.set_xscale("log")
    ax.set_xlabel("cycle life (cycles)")
    ax.set_ylabel("count")
    ax.set_title("(a) Sim-to-real lifetime distribution gap", loc="left")
    ax.axvline(np.median(y_all), color=C_REAL, ls="--", lw=1)
    ax.axvline(np.median(sim_eol), color=C_SIM, ls="--", lw=1)
    ax.text(np.median(y_all), ax.get_ylim()[1] * 0.95, "  real median %.0f" % np.median(y_all),
            color=C_REAL, fontsize=6)
    ax.text(np.median(sim_eol), ax.get_ylim()[1] * 0.85, "  sim median %.0f" % np.median(sim_eol),
            color=C_SIM, fontsize=6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_sev1_distribution_gap.png", dpi=600)
    plt.close(fig)

    # ---------------- fig_sev2: B2 复现散点 ----------------
    def feat(X):
        dq = X[:, :, 98] - X[:, :, 8]
        return np.log10(np.var(dq, axis=1))
    ftr, ft1, ft2 = feat(Xtr), feat(Xt1), feat(Xt2)
    yl = [np.log10(ytr), np.log10(yt1), np.log10(yt2)]
    A = np.stack([np.ones_like(ftr), ftr], 1)
    w, *_ = np.linalg.lstsq(A, yl[0], rcond=None)
    def pred(f): return 10 ** (w[0] + w[1] * f)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for j, (ax, (f, y, lab)) in enumerate(zip(axes, [(ft1, yt1, "primary test"),
                                                       (ft2, yt2, "secondary test")])):
        ax.scatter(pred(f), y, s=12, color=C_SIM, alpha=0.7, edgecolor="none")
        ax.plot([100, 2400], [100, 2400], "k--", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(100, 2400); ax.set_ylim(100, 2400)
        ax.set_xlabel("predicted cycle life"); ax.set_ylabel("measured cycle life")
        rmse = float(np.sqrt(np.mean((y - pred(f)) ** 2)))
        ax.set_title(f"({chr(97+j)}) {lab}: RMSE {rmse:.0f} cyc", loc="left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_sev2_reproduction.png", dpi=600)
    plt.close(fig)

    # ---------------- fig_sev3: A' 早期衰减 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    n = np.arange(2, 101)
    # 左: 样本 Qd 曲线 (先升后降)
    ax = axes[0]
    for X, tag, c in [(Xtr, "train", C_REAL), (Xt2, "test2", C_GREY)]:
        for i in [0, len(X)//2, len(X)-1]:
            qd = qd_series(X[i]); qd = qd / qd[0]
            ax.plot(n, (qd - 1) * 100, lw=0.9, color=c, alpha=0.7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("cycle"); ax.set_ylabel("ΔQd / Qd(2)  (%)")
    ax.set_title("(a) Early-life capacity: rise then fall (fast charge)", loc="left")
    # 右: 早期损失 vs 寿命
    ax = axes[1]
    el, lf = [], []
    for X, y in [(Xtr, ytr), (Xt1, yt1), (Xt2, yt2)]:
        for i in range(X.shape[0]):
            qd = qd_series(X[i])
            el.append(1 - qd[-1] / qd[0]); lf.append(y[i])
    el = np.array(el); lf = np.array(lf)
    ax.scatter(el * 100, lf, s=10, color=C_SIM, alpha=0.6, edgecolor="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("early capacity loss at cycle 100 (%)")
    ax.set_ylabel("cycle life")
    rho = float(np.corrcoef(np.argsort(np.argsort(el)), np.argsort(np.argsort(lf)))[0, 1])
    ax.set_title(f"(b) early loss → life  (Spearman {rho:+.2f})", loc="left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_sev3_early_fade.png", dpi=600)
    plt.close(fig)

    print("wrote fig_sev1/2/3")


if __name__ == "__main__":
    main()
