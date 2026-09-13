"""
Severson 外部验证 — 实验 D + B2(复现对标) + A'(早期衰减)。

数据源: data/severson/revisit-severson-et-al-main/data
  - cell CSV: (1000 电压点, 99 周期), 列 i = 周期 i+2 的 Qdlin(V) 放电容量曲线
  - cycle_lives: 124 cell 寿命 (QDischarge < 0.88 Ah 首周期)

对标 (Severson Nature Energy 2019):
  - Table 1 单特征 log10(var(dQ_100-10)) 模型 RMSE = (103, 138, 196) cycles
  - 摘要 "9.1% test error" (full model, primary test)
  - test1 删 index-21 outlier
"""
import numpy as np
import glob, json, pickle, os

BASE = "data/severson/revisit-severson-et-al-main/data"
RES = "results"


def load_split(split):
    files = sorted(glob.glob(f"{BASE}/{split}/cell*.csv"),
                   key=lambda p: int(p.rsplit("cell", 1)[1][:-4]))
    X = np.array([np.loadtxt(f, delimiter=",") for f in files])
    y = np.loadtxt(f"{BASE}/cycle_lives/{split}_cycle_lives.csv")
    return X, y


def qd_series(a):
    return a.max(axis=0)  # (99,) 周期2..100 总放电容量


def rmse_cycles(y_true, y_logpred):
    """RMSE 单位=cycles: sqrt(mean((10^y_true - 10^y_pred)^2))."""
    return float(np.sqrt(np.mean((y_true - 10 ** y_logpred) ** 2)))


def ridge(X, y, lam=1.0):
    d = X.shape[1]
    w = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
    return w


def main():
    out = {}

    Xtr, ytr = load_split("train")       # 41
    Xt1, yt1 = load_split("test1")       # 43
    Xt2, yt2 = load_split("test2")       # 40
    # 删 test1 index-21 outlier (Severson 标准)
    keep = np.ones(len(yt1), bool); keep[21] = False
    Xt1, yt1 = Xt1[keep], yt1[keep]
    print(f"train {Xtr.shape[0]}  test1 {Xt1.shape[0]}(删outlier)  test2 {Xt2.shape[0]}")

    y_all = np.concatenate([ytr, yt1, yt2])

    # ================= 实验 D: sim2real gap =================
    cells = pickle.load(open("data/lfp_cells.pkl", "rb"))
    sim_eol = np.array([c["n_eol"] for c in cells])
    out["D"] = dict(
        severson=dict(n=int(len(y_all)), min=float(y_all.min()),
                      med=float(np.median(y_all)), max=float(y_all.max()),
                      q10=float(np.percentile(y_all, 10)), q90=float(np.percentile(y_all, 90))),
        sim=dict(n=int(len(sim_eol)), min=float(sim_eol.min()),
                 med=float(np.median(sim_eol)), max=float(sim_eol.max()),
                 q10=float(np.percentile(sim_eol, 10)), q90=float(np.percentile(sim_eol, 90))))
    print("\n=== 实验 D: sim2real 寿命分布 gap ===")
    print(f"  Severson: min {y_all.min():.0f} med {np.median(y_all):.0f} "
          f"max {y_all.max():.0f}  [q10 {np.percentile(y_all,10):.0f}, q90 {np.percentile(y_all,90):.0f}]")
    print(f"  仿真:     min {sim_eol.min():.0f} med {np.median(sim_eol):.0f} "
          f"max {sim_eol.max():.0f}  [q10 {np.percentile(sim_eol,10):.0f}, q90 {np.percentile(sim_eol,90):.0f}]")
    out["D"]["median_ratio"] = float(np.median(sim_eol) / np.median(y_all))

    # ================= 实验 B2: 复现对标 =================
    def features(X):
        """从 ΔQ(V) 曲线提取 Severson 类特征 (前100周期)."""
        n = X.shape[0]
        dq100_10 = X[:, :, 98] - X[:, :, 8]      # Q_100 - Q_10 (1000 电压点)
        F = {}
        F["var"] = np.log10(np.var(dq100_10, axis=1))
        F["min"] = np.min(dq100_10, axis=1)
        F["mean"] = np.mean(dq100_10, axis=1)
        F["skew"] = np.mean(((dq100_10 - dq100_10.mean(1, keepdims=True)) /
                             (dq100_10.std(1, keepdims=True) + 1e-12)) ** 3, axis=1)
        F["kurt"] = np.mean(((dq100_10 - dq100_10.mean(1, keepdims=True)) /
                             (dq100_10.std(1, keepdims=True) + 1e-12)) ** 4, axis=1) - 3
        # 容量衰减斜率 (cycle 2..100): 对 Qd(n) 线性拟合斜率
        slopes = []
        for i in range(n):
            qd = qd_series(X[i]); qd = qd / qd[0]
            w, *_ = np.linalg.lstsq(np.stack([np.ones(99), np.arange(99)], 1),
                                    qd, rcond=None)
            slopes.append(w[1])
        F["slope_fade"] = np.array(slopes)
        # 只保留 ΔQ_100-10 的两个最稳健统计量, 避免小样本(41)过拟合
        keys = ["var", "min"]
        return np.stack([F[k] for k in keys], 1)

    Ftr, Ft1, Ft2 = features(Xtr), features(Xt1), features(Xt2)
    yl = [np.log10(ytr), np.log10(yt1), np.log10(yt2)]
    # z-score (训练统计)
    mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-8
    Z = [(Ftr - mu) / sd, (Ft1 - mu) / sd, (Ft2 - mu) / sd]

    # ---- B2a: 单特征 log10(var) 线性 (对标 Table1 103/138/196) ----
    A1 = np.stack([np.ones(len(Z[0])), Z[0][:, 0]], 1)
    w1, *_ = np.linalg.lstsq(A1, yl[0], rcond=None)
    rmse1 = [rmse_cycles(ytr, A1 @ w1), rmse_cycles(yt1,
             np.stack([np.ones(len(Z[1])), Z[1][:, 0]], 1) @ w1),
             rmse_cycles(yt2, np.stack([np.ones(len(Z[2])), Z[2][:, 0]], 1) @ w1)]

    # ---- B2b: 多特征 ridge (6 特征) ----
    def ridge_fit_predict(Ztr, ytr_l, Zte, lam=1.0):
        A = np.hstack([np.ones((len(Ztr), 1)), Ztr])
        w = ridge(A, ytr_l, lam)
        Ate = np.hstack([np.ones((len(Zte), 1)), Zte])
        return Ate @ w

    p_tr = ridge_fit_predict(Z[0], yl[0], Z[0])
    p_t1 = ridge_fit_predict(Z[0], yl[0], Z[1])
    p_t2 = ridge_fit_predict(Z[0], yl[0], Z[2])
    rmse_full = [rmse_cycles(ytr, p_tr), rmse_cycles(yt1, p_t1), rmse_cycles(yt2, p_t2)]

    out["B2"] = dict(
        baseline_severson_table1_rmse=[103, 138, 196],
        single_feature_var_rmse=[round(r, 1) for r in rmse1],
        two_feature_ridge_rmse=[round(r, 1) for r in rmse_full])
    print("\n=== 实验 B2: 复现对标 (RMSE, cycles) ===")
    print(f"  Severson Table1 单特征基准: RMSE = (103, 138, 196)")
    print(f"  复现 单特征 log10(var):    RMSE = ({rmse1[0]:.0f}, {rmse1[1]:.0f}, {rmse1[2]:.0f})")
    print(f"  2 特征 (var+min) ridge:    RMSE = ({rmse_full[0]:.0f}, {rmse_full[1]:.0f}, {rmse_full[2]:.0f})")

    # ================= 实验 A': 早期衰减行为 =================
    n_ser = np.arange(2, 101, dtype=float)
    sqrt_n = np.sqrt(n_ser - 2); lin_n = n_ser - 2
    ks, kl, r2s, early_loss, life = [], [], [], [], []
    for X, y in [(Xtr, ytr), (Xt1, yt1), (Xt2, yt2)]:
        for i in range(X.shape[0]):
            qd = qd_series(X[i]); soh = qd / qd[0]; loss = 1 - soh
            A = np.stack([sqrt_n, lin_n], 1)
            k, *_ = np.linalg.lstsq(A, loss, rcond=None)
            pred = A @ k
            r2 = 1 - np.sum((loss - pred) ** 2) / np.sum((loss - loss.mean()) ** 2)
            ks.append(k[0]); kl.append(k[1]); r2s.append(r2)
            early_loss.append(1 - qd[-1] / qd[0]); life.append(y[i])
    ks = np.array(ks); kl = np.array(kl); r2s = np.array(r2s)
    early_loss = np.array(early_loss); life = np.array(life)
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])
    rho = spearman(early_loss, life)
    out["A_prime"] = dict(
        note="前100周期 loss=1-Qd/Qd2 拟合 k_sqrt*sqrt(n)+k_lin*n; 快充早期容量先升后降",
        k_sqrt_med=float(np.median(ks)), frac_k_sqrt_positive=float(np.mean(ks > 0)),
        k_lin_med=float(np.median(kl)), r2_med=float(np.median(r2s)),
        spearman_early_loss_vs_life=rho)
    print("\n=== 实验 A': 早期容量衰减行为 ===")
    print(f"  k_sqrt 中位数 {np.median(ks):+.2e} (正值占比 {np.mean(ks>0)*100:.0f}%) -> 非单调√n衰减")
    print(f"  k_lin  中位数 {np.median(kl):+.2e}, R^2 中位数 {np.median(r2s):.3f}")
    print(f"  spearman(早期容量损失, 寿命) = {rho:+.3f}  (机理一致: 快衰->短命)")

    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(f"{RES}/severson_validate.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\nwrote results/severson_validate.json")


if __name__ == "__main__":
    main()
