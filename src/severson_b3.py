"""
实验 B3: Severson 重训主实验.

B3a 复现原文 full model (9特征 elastic-net), 对标原文 9.1% test error.
B3b 深度架构 (MLP) 吃同批特征, 对标 elastic-net.

数据: data/severson/revisit-severson-et-al-main/data (124 cell ΔQ(V) + cycle_life)
"""
import numpy as np
import glob, json, os
from sklearn.linear_model import ElasticNet, ElasticNetCV

BASE = "data/severson/revisit-severson-et-al-main/data"
RES = "results"


def load_split(split):
    files = sorted(glob.glob(f"{BASE}/{split}/cell*.csv"),
                   key=lambda p: int(p.rsplit("cell", 1)[1][:-4]))
    X = np.array([np.loadtxt(f, delimiter=",") for f in files])  # (n,1000,99)
    y = np.loadtxt(f"{BASE}/cycle_lives/{split}_cycle_lives.csv")
    return X, y


def extract_features(X):
    """从 ΔQ(V) 曲线提取 Severson 类特征 (原文 9 特征的核心子集)."""
    dq = X[:, :, 98] - X[:, :, 8]                     # ΔQ_100-10 (1000 电压点)
    n = X.shape[0]
    F = []
    F.append(np.log10(np.var(dq, axis=1)))            # var (log)
    F.append(np.min(dq, axis=1))                      # min
    F.append(np.mean(dq, axis=1))                     # mean
    mu = dq.mean(1, keepdims=True); sd = dq.std(1, keepdims=True) + 1e-12
    z = (dq - mu) / sd
    F.append(np.mean(z ** 3, axis=1))                 # skew
    F.append(np.mean(z ** 4, axis=1) - 3)             # kurt
    # 容量衰减斜率 + 截距 (cycle 2..100)
    qd = X.max(axis=1)                                # (n,99) 每 cell 每周期 Qd(n)
    slopes, intercepts = [], []
    for i in range(n):
        q = qd[i] / qd[i][0]
        w, *_ = np.linalg.lstsq(np.stack([np.ones(99), np.arange(99)], 1), q,
                                rcond=None)
        slopes.append(w[1]); intercepts.append(w[0])
    F.append(np.array(slopes))                        # slope_Qd
    F.append(np.array(intercepts))                    # intercept_Qd
    F.append(qd[:, 0])                                # Qd_2
    F.append(qd[:, -1])                               # Qd_100
    return np.stack(F, 1)                             # (n, 10)


def mape(y, p):
    """寿命域 MAPE: y,p 为 log10(life), 反变换后算百分比误差."""
    yl_, pl_ = 10 ** y, 10 ** p
    return 100.0 * np.mean(np.abs(pl_ - yl_) / (np.abs(yl_) + 1e-8))


def rmse(y, p):
    """寿命域 RMSE (cycles): y,p 为 log10(life)."""
    return float(np.sqrt(np.mean((10 ** y - 10 ** p) ** 2)))


def main():
    Xtr, ytr = load_split("train")       # 41
    Xt1, yt1 = load_split("test1")       # 43
    Xt2, yt2 = load_split("test2")       # 40
    keep = np.ones(len(yt1), bool); keep[21] = False
    Xt1, yt1 = Xt1[keep], yt1[keep]
    yl = [np.log10(ytr), np.log10(yt1), np.log10(yt2)]

    Ftr, Ft1, Ft2 = extract_features(Xtr), extract_features(Xt1), extract_features(Xt2)
    # z-score (训练统计)
    mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-8
    Z = [(Ftr - mu) / sd, (Ft1 - mu) / sd, (Ft2 - mu) / sd]

    # ---------- B3a: elastic-net (复现 full model) ----------
    cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                      alphas=np.logspace(-1.5, 0.5, 30), cv=5, random_state=0)
    cv.fit(Z[0], yl[0])
    en = ElasticNet(alpha=cv.alpha_, l1_ratio=cv.l1_ratio_, max_iter=20000)
    en.fit(Z[0], yl[0])
    p1, p2, ptr = en.predict(Z[1]), en.predict(Z[2]), en.predict(Z[0])

    rmse_en = [rmse(yl[0], ptr), rmse(yl[1], p1), rmse(yl[2], p2)]
    mape_en = [mape(yl[0], ptr), mape(yl[1], p1), mape(yl[2], p2)]
    n_nz = int(np.sum(np.abs(en.coef_) > 1e-8))
    print("=== B3a: elastic-net (复现 full model) ===")
    print(f"  alpha={cv.alpha_:.4f} l1_ratio={cv.l1_ratio_:.2f} 非零特征 {n_nz}/10")
    print(f"  RMSE(train,test1,test2) = ({rmse_en[0]:.0f}, {rmse_en[1]:.0f}, {rmse_en[2]:.0f})")
    print(f"  MAPE(test1) = {mape_en[1]:.1f}%  (原文 full model 9.1%)")

    # ---------- B3b: MLP 深度架构 ----------
    import torch, torch.nn as nn
    torch.set_num_threads(4)
    torch.manual_seed(0)

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(64, 1))
        def forward(self, x): return self.net(x).squeeze(-1)

    def train_mlp(Ztr, ytr, Zte):
        d = Ztr.shape[1]
        m = MLP(d); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
        Xt = torch.tensor(Ztr, dtype=torch.float32)
        yt = torch.tensor(ytr, dtype=torch.float32)
        Xte = torch.tensor(Zte, dtype=torch.float32)
        for ep in range(3000):
            m.train(); opt.zero_grad()
            loss = nn.functional.mse_loss(m(Xt), yt)
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            return m(Xte).numpy()

    p1m = train_mlp(Z[0], yl[0], Z[1]); p2m = train_mlp(Z[0], yl[0], Z[2])
    p0m = train_mlp(Z[0], yl[0], Z[0])
    rmse_mlp = [rmse(yl[0], p0m), rmse(yl[1], p1m), rmse(yl[2], p2m)]
    mape_mlp = [mape(yl[0], p0m), mape(yl[1], p1m), mape(yl[2], p2m)]
    print("\n=== B3b: MLP 深度架构 (同特征) ===")
    print(f"  RMSE(train,test1,test2) = ({rmse_mlp[0]:.0f}, {rmse_mlp[1]:.0f}, {rmse_mlp[2]:.0f})")
    print(f"  MAPE(test1) = {mape_mlp[1]:.1f}%")

    out = dict(
        baseline_severson_single=[103, 138, 196],
        baseline_severson_full_mape_9p1=9.1,
        elastic_net=dict(alpha=float(cv.alpha_), l1_ratio=float(cv.l1_ratio_),
                         n_nonzero=n_nz, rmse=rmse_en, mape_test1=mape_en[1]),
        mlp=dict(rmse=rmse_mlp, mape_test1=mape_mlp[1]))
    json.dump(out, open(f"{RES}/severson_b3.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote {RES}/severson_b3.json")


if __name__ == "__main__":
    main()
