"""
Severson struct 解析 + 实验 A(√n律完整拟合) + C(IR分配律) + B1(7通道零样本).

输入: data/severson_raw/2018-02-20_v2.mat (v7.3, batch1+2 ~96 cell)
      逐周期 summary: QDischarge / IR / Tavg / Tmax / Tmin / QCharge / cycle
"""
import h5py
import numpy as np
import json, os

RES = "results"


def parse_struct(path, verbose=False):
    """返回 list[dict], 每个 cell 的逐周期 summary 序列."""
    f = h5py.File(path, "r")
    batch = f["batch"]
    n = batch["summary"].shape[0]
    cells = []
    for i in range(n):
        ref = batch["summary"][i, 0]
        sg = f[ref]
        qd = np.array(sg["QDischarge"]).ravel().astype(float)
        ir = np.array(sg["IR"]).ravel().astype(float)
        tavg = np.array(sg["Tavg"]).ravel().astype(float)
        tmax = np.array(sg["Tmax"]).ravel().astype(float)
        tmin = np.array(sg["Tmin"]).ravel().astype(float)
        qc = np.array(sg["QCharge"]).ravel().astype(float)
        cyc = np.array(sg["cycle"]).ravel().astype(float)
        cells.append(dict(qd=qd, ir=ir, tavg=tavg, tmax=tmax, tmin=tmin,
                          qc=qc, cycle=cyc, n=len(qd)))
    f.close()
    if verbose:
        print(f"  parsed {n} cells; cycle counts: "
              f"min {min(c['n'] for c in cells)}, "
              f"med {int(np.median([c['n'] for c in cells]))}, "
              f"max {max(c['n'] for c in cells)}")
    return cells


def cycle_life(cell, thresh=0.88):
    """寿命 = SOH 首次 < thresh (0.88 Ah = 80% of 1.1) 的周期; 未达则 censored."""
    qd = cell["qd"]
    soh = qd / qd[0]
    idx = np.where(soh < thresh)[0]
    if len(idx) == 0:
        return None  # censored
    return int(idx[0])


def fit_sqrt_linear(cell, n_max=None):
    """拟合 SOH 损失 loss(n)=1-SOH(n)=k_s*sqrt(n)+k_l*n, 到 n_max (默认到80%)."""
    qd = cell["qd"]
    soh = qd / qd[0]
    loss = 1.0 - soh
    cl = cycle_life(cell)
    if n_max is None:
        n_max = cl if cl is not None else len(qd)
    n_max = min(n_max, len(qd))
    n = np.arange(1, n_max + 1, dtype=float)
    sqrt_n = np.sqrt(n); lin_n = n
    A = np.stack([sqrt_n, lin_n], 1)
    k, *_ = np.linalg.lstsq(A, loss[:n_max], rcond=None)
    pred = A @ k
    ss_res = np.sum((loss[:n_max] - pred) ** 2)
    ss_tot = np.sum((loss[:n_max] - loss[:n_max].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return k[0], k[1], r2, n_max


def main():
    import glob
    p = glob.glob("data/severson_raw/2018-02-20_v2.mat")
    if not p or os.path.getsize(p[0]) < 1.5e9:
        print("2GB struct 未就绪, 退出")
        return
    p = p[0]
    print(f"解析 {p} ...")
    cells = parse_struct(p, verbose=True)

    # ============ 实验 A: √n 律拟合完整 Qd 到 80% ============
    print("\n=== 实验 A: √n 律拟合完整容量衰减到 80% ===")
    ks, kl, r2s, lifes = [], [], [], []
    n_censored = 0
    for c in cells:
        cl = cycle_life(c)
        if cl is None:
            n_censored += 1
            continue
        k, l, r2, nm = fit_sqrt_linear(c)
        ks.append(k); kl.append(l); r2s.append(r2); lifes.append(cl)
    ks = np.array(ks); kl = np.array(kl); r2s = np.array(r2s)
    lifes = np.array(lifes)
    outA = dict(
        n_cells=len(cells), n_censored=n_censored, n_fit=len(ks),
        k_sqrt_med=float(np.median(ks)),
        k_sqrt_q10=float(np.percentile(ks, 10)),
        k_sqrt_q90=float(np.percentile(ks, 90)),
        frac_k_sqrt_positive=float(np.mean(ks > 0)),
        k_lin_med=float(np.median(kl)),
        r2_med=float(np.median(r2s)),
        r2_q10=float(np.percentile(r2s, 10)))
    print(f"  拟合 cell 数 {len(ks)} (删失 {n_censored})")
    print(f"  k_sqrt 中位 {np.median(ks):+.3e} (q10 {np.percentile(ks,10):+.3e}, "
          f"q90 {np.percentile(ks,90):+.3e})  正值占比 {np.mean(ks>0)*100:.0f}%")
    print(f"  k_lin 中位 {np.median(kl):+.3e}   R^2 中位 {np.median(r2s):.3f} "
          f"(q10 {np.percentile(r2s,10):.3f})")

    # 早期 vs 完整拟合: 前100周期 k_sqrt 是否仍为负?
    k100 = []
    for c in cells:
        if len(c["qd"]) >= 100:
            k, l, r2, nm = fit_sqrt_linear(c, n_max=100)
            k100.append(k)
    k100 = np.array(k100)
    outA["k_sqrt_first100_med"] = float(np.median(k100))
    outA["frac_k_sqrt_first100_pos"] = float(np.mean(k100 > 0))
    print(f"  前100周期 k_sqrt 中位 {np.median(k100):+.3e} (正值占比 "
          f"{np.mean(k100>0)*100:.0f}%)  <- 对比完整拟合")

    # ============ 实验 C: IR 分配律 (LLI vs CL 份额) ============
    print("\n=== 实验 C: IR 演化与两通道分解 ===")
    # IR(n) 相对上升; 早期斜率(界面主导) vs 后期斜率(接触损失主导)
    ir_slope_early, ir_slope_late, ir_ratio = [], [], []
    for c in cells:
        ir = c["ir"]
        valid = ir > 0
        ir = ir[valid]
        if len(ir) < 200:
            continue
        ir_rel = ir / ir[0]
        n = len(ir)
        # 早期 (前20%寿命) 斜率 vs 后期 (后20%) 斜率
        n_early = max(50, int(0.2 * n))
        n_late_start = int(0.8 * n)
        se = np.polyfit(np.arange(n_early), ir_rel[:n_early], 1)[0]
        sl = np.polyfit(np.arange(n_late_start, n), ir_rel[n_late_start:], 1)[0]
        ir_slope_early.append(se); ir_slope_late.append(sl)
        ir_ratio.append(ir_rel[-1])
    ir_slope_early = np.array(ir_slope_early); ir_slope_late = np.array(ir_slope_late)
    ir_ratio = np.array(ir_ratio)
    outC = dict(
        ir_slope_early_med=float(np.median(ir_slope_early)),
        ir_slope_late_med=float(np.median(ir_slope_late)),
        frac_late_faster=float(np.mean(ir_slope_late > ir_slope_early)),
        ir_end_ratio_med=float(np.median(ir_ratio)))
    print(f"  IR 早期斜率中位 {np.median(ir_slope_early):+.3e}, "
          f"后期斜率中位 {np.median(ir_slope_late):+.3e}")
    print(f"  后期加速占比 {np.mean(ir_slope_late>ir_slope_early)*100:.0f}% "
          f"(接触损失后期加速的指纹)")
    print(f"  IR 终点相对值中位 {np.median(ir_ratio):.3f}")

    out = dict(A=outA, C=outC)
    json.dump(out, open(f"{RES}/severson_struct_validate.json", "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nwrote {RES}/severson_struct_validate.json")


if __name__ == "__main__":
    main()
