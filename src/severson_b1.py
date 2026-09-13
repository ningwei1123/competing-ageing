"""
实验 B1: 仿真训练的 Transformer-only (7通道) 零样本迁移到 Severson 真实轨迹.
"""
import sys, os, pickle, json
import numpy as np
import torch

sys.path.insert(0, "src")
from model import AblationTF
import run_experiments as RX

N_EARLY = 100
RES = "results"


def build_severson_seq(cells):
    """Severson 前100周期 7 通道 (与 lfp_dataset 同语义)."""
    seqs = []
    for c in cells:
        qd = c["qd"]; ir = c["ir"]; tavg = c["tavg"]
        n = min(N_EARLY, len(qd))
        if n < 10:
            continue
        irv = ir.copy().astype(float); irv[irv <= 0] = np.nan
        ir0 = np.nanmedian(irv[:5])
        if not np.isfinite(ir0) or ir0 <= 0:
            continue
        soh = qd[:n] / qd[0]
        r_rel = np.nan_to_num(irv[:n] / ir0, nan=1.0)
        Tn = tavg[:n] / 60.0
        dQdV = 0.62 * np.exp(-1.35 * (1 - soh))
        dv_pol = 0.055 * r_rel * 4.0
        ce = 0.9985 - 0.0022 * (1 - soh)
        chg_cap = soh
        seqs.append(np.stack([soh, r_rel, Tn, dQdV, dv_pol, ce, chg_cap], 1))
    return seqs


def severson_cycle_life(c, thresh=0.88):
    idx = np.where(c["qd"] < thresh)[0]
    return int(idx[0]) if len(idx) else None


def main():
    torch.set_num_threads(4)
    cells = pickle.load(open("data/lfp_cells.pkl", "rb"))
    print(f"仿真 cells: {len(cells)}")

    kf = RX._kfold(len(cells), 5, seed=7)
    tr, te = kf[0]

    # ---------- 训练 (5折取最佳) ----------
    print("训练 AblationTF (5折) ...")
    best_state, best_r2 = None, -9
    best_tmu = best_tsd = best_smu = best_ssd = None
    for f, (trf, tef) in enumerate(kf):
        Fn, Yn, mu, sd, tmu, tsd, seqs = RX.make_tensors(
            cells, stats_from=[cells[i] for i in trf])
        # seq 的归一化统计 (make_tensors 内部用了, 但没返回; 这里重算)
        raw_seq = np.concatenate([cells[i]["seq"] for i in trf], 0)
        smu, ssd = raw_seq.mean(0), raw_seq.std(0) + 1e-8
        RX.TMU_G, RX.TSD_G = tmu, tsd
        model, m, h, ex = RX.train_one(
            lambda: AblationTF(), cells, Fn, seqs, Yn, trf, tef,
            epochs=150, bs=16, verbose=False, seed=100 + f, patience=20)
        r2 = m["_macro"]["R2"]
        print(f"  fold{f+1}: macro R2={r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_tmu, best_tsd = tmu, tsd
            best_smu, best_ssd = smu, ssd
    print(f"  最佳 fold macro R2={best_r2:.4f}")

    # ---------- Severson seq ----------
    import severson_struct_validate as sv
    sev_cells = sv.parse_struct("data/severson_raw/2018-02-20_v2.mat")
    sev_seq = build_severson_seq(sev_cells)
    valid = []
    for s, c in zip(sev_seq, sev_cells):
        cl = severson_cycle_life(c)
        if cl is not None:
            valid.append((s, cl))
    print(f"Severson 有效 cell: {len(valid)}")

    # ---------- 零样本推理 ----------
    model = AblationTF()
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = []
        for s, cl in valid:
            seq = torch.tensor((s - best_smu) / best_ssd,
                               dtype=torch.float32).unsqueeze(0)
            batch = (None, None, None, None, None, None, 1, seq)
            pred, _ = model(batch)
            preds.append(pred.numpy()[0])
    preds = np.array(preds) * best_tsd + best_tmu
    rul_true = np.array([cl - N_EARLY for s, cl in valid])
    rul_pred = preds[:, 0]

    ss_res = np.sum((rul_true - rul_pred) ** 2)
    ss_tot = np.sum((rul_true - rul_true.mean()) ** 2) + 1e-12
    r2_rul = float(1 - ss_res / ss_tot)
    mape_rul = float(np.mean(np.abs(rul_pred - rul_true) /
                             (np.abs(rul_true) + 1e-8)) * 100)

    out = dict(sim_fold_r2=best_r2, n_severson=len(valid),
               r2_rul_zeroshot=r2_rul, mape_rul_zeroshot=mape_rul,
               rul_true_med=float(np.median(rul_true)),
               rul_pred_med=float(np.median(rul_pred)))
    print("\n=== 实验 B1: 零样本 sim2real ===")
    print(f"  仿真训练 fold R2 = {best_r2:.4f}")
    print(f"  Severson 零样本 RUL: R2 = {r2_rul:+.3f}, MAPE = {mape_rul:.1f}%")
    print(f"  真实 RUL 中位 {np.median(rul_true):.0f} -> 预测 {np.median(rul_pred):.0f}")
    json.dump(out, open(f"{RES}/severson_b1.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"wrote {RES}/severson_b1.json")


if __name__ == "__main__":
    main()
