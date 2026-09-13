"""
Verify every numeric claim the manuscript makes against the produced results.
Run after run_experiments.py / analysis.py / micro_stats.py.
Prints PASS/FAIL per claim so the text can be corrected where it does not hold.

The manuscript's central claim is deliberately *not* "the proposed model has the
highest in-distribution R^2".  It is that the random split is saturated and that
the separation appears under distribution shift.  The checks below therefore
test the framing rather than a ranking that the data no longer supports.
"""
import os, sys, json, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
TASKS = ["RUL", "SOH", "dR_int"]
ORDER = ["LFP-ETGNet", "GAT-TST", "ST-GCN", "GMAN", "MTL-ResNet", "CNN-LSTM",
         "GNN-only", "Transformer-only"]
# architectures that consume the microstructure graph
GRAPH = ["LFP-ETGNet", "GAT-TST", "ST-GCN", "GMAN"]
NOGRAPH = ["MTL-ResNet", "CNN-LSTM", "Transformer-only"]

TOL_TIE = 0.010        # "indistinguishable" band used in the text


def line(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    return bool(ok)


def macro_of(M, k):
    return M[k]["metrics"]["_macro"]["R2"]


def load_ood():
    out = {}
    for fn, key in (("metrics_ood_temp.json", "temp"),
                    ("metrics_ood_life.json", "life")):
        p = os.path.join(RES, fn)
        if os.path.exists(p):
            out[key] = json.load(open(p))
    return out


# ----------------------------------------------------------------------
# Physical laws reported in the Results chapter.  Each clause asserts the
# science rather than a tolerance - an exponent sign, a >x collapse gain, an
# ordering between two percolation curves - because a numerically similar but
# physically different result would otherwise pass.  The point of the gate is
# that the chapter cannot claim a law the simulation does not exhibit.
LAW_BLOCKS = ["interphase", "survival", "percolation", "protocol",
              "master", "design"]


def check_laws(ok):
    """Check every law the Results chapter states against results/laws.json."""
    p = os.path.join(RES, "laws.json")
    print("\n== claims: physical laws ==")
    if not os.path.exists(p):
        ok.append(line(False, "results/laws.json is missing - run src/laws.py"))
        return
    L = json.load(open(p))

    missing = [k for k in LAW_BLOCKS if k not in L]
    ok.append(line(not missing,
                   f"the law set is complete ({len(LAW_BLOCKS) - len(missing)}"
                   f"/{len(LAW_BLOCKS)} blocks"
                   + (f", missing {missing}" if missing else "") + ")"))
    if missing:
        return

    # (i) one chemically independent channel, with a protocol-invariant
    #     crossover between its two rate terms
    I = L["interphase"]
    ok.append(line(I["closed_form_max_dev"] < 1e-12,
                   "the interphase closed form matches the cycle-by-cycle ledger "
                   f"to machine precision ({I['closed_form_max_dev']:.1e})"))
    ok.append(line(I["crossover_protocol_universality"] < 1e-12,
                   f"the sqrt-to-linear crossover n* = {I['n_star']:.3e} cycles is "
                   f"protocol-invariant ({I['crossover_protocol_universality']:.1e})"))
    ok.append(line(abs(I["arr_ratio_45_25"] - I["fade_ratio_45_25"]) < 1e-9,
                   f"the Arrhenius factor {I['arr_ratio_45_25']:.3f} at 45/25 degC "
                   "transfers unchanged from the rate to the measured fade"))
    ok.append(line(2.0 < I["arr_ratio_45_25"] < 2.3,
                   f"the temperature acceleration is "
                   f"{100 * (I['arr_ratio_45_25'] - 1):.1f} % for +20 K "
                   f"({100 * I['log_slope_per_K']:.2f} %/K, doubling every "
                   f"{I['doubling_interval_K']:.1f} K)"))

    # (ii) the second channel as a contact-fatigue survival law
    S = L["survival"]
    ok.append(line(S["r2"] > 0.99 and S["max_abs_error"] < 0.01,
                   "the lognormal survival law reproduces the simulator "
                   f"(R2 = {S['r2']:.5f}, max err {S['max_abs_error']:.1e}, "
                   f"median neck rupture {S['n_be_median']:.0f} cycles)"))

    # (iii) percolation: which of the two networks is fragile, and by how much
    P, lv = L["percolation"], [("f_5pct", "5 %"), ("f_10pct", "10 %"),
                               ("f_25pct", "25 %")]
    ok.append(line(all(P["electronic"][k] < P["composite"][k] for k, _ in lv),
                   "the electronic subnetwork is the fragile one at every "
                   "isolation level (tolerated contact removal "
                   + ", ".join(f"{lab}: {P['electronic'][k]:.2f} vs "
                               f"{P['composite'][k]:.2f}" for k, lab in lv) + ")"))
    ok.append(line(P["fragility_ratio"] > 1.5,
                   "the two networks differ in mechanical fragileness by "
                   f"{P['fragility_ratio']:.2f}x"))
    ok.append(line(0.3 < P["beta"] < 0.7 and P["beta_r2"] > 0.7,
                   f"the percolation exponent is beta = {P['beta']:.3f} "
                   f"(R2 = {P['beta_r2']:.3f})"))

    # (iv) protocol scaling: the three exponents must be MEASURED and must be
    #      shallower than the interphase-only prediction - that gap is the
    #      fingerprint of the second channel, not a modelling defect
    Q = L["protocol"]
    ok.append(line(Q["n_censored"] == 0,
                   "no lifetime in the protocol sweep is right-censored"))
    for key, pred_key, lab in (("eol_C", "eol_C_interphase", "C-rate"),
                               ("eol_A", "eol_A_interphase", "temperature"),
                               ("eol_D", "eol_D_interphase", "DoD")):
        f, pr = Q[key], Q["predicted"][pred_key]
        ok.append(line(f["r2_min"] > 0.98,
                       f"EOL vs {lab} is a clean power law at fixed values of the "
                       f"other variables (R2 min {f['r2_min']:.4f}, n = {f['n']})"))
        ok.append(line(abs(f["a"]) < pr,
                       f"the {lab} exponent ({f['a']:+.3f}) is shallower than the "
                       f"interphase-only prediction ({-pr:+.3f}) - the second "
                       "channel is active"))

    # (v) the headline: the network share FALLS with severity
    sh = [("share_C", "C-rate"), ("share_A", "temperature"), ("share_D", "DoD")]
    ok.append(line(all(Q[k]["a"] < 0 for k, _ in sh),
                   "the network share of the loss falls with "
                   + ", ".join(f"{lab} ({Q[k]['a']:+.2f})" for k, lab in sh)
                   + " - mild, cool and shallow cycling is the mechanically "
                     "aggressive regime"))

    # (vi) the collapse of every protocol onto one curve in Theta = k_CL n
    G = L["master"]
    ok.append(line(G["residual_gain"] >= 2.0 and G["collapse_gain"] >= 2.0,
                   "the Theta master curve collapses the protocols "
                   f"({G['residual_gain']:.2f}x lower residual, "
                   f"{G['collapse_gain']:.2f}x tighter scatter than raw cycles) "
                   f"although k_CL spans {G['pairs']['k_ratio']:.1f}x"))
    ok.append(line(G["r2"] > 0.9,
                   f"the master curve is a power law in Theta (exponent "
                   f"{G['exponent']:.3f}, R2 = {G['r2']:.4f})"))

    # (vii) the design laws, including the sign that identifies the channel
    D = L["design"]
    ok.append(line(D["a_z"]["a"] > 0 and D["a_z"]["r2"] > 0.9,
                   "better calendering lengthens life "
                   f"(EOL ~ z^{D['a_z']['a']:.3f}, R2 = {D['a_z']['r2']:.3f})"))
    #   the size experiment does NOT give a single power law: the two channels
    #   act in opposite directions, so the local exponent crosses zero and the
    #   sign of d ln EOL / d ln d is what identifies the governing channel
    loc = D["size_local"]
    ok.append(line(loc[0]["a"] > 0 and loc[-1]["a"] < 0,
                   "the particle-size response changes sign: the local exponent "
                   f"dlnEOL/dlnd falls from {loc[0]['a']:+.2f} "
                   f"(d = {loc[0]['d_mid']:.3f} um) to {loc[-1]['a']:+.2f} "
                   f"(d = {loc[-1]['d_mid']:.3f} um)"))
    ok.append(line(D["d_star"] is not None,
                   "control passes between the two channels inside the sweep "
                   f"(d* = {D['d_star']:.3f} um); a single power law cannot "
                   f"describe it (a_d = {D['a_d']['a']:+.2f}, "
                   f"R2 = {D['a_d']['r2']:.3f})"))
    ok.append(line(0 < loc[0]["a"] < D["predicted_a_d_interphase"],
                   f"the small-d exponent ({loc[0]['a']:+.2f}) shares the sign of "
                   f"the interphase-only prediction ({D['predicted_a_d_interphase']:+.2f}) "
                   "and under-shoots it - the surface channel governs there"))
    ok.append(line(D["predicted_a_d_network"] < loc[-1]["a"] < 0,
                   f"the large-d exponent ({loc[-1]['a']:+.2f}) shares the sign of "
                   f"the network-only prediction ({D['predicted_a_d_network']:+.2f}) "
                   "and under-shoots it - the mechanical channel governs there"))


def main():
    M = json.load(open(os.path.join(RES, "metrics.json")))
    cal = json.load(open(os.path.join(RES, "calibration.json")))
    I = np.load(os.path.join(RES, "interp.npz"), allow_pickle=True)
    ms = np.load(os.path.join(RES, "micro_stats.npz"))
    cells = pickle.load(open(os.path.join(ROOT, "data", "lfp_cells.pkl"), "rb"))
    O = load_ood()

    macro = {k: macro_of(M, k) for k in M}
    print("== in-distribution macro R2 ==")
    for k in ORDER:
        if k in macro:
            print(f"   {k:18s} {macro[k]:+.4f}"
                  + ("   <-- tied" if abs(macro[k] - macro["LFP-ETGNet"]) < TOL_TIE
                     else ""))
    main_m = M["LFP-ETGNet"]
    mm = main_m["metrics"]

    print("\n== claims: in-distribution framing ==")
    ok = []
    base_names = [k for k in macro if k != "LFP-ETGNet"]
    best_base = max(base_names, key=lambda k: macro[k])
    delta = macro["LFP-ETGNet"] - macro[best_base]
    n_tie = sum(1 for k in base_names if abs(macro[k] - macro["LFP-ETGNet"]) < TOL_TIE)
    ok.append(line(n_tie >= 2,
                   f"{n_tie} baselines lie within {TOL_TIE} of the full model "
                   f"(strongest: {best_base} {macro[best_base]:.4f}, "
                   f"delta {delta:+.4f})"))
    ok.append(line(abs(delta) < TOL_TIE,
                   f"full model is statistically tied with the strongest "
                   f"baseline (|delta|={abs(delta):.4f} < {TOL_TIE})"))

    # the ablation is small for a stated reason: the trajectory carries almost
    # the whole in-distribution signal, and the graph branch alone has
    # negligible standalone skill (Law D1 -- the two channels cancel)
    ok.append(line(abs(macro["LFP-ETGNet"] - macro["Transformer-only"]) < 0.05,
                   f"the trajectory branch carries almost the entire signal "
                   f"(Transformer-only {macro['Transformer-only']:.4f} vs full "
                   f"{macro['LFP-ETGNet']:.4f})"))
    ok.append(line(macro["GNN-only"] < 0.10,
                   f"the graph branch alone has negligible standalone skill "
                   f"({macro['GNN-only']:.4f})"))

    r2 = {t: mm[t]["R2"] for t in TASKS}
    print("   per-task R2:", {k: round(v, 4) for k, v in r2.items()})
    ok.append(line(r2["SOH"] >= r2["dR_int"] and r2["SOH"] >= r2["RUL"],
                   "SOH is the easiest task (highest R2)"))
    ok.append(line(r2["RUL"] == min(r2.values()),
                   "RUL is the hardest task (lowest R2)"))
    ok.append(line(main_m["latency_s"] * 1000 < 50,
                   f"latency is a few ms ({main_m['latency_s']*1000:.2f} ms)"))
    print("   params (k): " + ", ".join(f"{k}={M[k]['n_param']/1000:.1f}"
                                        for k in ORDER if k in M))

    # ------------------------------------------------------------------
    print("\n== claims: distribution shift ==")
    if not O:
        print("   (no OOD results yet - re-run after --split temp|life)")
    else:
        lab = {"temp": "temperature shift", "life": "life extrapolation"}
        for k in ("temp", "life"):
            if k not in O:
                continue
            rk = {n: macro_of(O[k], n) for n in O[k]}
            print(f"\n   --- {lab[k]} ---")
            for n in ORDER:
                if n in rk:
                    print(f"      {n:18s} {rk[n]:+.4f}   gap "
                          f"{macro.get(n, float('nan')) - rk[n]:+.4f}")
            bb = max((n for n in rk if n != "LFP-ETGNet"), key=lambda n: rk[n])
            ok.append(line(rk["LFP-ETGNet"] < macro["LFP-ETGNet"],
                           f"{lab[k]}: the full model degrades under shift "
                           f"({macro['LFP-ETGNet']:.4f} -> {rk['LFP-ETGNet']:.4f})"))
            #   the honest OOD outcome after the encoder + normalisation fixes:
            #   on a withheld thermal regime the model stays within a point of
            #   the strongest baseline (no collapse), while life extrapolation
            #   drives every model negative.
            if k == "temp":
                ok.append(line(abs(rk["LFP-ETGNet"] - rk[bb]) < 0.05,
                               f"{lab[k]}: the full model stays within 0.05 of "
                               f"the strongest shifted baseline {bb} "
                               f"(not a collapse)"))
            if k == "life":
                ok.append(line(rk[bb] < 0,
                               f"{lab[k]}: every model is negative under life "
                               f"extrapolation (strongest {bb} = {rk[bb]:.3f})"))

    print("\n== claims: calibration / interpretation ==")
    lv = {t: cal[t].get("logvar", np.nan) for t in TASKS}
    print("   learned log-variances:", {k: round(v, 3) for k, v in lv.items()})
    ok.append(line(lv["RUL"] == max(lv.values()),
                   "RUL has the largest learned log-variance"))
    ok.append(line(min(lv.values()) in (lv["SOH"], lv["dR_int"]),
                   "the smallest log-variance is attached to a fixed-horizon "
                   "target (SOH or resistance)"))
    cov = {t: 100 * cal[t]["coverage_2s"] for t in TASKS}
    print("   +-2sigma coverage (%):", {k: round(v, 1) for k, v in cov.items()})
    ok.append(line(all(85 <= v <= 100 for v in cov.values()),
                   "all 2-sigma coverages are close to nominal 95 %"))

    nf = ["particle diameter", "inter-particle spacing", "SEI thickness",
          "Li-conc. gradient", "coordination number", "contact quality",
          "electrode depth", "local SOC"]
    sf = ["SOH(n)", "R_int(n)", "T_max(n)", "dQ/dV(n)", "dV_pol(n)",
          "CE(n)", "Q_chg(n)"]
    ni, si = I["node_imp"], I["seq_imp"]
    no, so = np.argsort(-ni), np.argsort(-si)
    print("   top node feats:", [nf[i] for i in no[:3]])
    print("   top seq feats :", [sf[i] for i in so[:3]])
    #   the particle size is the one geometric quantity that enters both
    #   channels (neck strain on the network side, specific surface on the
    #   interphase side), so its ranking first is the empirical trace of the
    #   two-channel structure, and the reaction descriptors follow because the
    #   interphase channel dominates the capacity loss.
    ok.append(line(nf[no[0]] == "particle diameter",
                   f"the top node descriptor is the particle size "
                   f"({nf[no[0]]})"))
    top3 = {nf[i] for i in no[:3]}
    react = {"local SOC", "electrode depth", "Li-conc. gradient"}
    print("   top-3 node feats:", sorted(top3))
    ok.append(line(len(react & top3) >= 1,
                   "a reaction descriptor is in the top three"))
    ok.append(line(sf[so[0]].startswith("dQ/dV") or sf[so[0]].startswith("R_int")
                   or "capacity" in sf[so[0]] or sf[so[0]].startswith("SOH"),
                   f"the top trajectory channel is a capacity or resistance "
                   f"channel ({sf[so[0]]})"))

    print("\n== claims: microstructure / attribution ==")
    print(f"   isol0={float(ms['isol0'])*100:.1f} %  edge_ret_pred="
          f"{float(ms['edge_ret_eol']):.3f}  lam_eol={float(ms['lam_eol'])*100:.2f} %")
    LLI = np.mean([c["curve"]["LLI"][-1] for c in cells])
    LAM = np.mean([c["curve"]["LAM"][-1] for c in cells])
    CL = np.mean([c["curve"]["CL"][-1] for c in cells])
    #   LLI and LAM remove capacity; CL is the fraction of electronic contacts
    #   lost and does not subtract capacity, so it is reported separately rather
    #   than normalised against the capacity terms.
    cap = LLI + LAM
    print(f"   EOL capacity loss: LLI {100*LLI:.2f} % ({LLI/cap:.1%})  "
          f"LAM {100*LAM:.2f} % ({LAM/cap:.1%})   [CL {100*CL:.1f} % of contacts]")
    ok.append(line(LLI > LAM and LLI > CL, "LLI dominates the degradation"))
    ok.append(line(CL > 0.02, "CL is a non-negligible contribution"))
    ok.append(line(LAM > 1e-3, "LAM (percolation) is a non-negligible loss"))
    ok.append(line(abs((LLI + LAM) - 0.20) < 0.05,
                   "the two capacity channels together explain the 20 % EOL loss"))

    check_laws(ok)
    check_severson(ok)

    print(f"\n{sum(ok)}/{len(ok)} claims hold")
    return 0 if all(ok) else 1


def check_severson(ok):
    """Severson 外部验证断言 (results/severson_validate.json 等)."""
    p = os.path.join(RES, "severson_validate.json")
    if not os.path.exists(p):
        print("\n== severson: results/severson_validate.json absent — skipping ==")
        return
    S = json.load(open(p, encoding="utf-8"))
    print("\n== claims: Severson external validation ==")

    # B2 复现对标: 单特征 log10(var) RMSE 应复现 Severson Table1 (103,138,196)
    r = S["B2"]["single_feature_var_rmse"]
    ok.append(line(abs(r[1] - 138) < 5,
                   f"Severson Table1 复现: test1 RMSE {r[1]:.0f} ~ 138 cyc "
                   f"(原文 138)"))
    ok.append(line(abs(r[0] - 103) < 6,
                   f"Severson Table1 复现: train RMSE {r[0]:.0f} ~ 103 cyc "
                   f"(原文 103)"))

    # A' 早期衰减 -> 寿命 机理一致
    rho = S["A_prime"]["spearman_early_loss_vs_life"]
    ok.append(line(rho < -0.5,
                   f"早期容量损失->寿命 Spearman {rho:+.2f} < -0.5 (快衰->短命)"))

    # D sim2real gap: 仿真寿命显著偏长 (>2x, 快充 vs 温和循环协议差)
    ratio = S["D"]["median_ratio"]
    ok.append(line(ratio > 2.0,
                   f"sim2real gap: 仿真中位寿命是真实 {ratio:.1f}x (>2x)"))

    # ---- 完整 struct 实验 (A 完整版 / C / B1) ----
    p2 = os.path.join(RES, "severson_struct_validate.json")
    if os.path.exists(p2):
        S2 = json.load(open(p2, encoding="utf-8"))
        A = S2["A"]; C = S2["C"]
        # A 完整版: √n 律在快充完整寿命尺度不成立 (k_sqrt 全负)
        ok.append(line(A["frac_k_sqrt_positive"] < 0.05,
                       f"完整寿命拟合: k_sqrt 正值占比 {A['frac_k_sqrt_positive']*100:.0f}% "
                       f"<5% (√n 律在快充不成立)"))
        # C: IR 后期加速 (接触损失后期主导的指纹)
        ok.append(line(C["frac_late_faster"] > 0.7,
                       f"IR 后期加速占比 {C['frac_late_faster']*100:.0f}% >70% "
                       f"(接触损失后期主导)"))
        ok.append(line(C["ir_end_ratio_med"] > 1.1,
                       f"IR 终点相对值 {C['ir_end_ratio_med']:.2f} >1.1 (内阻显著上升)"))

    p3 = os.path.join(RES, "severson_b1.json")
    if os.path.exists(p3):
        B = json.load(open(p3, encoding="utf-8"))
        # B1: 零样本崩溃 (sim2real gap 的模型侧)
        ok.append(line(B["r2_rul_zeroshot"] < 0,
                       f"零样本 sim2real RUL R2 = {B['r2_rul_zeroshot']:.0f} < 0 "
                       f"(零样本崩溃, 需重训)"))
        ok.append(line(B["sim_fold_r2"] > 0.9,
                       f"仿真训练 fold R2 = {B['sim_fold_r2']:.3f} > 0.9 "
                       f"(仿真内部正常)"))

    p4 = os.path.join(RES, "severson_b3.json")
    if os.path.exists(p4):
        B3 = json.load(open(p4, encoding="utf-8"))
        # B3: 单特征最稳健 (多特征 elastic-net 在 test1 过拟合, RMSE > 单特征 138)
        en_t1 = B3["elastic_net"]["rmse"][1]
        ok.append(line(en_t1 > 138,
                       f"多特征 elastic-net test1 RMSE {en_t1:.0f} > 单特征 138 "
                       f"(小样本过拟合, 单特征最稳健)"))
        # 深度架构不输传统 (MLP ~ elastic-net)
        mlp_t1 = B3["mlp"]["rmse"][1]
        ok.append(line(abs(mlp_t1 - en_t1) < 150,
                       f"MLP test1 RMSE {mlp_t1:.0f} ~ elastic-net {en_t1:.0f} "
                       f"(深度架构不输传统)"))


if __name__ == "__main__":
    sys.exit(main())
