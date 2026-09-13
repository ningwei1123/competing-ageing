"""
================================================================================
 LFP-ETGNet  |  Module 2: Multi-cell dataset generation
================================================================================
 Builds a population of automotive LFP cells:

   * stochastic electrode microstructures  ->  microstructure graphs
   * fast-charging / thermal / DOD protocols
   * coupled LLI / LAM / CL ageing simulation with topology evolution
   * early-life (first N_EARLY cycles) electrochemical time series
   * three multi-task targets:  RUL , SOH , relative internal-resistance rise

 Calibration anchors (published automotive LFP data)
 ---------------------------------------------------
   25 C, 1C, 100 % DOD : SOH(1000 cyc) = 92.2 %       (20 Ah LFP/graphite pouch)
   45 C, 1C, 100 % DOD : SOH(1000 cyc) = 83.8 %
   SOH = 85 % at 45 C needs ~1/3 of the cycles needed at 25 C
   25 C, 1C, 100 % DOD : ~4500 cycles to 80 % SOH  (0.3C -> 10000, 0.5C -> 8000)
   LFP fade per 100 cycles (25 C) ~ 0.25 % life-averaged

 The coefficients below are *solved* by ``_probe/calibrate.py``, which requires
 the coupled model to reproduce the two anchors above simultaneously.  Both
 channels are measured through the generator's own stochastic physics, because
 the network contribution to q_LAM is not reproducible with a deterministic
 rupture law.  The verified outcome of the reference protocol is

     SOH(1000) = 0.9220                    (anchor 0.9220)
     SOH(4500) = 0.8020 +/- 0.0033         (anchor 0.8000)
     EOL       = 4540 +/- 102 cycles       (anchor 4500, +0.9 %)
     q_LAM(4500) = 0.0237   CL_el(4500) = 0.2158

 over 8 RVEs x 5 fatigue-scatter realisations.

 Two independent degradation channels drive every cell:

   LLI        interphase growth and parasitic side reactions; capacity only,
              with a proportional resistance contribution;
   network    contact fatigue of the solid-solid necks, which manifests as an
              increased series resistance (CL) and, when a particle loses its
              last electronic path, as isolated active material (LAM).

 LAM is therefore *not* an independent mechanism: it is the topological
 consequence of network degradation, which is exactly why a graph is needed to
 resolve it.
================================================================================
"""
from __future__ import annotations
import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfp_micro import (P, reconstruct_electrode, build_graph, simulate_aging,
                       assemble_node_features, graph_descriptors)
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

N_EARLY = 100          # prediction point (early life, ~ <10 % of total life)
N_HORIZON = 300        # fixed reporting horizon for SOH / dR targets
N_TARGET = 1200        # active particles per representative volume element

# ------- reference degradation coefficients (calibrated) -------------------
#   Solved by _probe/calibrate.py such that the coupled model reproduces
#   SOH(1000) = 0.922 and 80 % SOH at ~4500 cycles for 25 C / 1C / 100 % DOD.
#   The network channel (K_CL_BASE) is fixed first, from the requirement that
#   the electronic contact network lose 22 % of its solid-solid connections by
#   end of life; K_LLI_REF and K_LIN_REF then follow from the two capacity
#   anchors with q_LAM measured from the same stochastic simulator.
K_LLI_REF = 2.342097e-03    # sqrt-term, 25 C / 1C / 100 % DOD
K_LIN_REF = 3.936386e-06    # linear side-reaction term
K_CL_BASE = 2.123526e-04    # contact-fatigue / network-degradation rate
EA_EFF = P["Ea_eff"]        # effective activation energy (J/mol), from lfp_micro
R_GAS = P["R_gas"]
T_REF = P["T_ref"]

# ---- microstructure susceptibility scatter ---------------------------------
#   Cell-to-cell lifetime scatter under an identical protocol is well
#   documented for automotive LFP cells and is only partly explained by the
#   protocol itself.  The residual scatter is attributed here to electrode
#   manufacturing variation and is expressed as one multiplicative factor per
#   independent degradation channel, each derived from a measurable electrode
#   property.  The log-factors are standardised over the population and divided
#   by their mean, so their expectation is exactly one and the published
#   capacity-fade anchors are preserved.
S_MICRO = 0.26             # log-scatter of the microstructure factors
MICRO_CLIP = 0.60          # |log phi| clip


def _raw_micro_indices(graph: dict) -> np.ndarray:
    """Two microstructure indices, one per independent degradation channel.

      * LLI - interphase growth is surface-limited, so the rate scales with the
              specific surface area of the active phase, i.e. inversely with
              the mean particle diameter;
      * CL  - contact fatigue combines the mean cyclic neck strain,
              (r_i + r_j)/r_ref, with the mean contact quality of the
              solid-solid necks.

    Returns log-scale indices (larger value = faster degradation).
    """
    n_am, r_am = graph["n_am"], graph["r_am"]
    d_mean = float(graph["dia"][:n_am].mean())
    e, direct = graph["edges"], graph["direct"]
    if e.size:
        i0 = np.minimum(e[:, 0], n_am - 1)
        i1 = np.minimum(e[:, 1], n_am - 1)
        eps = np.clip((r_am[i0] + r_am[i1]) / P["r_neck_ref"], 1e-3, None)
        eps_d = eps[direct] if direct.any() else eps
        sig_d = graph["sigma_c"][direct]
        eps_mean = float(eps_d.mean())
        q_mean = float(sig_d.mean()) if sig_d.size else 0.5
    else:
        eps_mean, q_mean = 1.0, 0.5
    eps_mean = max(eps_mean, 1e-3)
    q_mean = max(q_mean, 1e-3)
    return np.array([-np.log(d_mean),
                     np.log(eps_mean) + np.log(1.0 / q_mean)])


def _protocol(rng: np.random.Generator, idx: int) -> dict:
    """Sample an operating protocol for one cell.

    Charge protocols follow the multi-step constant-current form used in
    automotive fast-charging studies (1-6 C), combined with a temperature and a
    depth-of-discharge window.
    """
    c_rate = float(np.clip(rng.lognormal(np.log(2.1), 0.42), 0.9, 6.2))
    T = float(rng.choice([25.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
                         p=[.18, .16, .13, .14, .13, .16, .10]))
    lo_choices = [(0.00, 1.00), (0.00, 1.00), (0.05, 0.95), (0.10, 0.90),
                  (0.20, 0.80), (0.30, 0.70)]
    soc_lo, soc_hi = lo_choices[rng.integers(0, len(lo_choices))]
    dod = soc_hi - soc_lo
    arr = float(np.exp(-EA_EFF / R_GAS * (1.0 / (T + 273.15) - 1.0 / T_REF)))
    k_lli = K_LLI_REF * arr * (c_rate ** 0.60) * (dod ** 0.85)
    k_lin = K_LIN_REF * arr * (c_rate ** 0.60) * (dod ** 0.85)
    k_cl = K_CL_BASE * arr * (c_rate ** 0.60) * (dod ** 1.45)
    return dict(cell_id=idx, c_rate=c_rate, T_celsius=T, soc_lo=soc_lo,
                soc_hi=soc_hi, dod=dod, k_lli_ref=k_lli, k_lin_ref=k_lin,
                k_cl_ref=k_cl, Q0=1.0, dt_rise=0.9, n_eol=8000)


def _apply_micro(proto: dict, phi: np.ndarray) -> dict:
    """Fold the microstructure susceptibility factors into the rate constants.

    Each *independent* channel is scaled by its own factor, so two cells cycled
    under the same protocol but with different electrodes do not follow the same
    degradation *composition*.  Because LAM is a topological consequence of the
    network channel, the network factor acts on both the resistance rise and the
    isolation of active material, and the two are separated by the graph rather
    than by two separate parameters.
    """
    p = dict(proto)
    p["phi_lli"], p["phi_cl"] = float(phi[0]), float(phi[1])
    p["k_lli_ref"] = proto["k_lli_ref"] * p["phi_lli"]
    p["k_cl_ref"] = proto["k_cl_ref"] * p["phi_cl"]
    # the linear side reaction shares the interphase channel
    p["k_lin_ref"] = proto["k_lin_ref"] * p["phi_lli"]
    return p


def _edge_damage(graph: dict, proto: dict, n_cycle: float,
                 scatter: np.ndarray | None = None) -> np.ndarray:
    """Contact-fatigue damage accumulated by every edge at cycle ``n_cycle``.

    D_ij(n) = n * k_cl * ((r_i + r_j)/r_ref)^m * s_ij , where ``s_ij`` is the
    per-edge log-normal scatter of the fatigue life and the rate coefficient is
    already scaled by temperature / C-rate / DOD in :func:`_protocol`.  The
    same ``scatter`` that :func:`simulate_aging` drew must be supplied here, so
    that the edge features seen by the network are consistent with the network
    degradation that produced the targets.
    """
    n_am, r_am, e = graph["n_am"], graph["r_am"], graph["edges"]
    if e.shape[0] == 0:
        return np.zeros(0)
    i0 = np.minimum(e[:, 0], n_am - 1)
    i1 = np.minimum(e[:, 1], n_am - 1)
    eps_edge = np.clip((r_am[i0] + r_am[i1]) / P["r_neck_ref"], 1e-3, None)
    D = n_cycle * proto["k_cl_ref"] * eps_edge ** P["m_fatigue"]
    return D if scatter is None else D * scatter


def _eol_from_analytic(proto: dict, p: dict = P, n_max: int = 8000) -> int:
    """Analytic EOL estimate (cycles to 80 % SOH) from the interphase channel
    alone.  It deliberately *ignores* the network channel, so it upper-bounds
    the true end of life and provides a safe cycle limit for the sweep.
    """
    n = np.arange(1, n_max + 1, dtype=float)
    q = proto["k_lli_ref"] * np.sqrt(n) + proto["k_lin_ref"] * n
    idx = np.where(q >= 0.20)[0]
    return int(n[idx[0]]) if idx.size else n_max


def build_cell(idx: int, graph: dict, proto: dict, phi, rng) -> dict:
    """Age one cell.  The microstructure graph and the protocol are supplied by
    :func:`build_dataset`, which has already used them to standardise the
    susceptibility factors across the population."""
    if phi is not None:
        proto = _apply_micro(proto, phi)
    proto["n_eol"] = int(min(_eol_from_analytic(proto), 8000))
    sim = simulate_aging(graph, proto, rng, n_check=25, n_pred=N_EARLY)

    rec = sim["record"]
    n_end = int(rec["n"][-1])
    if n_end < N_HORIZON + 60:                       # ensure targets defined
        proto["n_eol"] = max(N_HORIZON + 400, proto["n_eol"])
        sim = simulate_aging(graph, proto, rng, n_check=25, n_pred=N_EARLY)
        rec = sim["record"]
        n_end = int(rec["n"][-1])
    if n_end < N_HORIZON + 20:                       # still too short -> reject
        return None
    # ---- reject censored cells -------------------------------------------
    #   End of life is defined as the cycle at which the state of health
    #   reaches 80 %.  A cell whose simulated end is set by the sweep limit
    #   instead would carry a right-censored remaining-useful-life label, so it
    #   is dropped, which keeps a single consistent definition of the target
    #   for every cell retained in the population.
    if float(rec["SOH"][-1]) > 0.801:
        return None

    # ---- microstructure graph features at the prediction point ------------
    Fnode = assemble_node_features(graph, sim, soc_ref=0.5)
    Fnode = np.nan_to_num(Fnode, nan=0.0, posinf=0.0, neginf=0.0)
    desc = graph_descriptors(graph, sim)

    # ---- contact-quality degradation at the prediction point --------------
    #   edge conductance decays continuously before a neck finally ruptures:
    #        g_ij(n) = (1 - min(D_ij(n), 1))^1.5 ,  D_ij(n) = n * k_cl * eps_ij^m
    #   This gives the GNN an explicit, early-life topology/quality signal.
    D_pred = np.clip(_edge_damage(graph, proto, N_EARLY, sim["fat_scatter"]),
                     0.0, 1.0)
    gate = (1.0 - D_pred) ** 1.5
    eweight = np.stack([graph["sigma_c"] * gate, graph["ionic"] * gate,
                        graph["direct"].astype(float)], 1).astype(np.float32)

    # ---- early-life electrochemical time series --------------------------
    #   The measured trajectory is a partial observation of the same ledger
    #   that the simulator integrates, so it is sampled from the recorded curve
    #   rather than re-derived from the rate coefficients.  The observation and
    #   the target therefore come from one and the same degradation model and
    #   cannot drift apart.
    #   Over the first N_EARLY cycles the true fade is of the order of 2 %,
    #   which is comparable to the sensor calibration offset and to the slow
    #   thermal drift of a cycling rig; both are correlated errors that bias
    #   the apparent early slope and therefore cannot be averaged away.
    n_ser = np.arange(1, N_EARLY + 1, dtype=float)
    soh_true = np.interp(n_ser, rec["n"], rec["SOH"])
    r_rel_true = np.interp(n_ser, rec["n"], rec["R_rel"])
    b_cap = rng.normal(0.0, 2.0e-3)                       # capacity calibration
    d_cap = rng.normal(0.0, 2.6e-3)                       # thermal drift
    b_res = rng.normal(0.0, 0.010)                        # resistance calibration
    soh_n = (soh_true + b_cap + d_cap * (n_ser / N_EARLY) ** 1.5
             + rng.normal(0, 1.0e-3, N_EARLY))
    r_rel_n = r_rel_true * (1.0 + b_res
                            + rng.normal(0, 0.012, N_EARLY))
    Tn = proto["T_celsius"] + 0.9 * proto["c_rate"] * (1 + 0.4 * n_ser / n_ser.max())
    Tn = Tn + rng.normal(0, 0.45, N_EARLY)
    dQdV = 0.62 * np.exp(-1.35 * (1 - soh_n)) * (1 + rng.normal(0, 0.012, N_EARLY))
    dv_pol = 0.055 * r_rel_n * proto["c_rate"] * (1 + rng.normal(0, 0.015, N_EARLY))
    ce = 0.9985 - 0.0022 * (1 - soh_n) + rng.normal(0, 3.5e-4, N_EARLY)
    chg_cap = soh_n * (1 + rng.normal(0, 0.0020, N_EARLY))
    seq = np.stack([soh_n, r_rel_n, Tn / 60.0, dQdV, dv_pol, ce, chg_cap], 1)

    # ---- targets ----------------------------------------------------------
    #   RUL      : cycles remaining (from N_EARLY) until SOH = 80 %
    #   SOH      : state of health at the fixed horizon (cycle N_HORIZON)
    #   dR_int   : relative internal-resistance rise at cycle N_HORIZON
    nn = rec["n"]
    j_h = int(np.argmin(np.abs(nn - N_HORIZON)))
    soh_h = float(rec["SOH"][j_h])
    dr_h = float(rec["R_rel"][j_h] - 1.0)
    rul = float(max(n_end - N_EARLY, 1.0))

    return dict(
        cell_id=idx, proto={k: (v if not isinstance(v, np.generic) else v.item())
                            for k, v in proto.items()},
        Fnode=Fnode.astype(np.float32),
        edges=graph["edges"].astype(np.int32),
        eweight=eweight,
        depth=graph["depth"].astype(np.float32),
        kind=graph["kind"].astype(np.int8),
        n_am=int(graph["n_am"]), n_node=int(graph["n_node"]),
        seq=seq.astype(np.float32),
        y=np.array([rul, soh_h, dr_h], np.float32),
        n_eol=float(n_end), desc={k: float(v) for k, v in desc.items()},
        curve=dict(n=rec["n"], SOH=rec["SOH"], LLI=rec["LLI"],
                   LAM=rec["LAM"], CL=rec["CL"], edge_ret=rec["edge_ret"]),
        alive=sim["alive"], alive_pred=sim["alive_pred"],
        eps_amp=sim["eps_amp"],
    )


def build_dataset(n_cells: int = 340, seed0: int = 1000, verbose: bool = True):
    """Two-pass construction.

    Pass 1 draws the electrode microstructures so that the susceptibility
    indices can be standardised over the population; pass 2 runs the ageing
    simulation with the resulting factors.  Standardising before the
    simulation guarantees that the population-mean rate constants are exactly
    the calibrated reference values, so the published capacity-fade anchors
    remain valid while the cell-to-cell scatter is added on top.
    """
    cells, t0 = [], time.time()

    # ---- pass 1: microstructures, graphs, protocols, raw indices ----------
    stage, raw = [], []
    for i in range(n_cells):
        rng = np.random.default_rng(seed0 + i)
        micro = reconstruct_electrode(rng, n_target=N_TARGET)
        graph = build_graph(micro, rng=rng)
        proto = _protocol(rng, i)
        stage.append((graph, proto))
        raw.append(_raw_micro_indices(graph))
        if verbose and (i + 1) % 50 == 0:
            print(f"  pass1 {i+1}/{n_cells}  {time.time()-t0:.1f}s", flush=True)
    raw = np.asarray(raw)
    z = (raw - raw.mean(0)) / np.maximum(raw.std(0), 1e-12)
    phi = np.exp(np.clip(S_MICRO * z, -MICRO_CLIP, MICRO_CLIP))
    phi /= phi.mean(0)                        # exact unit mean per channel
    if verbose:
        print("  phi mean", np.round(phi.mean(0), 4).tolist(),
              "sd", np.round(phi.std(0), 4).tolist(), flush=True)

    # ---- pass 2: ageing simulation with the standardised factors ----------
    for i, (graph, proto) in enumerate(stage):
        rng = np.random.default_rng(seed0 + i + 900001)
        c = build_cell(i, graph, proto, phi[i], rng)
        if c is not None:
            cells.append(c)
        if verbose and (i + 1) % 20 == 0:
            print(f"  built {i+1}/{n_cells}  kept={len(cells)}  "
                  f"{time.time()-t0:.1f}s", flush=True)
    return cells


if __name__ == "__main__":
    import argparse, pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", type=str, default="../data/lfp_cells.pkl")
    a = ap.parse_args()
    cells = build_dataset(a.n)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(cells, f)
    ys = np.stack([c["y"] for c in cells])
    print("cells:", len(cells), "saved ->", out)
    print("RUL  min/med/max %.0f %.0f %.0f" % (ys[:, 0].min(), np.median(ys[:, 0]), ys[:, 0].max()))
    print("SOH  min/med/max %.4f %.4f %.4f" % (ys[:, 1].min(), np.median(ys[:, 1]), ys[:, 1].max()))
    print("dR   min/med/max %.4f %.4f %.4f" % (ys[:, 2].min(), np.median(ys[:, 2]), ys[:, 2].max()))
    print("nodes/cell mean %.0f" % np.mean([c["n_node"] for c in cells]))
    print("edges/cell mean %.0f" % np.mean([len(c["edges"]) for c in cells]))
    print("EOL min/med/max %.0f %.0f %.0f" % (
        min(c["n_eol"] for c in cells), np.median([c["n_eol"] for c in cells]),
        max(c["n_eol"] for c in cells)))
