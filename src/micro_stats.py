"""
================================================================================
 LFP-ETGNet  |  Module 3b: Microstructure population statistics
================================================================================
 Reports the statistics quoted in Section 4.1 of the manuscript:
   * as-built solid volume fraction, mean / median particle diameter
   * RVE edge length, node and edge counts per cell
   * as-built mean coordination number and as-built isolated fraction
   * edge retention and isolated fraction at end of life
 Writes results/micro_stats.npz
================================================================================
"""
from __future__ import annotations
import os, sys, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lfp_micro import (P, reconstruct_electrode, build_graph,
                       _percolating_nodes)

ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

N_SAMPLE = 120


def main():
    # ---- independent reconstructions: geometry and as-built topology ----
    box, phi, dmean, dmed, isol0, zdeg, dfrac = [], [], [], [], [], [], []
    all_dia = []
    for i in range(N_SAMPLE):
        rng = np.random.default_rng(50000 + i)
        micro = reconstruct_electrode(rng)
        g = build_graph(micro, rng=rng)
        v_solid = float(np.sum(np.pi / 6.0 * micro["radii"] ** 3 * 8.0))
        box.append(g["box"])
        phi.append(v_solid / g["box"] ** 3)
        dmean.append(float(micro["dia"].mean()))
        dmed.append(float(np.median(micro["dia"])))
        all_dia.append(micro["dia"])
        # as-built percolation, evaluated on the electronic subnetwork that
        # the aging simulator uses for LAM.  Using the full graph here (as an
        # earlier version did) mixes in electrolyte-wetted edges, which cannot
        # carry electronic current, and reports an isolated fraction that is
        # inconsistent with every q_LAM in the dataset.
        alive = np.ones(len(g["edges"]), bool)
        perc = _percolating_nodes(g["n_node"], g["edges"], alive,
                                 g["kind"], g["depth"], subnet=g["direct"])
        n_am = g["n_am"]
        isol0.append(1.0 - perc[:n_am].mean())
        dfrac.append(float(g["direct"].mean()) if g["edges"].size else 0.0)
        deg = np.zeros(g["n_node"])
        e = g["edges"]
        if e.size:
            np.add.at(deg, e[:, 0], 1); np.add.at(deg, e[:, 1], 1)
        zdeg.append(float(deg[:n_am].mean()))

    # ---- population statistics from the generated dataset ----------------
    cells = pickle.load(open(os.path.join(ROOT, "data", "lfp_cells.pkl"), "rb"))
    n_node = np.array([c["n_node"] for c in cells], float)
    n_edge = np.array([c["desc"]["n_edge"] for c in cells], float)
    z_pop = np.array([c["desc"]["mean_degree"] for c in cells], float)
    edge_ret = np.array([c["desc"]["edge_retention"] for c in cells], float)
    lam_eol = np.array([c["curve"]["LAM"][-1] for c in cells], float)

    out = dict(
        box_mean=float(np.mean(box)), box_std=float(np.std(box)),
        phi_real=float(np.mean(phi)),
        d_mean=float(np.mean(dmean)), d_med=float(np.mean(dmed)),
        d_share_sec=float(np.mean([np.mean(d > 0.45) for d in all_dia])),
        n_node_mean=float(n_node.mean()), n_edge_mean=float(n_edge.mean()),
        zmean=float(np.mean(z_pop)),
        zmean_geo=float(np.mean(zdeg)),
        isol0=float(np.mean(isol0)),
        direct_frac=float(np.mean(dfrac)),
        edge_ret_eol=float(edge_ret.mean()),
        lam_eol=float(lam_eol.mean()),
    )
    np.savez(os.path.join(RES, "micro_stats.npz"), **out)
    for k, val in out.items():
        print(f"  {k:14s} {val:.4f}")
    print("saved results/micro_stats.npz")


if __name__ == "__main__":
    main()
