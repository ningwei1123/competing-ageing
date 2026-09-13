"""
================================================================================
 LFP-ETGNet  |  Module 4b: Interpretability & physical-attribution analysis
================================================================================
 Extracts, from the trained physics-guided model:
   * temporal self-attention profiles over the cycle index
   * permutation importance of the 8 node descriptors and 7 series channels
   * node-level topology importance (GNN sensitivity)
   * uncertainty calibration (coverage / interval width)
   * correlation of graph descriptors with the ageing targets
================================================================================
"""
from __future__ import annotations
import os, sys, json, pickle, math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LFP_ETGNet, batch_graphs
from run_experiments import load_cells, make_tensors, to_batch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
TASKS = ["RUL", "SOH", "dR_int"]
NODE_FEATS = ["particle diameter", "inter-particle spacing", "SEI thickness",
              "Li-conc. gradient", "coordination number", "contact quality",
              "electrode depth", "local SOC"]
SEQ_FEATS = ["SOH(n)", "R_int(n)", "T_max(n)", "dQ/dV(n)", "ΔV_pol(n)",
             "CE(n)", "Q_chg(n)"]


def extract_attention(model, seq):
    """Recompute the temporal self-attention of the first encoder layer."""
    enc = model.tf.enc.layers[0]
    attn = enc.self_attn
    d = attn.embed_dim
    h = attn.num_heads
    x = model.tf.pe(model.tf.proj(seq))
    if enc.norm_first:
        xn = enc.norm1(x)
    else:
        xn = x
    W = attn.in_proj_weight
    b = attn.in_proj_bias
    q, k, v = nn.functional.linear(xn, W, b).chunk(3, dim=-1)
    B, T, _ = q.shape
    q = q.view(B, T, h, d // h).transpose(1, 2)
    k = k.view(B, T, h, d // h).transpose(1, 2)
    A = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d // h), -1)
    return A.detach().cpu().numpy()          # [B, h, T, T]


def run_analysis(folds_bundle=None):
    torch.manual_seed(0)
    cells = load_cells()
    bundle = pickle.load(open(os.path.join(OUT, "bundle.pkl"), "rb"))
    mu, sd = bundle["mu"], bundle["sd"]
    tmu, tsd = bundle["tmu"], bundle["tsd"]
    Fn = [np.clip((c["Fnode"] - mu) / sd, -6, 6).astype(np.float32) for c in cells]
    seqs = [c["seq"] for c in cells]

    model = LFP_ETGNet()
    model.load_state_dict(bundle["state"])
    model.eval()
    dev = "cpu"
    N = len(cells)
    print(f"cells {N}", flush=True)

    # ---------------- 1. temporal attention profiles ----------------------
    idx = np.arange(0, N, max(1, N // 64))[:64]
    batch = to_batch(cells, Fn, seqs, idx, dev)
    with torch.no_grad():
        A = extract_attention(model, batch[7])          # [B,h,T,T]
    # attention received by each cycle position from the final query position
    A_last = A[:, :, -1, :].mean(1)                     # [B,T]
    attn_profile = A_last.mean(0)                        # [T]
    attn_heads = A.mean(0)                               # [h,T,T]
    np.savez(os.path.join(OUT, "attention.npz"),
             profile=attn_profile, heads=attn_heads,
             matrix=A[:8].mean(1))
    print("attention profile saved", flush=True)

    # ---------------- 2. permutation feature importance -------------------
    Y = np.stack([c["y"] for c in cells])
    Yn = ((Y - tmu) / tsd).astype(np.float32)
    te = bundle["te_idx"]
    rng = np.random.default_rng(3)
    sub = te[rng.permutation(len(te))[:min(60, len(te))]]
    base = _predict(model, cells, Fn, seqs, sub)
    node_imp = np.zeros(8)
    for k in range(8):
        Fp = [f.copy() for f in Fn]
        for i in sub:
            v = Fp[i][:, k].copy()
            Fp[i][:, k] = v[rng.permutation(len(v))]
        p = _predict(model, cells, Fp, seqs, sub)
        node_imp[k] = float(np.mean((p - base) ** 2))
    seq_imp = np.zeros(7)
    for k in range(7):
        Sp = [s.copy() for s in seqs]
        for i in sub:
            v = Sp[i][:, k].copy()
            Sp[i][:, k] = v[rng.permutation(len(v))]
        p = _predict(model, cells, Fn, Sp, sub)
        seq_imp[k] = float(np.mean((p - base) ** 2))
    node_imp /= node_imp.sum() + 1e-12
    seq_imp /= seq_imp.sum() + 1e-12
    print("importance:", np.round(node_imp, 3), np.round(seq_imp, 3), flush=True)

    # ---------------- 3. uncertainty calibration --------------------------
    #   In this implementation the homoscedastic-uncertainty term (Eq. 19) is a
    #   global learned log-variance per task, i.e. a single 3-vector shared by
    #   every sample, so the intervals have a constant half-width per task.
    logvar = _get_logvar(model, cells, Fn, seqs, te)
    print("learned log-variances:", np.round(logvar, 4).tolist(), flush=True)
    pr = bundle["pred"]; tr = bundle["true"]
    calib = {}
    for k, t in enumerate(TASKS):
        s = float(np.sqrt(np.exp(float(logvar[k]))) * tsd[k])
        err = np.abs(pr[:, k] - tr[:, k])
        calib[t] = dict(sigma_mean=s,
                        coverage_1s=float(np.mean(err <= 1.0 * s)),
                        coverage_2s=float(np.mean(err <= 2.0 * s)),
                        mae=float(err.mean()),
                        sharpness=s,
                        logvar=float(logvar[k]))
    print("calibration:", json.dumps(calib, indent=1), flush=True)

    # ---------------- 4. descriptor-target correlations -------------------
    desc_names = list(cells[0]["desc"].keys())
    D = np.array([[c["desc"][k] for k in desc_names] for c in cells])
    corr = np.zeros((len(desc_names), 3))
    for i in range(len(desc_names)):
        for j in range(3):
            a, b = D[:, i], Y[:, j]
            corr[i, j] = np.corrcoef(a, b)[0, 1] if a.std() > 0 else 0.0

    np.savez(os.path.join(OUT, "interp.npz"),
             node_imp=node_imp, seq_imp=seq_imp, corr=corr,
             desc_names=np.array(desc_names), Y=Y, D=D)
    with open(os.path.join(OUT, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)
    return calib


def _predict(model, cells, Fn, seqs, idx):
    with torch.no_grad():
        b = to_batch(cells, Fn, seqs, idx, "cpu")
        p, _ = model(b)
    return p.cpu().numpy()


def _get_logvar(model, cells, Fn, seqs, idx):
    """Return the learned per-task log-variance (a 3-vector, not per sample)."""
    b = to_batch(cells, Fn, seqs, list(idx)[:1], "cpu")
    with torch.no_grad():
        _, lv = model(b)
    return np.asarray(lv.detach().cpu().numpy(), float).reshape(-1)


if __name__ == "__main__":
    run_analysis()
