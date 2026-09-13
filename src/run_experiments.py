"""
================================================================================
 LFP-ETGNet  |  Module 4: Training / evaluation driver
================================================================================
 Trains the physics-guided GNN-Transformer and all baselines, records
 per-epoch convergence, and reports MSE / RMSE / MAE / R^2 per task plus
 inference latency and throughput.
================================================================================
"""
from __future__ import annotations
import os, sys, json, time, pickle, argparse, copy
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (LFP_ETGNet, AblationGNN, AblationTF, MTLResNet, GAT_TST,
                   STGCN, GMAN, CNNLSTM, batch_graphs)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

TASKS = ["RUL", "SOH", "dR_int"]


# ---------------------------------------------------------------------------
def load_cells(path=None):
    path = path or os.path.join(DATA, "lfp_cells.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def make_tensors(cells, stats_from=None):
    """Z-score node features, the trajectory and targets.

    Normalisation statistics are computed **only** from ``stats_from``
    (the training split) and then applied to every cell, so no test-set
    information leaks into the model.
    """
    src = cells if stats_from is None else stats_from
    F = np.concatenate([c["Fnode"] for c in src], 0)
    mu, sd = F.mean(0), F.std(0) + 1e-8
    Ys = np.stack([c["y"] for c in src])
    tmu, tsd = Ys.mean(0), Ys.std(0) + 1e-8
    Fn = [np.clip((c["Fnode"] - mu) / sd, -6, 6).astype(np.float32)
          for c in cells]
    Yn = np.stack([(c["y"] - tmu) / tsd for c in cells]).astype(np.float32)
    #   The trajectory is z-scored per channel too.  It was previously fed to
    #   the encoders unnormalised, and the seven channels span three orders of
    #   magnitude (Coulombic efficiency ~0.998, SOH ~0.97, polarisation ~0.14),
    #   so the Transformer's linear projection had to balance wildly different
    #   scales; normalising per channel removes that imbalance.
    Ss = np.concatenate([c["seq"] for c in src], 0)
    smu, ssd = Ss.mean(0), Ss.std(0) + 1e-8
    seqs = [((c["seq"] - smu) / ssd).astype(np.float32) for c in cells]
    return Fn, Yn, mu, sd, tmu, tsd, seqs


def to_batch(cells, Fn, seqs, idx, device="cpu"):
    Gn = [Fn[i] for i in idx]
    ed = [cells[i]["edges"] for i in idx]
    ew = [cells[i]["eweight"] for i in idx]
    kd = [cells[i]["kind"] for i in idx]
    X, src, dst, e, K, b, ng, off = batch_graphs(Gn, ed, ew, kd, device)
    S = torch.tensor(np.stack([seqs[i] for i in idx]), dtype=torch.float32, device=device)
    return (X, src, dst, e, K, b, ng, S)


def batch_pred(model, cells, Fn, seqs, Y, idx, device="cpu", need_target=True):
    """Forward pass and uncertainty-weighted multi-task loss.

    Every model in this study predicts a per-task log-variance, so the
    homoscedastic-uncertainty objective (Eq. 19, Kendall & Gal) is evaluated
    from the model's own parameter.  A single parameter vector is used in the
    loss, which is what makes the learned variances the quantities that are
    later inspected for calibration.
    """
    batch = to_batch(cells, Fn, seqs, idx, device)
    pred, logvar = model(batch)
    if need_target and Y is not None:
        tgt = torch.tensor(Y[idx], dtype=torch.float32, device=device)
        se = (pred - tgt) ** 2
        lv = logvar if logvar is not None else torch.zeros(pred.shape[1],
                                                          device=device)
        loss = (0.5 * torch.exp(-lv) * se.mean(0) + 0.5 * lv).sum()
        return loss, pred.detach().cpu().numpy()
    return None, pred.detach().cpu().numpy()


# ---------------------------------------------------------------------------
def metrics(y_true, y_pred):
    """Per-task MSE / RMSE / MAE / R^2 / MAPE."""
    out = {}
    for k, t in enumerate(TASKS):
        yt, yp = y_true[:, k], y_pred[:, k]
        mse = float(np.mean((yt - yp) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(yt - yp)))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2) + 1e-12
        r2 = float(1 - ss_res / ss_tot)
        mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-8))) * 100)
        out[t] = dict(MSE=mse, RMSE=rmse, MAE=mae, R2=r2, MAPE=mape)
    out["_macro"] = dict(
        MSE=float(np.mean([out[t]["MSE"] for t in TASKS])),
        RMSE=float(np.mean([out[t]["RMSE"] for t in TASKS])),
        MAE=float(np.mean([out[t]["MAE"] for t in TASKS])),
        R2=float(np.mean([out[t]["R2"] for t in TASKS])),
        MAPE=float(np.mean([out[t]["MAPE"] for t in TASKS])))
    return out


# ---------------------------------------------------------------------------
def train_one(build_fn, cells, Fn, seqs, Y, tr, te, epochs=150, bs=16,
              lr=1e-3, wd=1e-4, device="cpu", verbose=False, seed=0,
              patience=20, val_frac=0.15):
    """Train one model on ``tr`` and evaluate on ``te``.

    A validation subset is carved out of the training split and is used for
    early stopping and for selecting the reported weights; the test split is
    never used for model selection.  Test metrics are still recorded every
    epoch, but only so that the convergence figure can be drawn.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = np.asarray(tr, int)
    perm = rng.permutation(len(tr))
    n_val = max(8, int(round(val_frac * len(tr))))
    if n_val >= len(tr) - 4:                      # degenerate tiny split
        n_val = max(1, len(tr) // 5)
    val = tr[perm[:n_val]]
    fit = tr[perm[n_val:]]

    model = build_fn().to(device)
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = dict(epoch=[], train_loss=[], test_MSE=[], test_RMSE=[], test_R2=[],
                task_MSE=[], lr=[])
    best = dict(r2=-9, state=None, ep=0)
    stale = 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(fit))
        tot, nb = 0.0, 0
        for i in range(0, len(fit), bs):
            idx = fit[order[i:i + bs]]
            opt.zero_grad()
            loss, _ = batch_pred(model, cells, Fn, seqs, Y, idx, device)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            _, pv = batch_pred(model, cells, Fn, seqs, Y, val, device)
            _, pt = batch_pred(model, cells, Fn, seqs, Y, te, device)
        pv = pv * TSD_G + TMU_G
        pt = pt * TSD_G + TMU_G
        mv = metrics(Y[val] * TSD_G + TMU_G, pv)
        mt = metrics(Y[te] * TSD_G + TMU_G, pt)
        hist["epoch"].append(ep); hist["train_loss"].append(tot / max(nb, 1))
        hist["test_MSE"].append(mt["_macro"]["MSE"])
        hist["test_RMSE"].append(mt["_macro"]["RMSE"])
        hist["test_R2"].append(mt["_macro"]["R2"])
        hist["task_MSE"].append([mt[t]["MSE"] for t in TASKS])
        hist["lr"].append(opt.param_groups[0]["lr"])
        if mv["_macro"]["R2"] > best["r2"] + 1e-4:
            best = dict(r2=mv["_macro"]["R2"], ep=ep,
                        state=copy.deepcopy(model.state_dict()))
            stale = 0
        else:
            stale += 1
        if verbose and (ep % 25 == 0 or ep == 1):
            print(f"    ep{ep:4d} loss {tot/max(nb,1):.4f} "
                  f"valR2 {mv['_macro']['R2']:.4f} "
                  f"testR2 {mt['_macro']['R2']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if stale >= patience:
            if verbose:
                print(f"    early stop at ep{ep} (best {best['ep']})", flush=True)
            break
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        _, pr = batch_pred(model, cells, Fn, seqs, Y, te, device)
    final = metrics(Y[te] * TSD_G + TMU_G, pr * TSD_G + TMU_G)
    # ---- inference latency (single-sample) --------------------------------
    lat = []
    with torch.no_grad():
        for i in te[:30]:
            t = time.perf_counter()
            batch_pred(model, cells, Fn, seqs, Y, [i], device, False)
            lat.append(time.perf_counter() - t)
    n_param = sum(p.numel() for p in model.parameters())
    return (model, final, hist,
            dict(best_epoch=best["ep"], latency_s=float(np.mean(lat)),
                 n_param=int(n_param), train_time_s=time.time() - t0,
                 te_idx=np.asarray(te, int), pred=pr,
                 true=Y[te].copy()))


TMU_G = None
TSD_G = None


def run(folds=5, epochs=150, bs=16, n_cells=None, models=None, verbose=True,
        patience=20, tag="metrics", split="random"):
    global TMU_G, TSD_G
    cells = load_cells()
    if n_cells:
        cells = cells[:n_cells]
    n = len(cells)
    print(f"cells: {n}  split={split}", flush=True)

    idx_all = np.arange(n)
    if split == "random":
        kf = _kfold(n, folds, seed=7)
    else:
        kf = _group_split(cells, split)

    ALL = {
        "LFP-ETGNet": lambda: LFP_ETGNet(),
        "GNN-only": lambda: AblationGNN(),
        "Transformer-only": lambda: AblationTF(),
        "MTL-ResNet": lambda: MTLResNet(),
        "GAT-TST": lambda: GAT_TST(),
        "ST-GCN": lambda: STGCN(),
        "GMAN": lambda: GMAN(),
        "CNN-LSTM": lambda: CNNLSTM(),
    }
    if models:
        ALL = {k: v for k, v in ALL.items() if k in models}

    results, bundle = {}, {}
    for name, fn in ALL.items():
        print(f"\n=== {name} ===", flush=True)
        fold_metrics, fold_hist, fold_extra = [], [], []
        for f, (tr, te) in enumerate(kf):
            Fn, Yn, mu, sd, tmu, tsd, seqs = make_tensors(
                cells, stats_from=[cells[i] for i in tr])
            TMU_G, TSD_G = tmu, tsd
            model, m, h, ex = train_one(fn, cells, Fn, seqs, Yn, tr, te,
                                        epochs=epochs, bs=bs, verbose=verbose,
                                        seed=100 + f, patience=patience)
            fold_metrics.append(m); fold_hist.append(h); fold_extra.append(ex)
            print(f"  fold{f+1}: R2={m['_macro']['R2']:.4f} "
                  f"MSE={m['_macro']['MSE']:.4f} lat={ex['latency_s']*1000:.1f}ms",
                  flush=True)
            if name == "LFP-ETGNet" and f == len(kf) - 1:
                bundle["te_idx"] = ex["te_idx"]
                bundle["pred"] = ex["pred"] * tsd + tmu
                bundle["true"] = ex["true"] * tsd + tmu
                bundle["state"] = {k: v.cpu() for k, v in model.state_dict().items()}
                bundle["mu"], bundle["sd"] = mu, sd
                bundle["tmu"], bundle["tsd"] = tmu, tsd
        agg = {}
        for t in TASKS + ["_macro"]:
            agg[t] = {k: float(np.mean([fm[t][k] for fm in fold_metrics]))
                      for k in fold_metrics[0][t]}
            agg[t + "_std"] = {k: float(np.std([fm[t][k] for fm in fold_metrics]))
                               for k in fold_metrics[0][t]}
        results[name] = dict(
            metrics=agg,
            folds=[fm for fm in fold_metrics],
            hist=fold_hist[0],
            latency_s=float(np.mean([e["latency_s"] for e in fold_extra])),
            n_param=int(fold_extra[0]["n_param"]),
            train_time_s=float(np.mean([e["train_time_s"] for e in fold_extra])),
            best_epoch=int(np.mean([e["best_epoch"] for e in fold_extra])))
        print(f"  -> macro R2 {agg['_macro']['R2']:.4f}  "
              f"MSE {agg['_macro']['MSE']:.4f}  "
              f"RMSE {agg['_macro']['RMSE']:.4f}", flush=True)

    with open(os.path.join(OUT, f"{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    if bundle and split == "random" and tag == "metrics":
        with open(os.path.join(OUT, "bundle.pkl"), "wb") as f:
            pickle.dump(bundle, f)
    print(f"\nsaved results/{tag}.json", flush=True)
    return results


def _kfold(n, k, seed=7):
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    folds = np.array_split(p, k)
    out = []
    for i in range(k):
        te = np.sort(folds[i])
        tr = np.sort(np.concatenate([folds[j] for j in range(k) if j != i]))
        out.append((tr, te))
    return out


def _strata(cells, mode):
    """Assign every cell to a discrete group used for out-of-distribution folds.

    ``temp``  -- tertiles of the ageing temperature (cold / mild / hot):
                 the model must extrapolate to an unseen thermal regime,
                 which stresses the Arrhenius kinetic term and the
                 temperature-driven growth of the resistive surface film.
    ``life``  -- tertiles of the end-of-life cycle count (short / medium /
                 long lived): the model must extrapolate the degradation
                 trajectory beyond the life range spanned by training cells,
                 the practically relevant setting for RUL prognosis.
    """
    if mode == "temp":
        v = np.array([c["proto"]["T_celsius"] for c in cells], float)
    elif mode == "life":
        #   The realised end of life, i.e. the cycle at which the coupled
        #   simulation reaches 80 % SOH -- NOT the protocol's analytic sweep
        #   cap (proto["n_eol"]), which is the interphase-only upper bound and
        #   excludes the graph-derived LAM.  The manuscript states that the
        #   tertile thresholds are taken from the end-of-life cycle count and
        #   make_paper.py reports them from the same quantity, so the split must
        #   use it too; grouping on the sweep cap would place the reported
        #   thresholds at different cells than the ones actually held out.
        v = np.array([c["n_eol"] for c in cells], float)
    else:
        raise ValueError(f"unknown strata mode {mode!r}")
    q = np.quantile(v, [1 / 3, 2 / 3])
    g = np.zeros(len(cells), int)
    g[v > q[0]] = 1
    g[v > q[1]] = 2
    return g, v


def _group_split(cells, mode, seed=7):
    """Leave-one-group-out folds under a distribution-shift protocol."""
    g, _ = _strata(cells, mode)
    groups = np.unique(g)
    out = []
    for u in groups:
        te = np.sort(np.where(g == u)[0])
        tr = np.sort(np.where(g != u)[0])
        if len(te) and len(tr):
            out.append((tr, te))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--models", type=str, default="")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--tag", type=str, default="metrics")
    ap.add_argument("--split", type=str, default="random",
                    choices=["random", "temp", "life"])
    a = ap.parse_args()
    ms = [m.strip() for m in a.models.split(",") if m.strip()] or None
    run(folds=a.folds, epochs=a.epochs, bs=a.bs,
        n_cells=(a.n or None), models=ms, patience=a.patience, tag=a.tag,
        split=a.split)
