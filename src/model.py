"""
================================================================================
 LFP-ETGNet  |  Module 3: Physics-guided GNN-Transformer network and baselines
================================================================================
 The network fuses two physically distinct views of an ageing LFP cell:

   (upper branch, microscale)  electrode microstructure graph  ->  GNN
       nodes = active-material particles / carbon-black domains
       edges = solid-solid conductive contacts and electrolyte-wetted paths
       The GNN propagates over the *current* degraded topology, so the loss of
       electronic percolation caused by the 6.8 % breathing strain is modelled
       directly as graph-topology evolution.

   (lower branch, macroscale)  early-life electrochemical trajectory -> Transformer
       multivariate series of capacity, resistance, temperature, dQ/dV,
       polarisation, Coulombic efficiency  ->  multi-head self-attention over
       cycle index captures long-range ageing patterns.

 A multi-task head then jointly predicts RUL, SOH and relative resistance rise
 with homoscedastic-uncertainty self-weighting.
================================================================================
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
#  Graph utilities (batched message passing on concatenated graphs)
# ============================================================================
def batch_graphs(Fnode, edges, eweight, kind, device="cpu"):
    """Concatenate a list of graphs into one block-diagonal batch."""
    ns = [f.shape[0] for f in Fnode]
    es = [e.shape[0] for e in edges]
    off = list(np.cumsum([0] + ns))
    X = torch.tensor(np.concatenate(Fnode, 0), dtype=torch.float32, device=device)
    src, dst, ew = [], [], []
    for i, (e, w) in enumerate(zip(edges, eweight)):
        if e.shape[0]:
            src.append(torch.tensor(e[:, 0] + off[i], dtype=torch.long))
            dst.append(torch.tensor(e[:, 1] + off[i], dtype=torch.long))
            ew.append(torch.tensor(w, dtype=torch.float32))
    src = torch.cat(src).to(device)
    dst = torch.cat(dst).to(device)
    ew = torch.cat(ew).to(device)
    K = torch.tensor(np.concatenate(kind, 0), dtype=torch.long, device=device)
    batch = torch.tensor(np.concatenate([[i] * n for i, n in enumerate(ns)]),
                         dtype=torch.long, device=device)
    return X, src, dst, ew, K, batch, len(ns), off


# ============================================================================
#  Physics-guided GraphSAGE encoder with edge gating
# ============================================================================
class EdgeGatedSAGE(nn.Module):
    """Mean-aggregation GraphSAGE with an edge-conditioned gate.

    Message:   m_ij = h_j  *  sigmoid(W_e [e_ij])
               (the gate encodes contact quality: a degraded neck transmits
                less current, so its message is attenuated)
    Update:    h_i' = ReLU( W_s h_i + W_n * mean_j( m_ij ) )
    """

    def __init__(self, d_in, d_hid, d_edge=3):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_hid)
        self.lin_neigh = nn.Linear(d_in, d_hid)
        self.gate = nn.Sequential(nn.Linear(d_edge, d_hid), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_hid)

    def forward(self, x, src, dst, ew):
        # The graph is undirected, so every edge must transmit in both
        # orientations.  Aggregating only src <- dst makes the receptive field
        # depend on the arbitrary node numbering and leaves part of the nodes
        # without any incoming message, which weakens the encoder.
        s2 = torch.cat([src, dst])
        d2 = torch.cat([dst, src])
        gate = self.gate(ew)                              # [E, H]
        gate2 = torch.cat([gate, gate], 0)                # [2E, H]
        msg = self.lin_neigh(x)[d2] * gate2               # [2E, H]
        agg = torch.zeros_like(self.lin_self(x))
        agg.index_add_(0, s2, msg)
        deg = torch.zeros(x.shape[0], device=x.device)
        deg.index_add_(0, s2, torch.ones(s2.shape[0], device=x.device))
        agg = agg / deg.clamp(min=1.0)[:, None]
        h = F.relu(self.lin_self(x) + agg)
        return self.norm(h)


class GraphEncoder(nn.Module):
    def __init__(self, d_in=8, d_hid=64, n_layers=3, d_edge=3):
        super().__init__()
        dims = [d_in] + [d_hid] * n_layers
        self.layers = nn.ModuleList(
            [EdgeGatedSAGE(dims[i], dims[i + 1], d_edge) for i in range(n_layers)])
        self.att_pool = nn.Sequential(nn.Linear(d_hid, d_hid // 2), nn.Tanh(),
                                      nn.Linear(d_hid // 2, 1))
        self.out_dim = d_hid * 2      # mean-pool  +  attention-pool

    def forward(self, x, src, dst, ew, batch, n_graph):
        for l in self.layers:
            x = l(x, src, dst, ew)
        # ---- mean pooling ---------------------------------------------------
        s = torch.zeros(n_graph, x.shape[1], device=x.device)
        s.index_add_(0, batch, x)
        cnt = torch.zeros(n_graph, device=x.device)
        cnt.index_add_(0, batch, torch.ones_like(batch, dtype=torch.float32))
        mean_p = s / cnt.clamp(min=1)[:, None]
        # ---- attention pooling ---------------------------------------------
        w = self.att_pool(x)                                  # [N,1]
        w = torch.exp(w - _segment_max(w, batch, n_graph)[batch])
        den = torch.zeros(n_graph, 1, device=x.device)
        den.index_add_(0, batch, w)
        att_p = torch.zeros(n_graph, x.shape[1], device=x.device)
        att_p.index_add_(0, batch, w * x)
        att_p = att_p / den.clamp(min=1e-8)
        return torch.cat([mean_p, att_p], -1)


def _segment_max(v, batch, n):
    out = torch.full((n, v.shape[1]), -1e9, device=v.device)
    out = out.scatter_reduce(0, batch[:, None].expand_as(v), v, reduce="amax",
                             include_self=True)
    return out


# ============================================================================
#  Transformer encoder for the electrochemical trajectory
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class TrajEncoder(nn.Module):
    def __init__(self, d_in=7, d_model=128, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pe = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model * 2      # last-token  +  mean-pool

    def forward(self, x, return_attn=False):
        h = self.pe(self.proj(x))
        h = self.enc(h)
        h = self.norm(h)
        mean_p = h.mean(1)
        last_p = h[:, -1]
        return torch.cat([mean_p, last_p], -1)


# ============================================================================
#  Feature fusion + multi-task head with uncertainty weighting
# ============================================================================
class Fusion(nn.Module):
    def __init__(self, d_g, d_l, d_out=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_g + d_l, d_out)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, hg, hl):
        h = self.proj(torch.cat([hg, hl], -1))
        return self.norm(F.relu(h) + h)          # residual + LayerNorm (Eq. 9)


class SharedEncoder(nn.Module):
    def __init__(self, d_in=128, d_hid=128, n_layers=2, dropout=0.1):
        super().__init__()
        layers, d = [], d_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, d_hid), nn.ReLU(), nn.Dropout(dropout)]
            d = d_hid
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiTaskHead(nn.Module):
    """Three task-specific branches; each task also predicts a log-variance
    used by the homoscedastic-uncertainty loss (Eq. 19)."""

    def __init__(self, d_in=128, d_hid=64, n_tasks=3):
        super().__init__()
        self.mu = nn.ModuleList([nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(), nn.Linear(d_hid, 1))
            for _ in range(n_tasks)])
        self.logvar = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, z):
        return torch.cat([m(z) for m in self.mu], -1), self.logvar


class LFP_ETGNet(nn.Module):
    """Full physics-guided multi-scale model."""

    def __init__(self, d_node=8, d_g=64, gnn_layers=3, d_model=128,
                 tf_layers=2, n_heads=4, d_fuse=128, d_share=128,
                 d_seq=7, dropout=0.1, use_gnn=True, use_tf=True):
        super().__init__()
        self.use_gnn, self.use_tf = use_gnn, use_tf
        d_g_out = d_g * 2 if use_gnn else 0
        d_l_out = d_model * 2 if use_tf else 0
        if use_gnn:
            self.gnn = AttentionGraphEncoder(d_node, d_g, gnn_layers)
        if use_tf:
            self.tf = TrajEncoder(d_seq, d_model, tf_layers, n_heads, dropout)
        self.fuse = Fusion(max(d_g_out, 1), max(d_l_out, 1), d_fuse, dropout)
        self.shared = SharedEncoder(d_fuse, d_share, 2, dropout)
        self.head = MultiTaskHead(d_share, 64, 3)

    def forward(self, batch):
        (X, src, dst, ew, K, bidx, n_graph, seq) = batch
        parts = []
        if self.use_gnn:
            parts.append(self.gnn(X, src, dst, ew, bidx, n_graph))
        else:
            parts.append(torch.zeros(n_graph, 1, device=X.device))
        if self.use_tf:
            parts.append(self.tf(seq))
        else:
            parts.append(torch.zeros(n_graph, 1, device=X.device))
        h = self.fuse(parts[0], parts[1])
        z = self.shared(h)
        return self.head(z)


# ============================================================================
#  Uncertainty-weighted multi-task loss (Kendall & Gal, Eq. 19)
# ============================================================================
#   The objective is
#       L = sum_t [ 0.5 * exp(-log sigma_t^2) * SE_t + 0.5 * log sigma_t^2 ]
#   In this implementation the per-task log-variance is a learnable parameter of
#   every model (``MultiTaskHead.logvar``), so the objective is evaluated
#   directly from the model output in ``run_experiments.batch_pred``.  The class
#   below is kept as the standalone form of the same objective; it is not used
#   in the training loop, to avoid having two divergent copies of the variance
#   parameters.
class UncertaintyLoss(nn.Module):
    def __init__(self, n_tasks=3):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, pred, target):
        # pred, target: [B, T]
        se = (pred - target) ** 2
        loss = 0.5 * torch.exp(-self.log_sigma) * se.mean(0) + 0.5 * self.log_sigma
        return loss.sum(), se.mean(0)


# ============================================================================
#  Baseline architectures
# ============================================================================
class MLP(nn.Module):
    """Plain multi-layer perceptron on pooled graph and trajectory statistics."""

    def __init__(self, d_in, d_hid=128, n=3, d_out=3, dropout=0.1):
        super().__init__()
        L, d = [], d_in
        for _ in range(n - 1):
            L += [nn.Linear(d, d_hid), nn.ReLU(), nn.Dropout(dropout)]; d = d_hid
        L += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*L)
        self.logvar = nn.Parameter(torch.zeros(d_out))

    def forward(self, batch):
        X, src, dst, ew, K, b, ng, seq = batch
        g = _flat_graph_mean(X, b, ng)
        s = seq.mean(1)
        return self.net(torch.cat([g, s], -1)), self.logvar


def _flat_graph_mean(X, batch, n):
    s = torch.zeros(n, X.shape[1], device=X.device)
    s.index_add_(0, batch, X)
    c = torch.zeros(n, device=X.device)
    c.index_add_(0, batch, torch.ones_like(batch, dtype=torch.float32))
    return s / c.clamp(min=1)[:, None]


class AblationGNN(nn.Module):
    """GNN branch only (microstructure topology, no trajectory)."""

    def __init__(self, d_node=8, d_g=64, layers=3, d_fuse=128):
        super().__init__()
        self.gnn = AttentionGraphEncoder(d_node, d_g, layers)
        self.shared = SharedEncoder(self.gnn.out_dim, d_fuse, 2)
        self.head = MultiTaskHead(d_fuse, 64, 3)

    def forward(self, batch):
        X, src, dst, ew, K, b, ng, seq = batch
        z = self.shared(self.gnn(X, src, dst, ew, b, ng))
        return self.head(z)


class AblationTF(nn.Module):
    """Transformer branch only (electrochemical trajectory)."""

    def __init__(self, d_seq=7, d_model=128, layers=2, heads=4, d_fuse=128):
        super().__init__()
        self.tf = TrajEncoder(d_seq, d_model, layers, heads)
        self.shared = SharedEncoder(self.tf.out_dim, d_fuse, 2)
        self.head = MultiTaskHead(d_fuse, 64, 3)

    def forward(self, batch):
        X, src, dst, ew, K, b, ng, seq = batch
        z = self.shared(self.tf(seq))
        return self.head(z)


class MTLResNet(nn.Module):
    """Residual multi-task network on concatenated hand-crafted features."""

    def __init__(self, d_node=8, d_seq=7, d=192, blocks=4, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(d_node + d_seq * 3, d)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout))
            for _ in range(blocks)])
        self.norm = nn.LayerNorm(d)
        self.head = MultiTaskHead(d, 64, 3)

    def forward(self, batch):
        X, src, dst, ew, K, b, ng, seq = batch
        g = _flat_graph_mean(X, b, ng)
        s_mean, s_std, s_last = seq.mean(1), seq.std(1), seq[:, -1]
        h = self.inp(torch.cat([g, s_mean, s_std, s_last], -1))
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(self.norm(h))


class GraphAttention(nn.Module):
    def __init__(self, d_in, d_out, heads=4):
        super().__init__()
        self.h, self.d = heads, d_out // heads
        self.lin = nn.Linear(d_in, d_out, bias=False)
        # residual path needs a shape-matched input (d_in may differ from d_out
        # in the first layer); without it the skip connection cannot be added
        self.proj = (nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out))
        self.a = nn.Parameter(torch.zeros(1, heads, 2 * self.d))
        self.out = nn.Linear(d_out, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x, src, dst, ew):
        h = self.lin(x).view(-1, self.h, self.d)
        # undirected graph: propagate along both orientations of every edge
        s2 = torch.cat([src, dst])
        d2 = torch.cat([dst, src])
        hs, hd = h[s2], h[d2]
        e = F.leaky_relu((torch.cat([hs, hd], -1) * self.a).sum(-1), 0.2)
        e = e - _segment_max(e, s2, x.shape[0])[s2]
        alpha = torch.exp(e)
        den = torch.zeros(x.shape[0], self.h, device=x.device)
        den.index_add_(0, s2, alpha)
        alpha = alpha / den.clamp(min=1e-8)[s2]
        m = hd * alpha[..., None]
        agg = torch.zeros(x.shape[0], self.h, self.d, device=x.device)
        agg.index_add_(0, s2, m)
        return self.norm(F.relu(self.out(agg.reshape(-1, self.h * self.d)))
                         + self.proj(x))


class AttentionGraphEncoder(nn.Module):
    """Graph branch whose message-passing layers are multi-head graph attention
    (GAT) instead of edge-gated mean aggregation.

    This is the encoder swap the distribution-shift study motivates: the
    mean-aggregation encoder degrades far more than the graph-attention
    baselines when a thermal regime is withheld, so the attention layers are
    substituted here while the pooling and the fusion are left unchanged, which
    isolates the change to the message-passing layer.  The output dimension and
    pooling are identical to :class:`GraphEncoder`, so the rest of the model
    (fusion, shared encoder, head) is untouched.
    """

    def __init__(self, d_in=8, d_hid=64, n_layers=3, heads=4):
        super().__init__()
        dims = [d_in] + [d_hid] * n_layers
        self.layers = nn.ModuleList(
            [GraphAttention(dims[i], dims[i + 1], heads)
             for i in range(n_layers)])
        self.att_pool = nn.Sequential(nn.Linear(d_hid, d_hid // 2), nn.Tanh(),
                                      nn.Linear(d_hid // 2, 1))
        self.out_dim = d_hid * 2      # mean-pool  +  attention-pool

    def forward(self, x, src, dst, ew, batch, n_graph):
        for l in self.layers:
            x = l(x, src, dst, ew)
        # ---- mean pooling ---------------------------------------------------
        s = torch.zeros(n_graph, x.shape[1], device=x.device)
        s.index_add_(0, batch, x)
        cnt = torch.zeros(n_graph, device=x.device)
        cnt.index_add_(0, batch, torch.ones_like(batch, dtype=torch.float32))
        mean_p = s / cnt.clamp(min=1)[:, None]
        # ---- attention pooling ---------------------------------------------
        w = self.att_pool(x)
        w = torch.exp(w - _segment_max(w, batch, n_graph)[batch])
        den = torch.zeros(n_graph, 1, device=x.device)
        den.index_add_(0, batch, w)
        att_p = torch.zeros(n_graph, x.shape[1], device=x.device)
        att_p.index_add_(0, batch, w * x)
        att_p = att_p / den.clamp(min=1e-8)
        return torch.cat([mean_p, att_p], -1)


class GAT_TST(nn.Module):
    """Graph-Attention Network + Time-Series Transformer baseline."""

    def __init__(self, d_node=8, d_hid=64, g_layers=3, d_model=128,
                 t_layers=2, heads=4, d_fuse=128):
        super().__init__()
        self.g = nn.ModuleList([GraphAttention(d_node if i == 0 else d_hid, d_hid, 4)
                                for i in range(g_layers)])
        self.tf = TrajEncoder(7, d_model, t_layers, heads)
        self.norm = nn.LayerNorm(d_hid)
        self.shared = SharedEncoder(d_hid + self.tf.out_dim, d_fuse, 2)
        self.head = MultiTaskHead(d_fuse, 64, 3)

    def forward(self, batch):
        X, src, dst, ew, K, b, ng, seq = batch
        for l in self.g:
            X = l(X, src, dst, ew)
        g = self.norm(_flat_graph_mean(X, b, ng))
        z = self.shared(torch.cat([g, self.tf(seq)], -1))
        return self.head(z)


class STGCN(nn.Module):
    """Spatio-temporal GCN: graph convolution followed by temporal convolution."""

    def __init__(self, d_node=8, d_hid=64, d_seq=7, d=128, dropout=0.1):
        super().__init__()
        self.g1 = EdgeGatedSAGE(d_node, d_hid, 3)
        self.g2 = EdgeGatedSAGE(d_hid, d_hid, 3)
        self.tconv = nn.Sequential(
            nn.Conv1d(d_seq, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, d, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.fuse = nn.Linear(d_hid + d, d)
        self.norm = nn.LayerNorm(d)
        self.head = MultiTaskHead(d, 64, 3)

    def forward(self, batch):
        X, src, dst, ew, K, b, ng, seq = batch
        X = self.g2(self.g1(X, src, dst, ew), src, dst, ew)
        g = _flat_graph_mean(X, b, ng)
        t = self.tconv(seq.transpose(1, 2)).squeeze(-1)
        h = self.norm(F.relu(self.fuse(torch.cat([g, t], -1))))
        return self.head(h)


class GMAN(nn.Module):
    """Graph Multi-Attention Network: multi-head graph attention + temporal attention."""

    def __init__(self, d_node=8, d_hid=64, g_layers=2, d_seq=7, d=128, heads=4):
        super().__init__()
        self.g = nn.ModuleList([GraphAttention(d_node if i == 0 else d_hid, d_hid, heads)
                                for i in range(g_layers)])
        # the 7 input channels are not divisible by the head count, so the
        # trajectory is projected to d before temporal multi-head attention
        self.tin = nn.Linear(d_seq, d)
        self.tatt = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_hid + d)
        self.head = MultiTaskHead(d_hid + d, 64, 3)

    def forward(self, batch):
        X, src, dst, ew, K, b, ng, seq = batch
        for l in self.g:
            X = l(X, src, dst, ew)
        g = _flat_graph_mean(X, b, ng)
        tq = self.tin(seq)
        ta, _ = self.tatt(tq, tq, tq)
        t = ta.mean(1)
        return self.head(self.norm(torch.cat([g, t], -1)))


class CNNLSTM(nn.Module):
    """CNN + LSTM on the electrochemical trajectory (no graph branch)."""

    def __init__(self, d_seq=7, d=128, hidden=96, layers=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(d_seq, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(64, hidden, layers, batch_first=True, bidirectional=True)
        self.head = MultiTaskHead(hidden * 2, 64, 3)

    def forward(self, batch):
        X, src, dst, ew, K, b, ng, seq = batch
        h = self.cnn(seq.transpose(1, 2)).transpose(1, 2)
        o, _ = self.lstm(h)
        return self.head(o[:, -1])
