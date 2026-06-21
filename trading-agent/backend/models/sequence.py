"""QuantileTCN — the headline neural model of the Phase 2 hybrid.

A causal TCN (parallel, full receptive field, less overfit-prone than a deep LSTM on weak
signal — the owner's own probe showed the LSTM overfit) over the 60-bar window, with a
symbol embedding and OPTIONAL explicit regime conditioning, emitting per-horizon QUANTILES
{p10, p50, p90} of the vol-normalized forward return:

    edge = p50            (calibrated directional conviction → drives μ̂/σ̂² sizing)
    uncertainty = p90-p10 (interval width → the gate/Kelly shrink; no MC-dropout needed)

The temporal core (TemporalConvNet) and attention are reused from
``backend.agents.improved_model`` so the trunk stays identical to the live policy net's TCN
option. The quantile heads use a CUMULATIVE-softplus parameterisation so the predicted
quantiles can never cross (p10 ≤ p50 ≤ p90 by construction).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from backend.agents.improved_model import TemporalConvNet, AttentionLayer
    from backend.signals import feature_spec as fs
except ImportError:  # pragma: no cover
    from agents.improved_model import TemporalConvNet, AttentionLayer
    from signals import feature_spec as fs


class QuantileTCN(nn.Module):
    def __init__(self, input_size: int = None, hidden: int = 128, num_symbols: int = 18,
                 symbol_embed_dim: int = 16, num_horizons: int = 3,
                 quantiles=(0.1, 0.5, 0.9), dropout: float = 0.1, regime_dim: int = 0):
        super().__init__()
        self.input_size = int(input_size if input_size is not None else fs.INPUT)
        self.quantiles = tuple(quantiles)
        self.num_horizons = int(num_horizons)
        self.regime_dim = int(regime_dim)
        Q = len(self.quantiles)

        self.symbol_embedding = nn.Embedding(num_symbols, symbol_embed_dim)
        self.tcn = TemporalConvNet(self.input_size + symbol_embed_dim, hidden, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden)
        self.attention = AttentionLayer(hidden)
        self.dropout = nn.Dropout(dropout)
        self.shared = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
        )
        # One quantile head per horizon. Head output is (B, Q) raw; forward() turns it into
        # NON-CROSSING quantiles via base + cumulative softplus increments.
        head_in = 64 + self.regime_dim
        self.q_heads = nn.ModuleList([nn.Linear(head_in, Q) for _ in range(self.num_horizons)])

    def _trunk(self, x: torch.Tensor, symbol_ids: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        emb = self.symbol_embedding(symbol_ids).unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([x, emb], dim=-1)
        core = self.tcn(x)                                  # (B, T, hidden)
        core = self.layer_norm(core)
        context, _ = self.attention(core)                   # (B, hidden)
        return self.shared(self.dropout(context))           # (B, 64)

    @staticmethod
    def _monotonic(raw: torch.Tensor) -> torch.Tensor:
        """(B, Q) raw → non-decreasing quantiles: q0 = raw[:,0]; q_k = q_{k-1} + softplus(raw[:,k])."""
        base = raw[:, :1]
        inc = F.softplus(raw[:, 1:])
        return torch.cat([base, base + torch.cumsum(inc, dim=1)], dim=1)

    def forward(self, x: torch.Tensor, symbol_ids: torch.Tensor, regime: torch.Tensor = None):
        """Returns list (len num_horizons) of (B, Q) monotonic quantile predictions."""
        shared = self._trunk(x, symbol_ids)
        if self.regime_dim:
            if regime is None:
                regime = torch.zeros(shared.shape[0], self.regime_dim,
                                     dtype=shared.dtype, device=shared.device)
            shared = torch.cat([shared, regime], dim=-1)
        return [self._monotonic(head(shared)) for head in self.q_heads]

    @torch.no_grad()
    def edge_and_uncertainty(self, x, symbol_ids, horizon_idx: int = 0, regime=None):
        """Convenience for inference: (edge=p50, uncertainty=p90-p10) at one horizon.
        Assumes quantiles are ordered low→high (e.g. 0.1, 0.5, 0.9)."""
        self.eval()
        q = self.forward(x, symbol_ids, regime)[horizon_idx]   # (B, Q)
        med_idx = len(self.quantiles) // 2
        edge = q[:, med_idx]
        uncertainty = q[:, -1] - q[:, 0]
        return edge, uncertainty
