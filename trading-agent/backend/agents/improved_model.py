"""Canonical live policy network.

This is the unified model the LIVE agent runs. Its *base* (symbol embedding,
LSTM, attention, shared trunk, per-horizon direction heads, size head,
temperature) is identical to the offline pretraining model in
``scripts/pretrain.py`` so offline-trained core weights load straight in.

On top of the base it adds three EXIT heads (stop-loss / take-profit / trailing)
that are trained ONLINE via RL (Phase 3/4), not during offline pretraining — so
when loading an offline checkpoint the exit-head keys are simply missing and are
left freshly initialised (see PersistentTradingModel._load_or_initialise).

NOTE (follow-up): the base class is currently duplicated in scripts/pretrain.py.
When the offline trainer is reworked (Phase 3) pretrain.py should import this base
so the two can never drift. They are identical today.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # works whether imported as backend.agents.* or with backend/ on path
    from signals import feature_spec as fs
except ImportError:  # pragma: no cover
    from backend.signals import feature_spec as fs

# --- architecture constants (must match scripts/pretrain.py for weight compat) ---
INPUT_SIZE = fs.INPUT          # 70
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 3
DROPOUT = 0.3
SYMBOL_EMBED_DIM = 16
HORIZONS = [3, 12, 48]         # candles ahead per direction head (5m → 15m/1h/4h)
NUM_CLASSES = 3                # 0=long, 1=short, 2=hold

# Phase 18 v2: next-K_FUTURE OHLC log-return deltas (small head on shared trunk).
# Emits log-returns; the agent rolls them up into absolute OHLC for the chart.
K_FUTURE = 5

# Symbol registry — shared id space across offline + live (extend, never reorder).
# Phase 7b: stock underlyings appended after the crypto block so existing
# crypto-only checkpoint embedding rows stay at the SAME indices and a v2.1
# checkpoint trained on the older 8-symbol vocabulary still loads cleanly.
SYMBOLS = [
    # ---- crypto (ids 0..7) ----
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "AAVEUSDT",
    "XLMUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    # ---- US stocks (ids 8..12) — Phase 7b ----
    "SNDK", "AMD", "MU", "AXTI", "BE",
]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}

# Exit-head output ranges.
#
# Phase 17: the heads now emit **ATR multiples** (regime-invariant) rather than
# raw fractions. The agent converts ``mult * atr_pct`` to an absolute fraction
# of entry price at inference time, so the SAME trained model adapts cleanly
# from a 0.4%-stdev stock (BE) to a 3%-stdev stock (AMD) without per-ticker
# retraining. Slugs of fixed fractions are kept for the (offline-only)
# legacy ranges so v2.0 checkpoints can still be deserialised — but the live
# loop reads ATR multiples from ``exits['sl_mult'/'tp_mult'/'trail_mult']``.
SL_FRAC_RANGE = (0.003, 0.10)         # legacy (v2.0)
TP_FRAC_RANGE = (0.005, 0.20)         # legacy (v2.0)
TRAIL_FRAC_RANGE = (0.003, 0.10)      # legacy (v2.0)

SL_ATR_MULT_RANGE = (0.5, 5.0)        # 0.5×ATR .. 5×ATR
TP_ATR_MULT_RANGE = (0.5, 10.0)       # 0.5×ATR .. 10×ATR
TRAIL_ATR_MULT_RANGE = (0.3, 5.0)     # 0.3×ATR .. 5×ATR


def _scale(sig: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + sig * (hi - lo)


class AttentionLayer(nn.Module):
    """Additive (Bahdanau-style) attention over the LSTM sequence."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor):
        weights = self.score(lstm_out)          # (B, T, 1)
        weights = F.softmax(weights, dim=1)
        context = (weights * lstm_out).sum(1)   # (B, H)
        return context, weights.squeeze(-1)


class ImprovedTradingLSTM(nn.Module):
    """Unified multi-horizon policy net with symbol embedding + learned exit heads."""

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LSTM_LAYERS,
        dropout: float = DROPOUT,
        num_symbols: int = len(SYMBOLS),
        symbol_embed_dim: int = SYMBOL_EMBED_DIM,
        num_horizons: int = len(HORIZONS),
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.num_horizons = num_horizons
        self.symbol_embedding = nn.Embedding(num_symbols, symbol_embed_dim)

        # A4: dropout + recurrent-core type are config-driven so the same code
        # can train/run an LSTM or a (faster, fewer-param) GRU. MUST match the
        # offline trainer (scripts/pretrain.py) for checkpoints to load.
        try:
            from backend.core.config import settings as _s
            dropout = float(getattr(_s, "NN_DROPOUT", dropout))
            rnn_type = str(getattr(_s, "NN_RNN_TYPE", "lstm")).lower()
        except Exception:
            rnn_type = "lstm"
        self.rnn_type = rnn_type

        lstm_in = input_size + symbol_embed_dim
        _rnn = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.lstm = _rnn(
            input_size=lstm_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,   # causal — no future peek
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attention = AttentionLayer(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # One direction head per horizon — multi-task learning
        self.direction_heads = nn.ModuleList(
            [nn.Linear(64, num_classes) for _ in range(num_horizons)]
        )

        # Position sizing (shared across horizons)
        self.size_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))

        # --- EXIT HEADS (live-only; trained online) ---
        def _exit_head():
            return nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.sl_head = _exit_head()
        self.tp_head = _exit_head()
        self.trail_head = _exit_head()

        # --- OHLC NEXT-K_FUTURE LOG-RETURN HEAD (Phase 18 v2) ---
        # Output is a flat (B, 4*K_FUTURE) tensor of log-returns; the live agent
        # reshapes to (B, K_FUTURE, 4) and rolls up to absolute OHLC for the chart.
        # Tanh bounds the head to [-1, 1] log-return per candle (~e^1 = 2.7× cap).
        self.next_candle_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4 * K_FUTURE), nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    # Names of parameters that only exist on the live model (absent from offline ckpts).
    EXIT_PARAM_PREFIXES = ("sl_head", "tp_head", "trail_head", "next_candle_head")

    def enable_dropout(self):
        """Keep dropout active under eval() for MC-dropout uncertainty sampling."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _trunk(self, x: torch.Tensor, symbol_ids: torch.Tensor):
        B, T, _ = x.shape
        emb = self.symbol_embedding(symbol_ids)        # (B, embed)
        emb = emb.unsqueeze(1).expand(-1, T, -1)       # (B, T, embed)
        x = torch.cat([x, emb], dim=-1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        context, attn_w = self.attention(lstm_out)
        context = self.dropout(context)
        return self.shared(context), attn_w            # (B, 64), (B, T)

    def forward(self, x: torch.Tensor, symbol_ids: torch.Tensor):
        shared, attn_w = self._trunk(x, symbol_ids)
        logits_list = [h(shared) / self.temperature for h in self.direction_heads]
        probs_list = [F.softmax(lg, dim=-1) for lg in logits_list]
        size = self.size_head(shared)                  # (B, 1)
        # Phase 17: each head emits a sigmoid → scale to its ATR-multiple range.
        # The legacy *_FRAC keys still emit fractions for back-compat with v2.0
        # callers (the agent will prefer *_mult and convert via live ATR).
        sl_sig = self.sl_head(shared)
        tp_sig = self.tp_head(shared)
        tr_sig = self.trail_head(shared)
        # Phase 18 v2: (B, K_FUTURE, 4) — log-return deltas for next K candles.
        next_candle_logret = self.next_candle_head(shared).view(-1, K_FUTURE, 4)
        exits = {
            "sl": _scale(sl_sig, *SL_FRAC_RANGE),
            "tp": _scale(tp_sig, *TP_FRAC_RANGE),
            "trail": _scale(tr_sig, *TRAIL_FRAC_RANGE),
            "sl_mult": _scale(sl_sig, *SL_ATR_MULT_RANGE),
            "tp_mult": _scale(tp_sig, *TP_ATR_MULT_RANGE),
            "trail_mult": _scale(tr_sig, *TRAIL_ATR_MULT_RANGE),
            "next_candle_logret": next_candle_logret,
        }
        return logits_list, probs_list, size, exits, attn_w
