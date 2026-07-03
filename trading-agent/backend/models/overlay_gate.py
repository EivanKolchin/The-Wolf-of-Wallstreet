"""Live overlay gate — the OOS-validated DMN ensemble scaling the TS-momentum sleeve.

Bridges the offline result (scripts/train_quantile_tcn.py A/B: DMN gate +0.37 Sharpe of
TIMING skill vs the constant-scaling control on the untouched test tail) to the live agent.

Contract (identical to the backtest, deliberately):
  * The gate only SCALES positions the primary strategy already wants — weight in
    [floor, 1] — never originates, never flips. An uninformative model ≈ the
    Sharpe-preserving constant-scaling control.
  * Ensemble = mean over the seed checkpoints saved by the trainer.
  * Online adaptation is scalar-only: ConvictionShrink (EWMA hit-rate) tempers the DMN's
    saturated ±1 head per symbol; OnlineConformal recalibrates quantile-model intervals.
    Weights are frozen — retrains stay scheduled and walk-forward-judged.

Data: the model consumes the 58-feature hybrid contract, which needs 5m/1h/4h history
(z-score warm-up 1000 + window 60 → ≥ ~1100 5m bars). BinanceMultiTFProvider (in
live_providers) serves exactly that; anything with ``get_multi_tf(symbol) -> {"5m","1h","4h"}``
duck-types in for tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import structlog
    logger = structlog.get_logger("overlay_gate")
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger("overlay_gate")

SEQ_LEN = 60
MIN_5M_BARS = 1100          # z-score warm-up (1000) + sequence window (60) + slack


def gate_weight(pos, edge, unc, *, ir_cap: float = 1.0, floor: float = 0.25):
    """The single source of the gate math (the trainer's ``gated_positions`` delegates here).

    conviction = clip(sign(pos)·edge/unc, ±ir_cap)/ir_cap ∈ [-1, 1]
    weight     = floor + (1-floor)·(conviction+1)/2      ∈ [floor, 1]
    Flat positions stay flat. Vectorized; scalars in → scalar out."""
    pos = np.asarray(pos, dtype=np.float64)
    unc = np.maximum(np.asarray(unc, dtype=np.float64), 1e-9)
    ir = np.clip(np.sign(pos) * np.asarray(edge, dtype=np.float64) / unc, -ir_cap, ir_cap) / ir_cap
    w = floor + (1.0 - floor) * (ir + 1.0) / 2.0
    out = np.where(pos == 0.0, 0.0, w)
    return float(out) if out.ndim == 0 else out


def load_overlay_ensemble(ckpt_glob: str = "models/quantile_tcn_overlay_sharpe_seed*.pt",
                          device: str = "cpu") -> List:
    """Load the seed ensemble saved by scripts/train_quantile_tcn.py. Empty list when no
    checkpoints exist (the gate then degrades to a no-op)."""
    import torch
    from backend.models.sequence import QuantileTCN, DMNPositionNet
    from backend.agents.improved_model import SYMBOL_TO_ID
    models = []
    for p in sorted(Path(".").glob(ckpt_glob)):
        try:
            ck = torch.load(p, map_location=device, weights_only=False)
            cls = DMNPositionNet if ck.get("objective") == "sharpe" else QuantileTCN
            m = cls(input_size=ck["feature_dim"], hidden=ck.get("hidden", 64),
                    num_symbols=len(SYMBOL_TO_ID), num_horizons=len(ck["horizons"]),
                    quantiles=tuple(ck["quantiles"]))
            m.load_state_dict(ck["model_state_dict"])
            m.eval()
            models.append(m)
        except Exception as e:
            logger.warning("overlay_ckpt_load_failed", path=str(p), error=str(e)[:100])
    return models


@dataclass
class OverlayGate:
    """Callable-by-the-agent gate: ``apply(targets, perp_symbols)`` returns adjusted targets.

    ``bar_provider`` must expose ``get_multi_tf(symbol) -> {"5m": df, "1h": df, "4h": df}``.
    Feature tails are cached per symbol for ``ttl_seconds`` so a rebalance costs one model
    forward, not one feature build, per symbol per 4h."""
    models: List = field(default_factory=list)
    bar_provider: object = None
    ir_cap: float = 1.0
    floor: float = 0.25
    horizon_idx: int = 1                      # H+48 (4h) — the sleeve's decision cadence
    ttl_seconds: float = 3000.0
    shrink: object = None                     # ConvictionShrink (built by default)
    conformal: object = None                  # OnlineConformal (quantile models only)
    _cache: Dict[str, tuple] = field(default_factory=dict)   # sym -> (t, edge, unc)
    _last_call: Dict[str, tuple] = field(default_factory=dict)  # sym -> (edge, close) for update()

    def __post_init__(self):
        if self.shrink is None:
            from backend.models.conformal import ConvictionShrink
            self.shrink = ConvictionShrink()
        if self.conformal is None:
            from backend.models.conformal import OnlineConformal
            self.conformal = OnlineConformal()

    # ------------------------------------------------------------- inference
    def _edge_unc(self, symbol: str) -> Optional[tuple]:
        import time as _t
        hit = self._cache.get(symbol)
        if hit and _t.time() - hit[0] < self.ttl_seconds:
            return hit[1], hit[2]
        if self.bar_provider is None or not self.models:
            return None
        try:
            tfs = self.bar_provider.get_multi_tf(symbol)
            if not tfs or any(k not in tfs or tfs[k] is None for k in ("5m", "1h", "4h")):
                return None
            if len(tfs["5m"]) < MIN_5M_BARS:
                logger.info("overlay_insufficient_history", symbol=symbol, bars=len(tfs["5m"]))
                return None
            import torch
            from backend.features.pipeline import build_hybrid_matrix
            from backend.models.sequence import DMNPositionNet
            from backend.agents.improved_model import SYMBOL_TO_ID
            X = build_hybrid_matrix(tfs["5m"], tfs["1h"], tfs["4h"])
            xb = torch.from_numpy(X[-SEQ_LEN:][None].astype(np.float32))
            sb = torch.tensor([SYMBOL_TO_ID.get(symbol, 0)], dtype=torch.long)
            es, us = [], []
            with torch.no_grad():
                for m in self.models:
                    if isinstance(m, DMNPositionNet):
                        es.append(float(m.forward_position(xb, sb)[0]))
                        us.append(1.0)
                    else:
                        q = m(xb, sb)[self.horizon_idx][0].numpy()
                        es.append(float(q[1]))
                        us.append(float(self.conformal.effective_unc(symbol, q[2] - q[0])))
            edge, unc = float(np.mean(es)), float(np.mean(us))
            self._cache[symbol] = (_t.time(), edge, unc)
            return edge, unc
        except Exception as e:
            logger.warning("overlay_inference_failed", symbol=symbol, error=str(e)[:120])
            return None

    # ------------------------------------------------------------- the gate
    def apply(self, targets: Dict[str, float], perp_symbols) -> Dict[str, dict]:
        """Scale each perp symbol's target weight in place; returns {symbol: note} for
        observability. Symbols with no model/data/inference are left UNTOUCHED — the gate
        fails open to the validated baseline sleeve, never to zero."""
        notes: Dict[str, dict] = {}
        if not self.models:
            return notes
        for s in perp_symbols:
            tw = float(targets.get(s, 0.0))
            if tw == 0.0:
                continue
            eu = self._edge_unc(s)
            if eu is None:
                continue
            edge, unc = eu
            shrunk = edge * self.shrink.shrink(s)          # online skill-tracking temper
            w = gate_weight(tw, shrunk, unc, ir_cap=self.ir_cap, floor=self.floor)
            targets[s] = tw * w                            # scale, never originate/flip
            self._last_call[s] = (edge, None)
            notes[s] = {"edge": round(edge, 4), "unc": round(unc, 4),
                        "shrink": round(self.shrink.shrink(s), 3), "weight": round(w, 3)}
        if notes:
            logger.info("overlay_gate_applied", n=len(notes))
        return notes

    # --------------------------------------------------------- online update
    def record_outcome(self, symbol: str, realized_vol_norm_return: float) -> None:
        """Feed the realized (vol-normalized) return for the last gated decision — updates
        the conviction EWMA (and conformal width for quantile models)."""
        last = self._last_call.get(symbol)
        if not last:
            return
        edge = last[0]
        self.shrink.update(symbol, edge, realized_vol_norm_return)
        self.conformal.update(symbol, edge, 1.0, realized_vol_norm_return)


def make_overlay_gate(settings=None):
    """Build from config, or None when disabled / no checkpoints (fail-open to baseline)."""
    if settings is None:
        from backend.core.config import settings as settings_
        settings = settings_
    if not bool(getattr(settings, "STRATEGY_AGENT_OVERLAY", False)):
        return None
    glob_pat = str(getattr(settings, "OVERLAY_CKPT_GLOB",
                           "models/quantile_tcn_overlay_sharpe_seed*.pt"))
    models = load_overlay_ensemble(glob_pat)
    if not models:
        logger.warning("overlay_enabled_but_no_checkpoints", glob=glob_pat)
        return None
    from backend.agents.live_providers import BinanceMultiTFProvider
    gate = OverlayGate(models=models, bar_provider=BinanceMultiTFProvider(),
                       ir_cap=float(getattr(settings, "OVERLAY_IR_CAP", 1.0)),
                       floor=float(getattr(settings, "OVERLAY_FLOOR", 0.25)))
    logger.info("overlay_gate_built", seeds=len(models))
    return gate
