"""Pure, vectorized, cost-aware backtest engine — no torch / pandas / network, so
it's fast and trivially unit-testable. This is the *measuring stick*: every model
change should be judged by what it does to these numbers, not by validation loss.

Convention (no look-ahead): ``signal[i]`` is the desired position for the bar that
*follows* — i.e. the position held over ``[i, i+1]`` earns ``close[i+1]/close[i]-1``.
Costs (exchange fee + slippage) are charged on every change in position size
(turnover), in basis points of notional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# 5-minute bars, 24/7 (crypto) — annualization factor for Sharpe/Sortino. The
# driver overrides this per asset (regular-hours equities trade far fewer bars/yr).
BARS_PER_YEAR_5M = 365 * 24 * 12  # 105_120


def directional_signal(p_long, p_short, *, min_confidence: float = 0.4,
                       min_edge: float = 0.0, allow_short: bool = True) -> np.ndarray:
    """Map per-bar class probabilities → a position in {-1, 0, +1}.

    Goes long when ``p_long`` is the top class, ``p_long ≥ min_confidence`` AND the
    edge ``p_long - p_short ≥ min_edge``; symmetric for short; otherwise flat. This
    mirrors the live uncertainty/edge gate so the backtest reflects how the agent
    actually decides. Tightening the gate ⇒ fewer, higher-quality trades."""
    p_long = np.asarray(p_long, dtype=np.float64).ravel()
    p_short = np.asarray(p_short, dtype=np.float64).ravel()
    edge = p_long - p_short
    sig = np.zeros_like(p_long)
    long_ok = (edge >= min_edge) & (p_long >= min_confidence) & (p_long >= p_short)
    sig[long_ok] = 1.0
    if allow_short:
        short_ok = (-edge >= min_edge) & (p_short >= min_confidence) & (p_short > p_long)
        sig[short_ok] = -1.0
    return sig


@dataclass
class BacktestResult:
    equity: np.ndarray                 # equity curve, starts at 1.0
    net_returns: np.ndarray            # per-bar net (after costs) strategy returns
    position: np.ndarray               # held position per bar in [-1, 1]
    trades: List[Dict] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def _max_drawdown(equity: np.ndarray) -> float:
    """Most negative peak-to-trough drawdown of an equity curve (≤ 0)."""
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / np.where(peak > 0, peak, 1.0) - 1.0
    return float(dd.min())


def compute_metrics(net_returns: np.ndarray, equity: np.ndarray,
                    trades: List[Dict], bars_per_year: float = BARS_PER_YEAR_5M) -> Dict[str, float]:
    """Risk/return summary from per-bar net returns + the equity curve + trade list."""
    r = np.asarray(net_returns, dtype=np.float64)
    n = r.size
    out: Dict[str, float] = {}
    mean = float(r.mean()) if n else 0.0
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    downside = r[r < 0]
    dstd = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
    ann = float(np.sqrt(bars_per_year))

    out["bars"] = float(n)
    out["total_return"] = float(equity[-1] - 1.0) if equity.size else 0.0
    out["ann_return"] = 0.0
    if equity.size and n and equity[-1] > 0:
        with np.errstate(over="ignore", invalid="ignore"):
            ann = equity[-1] ** (bars_per_year / n) - 1.0
        out["ann_return"] = float(ann) if np.isfinite(ann) else 0.0
    out["ann_vol"] = std * ann
    out["sharpe"] = (mean / std * ann) if std > 1e-12 else 0.0
    out["sortino"] = (mean / dstd * ann) if dstd > 1e-12 else 0.0
    out["max_drawdown"] = _max_drawdown(equity)
    # Trade-level stats
    wins = [t for t in trades if t["net"] > 0]
    losses = [t for t in trades if t["net"] < 0]
    gross_win = float(sum(t["net"] for t in wins))
    gross_loss = float(-sum(t["net"] for t in losses))
    out["num_trades"] = float(len(trades))
    out["hit_rate"] = float(len(wins) / len(trades)) if trades else 0.0
    out["profit_factor"] = float(gross_win / gross_loss) if gross_loss > 1e-12 else (np.inf if gross_win > 0 else 0.0)
    out["avg_trade"] = float(np.mean([t["net"] for t in trades])) if trades else 0.0
    return out


def _segment_trades(position: np.ndarray, net_returns: np.ndarray) -> List[Dict]:
    """Group consecutive same-SIGN position bars into trades; each trade's ``net``
    is the compounded net return earned while that position was held. Grouping by
    sign (not exact size) keeps a partial scale-out within the same trade."""
    sign = np.sign(np.asarray(position, dtype=np.float64))
    trades: List[Dict] = []
    n = len(position)
    i = 1
    while i < n:
        d = sign[i - 1]                           # sign of position held into bar i
        if d == 0:
            i += 1
            continue
        j = i
        # extend while the held position keeps the same sign
        while j < n and sign[j - 1] == d:
            j += 1
        seg = net_returns[i:j]
        trades.append({
            "entry": int(i - 1), "exit": int(j - 1), "dir": float(d),
            "bars": int(j - i),
            "net": float(np.prod(1.0 + seg) - 1.0) if seg.size else 0.0,
        })
        i = j
    return trades


def run_backtest(close, signal, *, fee_bps: float = 10.0, slippage_bps: float = 5.0,
                 allow_short: bool = True, bars_per_year: float = BARS_PER_YEAR_5M) -> BacktestResult:
    """Event-driven long/flat/short backtest.

    Parameters
    ----------
    close   : (N,) price series.
    signal  : (N,) desired position for the NEXT bar in [-1, 1]
              (sign = direction, magnitude = size); ``signal[i]`` is held over [i, i+1].
    fee_bps, slippage_bps : per-side costs in basis points, charged on turnover.
    allow_short : if False, negative signals are clipped to 0 (long/flat only).

    No look-ahead: a bar's return uses the position decided on the *previous* bar.
    """
    close = np.asarray(close, dtype=np.float64).ravel()
    pos = np.asarray(signal, dtype=np.float64).ravel().copy()
    n = close.shape[0]
    if pos.shape[0] != n:
        raise ValueError(f"close ({n}) and signal ({pos.shape[0]}) must be the same length")
    pos = np.clip(pos, -1.0, 1.0)
    if not allow_short:
        pos = np.clip(pos, 0.0, 1.0)

    bar_ret = np.zeros(n)
    bar_ret[1:] = close[1:] / np.where(close[:-1] != 0, close[:-1], np.nan)[:] - 1.0
    bar_ret = np.nan_to_num(bar_ret, nan=0.0, posinf=0.0, neginf=0.0)

    # Position held during (i-1 -> i) earns bar_ret[i]; turnover at i = |pos[i]-pos[i-1]|.
    gross = np.zeros(n)
    gross[1:] = pos[:-1] * bar_ret[1:]
    turn = np.zeros(n)
    turn[0] = abs(pos[0])
    turn[1:] = np.abs(pos[1:] - pos[:-1])
    cost = turn * (fee_bps + slippage_bps) / 1e4
    net = gross - cost
    equity = np.cumprod(1.0 + net)

    trades = _segment_trades(pos, net)
    metrics = compute_metrics(net, equity, trades, bars_per_year=bars_per_year)
    metrics["total_cost"] = float(cost.sum())
    metrics["turnover"] = float(turn.sum())
    return BacktestResult(equity=equity, net_returns=net, position=pos, trades=trades, metrics=metrics)


def atr_from_ohlc(high, low, close, window: int = 14) -> np.ndarray:
    """Wilder-ish ATR (rolling mean of True Range) in PRICE units, with a robust
    warm-up fill so stops are always defined."""
    high = np.asarray(high, float).ravel()
    low = np.asarray(low, float).ravel()
    close = np.asarray(close, float).ravel()
    n = len(close)
    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    out = np.copy(tr)
    if n:
        c = np.cumsum(tr)
        for i in range(n):
            lo = max(0, i - window + 1)
            out[i] = (c[i] - (c[lo - 1] if lo > 0 else 0.0)) / (i - lo + 1)
    med = float(np.median(out[out > 0])) if (out > 0).any() else float(close.mean() * 0.005 + 1e-9)
    return np.where(out > 0, out, med)


def run_exec_backtest(close, high, low, atr, signal, *, forecast_vol=None,
                      target_vol: float = 0.01, max_size: float = 1.0,
                      stop_atr: float = 2.0, trail_atr: float = 3.0, breakeven_atr: float = 1.0,
                      tp1_atr: float = 2.0, scale_out_frac: float = 0.5, max_hold: int = None,
                      fee_bps: float = 10.0, slippage_bps: float = 5.0, allow_short: bool = True,
                      bars_per_year: float = BARS_PER_YEAR_5M) -> BacktestResult:
    """Execution-aware backtest (Cycle 5): on top of the gated direction ``signal``
    it adds the realistic exit/sizing logic that wins or loses most of the edge —

      • vol-targeted, capped sizing: ``size = clip(target_vol/forecast_vol, 0, max_size)``
        (scales exposure INVERSELY with forecast volatility — your "risk map");
      • ATR stop-loss, then an ATR TRAILING stop ratcheting from the favourable extreme;
      • move-to-breakeven once the trade is ``breakeven_atr`` in profit;
      • partial scale-out of ``scale_out_frac`` at ``tp1_atr``, trailing the remainder;
      • optional time stop; exit/flip on an opposite or flat signal.

    A stateful loop builds the (signed, variable-size) position path; PnL/costs are
    then accounted exactly like ``run_backtest``. Bar-resolution: exits realise at the
    bar close (intra-bar fill prices aren't knowable from OHLC). Pass the same
    per-bar ``forecast_vol`` used for labels and ``atr`` from ``atr_from_ohlc``.
    """
    close = np.asarray(close, float).ravel(); high = np.asarray(high, float).ravel()
    low = np.asarray(low, float).ravel();     atr = np.asarray(atr, float).ravel()
    sig = np.asarray(signal, float).ravel()
    n = len(close)
    if forecast_vol is None:
        size_at = np.full(n, float(max_size))
    else:
        fv = np.asarray(forecast_vol, float).ravel()
        size_at = np.clip(target_vol / np.where(fv > 1e-9, fv, 1e-9), 0.0, float(max_size))

    position = np.zeros(n)
    in_pos = False; d = 0; size = 0.0; entry = 0.0; stop = 0.0; peak = 0.0; scaled = False; hold = 0
    blocked = 0   # direction we were stopped/timed out of — suppress immediate same-dir
                  # re-entry (else a persistent signal just re-buys and the stop is pointless)
                  # until the signal resets (goes flat or flips).
    for i in range(n):
        px = close[i]
        if blocked != 0 and (sig[i] == 0 or np.sign(sig[i]) != blocked):
            blocked = 0
        if in_pos:
            hold += 1
            peak = max(peak, high[i]) if d > 0 else min(peak, low[i])
            fav = (peak - entry) * d
            if breakeven_atr and fav >= breakeven_atr * atr[i]:           # lock to breakeven
                stop = max(stop, entry) if d > 0 else min(stop, entry)
            stop = (max(stop, peak - trail_atr * atr[i]) if d > 0          # ratchet trailing stop
                    else min(stop, peak + trail_atr * atr[i]))
            if (not scaled) and scale_out_frac > 0 and fav >= tp1_atr * atr[i]:
                size *= (1.0 - scale_out_frac); scaled = True              # bank partial profit
            stop_hit = (low[i] <= stop) if d > 0 else (high[i] >= stop)
            timed = (max_hold is not None and hold >= max_hold)
            flip = (sig[i] == 0) or (sig[i] != 0 and np.sign(sig[i]) != d)
            if stop_hit or timed:
                blocked = d                                               # don't re-buy into the same move
                in_pos = False; d = 0; size = 0.0; scaled = False; hold = 0
            elif flip:
                in_pos = False; d = 0; size = 0.0; scaled = False; hold = 0
        if (not in_pos) and sig[i] != 0:
            nd = int(np.sign(sig[i]))
            if nd != blocked and (nd > 0 or allow_short):
                in_pos, d, entry, peak, scaled, hold = True, nd, px, px, False, 0
                size = float(size_at[i])
                stop = entry - stop_atr * atr[i] if nd > 0 else entry + stop_atr * atr[i]
        position[i] = d * size if in_pos else 0.0

    bar_ret = np.zeros(n)
    bar_ret[1:] = close[1:] / np.where(close[:-1] != 0, close[:-1], np.nan) - 1.0
    bar_ret = np.nan_to_num(bar_ret)
    gross = np.zeros(n); gross[1:] = position[:-1] * bar_ret[1:]
    turn = np.zeros(n); turn[0] = abs(position[0]); turn[1:] = np.abs(position[1:] - position[:-1])
    cost = turn * (fee_bps + slippage_bps) / 1e4
    net = gross - cost
    equity = np.cumprod(1.0 + net)
    trades = _segment_trades(position, net)
    metrics = compute_metrics(net, equity, trades, bars_per_year=bars_per_year)
    metrics["total_cost"] = float(cost.sum()); metrics["turnover"] = float(turn.sum())
    metrics["avg_exposure"] = float(np.mean(np.abs(position)))
    return BacktestResult(equity=equity, net_returns=net, position=position, trades=trades, metrics=metrics)
