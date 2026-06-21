"""Multi-strategy portfolio backtester — the measuring stick for the strategy book.

Where ``engine.run_backtest`` scores ONE position path on ONE asset, this combines MANY
strategies across MANY assets into a single book, with the things that actually drive a
real Sharpe:
  • per-strategy PnL attribution (so decaying edges can be found and retired);
  • allocation across strategies (equal or inverse-vol "risk parity");
  • portfolio-level VOLATILITY TARGETING (causal — the biggest documented Sharpe lever);
  • the cross-strategy return correlation matrix (diversification IS the edge).

Pure numpy/pandas + the engine's metrics — fast and unit-testable. Costs are charged on
turnover per (strategy, symbol) exactly like the single-asset engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from backend.backtest.engine import (
    compute_metrics, regression_alpha_beta, _max_drawdown, BARS_PER_YEAR_5M,
)
from backend.strategies.base import Strategy


def align_panel(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Inner-join all symbols on timestamp → equal-length, same-calendar frames. REQUIRED
    before a cross-sectional / pairs strategy + its backtest, so that "position index t"
    means the same wall-clock bar across every symbol (otherwise ranking/aggregation mixes
    dates). Per-symbol-independent strategies (e.g. ts_momentum) don't need this."""
    frames = {}
    for s, df in data.items():
        d = df.copy()
        if "timestamp" not in d.columns:
            d = d.reset_index().rename(columns={d.index.name or "index": "timestamp"})
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        frames[s] = d.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    common = None
    for d in frames.values():
        common = d.index if common is None else common.intersection(d.index)
    if common is None or len(common) == 0:
        raise ValueError("align_panel: symbols share no common timestamps")
    common = common.sort_values()
    return {s: frames[s].loc[common].reset_index() for s in frames}


def _bar_returns(close: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64).ravel()
    r = np.zeros(close.shape[0])
    r[1:] = close[1:] / np.where(close[:-1] != 0, close[:-1], np.nan) - 1.0
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def strategy_net_returns(positions: Dict[str, np.ndarray], data: Dict[str, pd.DataFrame],
                         fee_bps: float, slippage_bps: float) -> np.ndarray:
    """Per-bar net return series for ONE strategy: dollar-equal across the symbols it trades.
    ``position[t-1]`` earns ``bar_ret[t]``; turnover ``|Δposition|`` is charged costs."""
    cost = (fee_bps + slippage_bps) / 1e4
    per_sym = []
    length = None
    for sym, pos in positions.items():
        if sym not in data or pos is None or len(pos) < 2:
            continue
        pos = np.asarray(pos, dtype=np.float64).ravel()
        ret = _bar_returns(data[sym]["close"].to_numpy())
        m = min(len(pos), len(ret))
        pos, ret = pos[:m], ret[:m]
        gross = np.zeros(m); gross[1:] = pos[:-1] * ret[1:]
        turn = np.zeros(m); turn[0] = abs(pos[0]); turn[1:] = np.abs(pos[1:] - pos[:-1])
        net = gross - turn * cost
        per_sym.append(net)
        length = m if length is None else min(length, m)
    if not per_sym:
        return np.zeros(0)
    stacked = np.vstack([s[:length] for s in per_sym])
    return stacked.mean(axis=0)            # equal-dollar across the strategy's symbols


def drawdown_degear(net: np.ndarray, dd_threshold: float = 0.10,
                    floor: float = 0.25) -> np.ndarray:
    """Causal drawdown de-gearing: when the book is in a peak-to-trough drawdown beyond
    ``dd_threshold``, scale the NEXT bar's exposure down (linearly toward ``floor``), and
    restore it as the equity recovers. This caps the depth of a losing run — the discipline
    the −90% mean-reversion DD showed was missing (vol-targeting alone will happily lever a
    loser). The scale at bar t uses only the drawdown realised through t-1 (no look-ahead).
    Returns the de-geared return series."""
    net = np.asarray(net, dtype=np.float64).ravel()
    out = np.zeros_like(net)
    eq = 1.0
    peak = 1.0
    for t in range(net.size):
        dd = 1.0 - eq / peak if peak > 0 else 0.0          # current drawdown (>=0), known pre-bar
        if dd > dd_threshold:
            scale = max(floor, 1.0 - (dd - dd_threshold) / max(1.0 - dd_threshold, 1e-9))
        else:
            scale = 1.0
        r = net[t] * scale
        out[t] = r
        eq *= (1.0 + r)
        peak = max(peak, eq)
    return out


def vol_target_scale(net: np.ndarray, target_ann_vol: float, bars_per_year: float,
                     window: int = 96, max_leverage: float = 3.0) -> np.ndarray:
    """Causal leverage path that scales returns toward a target annualised vol: leverage at
    bar t uses realised vol through t-1 only (shifted), clipped to [0, max_leverage]."""
    net = np.asarray(net, dtype=np.float64).ravel()
    if net.size == 0:
        return net
    roll = pd.Series(net).rolling(window, min_periods=max(10, window // 4)).std()
    ann = roll.to_numpy() * np.sqrt(bars_per_year)
    with np.errstate(divide="ignore", invalid="ignore"):
        lev = np.where(ann > 1e-9, target_ann_vol / ann, 0.0)
    lev = np.clip(np.nan_to_num(lev, nan=0.0), 0.0, max_leverage)
    lev = np.concatenate([[0.0], lev[:-1]])     # shift → only past vol scales the current bar
    return lev


@dataclass
class PortfolioResult:
    equity: np.ndarray
    net_returns: np.ndarray
    strategy_returns: Dict[str, np.ndarray]
    metrics: Dict[str, float]
    strategy_metrics: Dict[str, Dict[str, float]]
    correlation: pd.DataFrame = field(default_factory=pd.DataFrame)


def portfolio_backtest(strategies: Dict[str, Strategy], data: Dict[str, pd.DataFrame], *,
                       fee_bps: float = 10.0, slippage_bps: float = 5.0,
                       target_ann_vol: float = 0.12, bars_per_year: float = BARS_PER_YEAR_5M,
                       alloc: str = "risk_parity", market_symbol: str = None,
                       max_leverage: float = 2.0, degear_threshold: float = 0.15,
                       degear_floor: float = 0.25) -> PortfolioResult:
    """Combine strategies into one vol-targeted book. ``alloc`` ∈ {equal, risk_parity}
    (risk_parity = inverse-vol weights). ``market_symbol`` (e.g. BTCUSDT) is the benchmark
    for regression α/β; defaults to an equal-weight basket of all symbols.

    Risk overlay: vol-targeting is capped at ``max_leverage`` (conservative 2.0 default — so a
    low-vol/losing leg can't be levered into a blow-up) and the book is drawdown-de-geared
    beyond ``degear_threshold`` (0 disables). These cap the depth of a losing run."""
    # 1) per-strategy net return series (truncate to the common length) + turnover
    strat_rets: Dict[str, np.ndarray] = {}
    turnover_total = 0.0
    for name, strat in strategies.items():
        positions = strat.generate_positions(data)
        r = strategy_net_returns(positions, data, fee_bps, slippage_bps)
        if r.size:
            strat_rets[name] = r
            for pos in positions.values():
                p = np.asarray(pos, dtype=np.float64).ravel()
                if p.size:
                    turnover_total += float(np.abs(np.diff(p, prepend=0.0)).sum())
    if not strat_rets:
        raise RuntimeError("no strategy produced returns")
    L = min(len(r) for r in strat_rets.values())
    strat_rets = {k: v[-L:] for k, v in strat_rets.items()}

    # 2) allocate across strategies (equal, or inverse-vol risk parity)
    names = list(strat_rets)
    R = np.vstack([strat_rets[n] for n in names])          # (S, L)
    if alloc == "equal":
        w = np.full(len(names), 1.0 / len(names))
    else:
        vol = R.std(axis=1)
        inv = np.where(vol > 1e-12, 1.0 / vol, 0.0)
        w = inv / inv.sum() if inv.sum() > 0 else np.full(len(names), 1.0 / len(names))
    combined = (w[:, None] * R).sum(axis=0)                # (L,)

    # 3) portfolio volatility targeting (causal) + drawdown de-gearing
    lev = vol_target_scale(combined, target_ann_vol, bars_per_year, max_leverage=max_leverage)
    net = combined * lev
    if degear_threshold and degear_threshold > 0:
        net = drawdown_degear(net, dd_threshold=degear_threshold, floor=degear_floor)
    equity = np.cumprod(1.0 + net)

    # 4) metrics — portfolio + per-strategy attribution
    metrics = compute_metrics(net, equity, trades=[], bars_per_year=bars_per_year)
    if market_symbol and market_symbol in data:
        mkt = _bar_returns(data[market_symbol]["close"].to_numpy())[-L:]
    else:
        mkt = np.mean([_bar_returns(data[s]["close"].to_numpy())[-L:] for s in data], axis=0)
    a, b = regression_alpha_beta(net, mkt, bars_per_year=bars_per_year)
    metrics["reg_alpha"] = a
    metrics["beta"] = b
    metrics["leverage_mean"] = float(np.mean(lev))
    metrics["turnover"] = turnover_total

    strat_metrics: Dict[str, Dict[str, float]] = {}
    for n in names:
        eq = np.cumprod(1.0 + strat_rets[n])
        sm = compute_metrics(strat_rets[n], eq, trades=[], bars_per_year=bars_per_year)
        strat_metrics[n] = {k: sm[k] for k in ("sharpe", "sortino", "ann_return", "max_drawdown")}

    corr = pd.DataFrame(R.T, columns=names).corr() if len(names) > 1 else pd.DataFrame()
    return PortfolioResult(equity=equity, net_returns=net, strategy_returns=strat_rets,
                           metrics=metrics, strategy_metrics=strat_metrics, correlation=corr)
