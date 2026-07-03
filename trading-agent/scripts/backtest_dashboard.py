#!/usr/bin/env python
r"""Interactive streaming backtest dashboard — strategies vs RANDOM vs the S&P 500.

A FastAPI app to tick one-or-many strategies, run them on a chosen asset/window, and watch the
equity curves draw onto one chart IN REAL TIME (Server-Sent Events — each curve appears the moment
its backtest finishes). Two references are always available: the S&P 500 (buy-&-hold SPY, the
benchmark) and a RANDOM Monte-Carlo band (median + 10-90 pct envelope of random traders under the
same cost model — the "beat luck" bar).

Strategy library (all causal, cost-aware via backend.backtest.engine):
  * SINGLE-ASSET (≈30): the 12-strategy zoo (strategy_zoo_research.py) PLUS many more trend /
    breakout / momentum / mean-reversion / oscillator / volume families, a FIBONACCI-retracement
    turning-point strategy, and Connors RSI2.
  * PORTFOLIO BOOKS: the current operating model — ``managed_book`` (★ the validated 7-asset
    managed-beta book, Sharpe ≈1.1) — plus dollar-neutral cross-sectional momentum / 1-month
    reversal / low-vol books. These run their own universe and are overlaid (reindexed) on the chart.

NEWS: the project's honest finding is that news is a RISK OVERLAY (veto / de-gear / halt), not a
direction predictor, and there is no historical dated-news cache to backtest — so news is NOT a
historical equity line. Instead the dashboard exposes a LIVE news-risk readout (``/api/news``):
keyless GDELT tone for the asset, run through the real ``backend.risk.news_overlay.assess_news_risk``
overlay, showing how recent news would gate a long position right now.

Numeric stats stream into a sortable table: Sharpe / Sortino / Calmar / CAGR / maxDD / vol / alpha &
beta vs SPY / Deflated Sharpe, plus the annual-return distribution (mean, Q25, median, Q75, best,
worst). Everything offline from training_data/ parquet caches (news is the only networked feature).

Run:
    .\backend\.venv\Scripts\python.exe scripts\backtest_dashboard.py            # http://127.0.0.1:8765
    .\backend\.venv\Scripts\python.exe scripts\backtest_dashboard.py --port 9000
or use launch_backtester.bat / launch_backtester.sh.
"""
import argparse
import asyncio
import glob
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.backtest.engine import (  # noqa: E402
    run_backtest, compute_metrics, regression_alpha_beta, deflated_sharpe_ratio,
)
from backend.backtest.portfolio import (  # noqa: E402
    align_panel, strategy_net_returns, vol_target_scale, drawdown_degear, _bar_returns,
)
from backend.strategies.managed_beta import ManagedBeta, ManagedBetaParams  # noqa: E402
from backend.strategies.cross_sectional import (  # noqa: E402
    CrossSectionalMomentum, XSectionalMomentumParams, LowVolatility, LowVolParams,
)

import strategy_zoo_research as zoo  # noqa: E402  (12 vetted single-asset strategies + indicators)

PPY = 252
CACHE_DIR = ROOT / "training_data" / "equity_daily"
MAX_POINTS = 800

# Curated single-asset menu (re-checked against the cache at request time).
ASSET_MENU = ["SPY", "QQQ", "TQQQ", "IWM", "TLT", "IEF", "GLD", "DBC", "EFA", "EEM",
              "BTC-USD", "ETH-USD", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "META", "GOOGL"]

MANAGED_BOOK_SYMS = ["SPY", "QQQ", "TQQQ", "TLT", "GLD", "BTC-USD", "ETH-USD"]
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "AMD", "QCOM", "INTC", "MU",
    "TXN", "AMAT", "CRM", "ORCL", "ADBE", "CSCO", "IBM",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW",
    "XOM", "CVX", "COP", "SLB",
    "UNH", "JNJ", "PFE", "MRK", "ABBV", "LLY",
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "HD",
    "CAT", "DE", "BA", "GE", "HON", "DIS", "V", "MA",
]

# ───────────────────────────── indicator helpers (causal) ─────────────────────────────
ema, sma, rsi, atr = zoo.ema, zoo.sma, zoo.rsi, zoo.atr
rolling_max, rolling_min, rolling_std, true_range, adx = (
    zoo.rolling_max, zoo.rolling_min, zoo.rolling_std, zoo.true_range, zoo.adx)


def _cci(h, l, c, n=20):
    tp = (h + l + c) / 3.0
    ma = sma(tp, n)
    md = pd.Series(tp).rolling(n, min_periods=n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True).to_numpy()
    return (tp - ma) / (0.015 * md + 1e-12)


def _stoch_k(h, l, c, n=14):
    hh = rolling_max(h, n); ll = rolling_min(l, n)
    return 100.0 * (c - ll) / (hh - ll + 1e-12)


def _williams_r(h, l, c, n=14):
    hh = rolling_max(h, n); ll = rolling_min(l, n)
    return -100.0 * (hh - c) / (hh - ll + 1e-12)


def _mfi(h, l, c, v, n=14):
    tp = (h + l + c) / 3.0
    mf = tp * v
    dpos = np.where(np.diff(tp, prepend=tp[:1]) > 0, mf, 0.0)
    dneg = np.where(np.diff(tp, prepend=tp[:1]) < 0, mf, 0.0)
    pos = pd.Series(dpos).rolling(n, min_periods=n).sum().to_numpy()
    neg = pd.Series(dneg).rolling(n, min_periods=n).sum().to_numpy()
    return 100.0 - 100.0 / (1.0 + pos / (neg + 1e-12))


def _aroon(h, l, n=25):
    up = pd.Series(h).rolling(n + 1, min_periods=n + 1).apply(lambda x: 100.0 * np.argmax(x) / n, raw=True).to_numpy()
    dn = pd.Series(l).rolling(n + 1, min_periods=n + 1).apply(lambda x: 100.0 * np.argmin(x) / n, raw=True).to_numpy()
    return up, dn


def _vortex(h, l, c, n=14):
    tr = true_range(h, l, c)
    vmp = np.abs(h - np.roll(l, 1)); vmp[0] = 0.0
    vmm = np.abs(l - np.roll(h, 1)); vmm[0] = 0.0
    s_tr = pd.Series(tr).rolling(n, min_periods=n).sum().to_numpy()
    vip = pd.Series(vmp).rolling(n, min_periods=n).sum().to_numpy() / (s_tr + 1e-12)
    vim = pd.Series(vmm).rolling(n, min_periods=n).sum().to_numpy() / (s_tr + 1e-12)
    return vip, vim


def _trix(c, n=15):
    e1 = ema(c, n); e2 = ema(e1, n); e3 = ema(e2, n)
    return np.concatenate([[0.0], np.diff(e3) / (e3[:-1] + 1e-12)])


def _kama(c, n=10, fast=2, slow=30):
    nn = len(c)
    change = np.abs(c - np.roll(c, n)); change[:n] = 0.0
    vol = pd.Series(np.abs(np.diff(c, prepend=c[:1]))).rolling(n, min_periods=1).sum().to_numpy()
    er = change / (vol + 1e-12)
    sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
    kama = np.copy(c)
    for i in range(1, nn):
        kama[i] = kama[i - 1] + sc[i] * (c[i] - kama[i - 1])
    return kama


# ───────────────────── single-asset strategies → position in {-1,0,1} (pre-shift) ─────────────────────
def s_faber_trend(o, h, l, c, v, n=200):
    return np.where(c > ema(c, n), 1.0, 0.0)


def s_sma200(o, h, l, c, v, n=200):
    return np.where(c > sma(c, n), 1.0, 0.0)


def s_golden_cross(o, h, l, c, v):
    return np.where(sma(c, 50) > sma(c, 200), 1.0, 0.0)


def s_tsmom_12m(o, h, l, c, v, n=252):
    past = np.roll(c, n); pos = np.where(c / np.where(past > 0, past, np.nan) - 1.0 > 0, 1.0, 0.0)
    pos[:n] = 0.0; return np.nan_to_num(pos)


def s_mom_6m(o, h, l, c, v, n=126):
    past = np.roll(c, n); pos = np.where(c / np.where(past > 0, past, np.nan) - 1.0 > 0, 1.0, 0.0)
    pos[:n] = 0.0; return np.nan_to_num(pos)


def s_donchian_55(o, h, l, c, v):
    return zoo.s_donchian(o, h, l, c, v, n=55)


def s_kama_trend(o, h, l, c, v):
    return np.where(c > _kama(c), 1.0, 0.0)


def s_trix(o, h, l, c, v):
    return np.where(_trix(c) > 0, 1.0, -1.0)


def s_aroon(o, h, l, c, v, n=25):
    up, dn = _aroon(h, l, n)
    return np.nan_to_num(np.where(up > dn, 1.0, -1.0))


def s_vortex(o, h, l, c, v, n=14):
    vip, vim = _vortex(h, l, c, n)
    return np.nan_to_num(np.where(vip > vim, 1.0, -1.0))


def s_keltner(o, h, l, c, v, n=20, k=2.0):
    mid = ema(c, n); a = atr(h, l, c, n)
    up = mid + k * a; dn = mid - k * a
    pos = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i] > up[i]: pos[i] = 1.0
        elif c[i] < dn[i]: pos[i] = -1.0
        else: pos[i] = pos[i - 1]
    return pos


def s_bollinger_breakout(o, h, l, c, v, n=20, k=2.0):
    m = sma(c, n); sd = rolling_std(c, n)
    up = m + k * sd; dn = m - k * sd
    pos = np.zeros(len(c))
    for i in range(1, len(c)):
        if np.isfinite(up[i]) and c[i] > up[i]: pos[i] = 1.0
        elif np.isfinite(dn[i]) and c[i] < dn[i]: pos[i] = -1.0
        else: pos[i] = pos[i - 1]
    return pos


def s_zscore_revert(o, h, l, c, v, n=20):
    z = (c - sma(c, n)) / (rolling_std(c, n) + 1e-12)
    pos = np.zeros(len(c))
    for i in range(1, len(c)):
        if not np.isfinite(z[i]): pos[i] = pos[i - 1]; continue
        if z[i] < -2.0: pos[i] = 1.0
        elif z[i] > 2.0: pos[i] = -1.0
        elif abs(z[i]) < 0.5: pos[i] = 0.0
        else: pos[i] = pos[i - 1]
    return pos


def s_cci(o, h, l, c, v, n=20):
    x = _cci(h, l, c, n); pos = np.zeros(len(c))
    for i in range(1, len(c)):
        if not np.isfinite(x[i]): pos[i] = pos[i - 1]; continue
        if x[i] < -100: pos[i] = 1.0
        elif x[i] > 100: pos[i] = -1.0
        elif abs(x[i]) < 20: pos[i] = 0.0
        else: pos[i] = pos[i - 1]
    return pos


def s_stochastic(o, h, l, c, v, n=14):
    k = _stoch_k(h, l, c, n); pos = np.zeros(len(c))
    for i in range(1, len(c)):
        if not np.isfinite(k[i]): pos[i] = pos[i - 1]; continue
        if k[i] < 20: pos[i] = 1.0
        elif k[i] > 80: pos[i] = -1.0
        else: pos[i] = pos[i - 1]
    return pos


def s_williams_r(o, h, l, c, v, n=14):
    r = _williams_r(h, l, c, n); pos = np.zeros(len(c))
    for i in range(1, len(c)):
        if not np.isfinite(r[i]): pos[i] = pos[i - 1]; continue
        if r[i] < -80: pos[i] = 1.0
        elif r[i] > -20: pos[i] = -1.0
        else: pos[i] = pos[i - 1]
    return pos


def s_mfi(o, h, l, c, v, n=14):
    m = _mfi(h, l, c, v, n); pos = np.zeros(len(c))
    for i in range(1, len(c)):
        if not np.isfinite(m[i]): pos[i] = pos[i - 1]; continue
        if m[i] < 20: pos[i] = 1.0
        elif m[i] > 80: pos[i] = -1.0
        else: pos[i] = pos[i - 1]
    return pos


def s_rsi2(o, h, l, c, v):
    r = rsi(c, 2); sma200 = sma(c, 200); sma5 = sma(c, 5)
    pos = np.zeros(len(c))
    for i in range(1, len(c)):
        up = np.isfinite(sma200[i]) and c[i] > sma200[i]
        if pos[i - 1] == 0.0:
            pos[i] = 1.0 if (up and r[i] < 10.0) else 0.0
        else:
            pos[i] = 0.0 if (np.isfinite(sma5[i]) and c[i] > sma5[i]) else 1.0
    return pos


def s_fibonacci(o, h, l, c, v, window=60, trend_n=100):
    """Fibonacci-retracement turning-point strategy. Find the recent swing high/low over ``window``
    bars (causal). In an uptrend (price above its long EMA), BUY when price pulls back into the
    50-78.6% retracement support zone and turns up (bounce off support), hold until it breaks the
    swing high (target) or the swing low (stop). Symmetric short in a downtrend. The classic
    "retracements find turning points" idea — measured here under costs + the deflated-Sharpe bar."""
    n = len(c)
    hh = rolling_max(h, window); ll = rolling_min(l, window)
    rng = hh - ll
    fib50 = hh - 0.5 * rng         # 50% level
    fib786 = hh - 0.786 * rng      # deep retracement
    trend = ema(c, trend_n)
    pos = np.zeros(n)
    for i in range(1, n):
        if not (np.isfinite(rng[i]) and rng[i] > 0 and np.isfinite(trend[i])):
            pos[i] = pos[i - 1]; continue
        up = c[i] > trend[i]
        in_long_zone = fib786[i] <= c[i] <= fib50[i]        # pulled back to 50-78.6% support
        in_short_zone = (hh[i] - 0.5 * rng[i]) <= c[i] <= (hh[i] - 0.214 * rng[i])
        if pos[i - 1] == 0.0:
            if up and in_long_zone and c[i] > c[i - 1]:     # bounce up off support
                pos[i] = 1.0
            elif (not up) and in_short_zone and c[i] < c[i - 1]:
                pos[i] = -1.0
        elif pos[i - 1] > 0:
            pos[i] = 0.0 if (c[i] >= hh[i] or c[i] <= ll[i]) else 1.0   # target or stop
        else:
            pos[i] = 0.0 if (c[i] <= ll[i] or c[i] >= hh[i]) else -1.0
    return pos


# name -> (callable, family). Zoo first, then the extras.
SINGLE_STRATEGIES: Dict[str, Callable] = dict(zoo.STRATEGIES)
SINGLE_STRATEGIES.update({
    "faber_trend": s_faber_trend, "sma200": s_sma200, "golden_cross": s_golden_cross,
    "kama_trend": s_kama_trend, "trix": s_trix, "aroon": s_aroon, "vortex": s_vortex,
    "donchian_55": s_donchian_55, "keltner": s_keltner, "bollinger_breakout": s_bollinger_breakout,
    "tsmom_12m": s_tsmom_12m, "mom_6m": s_mom_6m,
    "rsi2": s_rsi2, "zscore_revert": s_zscore_revert, "cci": s_cci, "stochastic": s_stochastic,
    "williams_r": s_williams_r, "mfi": s_mfi, "fib_retrace": s_fibonacci,
})

FAMILY = {
    "ema_cross": "trend", "macd": "trend", "supertrend": "trend", "ichimoku": "trend",
    "psar": "trend", "adx_trend": "trend", "faber_trend": "trend", "sma200": "trend",
    "golden_cross": "trend", "kama_trend": "trend", "trix": "trend", "aroon": "trend",
    "vortex": "trend",
    "donchian": "breakout", "atr_channel": "breakout", "donchian_55": "breakout",
    "keltner": "breakout", "bollinger_breakout": "breakout",
    "roc_mom": "momentum", "tsmom_12m": "momentum", "mom_6m": "momentum",
    "rsi_mr": "mean-rev", "bollinger_mr": "mean-rev", "rsi2": "mean-rev",
    "zscore_revert": "mean-rev", "cci": "oscillator", "stochastic": "oscillator",
    "williams_r": "oscillator", "mfi": "volume", "obv_trend": "volume",
    "fib_retrace": "fibonacci",
}

BOOK_LABELS = {
    "managed_book": "★ Managed book (current model)",
    "xs_momentum": "XS momentum (12-1, β≈0)",
    "xs_reversal": "XS reversal (1-month, β≈0)",
    "low_vol": "Low-vol book (β≈0)",
}


# ───────────────────────────── data loading (offline cache) ─────────────────────────────
_UNIVERSE_CACHE: Dict[tuple, Dict[str, pd.DataFrame]] = {}


def load_asset(symbol: str, start_year: int) -> Optional[pd.DataFrame]:
    files = sorted(glob.glob(str(CACHE_DIR / f"{symbol}_*_1d.parquet")))
    if not files:
        return None
    df = pd.read_parquet(files[0]).copy()
    if "timestamp" not in df.columns or "close" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"] >= pd.Timestamp(year=start_year, month=1, day=1)].reset_index(drop=True)
    return df if len(df) > 260 else None


def load_universe(symbols: List[str], start_year: int) -> Dict[str, pd.DataFrame]:
    key = (tuple(symbols), start_year)
    if key in _UNIVERSE_CACHE:
        return _UNIVERSE_CACHE[key]
    out = {}
    for s in symbols:
        df = load_asset(s, start_year)
        if df is not None:
            out[s] = df
    _UNIVERSE_CACHE[key] = out
    return out


# ───────────────────────────── pure compute layer (testable) ─────────────────────────────
def _calendar_year_distribution(net: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    s = pd.Series(net, index=pd.DatetimeIndex(dates[:len(net)]))
    yearly = s.groupby(s.index.year).apply(lambda r: float(np.prod(1.0 + r.to_numpy()) - 1.0))
    arr = yearly.to_numpy()
    if arr.size == 0:
        return {}
    return {"mean": float(arr.mean()), "q25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)), "q75": float(np.percentile(arr, 75)),
            "best": float(arr.max()), "worst": float(arr.min()),
            "pos_years": int((arr > 0).sum()), "n_years": int(arr.size)}


def _stats(net: np.ndarray, equity: np.ndarray, trades, spy_ret: np.ndarray,
           dates: pd.DatetimeIndex) -> dict:
    m = compute_metrics(net, equity, trades, bars_per_year=PPY)
    alpha, beta = regression_alpha_beta(net, spy_ret[:len(net)], bars_per_year=PPY)
    sd = float(net.std()); dd = m["max_drawdown"]
    return {"sharpe": m["sharpe"], "sortino": m["sortino"],
            "calmar": (m["ann_return"] / abs(dd)) if abs(dd) > 1e-9 else 0.0,
            "cagr": m["ann_return"], "total_return": m["total_return"],
            "max_drawdown": dd, "ann_vol": m["ann_vol"], "alpha": alpha, "beta": beta,
            "win_rate": float((net > 0).mean()), "sr_bar": float(net.mean() / sd) if sd > 1e-12 else 0.0,
            "skew": float(pd.Series(net).skew()), "kurt": float(pd.Series(net).kurt() + 3.0),
            "n_obs": int(net.size), "annual": _calendar_year_distribution(net, dates)}


def run_strategy(name: str, df: pd.DataFrame, spy_ret: np.ndarray, *,
                 fee_bps: float, slippage_bps: float, allow_short: bool) -> dict:
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float); v = df["volume"].to_numpy(float)
    pos = np.nan_to_num(np.asarray(SINGLE_STRATEGIES[name](o, h, l, c, v), dtype=float))
    res = run_backtest(c, pos, fee_bps=fee_bps, slippage_bps=slippage_bps,
                       allow_short=allow_short, bars_per_year=PPY)
    dates = pd.DatetimeIndex(df["timestamp"])
    return {"equity": res.equity, "stats": _stats(res.net_returns, res.equity, res.trades, spy_ret, dates)}


def buy_hold(df: pd.DataFrame, spy_ret: np.ndarray) -> dict:
    c = df["close"].to_numpy(float)
    res = run_backtest(c, np.ones(len(c)), fee_bps=0.0, slippage_bps=0.0, allow_short=False, bars_per_year=PPY)
    dates = pd.DatetimeIndex(df["timestamp"])
    return {"equity": res.equity, "stats": _stats(res.net_returns, res.equity, res.trades, spy_ret, dates)}


def spy_reference(dates: pd.DatetimeIndex, start_year: int) -> Optional[dict]:
    spy = load_asset("SPY", start_year)
    if spy is None:
        return None
    s = pd.Series(spy["close"].to_numpy(float), index=pd.DatetimeIndex(spy["timestamp"]).normalize())
    s = s[~s.index.duplicated(keep="last")]
    aligned = s.reindex(pd.DatetimeIndex(dates).normalize()).ffill().bfill()
    close = aligned.to_numpy(float)
    bar_ret = np.zeros(len(close)); bar_ret[1:] = close[1:] / np.where(close[:-1] > 0, close[:-1], np.nan) - 1.0
    bar_ret = np.nan_to_num(bar_ret); equity = np.cumprod(1.0 + bar_ret)
    return {"equity": equity, "bar_ret": bar_ret,
            "stats": _stats(bar_ret, equity, [], bar_ret, pd.DatetimeIndex(dates))}


def _random_positions(n: int, flip_prob: float, rng: np.random.Generator) -> np.ndarray:
    flips = rng.random(n) < flip_prob
    choices = rng.choice(np.array([-1.0, 1.0]), size=n)
    pos = np.where(flips, choices, np.nan); pos[0] = choices[0]
    return pd.Series(pos).ffill().to_numpy()


def random_band(df: pd.DataFrame, spy_ret: np.ndarray, *, fee_bps: float, slippage_bps: float,
                allow_short: bool, n_runs: int = 40, flip_prob: float = 0.05, seed: int = 7) -> dict:
    c = df["close"].to_numpy(float); n = len(c)
    rng = np.random.default_rng(seed)
    eqs = np.empty((n_runs, n)); sharpes = []; mid_net = None
    for k in range(n_runs):
        res = run_backtest(c, _random_positions(n, flip_prob, rng), fee_bps=fee_bps,
                           slippage_bps=slippage_bps, allow_short=allow_short, bars_per_year=PPY)
        eqs[k] = res.equity; sharpes.append(res.metrics["sharpe"])
        if k == n_runs // 2:
            mid_net = res.net_returns
    dates = pd.DatetimeIndex(df["timestamp"])
    stats = _stats(mid_net if mid_net is not None else np.zeros(n), np.median(eqs, axis=0), [], spy_ret, dates)
    stats["sharpe"] = float(np.median(sharpes))
    return {"median": np.median(eqs, axis=0), "p10": np.percentile(eqs, 10, axis=0),
            "p90": np.percentile(eqs, 90, axis=0), "stats": stats}


# ───────────────────────────── portfolio / book strategies ─────────────────────────────
def _managed_sleeve_net(pos: np.ndarray, ret: np.ndarray, cost: float) -> np.ndarray:
    m = min(len(pos), len(ret)); pos, ret = pos[:m], ret[:m]
    gross = np.zeros(m); gross[1:] = pos[:-1] * ret[1:]
    turn = np.zeros(m); turn[0] = abs(pos[0]); turn[1:] = np.abs(pos[1:] - pos[:-1])
    return gross - turn * cost


def book_managed(start_year: int, fee_bps: float, slippage_bps: float) -> Optional[Tuple]:
    """The current operating model: 7-asset managed-beta book (equal-weight active managed sleeves,
    vol-targeted to 15%, drawdown de-geared) — the validated Sharpe≈1.1 deliverable."""
    data = load_universe(MANAGED_BOOK_SYMS, start_year)
    if len(data) < 3:
        return None
    cost = (fee_bps + slippage_bps) / 1e4
    aligned = align_panel(data, how="outer")
    pos = ManagedBeta(ManagedBetaParams(trend_ema=200)).generate_positions(aligned)
    per = [_managed_sleeve_net(pos[s], _bar_returns(aligned[s]["close"].to_numpy()), cost) for s in aligned]
    L = min(len(x) for x in per)
    R = np.vstack([x[:L] for x in per])
    roll = pd.DataFrame(R.T).rolling(63, min_periods=10).std().to_numpy()
    active = (roll > 1e-9).astype(float); rs = active.sum(axis=1, keepdims=True)
    w = np.where(rs > 0, active / np.where(rs > 0, rs, 1.0), 0.0)
    w = np.nan_to_num(pd.DataFrame(w).shift(1).to_numpy()).T
    combined = (w * R).sum(axis=0)
    net = combined * vol_target_scale(combined, 0.15, PPY, max_leverage=2.0)
    net = drawdown_degear(net, 0.15, 0.25)
    dates = pd.DatetimeIndex(aligned[list(aligned)[0]]["timestamp"])[:L]
    return dates, np.cumprod(1.0 + net), net


def book_cross_sectional(kind: str, start_year: int, fee_bps: float, slippage_bps: float) -> Optional[Tuple]:
    data = load_universe(STOCK_UNIVERSE, start_year)
    if len(data) < 10:
        return None
    aligned = align_panel(data, how="outer")
    if kind == "low_vol":
        strat = LowVolatility(LowVolParams(vol_window=60, hold=21, quantile=0.2))
    elif kind == "xs_reversal":
        strat = CrossSectionalMomentum(XSectionalMomentumParams(lookback=21, skip=0, hold=5,
                                                                quantile=0.2, kind="reversal"))
    else:
        strat = CrossSectionalMomentum(XSectionalMomentumParams(lookback=252, skip=21, hold=21,
                                                                quantile=0.2, kind="momentum"))
    positions = strat.generate_positions(aligned)
    raw = strategy_net_returns(positions, aligned, fee_bps, slippage_bps)
    if raw.size == 0:
        return None
    net = raw * vol_target_scale(raw, 0.10, PPY, max_leverage=2.0)
    net = drawdown_degear(net, 0.15, 0.25)
    dates = pd.DatetimeIndex(aligned[list(aligned)[0]]["timestamp"])[-len(net):]
    return dates, np.cumprod(1.0 + net), net


def run_book(name: str, start_year: int, fee_bps: float, slippage_bps: float) -> Optional[dict]:
    if name == "managed_book":
        res = book_managed(start_year, fee_bps, slippage_bps)
    elif name in ("xs_momentum", "xs_reversal", "low_vol"):
        res = book_cross_sectional(name, start_year, fee_bps, slippage_bps)
    else:
        return None
    if res is None:
        return None
    dates, equity, net = res
    spy = spy_reference(dates, start_year)
    spy_ret = spy["bar_ret"] if spy else np.zeros(len(net))
    return {"dates": dates, "equity": equity, "stats": _stats(net, equity, [], spy_ret, dates)}


# ───────────────────────────── sampling / reindex helpers ─────────────────────────────
def downsample_index(n: int, max_points: int = MAX_POINTS) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).astype(int))


def _take(arr: np.ndarray, idx: np.ndarray) -> list:
    out = []
    for x in np.asarray(arr, dtype=float)[idx]:
        out.append(None if not np.isfinite(x) else round(float(x), 5))
    return out


def reindex_book(dates_b: pd.DatetimeIndex, equity_b: np.ndarray, full_dates: pd.DatetimeIndex) -> np.ndarray:
    """Overlay a book's equity onto the display calendar: reindex by date, ffill within its live
    span, leave NaN (→ gap) before it starts, and rebase to 1.0 at its first visible point."""
    s = pd.Series(equity_b, index=pd.DatetimeIndex(dates_b).normalize())
    s = s[~s.index.duplicated(keep="last")]
    r = s.reindex(pd.DatetimeIndex(full_dates).normalize()).ffill()
    first = r.first_valid_index()
    if first is not None and r[first] not in (0, np.nan):
        r = r / r[first]
    return r.to_numpy(dtype=float)


# ───────────────────────────── live news risk overlay (GDELT, keyless) ─────────────────────────────
NEWS_QUERY = {"SPY": "S&P 500 stock market", "QQQ": "Nasdaq technology stocks", "TQQQ": "Nasdaq technology stocks",
              "IWM": "small cap stocks", "TLT": "US Treasury bonds", "GLD": "gold price",
              "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "AAPL": "Apple", "MSFT": "Microsoft",
              "NVDA": "Nvidia", "AMD": "AMD", "TSLA": "Tesla", "AMZN": "Amazon", "META": "Meta Facebook",
              "GOOGL": "Google Alphabet"}


def fetch_news_risk(symbol: str) -> dict:
    """Live, keyless news-risk readout for one asset: average GDELT tone over the last 3 days →
    signed sentiment → run through the project's real assess_news_risk overlay (the architecturally
    honest way news enters: it gates/sizes a trade, it does NOT predict direction). Best-effort."""
    import types
    from datetime import datetime, timezone
    try:
        import requests
    except Exception:
        return {"available": False, "msg": "requests not installed"}
    q = NEWS_QUERY.get(symbol, symbol.replace("-USD", ""))
    try:
        base = "https://api.gdeltproject.org/api/v2/doc/doc"
        tc = requests.get(base, params={"query": q, "mode": "tonechart", "timespan": "3d",
                                        "format": "json"}, timeout=12).json()
        bins = tc.get("tonechart", []) if isinstance(tc, dict) else []
        total = sum(b.get("count", 0) for b in bins)
        avg_tone = (sum(b.get("bin", 0) * b.get("count", 0) for b in bins) / total) if total else 0.0
        arts = requests.get(base, params={"query": q, "mode": "artlist", "maxrecords": 6,
                                          "timespan": "3d", "sort": "datedesc", "format": "json"},
                            timeout=12).json()
        headlines = [{"title": a.get("title", ""), "url": a.get("url", ""), "domain": a.get("domain", "")}
                     for a in (arts.get("articles", []) if isinstance(arts, dict) else [])][:6]
    except Exception as e:
        return {"available": False, "msg": f"GDELT unreachable: {str(e)[:120]}"}
    if total == 0:
        return {"available": True, "n_articles": 0, "avg_tone": 0.0, "signed_score": 0.0,
                "action": "allow", "size_scale": 1.0, "reason": "no recent news found", "headlines": []}
    score = float(np.tanh(avg_tone / 3.0))                 # tone → [-1,1] sentiment
    direction = "up" if score >= 0 else "down"
    severity = "SEVERE" if abs(avg_tone) >= 4.0 else ("SIGNIFICANT" if abs(avg_tone) >= 1.0 else "NEUTRAL")
    impact = types.SimpleNamespace(severity=severity, direction=direction, confidence=min(1.0, total / 200.0),
                                   trust_score=0.7, asset=symbol, magnitude_pct_low=0.0,
                                   magnitude_pct_high=abs(avg_tone), t_max_minutes=240.0,
                                   created_at=datetime.now(timezone.utc).isoformat(), symbol_relevance={symbol: 1.0})
    from backend.risk.news_overlay import assess_news_risk
    action = assess_news_risk(symbol, "long", [impact], now=datetime.now(timezone.utc))
    return {"available": True, "n_articles": int(total), "avg_tone": round(avg_tone, 2),
            "signed_score": round(score, 3), "severity": severity, "action": action.action,
            "size_scale": round(action.size_scale, 3), "reason": action.reason, "headlines": headlines}


# ───────────────────────────── FastAPI / SSE shell ─────────────────────────────
def build_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

    app = FastAPI(title="Backtest dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.get("/api/options")
    def options():
        singles = [{"name": n, "family": FAMILY.get(n, "other"), "kind": "single"} for n in SINGLE_STRATEGIES]
        books = [{"name": n, "label": BOOK_LABELS[n], "family": "book", "kind": "book"} for n in BOOK_LABELS]
        assets = [a for a in ASSET_MENU if glob.glob(str(CACHE_DIR / f"{a}_*_1d.parquet"))]
        return JSONResponse({"strategies": singles + books, "assets": assets})

    @app.get("/api/news")
    def news(symbol: str = "SPY"):
        from fastapi.responses import JSONResponse as _J
        return _J(fetch_news_risk(symbol))

    @app.get("/api/run")
    async def run(request: Request, asset: str = "SPY", start: int = 2010, strats: str = "",
                  fee: float = 5.0, slippage: float = 5.0, allow_short: int = 1,
                  random_runs: int = 40, want_random: int = 1):
        selected = [s for s in strats.split(",") if s]
        singles = [s for s in selected if s in SINGLE_STRATEGIES]
        books = [s for s in selected if s in BOOK_LABELS]

        async def gen():
            df = load_asset(asset, start)
            if df is None:
                yield _sse({"type": "error", "msg": f"No cached data for {asset} from {start}."}); return
            full_dates = pd.DatetimeIndex(df["timestamp"])
            idx = downsample_index(len(df))
            labels = [d.strftime("%Y-%m-%d") for d in full_dates[idx]]
            spy = spy_reference(full_dates, start)
            spy_ret = spy["bar_ret"] if spy else np.zeros(len(df))

            yield _sse({"type": "init", "asset": asset, "labels": labels, "n_bars": int(len(df)),
                        "n_series": len(singles) + len(books) + (1 if want_random else 0) + 2})
            await asyncio.sleep(0)

            if spy is not None:
                yield _sse({"type": "series", "kind": "benchmark", "name": "S&P 500 (SPY)",
                            "equity": _take(spy["equity"], idx), "stats": _round(spy["stats"])})
                await asyncio.sleep(0)
            if want_random:
                rb = random_band(df, spy_ret, fee_bps=fee, slippage_bps=slippage,
                                 allow_short=bool(allow_short), n_runs=int(random_runs))
                yield _sse({"type": "band", "name": "Random (MC)", "median": _take(rb["median"], idx),
                            "p10": _take(rb["p10"], idx), "p90": _take(rb["p90"], idx),
                            "stats": _round(rb["stats"])})
                await asyncio.sleep(0)
            bh = buy_hold(df, spy_ret)
            yield _sse({"type": "series", "kind": "reference", "name": f"Buy & Hold {asset}",
                        "equity": _take(bh["equity"], idx), "stats": _round(bh["stats"])})
            await asyncio.sleep(0)

            collected = []
            for name in singles:
                if await request.is_disconnected():
                    return
                try:
                    r = run_strategy(name, df, spy_ret, fee_bps=fee, slippage_bps=slippage,
                                     allow_short=bool(allow_short))
                except Exception as e:  # pragma: no cover
                    yield _sse({"type": "warn", "name": name, "msg": str(e)[:160]}); continue
                collected.append((name, r["stats"]))
                yield _sse({"type": "series", "kind": "strategy", "name": name,
                            "family": FAMILY.get(name, "other"), "equity": _take(r["equity"], idx),
                            "stats": _round(r["stats"])})
                await asyncio.sleep(0)

            for name in books:
                if await request.is_disconnected():
                    return
                try:
                    rb2 = run_book(name, start, fee, slippage)
                except Exception as e:  # pragma: no cover
                    yield _sse({"type": "warn", "name": name, "msg": str(e)[:160]}); continue
                if rb2 is None:
                    yield _sse({"type": "warn", "name": name, "msg": "insufficient cached data for this book"}); continue
                overlay = reindex_book(rb2["dates"], rb2["equity"], full_dates)
                collected.append((name, rb2["stats"]))
                yield _sse({"type": "series", "kind": "book", "name": name, "label": BOOK_LABELS[name],
                            "family": "book", "equity": _take(overlay, idx), "stats": _round(rb2["stats"])})
                await asyncio.sleep(0)

            if collected:
                srs = [s["sr_bar"] for _, s in collected]
                tstd = float(np.std(srs)) if len(srs) > 1 else 0.0
                dsr = {n: round(deflated_sharpe_ratio(s["sr_bar"], s["n_obs"], len(collected),
                                                      s["skew"], s["kurt"], tstd), 3)
                       for n, s in collected}
                yield _sse({"type": "dsr", "values": dsr})
            yield _sse({"type": "done"})

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj) + "\n\n"


def _round(stats: dict) -> dict:
    out = {}
    for k, val in stats.items():
        if k == "annual":
            out[k] = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in val.items()}
        elif isinstance(val, float):
            out[k] = round(val, 4)
        else:
            out[k] = val
    return out


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest dashboard — strategies vs random vs S&P 500</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root{ --bg:#0b0e14; --panel:#11151d; --panel2:#161b25; --line:#232a36; --txt:#e8ebf2;
         --mut:#8b94a6; --accent:#5b8cff; --good:#43d39e; --bad:#ff6b6b; --gold:#ffce4d; --radius:9px; }
  *{ box-sizing:border-box; }
  body{ margin:0; background:var(--bg); color:var(--txt);
        font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; font-size:14px; }
  header{ padding:12px 18px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; }
  header h1{ font-size:15px; font-weight:600; margin:0; letter-spacing:.2px; }
  header .sub{ color:var(--mut); font-size:12px; }
  header .pill{ margin-left:auto; font-size:11px; color:var(--mut); }
  .wrap{ display:grid; grid-template-columns:300px 1fr; min-height:calc(100vh - 50px); }
  .side{ background:var(--panel); border-right:1px solid var(--line); padding:14px; overflow:auto;
         max-height:calc(100vh - 50px); position:sticky; top:0; }
  .main{ padding:14px 18px; display:flex; flex-direction:column; gap:12px; }
  .field{ margin-bottom:12px; }
  label.lab{ display:block; color:var(--mut); font-size:11px; margin-bottom:5px; text-transform:uppercase; letter-spacing:.05em; }
  select,input[type=number],input[type=text]{ width:100%; background:var(--panel2); color:var(--txt);
        border:1px solid var(--line); border-radius:7px; padding:8px 9px; font-size:13px; }
  .row2{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .mini{ display:flex; gap:6px; flex-wrap:wrap; margin:6px 0; }
  .mini button{ background:var(--panel2); color:var(--mut); border:1px solid var(--line);
        border-radius:6px; padding:4px 9px; font-size:11px; cursor:pointer; }
  .mini button:hover{ color:var(--txt); border-color:var(--accent); }
  .group{ border:1px solid var(--line); border-radius:8px; margin-bottom:7px; background:var(--panel2); overflow:hidden; }
  .group>summary{ cursor:pointer; padding:7px 10px; font-size:12px; color:var(--txt); list-style:none;
        display:flex; align-items:center; gap:6px; user-select:none; }
  .group>summary::-webkit-details-marker{ display:none; }
  .group>summary .cnt{ margin-left:auto; color:var(--mut); font-size:11px; }
  .items{ padding:4px 6px 7px; display:flex; flex-direction:column; gap:1px; }
  .it{ display:flex; align-items:center; gap:8px; padding:4px 6px; border-radius:6px; cursor:pointer; font-size:12.5px; }
  .it:hover{ background:#1d2430; }
  .it input{ accent-color:var(--accent); }
  .it.book{ color:var(--gold); }
  .runbtn{ width:100%; background:var(--accent); color:#fff; border:0; border-radius:8px; padding:11px;
        font-size:14px; font-weight:600; cursor:pointer; margin-top:6px; }
  .runbtn:disabled{ opacity:.5; cursor:default; }
  .card{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; }
  .ctrls{ display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; margin-bottom:10px; font-size:12px; color:var(--mut); }
  .ctrls select{ width:auto; padding:5px 8px; }
  .ctrls label{ display:flex; align-items:center; gap:5px; cursor:pointer; }
  .chartbox{ position:relative; height:440px; }
  .legend{ display:flex; flex-wrap:wrap; gap:6px 12px; margin-bottom:8px; font-size:12px; }
  .legend .lg{ display:flex; align-items:center; gap:5px; color:var(--mut); cursor:pointer; padding:2px 4px; border-radius:5px; }
  .legend .lg:hover{ background:#1d2430; }
  .legend .lg.off{ opacity:.4; text-decoration:line-through; }
  .legend i{ width:14px; height:3px; border-radius:2px; display:inline-block; }
  table{ width:100%; border-collapse:collapse; font-size:12.5px; }
  th,td{ text-align:right; padding:7px 9px; border-bottom:1px solid var(--line); white-space:nowrap; }
  th{ color:var(--mut); font-weight:500; font-size:11px; text-transform:uppercase; letter-spacing:.03em;
      position:sticky; top:0; background:var(--panel); cursor:pointer; user-select:none; }
  th:hover{ color:var(--txt); }
  td:first-child,th:first-child{ text-align:left; }
  .nm{ display:flex; align-items:center; gap:7px; }
  .nm i{ width:11px; height:11px; border-radius:3px; flex:none; }
  .pos{ color:var(--good); } .neg{ color:var(--bad); }
  tr.book td:first-child{ color:var(--gold); }
  .detail td{ background:var(--panel2); color:var(--mut); font-size:12px; text-align:left; }
  .status{ color:var(--mut); font-size:12px; min-height:18px; display:flex; align-items:center; gap:8px; }
  .bar{ height:4px; background:var(--line); border-radius:3px; flex:1; overflow:hidden; max-width:200px; }
  .bar>i{ display:block; height:100%; width:0; background:var(--accent); transition:width .2s; }
  .spin{ width:12px; height:12px; border:2px solid var(--line); border-top-color:var(--accent);
         border-radius:50%; animation:sp .7s linear infinite; }
  @keyframes sp{ to{ transform:rotate(360deg); } }
  .clk{ cursor:pointer; } .clk:hover td{ background:#1b2230; }
  .news{ font-size:12.5px; }
  .news .badge{ display:inline-block; padding:3px 9px; border-radius:20px; font-size:11px; font-weight:600; }
  .news .hl{ color:var(--mut); padding:3px 0; border-bottom:1px solid var(--line); }
  .news a{ color:var(--accent); text-decoration:none; }
  .muted{ color:var(--mut); font-size:11.5px; }
</style></head>
<body>
<header>
  <h1>Backtest dashboard</h1>
  <span class="sub">strategies · random · S&amp;P 500 — streamed live, cost-aware, causal</span>
  <span class="pill" id="ptag"></span>
</header>
<div class="wrap">
  <aside class="side">
    <div class="field"><label class="lab">Asset (chart axis)</label><select id="asset"></select></div>
    <div class="field"><label class="lab">Start year</label><input type="number" id="start" value="2010" min="2000" max="2025"></div>
    <div class="field row2">
      <div><label class="lab">Fee bps</label><input type="number" id="fee" value="5"></div>
      <div><label class="lab">Slippage bps</label><input type="number" id="slip" value="5"></div>
    </div>
    <div class="field row2">
      <div><label class="lab">Random runs</label><input type="number" id="rruns" value="40" min="5" max="200"></div>
      <div><label class="lab" style="visibility:hidden">x</label>
        <label class="muted" style="display:flex;gap:6px;align-items:center;padding-top:7px"><input type="checkbox" id="short" checked>allow short</label></div>
    </div>
    <div class="field">
      <label class="lab">Strategies <span id="selcount" class="muted"></span></label>
      <input type="text" id="search" placeholder="filter…" style="margin-bottom:7px">
      <div class="mini">
        <button data-pick="all">all</button><button data-pick="none">none</button>
        <button data-pick="books">books</button><button data-pick="trend">trend</button>
        <button data-pick="momentum">momentum</button><button data-pick="mean-rev">mean-rev</button>
      </div>
      <div id="groups"></div>
    </div>
    <button class="runbtn" id="run">Run backtest</button>
    <div class="status" id="status" style="margin-top:10px;"></div>
  </aside>
  <main class="main">
    <div class="card">
      <div class="ctrls">
        <label>view
          <select id="ymode">
            <option value="growth_log">Growth of $1 (log)</option>
            <option value="growth_lin">Growth of $1 (linear)</option>
            <option value="cum">Cumulative return %</option>
            <option value="dd">Drawdown %</option>
          </select>
        </label>
        <label><input type="checkbox" id="tgRandom" checked> random band</label>
        <label><input type="checkbox" id="tgBench" checked> S&amp;P 500</label>
        <label><input type="checkbox" id="tgBH" checked> buy &amp; hold</label>
        <span class="muted" style="margin-left:auto">click legend / table headers to toggle / sort</span>
      </div>
      <div class="legend" id="legend"></div>
      <div class="chartbox"><canvas id="chart"></canvas></div>
    </div>
    <div class="card news">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
        <strong style="font-weight:600">News risk overlay</strong>
        <span class="muted">live · gates/sizes a long, does not predict direction (no dated-news history to backtest)</span>
        <button class="mini" id="newsbtn" style="margin-left:auto;border:1px solid var(--line);background:var(--panel2);color:var(--txt);padding:5px 11px;border-radius:6px;cursor:pointer">Check live news</button>
      </div>
      <div id="newsout" class="muted">Pick an asset and check recent GDELT news sentiment, run through the project's risk overlay.</div>
    </div>
    <div class="card" style="padding:0; overflow:auto;">
      <table id="stats">
        <thead><tr>
          <th data-k="name">Strategy</th><th data-k="sharpe">Sharpe</th><th data-k="sortino">Sortino</th>
          <th data-k="calmar">Calmar</th><th data-k="cagr">CAGR</th><th data-k="max_drawdown">Max DD</th>
          <th data-k="ann_vol">Vol</th><th data-k="alpha">α vs SPY</th><th data-k="beta">β</th><th data-k="dsr">DSR</th>
        </tr></thead>
        <tbody id="sbody"></tbody>
      </table>
    </div>
  </main>
</div>
<script>
const COLORS=["#5b8cff","#43d39e","#ff9f43","#b07bff","#ff6b9d","#2ee6d6","#ffd23f","#7bd651","#ff6b6b",
  "#5fa8ff","#c08457","#e879f9","#34d399","#fb923c","#60a5fa","#f472b6","#a3e635","#22d3ee","#fca5a5","#c4b5fd"];
let chart, labels=[], cidx=0, esrc=null, rows=[], famOf={}, bookLabel={};

async function loadOptions(){
  const o=await (await fetch('/api/options')).json();
  const asel=document.getElementById('asset');
  o.assets.forEach(a=>{ const op=document.createElement('option'); op.value=a; op.textContent=a; asel.appendChild(op); });
  const order=["book","trend","breakout","momentum","mean-rev","oscillator","volume","fibonacci","other"];
  const titles={book:"★ Portfolio books",trend:"Trend",breakout:"Breakout",momentum:"Momentum",
    "mean-rev":"Mean reversion",oscillator:"Oscillator",volume:"Volume",fibonacci:"Fibonacci"};
  const byFam={}; o.strategies.forEach(s=>{ famOf[s.name]=s.family; if(s.label) bookLabel[s.name]=s.label;
    (byFam[s.family]=byFam[s.family]||[]).push(s); });
  const G=document.getElementById('groups');
  order.forEach(f=>{ if(!byFam[f]) return; const d=document.createElement('details'); d.className='group';
    if(f==='book'||f==='trend') d.open=true;
    d.innerHTML=`<summary>${titles[f]||f}<span class="cnt">${byFam[f].length}</span></summary>`;
    const box=document.createElement('div'); box.className='items';
    byFam[f].forEach(s=>{ const it=document.createElement('label'); it.className='it'+(s.kind==='book'?' book':'');
      it.dataset.fam=f; it.dataset.name=s.name;
      it.innerHTML=`<input type="checkbox" value="${s.name}"><span>${s.label||s.name}</span>`;
      box.appendChild(it); });
    d.appendChild(box); G.appendChild(d); });
  ['managed_book','faber_trend','macd','fib_retrace','rsi2'].forEach(n=>{ const c=document.querySelector(`input[value="${n}"]`); if(c)c.checked=true; });
  updateCount();
}
function updateCount(){ const n=document.querySelectorAll('.it input:checked').length;
  document.getElementById('selcount').textContent=`(${n})`; }
document.getElementById('groups').addEventListener('change',updateCount);
document.getElementById('search').oninput=e=>{ const q=e.target.value.toLowerCase();
  document.querySelectorAll('.it').forEach(it=>{ it.style.display=it.dataset.name.toLowerCase().includes(q)?'':'none'; }); };
document.querySelectorAll('.mini button').forEach(b=> b.onclick=()=>{ const p=b.dataset.pick;
  document.querySelectorAll('.it').forEach(it=>{ const cb=it.querySelector('input');
    if(p==='all') cb.checked=true; else if(p==='none') cb.checked=false;
    else if(p==='books') cb.checked=(it.dataset.fam==='book')||cb.checked;
    else cb.checked=(it.dataset.fam===p)||cb.checked; }); updateCount(); });

function yconf(){ const m=document.getElementById('ymode').value;
  if(m==='growth_log') return {type:'logarithmic',fmt:v=>v.toFixed(2)+'x',lab:'×'};
  if(m==='growth_lin') return {type:'linear',fmt:v=>v.toFixed(2)+'x',lab:'×'};
  if(m==='cum') return {type:'linear',fmt:v=>(v>=0?'+':'')+v.toFixed(0)+'%',lab:'%'};
  return {type:'linear',fmt:v=>v.toFixed(0)+'%',lab:'%'}; }
function transform(eq){ const m=document.getElementById('ymode').value;
  if(m==='cum') return eq.map(v=>v==null?null:(v-1)*100);
  if(m==='dd'){ let pk=-1e9; return eq.map(v=>{ if(v==null) return null; pk=Math.max(pk,v); return (v/pk-1)*100; }); }
  return eq; }
function initChart(){ if(chart) chart.destroy(); const yc=yconf();
  chart=new Chart(document.getElementById('chart'),{ type:'line', data:{labels,datasets:[]},
    options:{ responsive:true, maintainAspectRatio:false, animation:false, normalized:true, spanGaps:false,
      interaction:{mode:'index',intersect:false}, elements:{point:{radius:0},line:{borderWidth:1.8}},
      plugins:{ legend:{display:false}, tooltip:{ callbacks:{ label:c=>` ${c.dataset.label}: ${yconf().fmt(c.parsed.y)}` } } },
      scales:{ x:{ ticks:{color:'#8b94a6',maxTicksLimit:9,autoSkip:true}, grid:{color:'#161b25'} },
        y:{ type:yc.type, ticks:{color:'#8b94a6',callback:yc.fmt}, grid:{color:'#161b25'} } } } });
}
function applyY(){ const yc=yconf(); chart.options.scales.y.type=yc.type;
  chart.data.datasets.forEach(d=>{ if(d._eq) d.data=transform(d._eq); }); chart.update('none'); }
document.getElementById('ymode').onchange=applyY;

function addLine(id,name,eq,color,dash,width){
  const ds={label:name,data:transform(eq),borderColor:color,backgroundColor:color,borderDash:dash||[],
    borderWidth:width||1.8,tension:0,fill:false,_eq:eq,_id:id}; chart.data.datasets.push(ds); chart.update('none');
  addLeg(id,name,color,dash); }
function addBand(id,name,median,p10,p90){ const g='#6b7280';
  chart.data.datasets.push({label:'_p10'+id,data:transform(p10),borderColor:'transparent',pointRadius:0,fill:false,_eq:p10,_grp:id});
  chart.data.datasets.push({label:'_p90'+id,data:transform(p90),borderColor:'transparent',pointRadius:0,
    backgroundColor:'rgba(107,114,128,.16)',fill:'-1',_eq:p90,_grp:id});
  chart.data.datasets.push({label:name,data:transform(median),borderColor:g,borderDash:[5,4],borderWidth:1.5,fill:false,_eq:median,_id:id,_grp:id});
  chart.update('none'); addLeg(id,name,g,[5,4]); }
function addLeg(id,name,color,dash){ const L=document.getElementById('legend'); const s=document.createElement('span');
  s.className='lg'; s.dataset.id=id; s.innerHTML=`<i style="background:${color}"></i>${name}`;
  s.onclick=()=>toggle(id,s); L.appendChild(s); }
function toggle(id,el){ const off=!el.classList.contains('off'); el.classList.toggle('off',off);
  chart.data.datasets.forEach(d=>{ if(d._id===id||d._grp===id) d.hidden=off; }); chart.update('none');
  const tr=rows.find(r=>r.id===id); if(tr&&tr.el) tr.el.style.opacity=off?.45:1; }

const pct=v=>(v*100).toFixed(1)+'%', f2=v=>v.toFixed(2), cl=v=>v>=0?'pos':'neg';
function addRow(id,name,color,st,kind){
  const tb=document.getElementById('sbody'); const tr=document.createElement('tr'); tr.className='clk'+(kind==='book'?' book':'');
  const dispName=(kind==='book'&&bookLabel[name])?bookLabel[name]:name;
  tr.innerHTML=`<td><div class="nm"><i style="background:${color}"></i>${dispName}</div></td>
    <td>${f2(st.sharpe)}</td><td>${f2(st.sortino)}</td><td>${f2(st.calmar)}</td>
    <td class="${cl(st.cagr)}">${pct(st.cagr)}</td><td class="neg">${pct(st.max_drawdown)}</td>
    <td>${pct(st.ann_vol)}</td><td class="${cl(st.alpha)}">${pct(st.alpha)}</td>
    <td>${f2(st.beta)}</td><td data-dsr="${name}">${(kind==='strategy'||kind==='book')?'…':'—'}</td>`;
  const a=st.annual||{}; const det=document.createElement('tr'); det.className='detail'; det.style.display='none';
  det.innerHTML=`<td colspan="10">Annual return — mean ${a.mean!=null?pct(a.mean):'–'}, Q25 ${a.q25!=null?pct(a.q25):'–'},
    median ${a.median!=null?pct(a.median):'–'}, Q75 ${a.q75!=null?pct(a.q75):'–'}, best ${a.best!=null?pct(a.best):'–'},
    worst ${a.worst!=null?pct(a.worst):'–'} · positive years ${a.pos_years!=null?a.pos_years+'/'+a.n_years:'–'}
    · total ${pct(st.total_return)} · win rate ${pct(st.win_rate)}</td>`;
  tr.onclick=()=>{ det.style.display=det.style.display==='none'?'table-row':'none'; };
  tb.appendChild(tr); tb.appendChild(det);
  rows.push({id,name,color,st,kind,el:tr,det,dsr:null});
}
document.querySelectorAll('#stats th').forEach(th=> th.onclick=()=>sortBy(th.dataset.k));
let sortK=null,sortAsc=false;
function sortBy(k){ if(sortK===k) sortAsc=!sortAsc; else {sortK=k;sortAsc=false;}
  const val=r=> k==='name'? (bookLabel[r.name]||r.name) : (k==='dsr'? (r.dsr??-1) : r.st[k]);
  rows.sort((a,b)=>{ let x=val(a),y=val(b); if(typeof x==='string'){return sortAsc?x.localeCompare(y):y.localeCompare(x);}
    return sortAsc?x-y:y-x; });
  const tb=document.getElementById('sbody'); tb.innerHTML=''; rows.forEach(r=>{ tb.appendChild(r.el); tb.appendChild(r.det); }); }

['tgRandom','tgBench','tgBH'].forEach(t=>{ document.getElementById(t).addEventListener('change',e=>{
  const id={tgRandom:'random',tgBench:'bench',tgBH:'bh'}[t]; const el=document.querySelector(`.lg[data-id="${id}"]`);
  if(el){ const off=el.classList.contains('off'); if(off===e.target.checked) toggle(id,el); } }); });

document.getElementById('run').onclick=()=>{ if(esrc) esrc.close();
  const g=id=>document.getElementById(id).value;
  const strats=[...document.querySelectorAll('.it input:checked')].map(c=>c.value);
  if(!strats.length){ document.getElementById('status').innerHTML='<span class="neg">pick at least one strategy</span>'; return; }
  document.getElementById('legend').innerHTML=''; document.getElementById('sbody').innerHTML=''; rows=[]; cidx=0;
  const st=document.getElementById('status'); const want=document.getElementById('tgRandom').checked?1:0;
  st.innerHTML='<span class="spin"></span><span>loading…</span><span class="bar"><i id="pb"></i></span>';
  document.getElementById('run').disabled=true; let done=0,total=1;
  const qs=`asset=${g('asset')}&start=${g('start')}&fee=${g('fee')}&slippage=${g('slip')}&allow_short=${document.getElementById('short').checked?1:0}&random_runs=${g('rruns')}&want_random=${want}&strats=${strats.join(',')}`;
  esrc=new EventSource('/api/run?'+qs);
  esrc.onmessage=ev=>{ const m=JSON.parse(ev.data);
    if(m.type==='init'){ labels=m.labels; total=m.n_series; initChart();
      document.getElementById('ptag').textContent=`${m.asset} · ${m.n_bars} bars`;
      st.querySelector('span:nth-child(2)').textContent=`running on ${m.asset}…`; }
    else if(m.type==='series'){ const isB=m.kind==='book';
      const col=m.kind==='benchmark'?'#ffce4d':(m.kind==='reference'?'#8b93a7':(isB?'#ffce4d':COLORS[cidx++%COLORS.length]));
      const dash=m.kind==='benchmark'?[6,3]:(m.kind==='reference'?[2,3]:(isB?[8,3]:[]));
      const id=m.kind==='benchmark'?'bench':(m.kind==='reference'?'bh':m.name);
      addLine(id,m.label||m.name,m.equity,isB?'#ffce4d':col,dash,(m.kind==='strategy')?1.9:2.4);
      addRow(id,m.name,isB?'#ffce4d':col,m.stats,m.kind); done++; pbar(done,total); }
    else if(m.type==='band'){ addBand('random',m.name,m.median,m.p10,m.p90); addRow('random',m.name,'#6b7280',m.stats,'band'); done++; pbar(done,total); }
    else if(m.type==='dsr'){ for(const[n,v]of Object.entries(m.values)){ const c=document.querySelector(`td[data-dsr="${n}"]`);
      if(c){ c.textContent=v.toFixed(2); c.className=v>0.95?'pos':''; } const r=rows.find(x=>x.name===n); if(r) r.dsr=v; } }
    else if(m.type==='warn'){ console.warn(m.name,m.msg); }
    else if(m.type==='error'){ st.innerHTML='<span class="neg">'+m.msg+'</span>'; esrc.close(); document.getElementById('run').disabled=false; }
    else if(m.type==='done'){ st.innerHTML='<span>done — '+rows.length+' series.</span>'; esrc.close();
      document.getElementById('run').disabled=false; applyVisToggles(); } };
  esrc.onerror=()=>{ st.innerHTML='<span class="neg">connection lost</span>'; esrc.close(); document.getElementById('run').disabled=false; };
};
function pbar(d,t){ const e=document.getElementById('pb'); if(e) e.style.width=Math.min(100,100*d/t)+'%'; }
function applyVisToggles(){ [['tgBench','bench'],['tgBH','bh'],['tgRandom','random']].forEach(([t,id])=>{
  if(!document.getElementById(t).checked){ const el=document.querySelector(`.lg[data-id="${id}"]`); if(el&&!el.classList.contains('off')) toggle(id,el); } }); }

document.getElementById('newsbtn').onclick=async()=>{ const sym=document.getElementById('asset').value;
  const out=document.getElementById('newsout'); out.innerHTML='<span class="spin"></span> checking GDELT…';
  try{ const d=await (await fetch('/api/news?symbol='+sym)).json();
    if(!d.available){ out.innerHTML=`<span class="neg">live news unavailable</span> <span class="muted">${d.msg||''}</span>`; return; }
    if(d.n_articles===0){ out.innerHTML=`<span class="muted">No recent news found for ${sym}.</span>`; return; }
    const col=d.signed_score>0.1?'var(--good)':(d.signed_score<-0.1?'var(--bad)':'var(--mut)');
    const act={allow:'var(--good)',resize:'var(--gold)',veto:'var(--bad)',halt_asset:'var(--bad)'}[d.action]||'var(--mut)';
    let html=`<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:8px">
      <span class="badge" style="background:${col};color:#0b0e14">tone ${d.avg_tone} · score ${d.signed_score}</span>
      <span class="badge" style="background:${act};color:#0b0e14">overlay: ${d.action} ×${d.size_scale}</span>
      <span class="muted">${d.severity} · ${d.n_articles} articles (3d) · for a LONG ${sym}</span></div>
      <div class="muted" style="margin-bottom:6px">${d.reason}</div>`;
    d.headlines.forEach(h=>{ html+=`<div class="hl"><a href="${h.url}" target="_blank" rel="noopener">${h.title||h.domain}</a> <span class="muted">— ${h.domain}</span></div>`; });
    out.innerHTML=html;
  }catch(e){ out.innerHTML='<span class="neg">news fetch failed</span>'; } };

loadOptions();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="Streaming backtest dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    import uvicorn
    print(f"\n  Backtest dashboard -> http://{args.host}:{args.port}\n")
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
