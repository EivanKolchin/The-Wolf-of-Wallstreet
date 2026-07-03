#!/usr/bin/env python
"""Strategy-zoo breadth scan — the multiple-testing-HONEST test of "dead for most assets but not all".

The 40-strategy TA cookbook collapses into ~5 signal families. We implement ~12 DISTINCT, vectorized,
causal strategies spanning all of them, scan them across a sector-diverse universe (tech / healthcare /
financials / energy / staples + ETFs + crypto), and score every (strategy, asset) trial with the
DEFLATED SHARPE RATIO (Bailey & López de Prado) using the ACTUAL trial count and the dispersion of the
trial Sharpes. The whole point: a few combos ALWAYS look great by luck; DSR tells us how many survive
beyond chance, and whether any survivor adds TIMING value (beats its own buy-&-hold) rather than just
re-collecting the equity risk premium.

Families covered: trend (EMA-cross, MACD, Supertrend, Ichimoku, PSAR, ADX), breakout (Donchian,
ATR-channel), momentum (ROC/abs-momentum), oscillator/mean-reversion (RSI, Bollinger), volume (OBV).
Also: a Fourier dominant-cycle predictive test (is there exploitable PERIODICITY, or is that a myth).

Measurement only; realistic costs; positions shifted 1 bar (causal). No fitting → the multiple-testing
risk is the zoo size itself, which is exactly what DSR(n_trials) corrects for.
"""
from __future__ import annotations

import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.backtest.engine import deflated_sharpe_ratio

EQ = str(ROOT / "training_data" / "equity_daily")
RAW = str(ROOT / "training_data" / "raw")
BPY = 252
FEE_BPS, SLIP_BPS = 5.0, 5.0


# ───────────────────────── indicators (vectorized + causal) ─────────────────────────
def ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()
def sma(x, n): return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()
def rolling_max(x, n): return pd.Series(x).rolling(n, min_periods=n).max().to_numpy()
def rolling_min(x, n): return pd.Series(x).rolling(n, min_periods=n).min().to_numpy()
def rolling_std(x, n): return pd.Series(x).rolling(n, min_periods=n).std().to_numpy()

def true_range(h, l, c):
    pc = np.roll(c, 1); pc[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))

def atr(h, l, c, n=14): return ema(true_range(h, l, c), n)

def rsi(c, n=14):
    d = np.diff(c, prepend=c[:1]); up = np.clip(d, 0, None); dn = np.clip(-d, 0, None)
    rs = ema(up, n) / (ema(dn, n) + 1e-12)
    return 100 - 100 / (1 + rs)

def adx(h, l, c, n=14):
    up = h - np.roll(h, 1); dn = np.roll(l, 1) - l; up[0] = 0; dn[0] = 0
    plus = np.where((up > dn) & (up > 0), up, 0.0); minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(h, l, c); atrn = ema(tr, n) + 1e-12
    pdi = 100 * ema(plus, n) / atrn; mdi = 100 * ema(minus, n) / atrn
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi + 1e-12)
    return ema(dx, n), pdi, mdi


# ───────────────────────── strategies → position in {-1,0,1} (pre-shift) ─────────────────────────
def s_ema_cross(o, h, l, c, v):
    return np.where(ema(c, 20) > ema(c, 100), 1.0, -1.0)

def s_macd(o, h, l, c, v):
    macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9)
    return np.where(macd > sig, 1.0, -1.0)

def s_supertrend(o, h, l, c, v, n=10, mult=3.0):
    a = atr(h, l, c, n); hl2 = (h + l) / 2
    ub = hl2 + mult * a; lb = hl2 - mult * a
    st = np.zeros(len(c)); dirn = np.ones(len(c))
    fub, flb = ub.copy(), lb.copy()
    for i in range(1, len(c)):
        fub[i] = ub[i] if (ub[i] < fub[i-1] or c[i-1] > fub[i-1]) else fub[i-1]
        flb[i] = lb[i] if (lb[i] > flb[i-1] or c[i-1] < flb[i-1]) else flb[i-1]
        dirn[i] = 1.0 if c[i] > fub[i-1] else (-1.0 if c[i] < flb[i-1] else dirn[i-1])
    return dirn

def s_ichimoku(o, h, l, c, v):
    conv = (rolling_max(h, 9) + rolling_min(l, 9)) / 2
    base = (rolling_max(h, 26) + rolling_min(l, 26)) / 2
    spanA = (conv + base) / 2
    spanB = (rolling_max(h, 52) + rolling_min(l, 52)) / 2
    top = np.maximum(spanA, spanB); bot = np.minimum(spanA, spanB)
    return np.where(c > top, 1.0, np.where(c < bot, -1.0, 0.0))

def s_psar(o, h, l, c, v, af0=0.02, afmax=0.2):
    n = len(c); ps = np.zeros(n); pos = np.ones(n)
    af = af0; ep = h[0]; sar = l[0]; up = True
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            if l[i] < sar:
                up = False; sar = ep; ep = l[i]; af = af0
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af0, afmax)
        else:
            if h[i] > sar:
                up = True; sar = ep; ep = h[i]; af = af0
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af0, afmax)
        pos[i] = 1.0 if up else -1.0
    return pos

def s_adx_trend(o, h, l, c, v, thr=20.0):
    ax, pdi, mdi = adx(h, l, c, 14)
    return np.where(ax > thr, np.where(pdi > mdi, 1.0, -1.0), 0.0)

def s_donchian(o, h, l, c, v, n=20):
    hi = rolling_max(h, n); lo = rolling_min(l, n)
    pos = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i] >= hi[i-1] and np.isfinite(hi[i-1]): pos[i] = 1.0
        elif c[i] <= lo[i-1] and np.isfinite(lo[i-1]): pos[i] = -1.0
        else: pos[i] = pos[i-1]
    return pos

def s_atr_channel(o, h, l, c, v, n=20, k=2.0):
    a = atr(h, l, c, n); mid = ema(c, n)
    return np.where(c > mid + k * a, 1.0, np.where(c < mid - k * a, -1.0, 0.0))

def s_roc_mom(o, h, l, c, v, n=126):
    roc = c / np.roll(c, n) - 1.0; roc[:n] = 0.0
    return np.where(roc > 0, 1.0, -1.0)            # absolute / time-series momentum

def s_rsi_mr(o, h, l, c, v, n=14, lo=30, hi=70):
    r = rsi(c, n); pos = np.zeros(len(c))
    for i in range(1, len(c)):
        if r[i] < lo: pos[i] = 1.0
        elif r[i] > hi: pos[i] = -1.0
        else: pos[i] = pos[i-1] if abs(r[i] - 50) > 5 else 0.0   # exit near neutral
    return pos

def s_bollinger_mr(o, h, l, c, v, n=20, k=2.0):
    m = sma(c, n); sd = rolling_std(c, n)
    up = m + k * sd; dn = m - k * sd; pos = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i] < dn[i]: pos[i] = 1.0
        elif c[i] > up[i]: pos[i] = -1.0
        elif (c[i] >= m[i] and pos[i-1] > 0) or (c[i] <= m[i] and pos[i-1] < 0): pos[i] = 0.0
        else: pos[i] = pos[i-1]
    return pos

def s_obv_trend(o, h, l, c, v, n=20):
    d = np.sign(np.diff(c, prepend=c[:1])); obv = np.cumsum(d * v)
    return np.where(obv > ema(obv, n), 1.0, -1.0)

STRATEGIES = {
    "ema_cross": s_ema_cross, "macd": s_macd, "supertrend": s_supertrend,
    "ichimoku": s_ichimoku, "psar": s_psar, "adx_trend": s_adx_trend,
    "donchian": s_donchian, "atr_channel": s_atr_channel, "roc_mom": s_roc_mom,
    "rsi_mr": s_rsi_mr, "bollinger_mr": s_bollinger_mr, "obv_trend": s_obv_trend,
}


# ───────────────────────── backtest one (strategy, asset) ─────────────────────────
def backtest(pos, c):
    pos = np.nan_to_num(pos)
    posL = np.concatenate([[0.0], pos[:-1]])            # shift → causal (act on next bar)
    ret = np.diff(c, prepend=c[:1]) / (np.roll(c, 1) + 1e-12); ret[0] = 0.0
    cost = (FEE_BPS + SLIP_BPS) / 1e4 * np.abs(np.diff(posL, prepend=0.0))
    net = posL * ret - cost
    net = net[np.isfinite(net)]
    return net


def sharpe_ann(net):
    sd = net.std()
    return float(net.mean() / sd * np.sqrt(BPY)) if sd > 1e-12 else np.nan

def buy_hold_sharpe(c):
    r = np.diff(c, prepend=c[:1]) / (np.roll(c, 1) + 1e-12); r = r[1:]
    sd = r.std(); return float(r.mean() / sd * np.sqrt(BPY)) if sd > 1e-12 else np.nan


# ───────────────────────── Fourier dominant-cycle predictive test ─────────────────────────
def fourier_cycle_ic(c, window=252, horizon=5):
    """Rolling FFT of detrended log-price; project the dominant low-freq cycle one step and test its
    rank-IC vs the forward `horizon` return. If markets had exploitable periodicity, this would be > 0."""
    lp = np.log(c + 1e-12)
    n = len(lp); pred = np.full(n, np.nan)
    for t in range(window, n):
        w = lp[t-window:t]; w = w - np.linspace(w[0], w[-1], window)   # detrend (remove the ramp)
        sp = np.fft.rfft(w * np.hanning(window))
        k = np.argmax(np.abs(sp[2:])) + 2                              # dominant non-DC frequency
        # phase slope of that component → next-step direction
        ang = np.angle(sp[k]); freq = k / window
        pred[t] = np.sin(2 * np.pi * freq * (window) + ang) - np.sin(2 * np.pi * freq * (window - 1) + ang)
    fwd = np.concatenate([c[horizon:] / c[:-horizon] - 1.0, np.full(horizon, np.nan)])
    m = np.isfinite(pred) & np.isfinite(fwd)
    if m.sum() < 200: return np.nan
    a = pd.Series(pred[m]).rank().to_numpy(); b = pd.Series(fwd[m]).rank().to_numpy()
    a -= a.mean(); b -= b.mean(); d = np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/d) if d > 0 else np.nan


# ───────────────────────── universe ─────────────────────────
UNIVERSE = {
    "tech":    ["AAPL","MSFT","NVDA","AMD","GOOGL","META","AVGO","ADBE","CRM","ORCL"],
    "health":  ["UNH","JNJ","PFE","MRK","LLY","ABBV","TMO","ABT","BMY","AMGN"],
    "finance": ["JPM","BAC","WFC","GS","MS","C","AXP","BLK","SCHW","SPGI"],
    "other":   ["XOM","CVX","CAT","GE","BA","HON","PG","KO","WMT","HD","MCD"],
    "etf":     ["SPY","QQQ","TQQQ","TLT","GLD"],
}
CRYPTO = ["BTCUSDT","ETHUSDT","SOLUSDT"]


def load_eq(sym):
    fs = sorted(glob.glob(f"{EQ}/{sym}_*_1d.parquet"))
    if not fs: return None
    df = pd.read_parquet(fs[0])
    return df if {"open","high","low","close","volume"}.issubset(df.columns) and len(df) > 600 else None

def load_cr(sym):
    fs = sorted(glob.glob(f"{RAW}/{sym}_1h_*.parquet"))
    if not fs: return None
    raw = pd.read_parquet(fs[0]); raw["d"] = pd.to_datetime(raw["timestamp"]).dt.normalize()
    g = raw.groupby("d")
    df = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                       "close": g["close"].last(), "volume": g["volume"].sum()}).reset_index(drop=True)
    return df if len(df) > 600 else None


def main():
    assets = []
    for sector, syms in UNIVERSE.items():
        for s in syms:
            df = load_eq(s)
            if df is not None: assets.append((s, sector, df))
    for s in CRYPTO:
        df = load_cr(s)
        if df is not None: assets.append((s, "crypto", df))
    print(f"loaded {len(assets)} assets across {len(set(a[1] for a in assets))} sectors")

    rows = []
    for sym, sector, df in assets:
        o=df["open"].to_numpy(float); h=df["high"].to_numpy(float); l=df["low"].to_numpy(float)
        c=df["close"].to_numpy(float); v=df["volume"].to_numpy(float)
        bh = buy_hold_sharpe(c)
        for name, fn in STRATEGIES.items():
            try:
                net = backtest(fn(o, h, l, c, v), c)
            except Exception:
                continue
            if net.size < 250: continue
            sd = net.std()
            rows.append(dict(sym=sym, sector=sector, strat=name, net=net,
                             sr_bar=(net.mean()/sd if sd > 1e-12 else 0.0),
                             sharpe=sharpe_ann(net), bh=bh, n=net.size,
                             skew=float(pd.Series(net).skew()), kurt=float(pd.Series(net).kurt()+3.0)))
    n_trials = len(rows)
    trial_sr_std = float(np.std([r["sr_bar"] for r in rows])) if rows else 0.0
    print(f"\nran {n_trials} (strategy × asset) trials | trial-Sharpe dispersion (per-bar) = {trial_sr_std:.4f}")

    for r in rows:
        r["dsr"] = deflated_sharpe_ratio(r["sr_bar"], r["n"], n_trials, r["skew"], r["kurt"], trial_sr_std)
        r["beats_bh"] = bool(np.isfinite(r["sharpe"]) and np.isfinite(r["bh"]) and r["sharpe"] > r["bh"])

    survivors = [r for r in rows if r["dsr"] > 0.95]
    real = [r for r in survivors if r["beats_bh"]]
    print(f"\n================  MULTIPLE-TESTING RESULT  ================")
    print(f"  trials: {n_trials}   |   DSR>0.95 survivors: {len(survivors)}   |   "
          f"survivors that ALSO beat own buy&hold (timing edge): {len(real)}")
    print(f"  by-chance expectation at the 0.95 bar is near-zero AFTER deflation, so survivors are the candidates.")

    print(f"\n  {'strategy':12s} {'asset':8s} {'sector':8s}  {'Sharpe':>7s}  {'B&H':>6s}  {'DSR':>6s}  beats_B&H")
    for r in sorted(rows, key=lambda x: -x["dsr"])[:18]:
        print(f"  {r['strat']:12s} {r['sym']:8s} {r['sector']:8s}  {r['sharpe']:>+7.2f}  {r['bh']:>+6.2f}  "
              f"{r['dsr']:>6.3f}  {'YES' if r['beats_bh'] else 'no'}")

    # per-strategy and per-sector hit summary
    print(f"\n  per-strategy: median Sharpe across assets, and # DSR-survivors")
    for name in STRATEGIES:
        rs = [r for r in rows if r["strat"] == name]
        med = np.nanmedian([r["sharpe"] for r in rs]) if rs else np.nan
        ns = sum(1 for r in rs if r["dsr"] > 0.95)
        print(f"    {name:12s}  median Sharpe {med:>+5.2f}   survivors {ns}/{len(rs)}")

    print(f"\n  per-sector: # DSR-survivors / trials")
    for sec in sorted(set(a[1] for a in assets)):
        rs = [r for r in rows if r["sector"] == sec]
        ns = sum(1 for r in rs if r["dsr"] > 0.95)
        print(f"    {sec:8s}  {ns}/{len(rs)}")

    # Fourier dominant-cycle predictive IC (pooled)
    print(f"\n================  FOURIER dominant-cycle predictive rank-IC  ================")
    ics = []
    for sym, sector, df in assets:
        ic = fourier_cycle_ic(df["close"].to_numpy(float))
        if np.isfinite(ic): ics.append(ic)
    print(f"  mean |IC| across {len(ics)} assets = {np.nanmean(np.abs(ics)):.4f}   "
          f"signed mean = {np.nanmean(ics):+.4f}")
    print(f"  (|IC| rivals top features only if ≳0.02; ~0 ⇒ no exploitable periodicity, as theory predicts)")


if __name__ == "__main__":
    main()
