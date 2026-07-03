#!/usr/bin/env python
"""Managed-beta research — does disciplined risk management beat BUY-AND-HOLD?

This is the pivot the alpha hunt earned: stop chasing ~0 market-neutral alpha and instead deliver an
asset's risk premium BETTER than holding it. The claim is narrow and falsifiable: a trend-filtered,
de-geared (per-asset) and vol-targeted (combined) book should show HIGHER Sharpe and SHALLOWER max
drawdown than buy-and-hold of the same assets — i.e. POSITIVE regression alpha versus buy-and-hold.

Per asset it prints buy-and-hold vs managed at EQUAL notional (isolating the trend overlay's
crash-avoidance) with alpha-vs-B&H and the upside-capture beta; then the combined multi-asset book
(vol-targeted + de-geared) vs an equal-weight buy-and-hold basket. Run across the risk spectrum —
index ETFs, a 3x leveraged ETP (where the effect is most dramatic), and crypto.

    python scripts/managed_beta_research.py --start 2010-01-01
    python scripts/managed_beta_research.py --symbols "SPY QQQ TQQQ BTC-USD ETH-USD" --down-exposure -0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.data.equity_daily import load_daily  # noqa: E402
from backend.backtest.engine import compute_metrics, regression_alpha_beta  # noqa: E402
from backend.backtest.portfolio import (  # noqa: E402
    portfolio_backtest, align_panel, drawdown_degear, vol_target_scale, _bar_returns,
)
from backend.strategies.managed_beta import ManagedBeta, ManagedBetaParams  # noqa: E402

PPY = 252
# A breadth universe spanning the risk spectrum + low-correlation sleeves (US large/small, intl,
# EM, long/intermediate Treasuries, gold, broad commodities, a 3x ETP, crypto) — the diversity that
# lets risk-parity weighting actually help.
DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "TQQQ",
                   "BTC-USD", "ETH-USD"]


def _calmar(ann_return: float, max_dd: float) -> float:
    return ann_return / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0


def _metrics(net: np.ndarray) -> dict:
    eq = np.cumprod(1.0 + net)
    m = compute_metrics(net, eq, trades=[], bars_per_year=PPY)
    m["calmar"] = _calmar(m["ann_return"], m["max_drawdown"])
    return m


def _managed_net(pos: np.ndarray, ret: np.ndarray, cost: float,
                 degear_thr: float, degear_floor: float) -> np.ndarray:
    """Per-asset managed return at EQUAL notional to buy-and-hold: pos[t-1] earns ret[t], turnover
    charged, then drawdown-de-geared. No vol-targeting here (applied at the combined-book layer) so
    the trend overlay's effect is compared apples-to-apples against holding the asset."""
    m = min(len(pos), len(ret))
    pos, ret = pos[:m], ret[:m]
    gross = np.zeros(m); gross[1:] = pos[:-1] * ret[1:]
    turn = np.zeros(m); turn[0] = abs(pos[0]); turn[1:] = np.abs(pos[1:] - pos[:-1])
    net = gross - turn * cost
    if degear_thr and degear_thr > 0:
        net = drawdown_degear(net, dd_threshold=degear_thr, floor=degear_floor)
    return net


def _fold_sharpes(net, folds, oos_frac):
    n = len(net)
    edges = np.linspace(int(n * (1 - oos_frac)), n, folds + 1).astype(int)
    out = []
    for k in range(folds):
        seg = net[int(edges[k]):int(edges[k + 1])]
        out.append(float(seg.mean() / (seg.std() + 1e-12) * np.sqrt(PPY)) if seg.size > 2 else 0.0)
    return out


def causal_weights(R: np.ndarray, window: int, mode: str) -> np.ndarray:
    """(S, L) causal per-asset weights for combining managed sleeves. ``equal`` = equal weight among
    the assets ACTIVE at each bar (an asset is active once its trailing vol is defined — i.e. it's
    listed and past warm-up); ``risk_parity`` = inverse trailing-vol among the active assets (so a
    70%-vol crypto sleeve doesn't dominate the book's risk). Weights at bar t use vol through t-1
    (shifted) → strictly causal; not-yet-listed (flat) sleeves get weight 0 automatically."""
    S, L = R.shape
    roll = pd.DataFrame(R.T).rolling(window, min_periods=max(10, window // 4)).std().to_numpy()  # (L,S)
    active = roll > 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        base = np.where(active, 1.0 / roll, 0.0) if mode == "risk_parity" else active.astype(float)
    rs = base.sum(axis=1, keepdims=True)
    w = np.where(rs > 0, base / np.where(rs > 0, rs, 1.0), 0.0)        # (L, S), each row sums to 1
    w = pd.DataFrame(w).shift(1).to_numpy()                            # causal: bar t uses t-1 vol
    return np.nan_to_num(w, nan=0.0).T                                # (S, L)


def combine_book(per_asset_net, alloc, window, target_vol, max_lev, dg_thr, dg_floor):
    """Combine raw per-asset managed sleeves (no per-asset de-gear) into a book: weight (equal /
    risk_parity), then portfolio vol-target, then ONE drawdown de-gear at the book level."""
    R = np.vstack(per_asset_net)
    w = causal_weights(R, window, alloc)
    combined = (w * R).sum(axis=0)
    net = combined * vol_target_scale(combined, target_vol, PPY, max_leverage=max_lev)
    if dg_thr and dg_thr > 0:
        net = drawdown_degear(net, dg_thr, dg_floor)
    return net


def main():
    ap = argparse.ArgumentParser(description="Managed-beta vs buy-and-hold research")
    ap.add_argument("--symbols", default=" ".join(DEFAULT_SYMBOLS))
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--trend-ema", type=int, default=200)
    ap.add_argument("--down-exposure", type=float, default=0.0)   # <0 = trend-follow short leg
    ap.add_argument("--target-vol", type=float, default=0.15)
    ap.add_argument("--max-leverage", type=float, default=2.0)
    ap.add_argument("--degear-threshold", type=float, default=0.15)
    ap.add_argument("--degear-floor", type=float, default=0.25)
    ap.add_argument("--fee-bps", type=float, default=2.0)
    ap.add_argument("--slippage-bps", type=float, default=3.0)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--oos-frac", type=float, default=0.5)
    ap.add_argument("--use-macro", action="store_true", help="apply VIX + cross-asset correlation de-risk overlay")
    ap.add_argument("--vix-lo", type=float, default=18.0)
    ap.add_argument("--vix-hi", type=float, default=32.0)
    ap.add_argument("--corr-hi", type=float, default=0.65)
    ap.add_argument("--macro-floor", type=float, default=0.25)
    args = ap.parse_args()

    symbols = args.symbols.split()
    data = load_daily(symbols, start=args.start, skip_download=args.skip_download)
    if not data:
        print("no data loaded (drop --skip-download to fetch)")
        return
    cost = (args.fee_bps + args.slippage_bps) / 1e4
    params = ManagedBetaParams(trend_ema=args.trend_ema, down_exposure=args.down_exposure)
    strat = ManagedBeta(params)
    positions = strat.generate_positions(data)

    print(f"\nManaged beta vs buy-and-hold | trend EMA {args.trend_ema} | down-exposure "
          f"{args.down_exposure} | de-gear@{args.degear_threshold:.0%}")
    print(f"{'asset':10s} {'   BH Sharpe':>12s} {'  Mgd Sharpe':>12s} {'  BH maxDD':>10s} "
          f"{' Mgd maxDD':>10s} {'BH Calmar':>9s} {'MgdCalmar':>9s} {'a-vs-BH':>8s} {'capture':>7s}")
    for s in symbols:
        if s not in data:
            continue
        ret = _bar_returns(data[s]["close"].to_numpy())
        bh = _metrics(ret)
        net = _managed_net(positions[s], ret, cost, args.degear_threshold, args.degear_floor)
        mg = _metrics(net)
        a, b = regression_alpha_beta(net, ret[:len(net)], bars_per_year=PPY)
        print(f"{s:10s} {bh['sharpe']:>12.2f} {mg['sharpe']:>12.2f} {bh['max_drawdown']*100:>9.1f}% "
              f"{mg['max_drawdown']*100:>9.1f}% {bh['calmar']:>9.2f} {mg['calmar']:>9.2f} "
              f"{a*100:>+7.1f}% {b:>7.2f}")

    # combined book: equal-weight vs RISK-PARITY (inverse-vol) weighting across the managed sleeves
    aligned = align_panel(data, how="outer")
    apos = ManagedBeta(params).generate_positions(aligned)
    per_asset_raw = [_managed_net(apos[s], _bar_returns(aligned[s]["close"].to_numpy()), cost,
                                  0.0, args.degear_floor) for s in aligned]   # no per-asset de-gear
    ew_bh = np.mean([_bar_returns(aligned[s]["close"].to_numpy()) for s in aligned], axis=0)
    ew = _metrics(ew_bh)

    def _book(alloc):
        return combine_book(per_asset_raw, alloc, 63, args.target_vol, args.max_leverage,
                            args.degear_threshold, args.degear_floor)

    def _line(label, net):
        m = _metrics(net)
        a, b = regression_alpha_beta(net, ew_bh[:len(net)], bars_per_year=PPY)
        fs = _fold_sharpes(net, args.folds, args.oos_frac); p = sum(1 for x in fs if x > 0)
        print(f"  {label:24s}: Sharpe {m['sharpe']:+.2f}  CAGR {m['ann_return']*100:+5.1f}%  "
              f"maxDD {m['max_drawdown']*100:+6.1f}%  Calmar {m['calmar']:.2f}  "
              f"a-vs-BH {a*100:+.1f}%/yr beta {b:.2f}  ({p}/{args.folds}+)")
        return net

    print("\n=== COMBINED managed book vs equal-weight buy-and-hold ===")
    print(f"  {'equal-weight BUY & HOLD':24s}: Sharpe {ew['sharpe']:+.2f}  CAGR {ew['ann_return']*100:+5.1f}%"
          f"  maxDD {ew['max_drawdown']*100:+6.1f}%  Calmar {ew['calmar']:.2f}")
    _line("managed (equal-weight)", _book("equal"))
    rp_net = _line("managed (RISK-PARITY)", _book("risk_parity"))

    if args.use_macro:
        from backend.signals.macro_regime import combined_macro_scalar  # noqa: E402
        asof = pd.to_datetime(next(iter(aligned.values()))["timestamp"])
        vix = None                       # VIX aligned to the book calendar (weekend NaN → ffill)
        vdata = load_daily(["^VIX"], start=args.start, skip_download=args.skip_download)
        if "^VIX" in vdata:
            vser = (vdata["^VIX"].assign(timestamp=lambda d: pd.to_datetime(d["timestamp"]))
                    .set_index("timestamp")["close"].reindex(asof).ffill())
            vix = vser.to_numpy(np.float64)
        rets_panel = np.column_stack([_bar_returns(aligned[s]["close"].to_numpy()) for s in aligned])
        scalar = combined_macro_scalar(vix=vix, returns=rets_panel, floor=args.macro_floor,
                                       vix_kw={"lo": args.vix_lo, "hi": args.vix_hi},
                                       corr_kw={"hi": args.corr_hi})
        scalar = scalar[-len(rp_net):]
        _line("risk-parity + MACRO", rp_net * scalar)
        print(f"  (macro: {'VIX+' if vix is not None else ''}correlation de-risk, "
              f"mean exposure {scalar.mean():.2f}, min {scalar.min():.2f})")

    print("\nWIN = Managed Sharpe > B&H Sharpe AND shallower maxDD. Risk-parity should raise Sharpe by")
    print("balancing risk across sleeves (low-vol bonds/gold get more weight, crypto less).")


if __name__ == "__main__":
    main()
