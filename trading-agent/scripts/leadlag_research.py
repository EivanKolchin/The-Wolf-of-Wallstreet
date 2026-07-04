#!/usr/bin/env python
"""Lead-lag network probe — does "asset A moves asset B with a delay" survive OOS and costs?

Sweeps FIVE timescales (5m/15m/30m/4h on the cached crypto panel; daily on the mixed
ETF+crypto panel) because the hypothesis has opposite priors by speed: fast lead-lag is
the documented hunting ground of HFT latency racers (expect: raw signal maybe, net of
15bps nothing), slow lead-lag (sector/supply-chain diffusion) may persist at retail cost.

Per timescale: chronological walk-forward — the directed edge network is re-estimated on
each fold's train prefix only, the follower book trades the test slice, costs charged on
turnover. Reported: in-sample edge strength, OOS edge persistence (the SAME edges' signed
correlation on test — the honest number), gross vs net Sharpe, DSR at the true trial count.

    python scripts/leadlag_research.py                # all five timescales, cached data only
    python scripts/leadlag_research.py --scales 4h 1d
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
sys.path.insert(0, str(ROOT / "scripts"))

if hasattr(sys.stdout, "reconfigure"):          # Windows cp1252 console → UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.signals.leadlag import LeadLagConfig, walk_forward          # noqa: E402
from backend.backtest.engine import deflated_sharpe_ratio               # noqa: E402

CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
          "LINKUSDT", "LTCUSDT", "BCHUSDT", "DOTUSDT", "ATOMUSDT", "ETCUSDT", "FILUSDT",
          "NEARUSDT", "UNIUSDT", "XLMUSDT", "ALGOUSDT", "AAVEUSDT", "MATICUSDT"]
DAILY = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "TQQQ",
         "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD",
         "DOGE-USD", "LTC-USD", "BCH-USD", "SLV", "USO", "EWJ", "FXI", "EWZ",
         "HYG", "LQD", "SHY", "VNQ"]

# timescale → (pandas resample rule or None for native/daily, bars/year, cost bps round-trip)
SCALES = {
    "5m":  (None,    365 * 24 * 12, 15.0),
    "15m": ("15min", 365 * 24 * 4,  15.0),
    "30m": ("30min", 365 * 24 * 2,  15.0),
    "4h":  ("4h",    365 * 6,       15.0),
    "1d":  ("daily", 252,           10.0),
}


def crypto_panel(rule: str | None, start_year: int) -> pd.DataFrame:
    """(T, N) close-price panel from the cached 5m parquet, resampled to ``rule``."""
    import pretrain as pre
    frames = {}
    for s in CRYPTO:
        try:
            df = pre.load_full_history(s, start_year, 1, skip_download=True)["5m"]
        except Exception as e:
            print(f"  skip {s}: {str(e)[:60]}")
            continue
        ser = df.set_index(pd.to_datetime(df["timestamp"]))["close"].astype(float)
        if rule:
            ser = ser.resample(rule).last()
        frames[s] = ser
    panel = pd.DataFrame(frames).dropna(how="all")
    return panel.dropna(axis=1, thresh=int(len(panel) * 0.9)).dropna()


def daily_panel(start: str = "2015-01-01") -> pd.DataFrame:
    from backend.data.equity_daily import load_daily
    data = load_daily(DAILY, start="2010-01-01", skip_download=True)
    frames = {s: d.set_index(pd.to_datetime(d["timestamp"]))["close"].astype(float)
              for s, d in data.items()}
    panel = pd.DataFrame(frames)
    panel = panel[panel.index >= start]
    # ffill BEFORE the coverage check: the union calendar contains crypto weekends, where every
    # ETF is "missing" ~29% of rows — without the ffill the whole equity side gets dropped and
    # the panel collapses to 2 crypto columns. After ffill only pre-listing NaN remains.
    panel = panel.ffill()
    return panel.dropna(axis=1, thresh=int(len(panel) * 0.7)).dropna()


def run_scale(name: str, args) -> dict | None:
    rule, bpy, cost = SCALES[name]
    print(f"\n================ {name}  (cost {cost:.0f}bps rt, {bpy:,} bars/yr) ================")
    if rule == "daily":
        panel = daily_panel()
    else:
        panel = crypto_panel(rule, args.start_year)
    if panel.shape[1] < 5 or len(panel) < 1000:
        print(f"  insufficient data ({panel.shape}) — skipped")
        return None
    R = panel.pct_change().iloc[1:].to_numpy()
    print(f"  panel: {panel.shape[1]} assets × {len(R):,} bars "
          f"({panel.index.min().date()} → {panel.index.max().date()})")

    cfg = LeadLagConfig(min_abs_corr=args.min_corr, max_edges_per_follower=args.max_edges)
    folds = walk_forward(R, folds=args.folds, cost_bps=cost, bars_per_year=bpy, cfg=cfg)
    if not folds:
        print("  no valid folds — skipped")
        return None
    for k, f in enumerate(folds):
        print(f"  fold {k+1}: edges={f.n_edges:4d}  IC_train={f.ic_insample:+.3f}  "
              f"IC_oos={f.ic_oos:+.3f}  Sharpe gross={f.sharpe_gross:+.2f}  "
              f"NET={f.sharpe_net:+.2f}  turn/bar={f.turnover:.3f}")
    m = {k: float(np.mean([getattr(f, k) for f in folds]))
         for k in ("ic_insample", "ic_oos", "sharpe_gross", "sharpe_net")}
    pos_folds = sum(1 for f in folds if f.sharpe_net > 0)
    # DSR on the mean net Sharpe: per-bar SR over the pooled OOS length, trials = all scales
    n_obs = sum(1 for _ in folds) * (len(R) // (args.folds + 1))
    sr_bar = m["sharpe_net"] / np.sqrt(bpy)
    dsr = deflated_sharpe_ratio(sr_bar, n_obs=max(n_obs, 10), n_trials=args.n_trials)
    print(f"  MEAN: IC_train {m['ic_insample']:+.3f} → IC_oos {m['ic_oos']:+.3f}   "
          f"gross {m['sharpe_gross']:+.2f} → NET {m['sharpe_net']:+.2f}   "
          f"{pos_folds}/{len(folds)} folds +   DSR {dsr:.3f}")
    return {"scale": name, **m, "pos_folds": f"{pos_folds}/{len(folds)}", "dsr": round(dsr, 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scales", nargs="+", default=list(SCALES), choices=list(SCALES))
    ap.add_argument("--start-year", type=int, default=2022)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--min-corr", type=float, default=0.03)
    ap.add_argument("--max-edges", type=int, default=5)
    ap.add_argument("--n-trials", type=int, default=5,
                    help="trial count for the DSR (all timescales tested = honest penalty)")
    args = ap.parse_args()

    rows = [r for s in args.scales if (r := run_scale(s, args))]
    if rows:
        print("\n================ SUMMARY (promote nothing below DSR 0.95) ================")
        df = pd.DataFrame(rows).set_index("scale")
        print(df.to_string(float_format=lambda v: f"{v:+.3f}"))
        print("\nREAD: IC_oos ≈ 0 with IC_train > 0 = the edges are noise-mined; "
              "gross>0 but NET<0 = real diffusion that HFT/costs already ate.")


if __name__ == "__main__":
    main()
