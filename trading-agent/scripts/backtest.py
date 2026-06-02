#!/usr/bin/env python
"""Out-of-sample backtest of a trained checkpoint — the measuring stick (Cycle 4).

Builds each symbol's features the SAME way as training, runs the model over the
held-out tail, turns primary-horizon probabilities into long/flat/short signals
(uncertainty/edge-gated, mirroring the live agent), and simulates with realistic
fees + slippage via ``backend.backtest.engine``. Reports Sharpe / Sortino / max-DD /
turnover / hit-rate per symbol and aggregate.

Run AFTER training, e.g.:
    python scripts/backtest.py --checkpoint models/trading_lstm_latest.pt \
        --start-year 2024 --fee-bps 10 --slippage-bps 5 --min-confidence 0.45 --min-edge 0.05
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

# Reuse the EXACT training feature pipeline so backtest features == training features.
_spec = importlib.util.spec_from_file_location("pretrain_bt_mod", str(ROOT / "scripts" / "pretrain.py"))
pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pre)

from backend.backtest.engine import (  # noqa: E402
    run_backtest, run_exec_backtest, directional_signal, atr_from_ohlc, BARS_PER_YEAR_5M,
)

PRIMARY_H = 0   # horizon head to trade (0 = nearest). probs columns: [long, short, hold]
BARS_PER_YEAR_STOCK = 252 * 78   # ~regular-hours 5m bars/yr (annualization only)


def load_model(checkpoint: str, device) -> torch.nn.Module:
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model = pre.ImprovedTradingLSTM(
        input_size=pre.INPUT_SIZE, hidden_size=pre.HIDDEN_SIZE, num_layers=pre.NUM_LSTM_LAYERS,
        dropout=pre.DROPOUT, num_symbols=len(pre.SYMBOLS), symbol_embed_dim=pre.SYMBOL_EMBED_DIM,
        num_horizons=len(pre.HORIZONS), num_classes=pre.NUM_CLASSES,
    ).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    return model


def build_symbol_features(sym: str, start_year: int, start_month: int, skip_download: bool):
    """(features (N,86), close, high, low) — identical pipeline to pretrain.build_dataset."""
    dfs = pre.load_full_history(sym, start_year, start_month, skip_download)
    df5 = dfs["5m"]
    base = pre.detect_regime(df5, pre.build_feature_matrix(df5))
    htf = pre.build_htf_features(df5, dfs["1h"], dfs["4h"])
    feats = pre.apply_rolling_zscore(np.concatenate([base, htf], axis=1))
    feats = np.concatenate([feats, pre.build_news_embed_matrix(df5, sym)], axis=1)
    return (feats.astype(np.float32), df5["close"].to_numpy(np.float64),
            df5["high"].to_numpy(np.float64), df5["low"].to_numpy(np.float64))


@torch.no_grad()
def probs_for_starts(model, feats, starts, sym_id, device, seq_len=None, batch=2048):
    """Primary-horizon class probs for each window starting at ``starts`` — batched
    via a sliding-window view so we never materialise all sequences at once."""
    seq_len = seq_len or pre.SEQ_LEN
    sw = sliding_window_view(feats, seq_len, axis=0)   # (·, F, seq_len) zero-copy
    out = []
    for s in range(0, len(starts), batch):
        b = starts[s:s + batch]
        block = sw[b].transpose(0, 2, 1).astype(np.float32)   # (cs, seq_len, F)
        bx = torch.from_numpy(block).to(device)
        bs = torch.full((len(b),), int(sym_id), dtype=torch.long, device=device)
        out.append(model(bx, bs)[1][PRIMARY_H].float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, pre.NUM_CLASSES), np.float32)


def backtest_symbol(model, sym, args, device):
    feats, close, high, low = build_symbol_features(sym, args.start_year, args.start_month, args.skip_download)
    seq_len = pre.SEQ_LEN
    n = len(feats)
    if n <= seq_len + 2:
        return None
    starts = np.arange(0, n - seq_len)                 # window k → decision bar k+seq_len-1
    bar = starts + seq_len - 1
    close_seq, high_seq, low_seq = close[bar], high[bar], low[bar]
    cut = int(len(starts) * (1.0 - args.val_frac))     # out-of-sample tail
    starts_oos = starts[cut:]
    close_oos, high_oos, low_oos = close_seq[cut:], high_seq[cut:], low_seq[cut:]
    if len(starts_oos) < 3:
        return None
    probs = probs_for_starts(model, feats, starts_oos, pre.SYMBOL_TO_ID[sym], device)
    sig = directional_signal(probs[:, 0], probs[:, 1], min_confidence=args.min_confidence,
                             min_edge=args.min_edge, allow_short=not args.long_only)
    ppy = BARS_PER_YEAR_STOCK if pre._is_stock_symbol(sym) else BARS_PER_YEAR_5M
    if args.exec:
        # Cycle 5 execution layer: vol-targeted sizing + ATR trailing/breakeven/scale-out.
        atr = atr_from_ohlc(high_oos, low_oos, close_oos, window=14)
        fvol = pre._rolling_vol(close_oos)
        return run_exec_backtest(
            close_oos, high_oos, low_oos, atr, sig, forecast_vol=fvol,
            target_vol=args.target_vol, max_size=1.0, stop_atr=args.stop_atr,
            trail_atr=args.trail_atr, breakeven_atr=args.breakeven_atr, tp1_atr=args.tp1_atr,
            scale_out_frac=args.scale_out, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
            allow_short=not args.long_only, bars_per_year=ppy)
    return run_backtest(close_oos, sig, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                        allow_short=not args.long_only, bars_per_year=ppy)


def _fmt(m):
    return (f"ret={m['total_return']*100:+6.1f}%  ann={m['ann_return']*100:+6.1f}%  "
            f"Sharpe={m['sharpe']:5.2f}  Sortino={m['sortino']:5.2f}  "
            f"maxDD={m['max_drawdown']*100:6.1f}%  trades={int(m['num_trades']):4d}  "
            f"hit={m['hit_rate']*100:4.0f}%  PF={m['profit_factor']:.2f}")


def main():
    ap = argparse.ArgumentParser(description="Out-of-sample backtest of a trained checkpoint")
    ap.add_argument("--checkpoint", default="models/trading_lstm_latest.pt")
    ap.add_argument("--symbols", nargs="+", default=pre.SYMBOLS)
    ap.add_argument("--start-year", type=int, default=2024)
    ap.add_argument("--start-month", type=int, default=1)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.2, help="out-of-sample tail fraction")
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--min-confidence", type=float, default=0.45)
    ap.add_argument("--min-edge", type=float, default=0.05)
    ap.add_argument("--long-only", action="store_true")
    # Cycle 5 execution layer (ATR-scaled stops + vol-targeted sizing).
    ap.add_argument("--exec", action="store_true",
                    help="Use the execution backtester (trailing stop, breakeven, scale-out, vol sizing)")
    ap.add_argument("--target-vol", type=float, default=0.01, help="per-bar vol target for sizing")
    ap.add_argument("--stop-atr", type=float, default=2.0)
    ap.add_argument("--trail-atr", type=float, default=3.0)
    ap.add_argument("--breakeven-atr", type=float, default=1.0)
    ap.add_argument("--tp1-atr", type=float, default=2.0)
    ap.add_argument("--scale-out", type=float, default=0.5, help="fraction taken off at tp1")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} on {device}\n")

    sharpes, rets = [], []
    print(f"{'SYMBOL':10s}  out-of-sample backtest (fees {args.fee_bps}+{args.slippage_bps}bps)")
    print("-" * 110)
    for sym in args.symbols:
        if sym not in pre.SYMBOL_TO_ID:
            continue
        try:
            res = backtest_symbol(model, sym, args, device)
        except Exception as e:  # data gap, etc. — keep going
            print(f"{sym:10s}  SKIPPED ({str(e)[:60]})")
            continue
        if res is None:
            print(f"{sym:10s}  (insufficient data)")
            continue
        print(f"{sym:10s}  {_fmt(res.metrics)}")
        sharpes.append(res.metrics["sharpe"])
        rets.append(res.metrics["total_return"])

    if sharpes:
        print("-" * 110)
        print(f"{'AGGREGATE':10s}  mean Sharpe={np.mean(sharpes):5.2f}  "
              f"median Sharpe={np.median(sharpes):5.2f}  mean OOS return={np.mean(rets)*100:+.1f}%  "
              f"symbols={len(sharpes)}")
        print("\nRead: Sharpe>1 and positive return AFTER costs across most symbols = a real edge. "
              "Compare runs to judge whether a change actually helped.")


if __name__ == "__main__":
    main()
