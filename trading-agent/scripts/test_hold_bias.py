"""Hold bias controlled test — Step 1 of the recovery plan.

Measures the model's hold ratio with and without inference-time manipulations.
Runs the pretraining pipeline with NEUTRAL settings, reports class distribution.

Usage:
    python scripts/test_hold_bias.py --epochs 10

Output: hold ratio on the validation set (H+12 horizon).
If hold ratio > 0.7 without hacks, the bias is structural (architecture issue).
If < 0.5, the hacks were doing active harm.
"""
import os
import argparse
import sys
import tempfile
from pathlib import Path

# Must set BEFORE any config/model import
os.environ["NN_HOLD_PROB_MULTIPLIER"] = "1.0"   # neutral: no inference-time hold scaling
os.environ["NN_IDLE_HOLD_ENABLED"] = "False"     # neutral: no idle-pressure ramp
os.environ["PRETRAIN_NUM_WORKERS"] = "0"         # Windows page file limit: no multiprocessing

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import torch
from structlog import get_logger

log = get_logger("test_hold_bias")


def main():
    parser = argparse.ArgumentParser(description="Hold bias controlled test")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--start-year", type=int, default=2026)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    parser.add_argument("--mmap", action="store_true", help="Use disk memmap to reduce RAM")
    parser.add_argument("--mmap-dir", default=None,
                        help="Temp dir for memmap (default: system temp)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Reuse cached parquet files")
    args = parser.parse_args()

    log.info("Hold bias test starting", epochs=args.epochs,
             note="ALL inference-time hold manipulations DISABLED",
             hold_mult=os.environ.get("NN_HOLD_PROB_MULTIPLIER"),
             idle_enabled=os.environ.get("NN_IDLE_HOLD_ENABLED"),
             start=f"{args.start_year}-{args.start_month:02d}")

    from scripts.pretrain import (
        build_dataset, make_split_loaders, make_weighted_loss,
        train_epoch, eval_epoch, save_checkpoint,
        ImprovedTradingLSTM, HORIZONS, INPUT_SIZE, HIDDEN_SIZE,
        NUM_LSTM_LAYERS, DROPOUT, SYMBOLS, SYMBOL_EMBED_DIM,
        BATCH_SIZE, LR, WEIGHT_DECAY, MODELS_DIR,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device", device=str(device))

    use_mmap = args.mmap
    mmap_dir = args.mmap_dir or (tempfile.mkdtemp(prefix="holdbias_") if use_mmap else None)
    if use_mmap:
        log.info("Using mmap", dir=mmap_dir)

    build_out = build_dataset(
        args.symbols, args.start_year, args.start_month,
        skip_download=args.skip_download,
        mmap_dir=mmap_dir,
    )

    train_loader, val_loader, y_tr, n_tr, n_va = make_split_loaders(
        build_out, BATCH_SIZE, val_frac=0.15, test_frac=0.20,
    )
    log.info("Dataset ready", train=n_tr, val=n_va)

    # Class distribution BEFORE training
    for hi, h in enumerate(HORIZONS):
        counts = np.bincount(y_tr[:, hi], minlength=3)
        hold_frac = counts[2] / counts.sum()
        log.info(f"Train H+{h} hold fraction: {hold_frac:.4f} "
                 f"(distribution: long={counts[0]}, short={counts[1]}, hold={counts[2]})")

    model = ImprovedTradingLSTM(
        input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LSTM_LAYERS, dropout=DROPOUT,
        num_symbols=len(SYMBOLS), symbol_embed_dim=SYMBOL_EMBED_DIM,
        num_horizons=len(HORIZONS), num_classes=3,
    ).to(device)

    loss_fns = make_weighted_loss(y_tr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fns, device)
        val_loss, preds_list, labels_list, probs_list, _ = eval_epoch(
            model, val_loader, loss_fns, device)

        # Measure hold ratio on H+12 (index 1 == the selected trading horizon)
        h_idx = 1
        preds = preds_list[h_idx]
        labels = labels_list[h_idx]
        hold_pred = (preds == 2).mean()
        hold_true = (labels == 2).mean()
        accuracy = (preds == labels).mean()

        traded = preds != 2
        trade_correct = ((preds == labels) & traded).sum()
        trade_total = traded.sum()
        trade_acc = trade_correct / trade_total if trade_total > 0 else 0.0

        log.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"val_loss={val_loss:.4f} | acc={accuracy:.4f} | "
            f"hold_pred={hold_pred:.4f} hold_true={hold_true:.4f} | "
            f"trade_acc={trade_acc:.4f} trade_frac={trade_total/len(preds):.4f}"
        )

    # Final hold bias report
    log.info("===== HOLD BIAS TEST RESULTS =====")
    for hi, h in enumerate(HORIZONS):
        preds = preds_list[hi]
        labels = labels_list[hi]
        hold_pred = (preds == 2).mean()
        hold_true = (labels == 2).mean()
        long_pred = (preds == 0).mean()
        short_pred = (preds == 1).mean()
        log.info(f"H+{h}: hold_pred={hold_pred:.4f} hold_true={hold_true:.4f} "
                 f"long_pred={long_pred:.4f} short_pred={short_pred:.4f}")

    bias = hold_pred - hold_true
    verdict = ""
    if bias > 0.15:
        verdict = "STRUCTURAL HOLD BIAS — architecture or class weighting needs changing"
    elif bias < -0.10:
        verdict = "ANTI-HOLD BIAS — model under-predicts hold (over-trading)"
    elif abs(bias) < 0.05:
        verdict = "NO SIGNIFICANT HOLD BIAS — the inference-time hacks were doing active harm"
    else:
        verdict = "MILD BIAS — tune CLASS_WEIGHT_POWER closer to uniform"
    log.info(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
