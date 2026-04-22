# Training Guide: NN + News + Online Adaptation Backbone

This document is a practical guide for training and improving the trading model in this repository.

## 1) Objectives

You want a model that:
- Generalizes across market regimes (trend / chop / high-volatility / news shocks)
- Handles multiple assets without overfitting
- Uses news impact in a symbol-aware way
- Learns slowly online from real outcomes without catastrophic drift

---

## 2) Current Backbone Added

The codebase now includes a training backbone to make future training easier:

- `backend/training/backbone.py`
  - `CRYPTO_KEYWORD_BANK`: editable keyword bank per symbol.
  - `extract_symbol_relevance(...)`: keyword relevance scoring per symbol.
  - `TrainingBackbone`: JSONL decision/outcome recorder.

- News pipeline enhancement:
  - `LLMNewsAgent` now augments each `NewsImpact` with:
    - `symbol_relevance` (per-symbol score)
    - `matched_keywords` (matched terms map)

- NN agent enhancement:
  - Uses symbol-specific news filtering for feature generation.
  - Logs decision snapshots (`training_data/decision_log.jsonl`).
  - Logs outcomes (`training_data/outcome_log.jsonl`).
  - Stores feature sequence at entry and reuses it on close for online updates.

---

## 3) Data Strategy (Most Important)

Train on diverse historical windows, not one recent period:

1. Bull expansion windows
2. Bear/downtrend windows
3. Ranging low-vol windows
4. High-volatility event windows (macro / regulatory / exchange shocks)

Use walk-forward validation:
- Train on window A
- Validate on next unseen window B
- Roll and repeat

Do **not** shuffle across time globally.

---

## 4) Candle Timeframes and Labels

Recommended setup:

- Execution timeframe: **5m**
- Add higher-timeframe context features:
  - 15m
  - 1h
  - 4h

Label suggestions:
- Multi-horizon labels (e.g. +3, +12, +48 candles)
- Or single target horizon aligned to your trade holding logic

If you keep a single horizon, stay consistent across assets and backtests.

---

## 5) Single Model vs Per-Asset Models

Start with one shared model + asset conditioning:
- Add asset ID embedding or one-hot symbol features.
- Keep shared representation for sample efficiency.

Split to separate models only if:
- You have enough per-asset data, and
- The assets have very different microstructure behavior.

---

## 6) News Optimization Workflow

### Keyword Bank
Edit `CRYPTO_KEYWORD_BANK` in:
- `backend/training/backbone.py`

For each symbol:
- Keep 15–40 high-signal phrases
- Include ecosystem terms, protocol names, legal topics, ETF terms, major entities
- Remove noisy generic terms that match everything

### LLM + Keyword Fusion
Current flow:
1. LLM classifies severity/direction/magnitude/timing.
2. Keyword matcher scores symbol relevance.
3. NN agent only feeds relevant news into each symbol’s features.

This keeps cross-symbol noise lower.

---

## 7) Online Learning / Reinforcement-Like Updates

Current live-safe approach:
- Record entry sequence.
- On close, update with realized outcome.
- Keep updates small/frequent rather than large/rare.

Recommended safeguards:
- Tiny LR for online updates
- Update every N trades (already batched in model)
- Rollback triggers on drawdown windows
- Weekly full retrain from logs

---

## 8) Training Data Files

Backbone logs are written to:
- `training_data/decision_log.jsonl`
- `training_data/outcome_log.jsonl`

Use these to build supervised or RL-style datasets.

Suggested schema usage:
- Join decision and outcome by symbol + nearest timestamp or trade_id.
- Build reward-adjusted labels:
  - positive pnl => reinforce direction
  - negative pnl => penalize direction
  - near-zero => favor hold or low-size

---

## 9) Suggested Experiments (Order)

1. **Baseline**: current model + improved historical windows
2. Add symbol embedding
3. Add multi-horizon outputs
4. Add news-gated loss weighting
5. Add confidence calibration (temperature scaling)
6. Add conservative online updates from outcome logs

Track:
- Win rate
- Expectancy
- Max DD
- Sharpe / Sortino
- Turnover and slippage sensitivity

---

## 10) Practical “Do / Don’t”

Do:
- Keep validation strictly out-of-time
- Maintain checkpoint + rollback discipline
- Version feature pipeline with model checkpoints

Don’t:
- Train only on one regime
- Change feature definitions without versioning
- Let online updates run with high LR in live mode

---

## 11) Next Steps You Can Do Immediately

1. Edit keyword bank per symbol in `backend/training/backbone.py`.
2. Run paper trading and collect decision/outcome logs for 1–2 weeks.
3. Build offline retraining pipeline from JSONL logs.
4. Compare champion vs challenger checkpoints before live promotion.


---

## 12) Offline Retraining Script (Added)

You can now retrain directly from collected logs using:

```bash
cd trading-agent
python scripts/train_from_logs.py --data-dir training_data --epochs 8 --batch-size 64 --lr 5e-5 --min-abs-pnl-pct 0.001
```

What it does:
- Loads `training_data/decision_log.jsonl` and `training_data/outcome_log.jsonl`
- Aligns each approved decision with the next outcome for the same symbol
- Builds supervised direction targets from realized PnL sign
- Fine-tunes the current `PersistentTradingModel`
- Saves a runtime-compatible checkpoint via `safe_checkpoint(label=\"offline_logs\")`

Tip:
- Start with low learning rates (`1e-5` to `5e-5`) and few epochs.
- Track out-of-sample performance before deploying new checkpoints.
