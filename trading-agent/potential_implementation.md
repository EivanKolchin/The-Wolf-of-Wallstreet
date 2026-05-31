# Potential Implementation — Option B: Autoregressive Candle Transformer

> **Status:** Deferred. This is the exploratory **v3** path for Phase 18 (predicted-price chart) — gated on empirical evidence that the hybrid (v1 MC-dropout bands + v2 OHLC delta head) is insufficient. See `~/.claude/plans/ok-please-wire-the-kind-pie.md` Phase 18 for the comparison and the trigger criteria.
>
> Saving this design here so it can be picked up later without re-deriving.

---

## Why this exists (not yet, but maybe)

Option A (the hybrid being built) gives ≈ 80 % of this transformer's forecasting quality at ≈ 20 % of the cost, plus *native* uncertainty bands. The transformer's marginal MAE win on noisy 5 m crypto data is ~3–8 % absolute (next-1) — small. But it has two genuine advantages worth picking up *if and only if* the hybrid underperforms in production:

1. **Path-dependent forecasts.** An autoregressive model can natively express "if price breaks resistance, big move; if not, mean revert" — the hybrid's regression head smooths this into the mean.
2. **Smoother degradation past candle 5.** The hybrid's direct regression flattens fast; the transformer's autoregressive conditioning holds shape better for K_FUTURE > 5.

## Empirical trigger to build this

Per the plan's Phase 18 gate:

- After v1 + v2 are live, measure 1-candle / 5-candle MAE on a held-out window per symbol.
- **Build this** if **both** of the following are true:
  - v2's 5-candle MAE > **0.95 × random walk** (i.e. the OHLC head is barely better than naive forecast at K=5), AND
  - v2's gain over v1-only is **< ~3 %** (i.e. the head isn't learning useful path-dependence — that's the smoking gun that a more expressive model is warranted).
- Otherwise the ~3–8 % marginal MAE gain does not justify ~5–7 days of build + ongoing tokeniser maintenance.

## Architecture

```
Past 60 candles → per-candle token embedding → causal transformer → autoregressive next-K candles
```

### Tokenisation

Each candle is encoded as a fixed-vocabulary token computed from its OHLC values relative to recent history.

**Recommended scheme — per-channel quantile binning:**
- For each of O, H, L, C compute the log-return vs the previous close: `log(x_t / close_{t-1})`.
- Discretise each into K=64 bins by quantile, fit on a rolling window of the last ~10k candles per symbol.
- Token = (open_bin, high_bin, low_bin, close_bin); embed each channel separately then sum: `emb = E_o[bin_o] + E_h[bin_h] + E_l[bin_l] + E_c[bin_c]`, each `E_*: (64, d_model)`.

**Alternative — joint hash:**
- One big vocabulary V = 64^4 = ~16 M is unworkable.
- 64^2 = 4096 (e.g. close × range) with the other two as continuous side-features — simpler but loses some information.

Per-channel summed embeddings are the right starting point: ~64 × 4 × 128 = 32 k embedding params per channel × 4 ≈ 128 k for embeddings.

### Model

- 2 layers × 4 heads × hidden 128 × FFN 4× ≈ 400 k transformer body params.
- Causal mask (standard GPT-style).
- 4 separate output heads (one per channel), each `Linear(d_model, K=64)`. ≈ 4 × 128 × 64 ≈ 32 k per head × 4 = 128 k output projection.
- **Total: ≈ 700 k – 1 M params**, ≈ 3–4 MB float32, ≈ 1 MB int8 if quantised at load (IPEX supports this).

### Inference (autoregressive K_FUTURE candles)

1. Encode last 60 candles → context.
2. For step k = 1..K_FUTURE:
   - Forward the model on `context + predictions_so_far` (use KV-cache so we only pay incremental cost per step).
   - Sample each channel head independently (argmax or top-k+temperature sampling depending on whether we want deterministic mode or uncertainty).
   - Convert the 4 chosen bin indices back to a (O, H, L, C) tuple via the bin centres.
   - Append the new candle to the running prediction; update the context.

CPU latency estimate with KV-cache reuse on Intel Iris Xe + IPEX: ≈ 15–40 ms for the initial pass, ≈ 10–20 ms per subsequent step. K_FUTURE=5 ≈ 60–120 ms total. Comfortable inside the visualisation loop budget. Becomes tight if K_FUTURE > 8.

## Training

- **Data source:** the same offline pretrain dataset used by the main policy (`scripts/pretrain.py` or `scripts/pretrain_stocks.py`) — no new fetch pipeline.
- **Loss:** sum of cross-entropies over the four channel heads per next-candle. Teacher-forcing during training (next-step targets are ground truth, not own predictions).
- **Batching:** standard transformer LM batching with causal mask.
- **Regularisation:** dropout 0.1, label smoothing 0.05 — the bin quantisation is already a form of regularisation.
- **Compute:** GPU strongly recommended for training (Colab T4 or EC2 spot). On CPU: 50–200× slower; impractical for full retraining cycles.
- **Recalibration cadence:** the **tokeniser bins themselves must be refit on a rolling window** (weekly is a reasonable cadence) — non-stationary markets shift quantile boundaries, and a stale tokeniser silently degrades the model. This is the silent footgun of this approach; document the recalibration step in the deploy runbook.

## Integration with the rest of the system

- Separate checkpoint from the policy net: `models/candle_transformer_latest.pt`.
- Loaded by `_background_predictions_loop` alongside `ImprovedTradingLSTM`. The policy net still drives **trading decisions**; the candle transformer only drives the **visualisation**. They are *not* coupled.
- `FEATURE_SCHEMA_VERSION` does not change (this is a separate model on top of raw OHLC, not a feature-vector change).
- Tokeniser bins are persisted in a tiny table `CandleTokeniser(symbol, channel, version, bin_edges_json, fit_date)` — versioned, append-only (so an old model can still decode old bins).

## Open questions if we build this

1. **Sampling vs argmax at inference.** Argmax gives one deterministic path; sampling K=20 paths gives a price distribution at each future step (better for the UI bands). Sampling adds ~20× to inference cost — likely worth it given we already accepted MC-dropout's K factor on the hybrid path.
2. **Per-symbol vs unified tokeniser.** Per-symbol bins respect each instrument's return distribution; one shared tokeniser is simpler but biased to whichever instrument has more data. Per-symbol is recommended; cost is small (a few KB of bin edges per symbol).
3. **Horizon-aware training.** If we sometimes care about next-1 candle and sometimes about next-5, a horizon-conditioned variant ("predict candle k") may outperform pure autoregression. Defer until needed.
4. **Joint training with the policy net.** Sharing the LSTM trunk between the policy net and the transformer's context encoder could amortise compute and improve both. Significantly more complex; only worth investigating if the standalone transformer ships and is the visualisation source of truth for several months.

## Estimated cost when built

- **Code:** ~5–7 days for a careful first version (tokeniser fit + persistence, model, training loop, inference with KV-cache reuse, sampling, chart wiring, tests).
- **Compute:** initial training ~1–4 GPU-hours per universe; weekly tokeniser refit + light retrain ~30 min GPU per week.
- **Maintenance:** moderate — separate checkpoint, separate version, the tokeniser non-stationarity must be monitored.
- **Memory:** ~3–4 MB model + ~10 KB tokeniser per symbol.

## TL;DR

Solid idea. Real but small marginal quality gain over the hybrid (~3–8 % absolute MAE) on this data regime. Build it **only** when the hybrid path's measured forecast quality on a held-out window meets the gate criteria above. Until then, this document is enough to pick the work up cleanly without re-deriving the design.
