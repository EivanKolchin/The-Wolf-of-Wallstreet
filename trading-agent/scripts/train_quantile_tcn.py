#!/usr/bin/env python
"""QuantileTCN overlay trainer + book-level A/B — ML as a SIZING/GATING layer, not a direction picker.

The project's own evidence (0.53 AUC direction, meta-label OOS 0.50) says standalone
direction prediction doesn't clear costs. This trains the Phase-2 QuantileTCN
(58-feature hybrid contract, pinball loss, vol-normalized forward returns at 1h/4h)
and judges it the ONLY way that matters: does scaling the validated 4h TS-momentum
sleeve's exposure by the model's edge/uncertainty LIFT the sleeve's out-of-sample
Sharpe vs. no overlay? The overlay can only shrink or keep exposure the primary
strategy already wants — it never originates trades — so an uninformative model
degrades gracefully to ~the unmodified sleeve.

Multi-seed: train k seeds, average the predicted quantiles at inference (the cheapest
reliable variance reduction). Selection inside training uses a cost-aware edge score
on the val slice, NOT the pinball loss.

Usage (offline, needs the raw 5m/1h/4h parquet caches):
    python scripts/train_quantile_tcn.py --symbols BTCUSDT ETHUSDT SOLUSDT --epochs 8
    python scripts/train_quantile_tcn.py --seeds 0,1,2 --stride 6
    python scripts/train_quantile_tcn.py --ab-only          # reuse saved checkpoints
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.features.pipeline import build_hybrid_matrix, HYBRID_FEATURE_NAMES  # noqa: E402
from backend.models.sequence import QuantileTCN, DMNPositionNet                   # noqa: E402
from backend.models.losses import pinball_loss                                    # noqa: E402
from backend.models.targets import vol_normalized_forward_return                  # noqa: E402
from backend.strategies.ts_momentum import TSMomentumBreakout, TSMomentumParams   # noqa: E402
from backend.agents.improved_model import SYMBOL_TO_ID                            # noqa: E402

SEQ_LEN = 60
HORIZONS = [12, 48]              # 1h / 4h ahead on 5m bars — the horizons with measurable edge
QUANTILES = (0.1, 0.5, 0.9)
VAL_FRAC, TEST_FRAC = 0.15, 0.20  # same honest per-symbol chronological split as pretrain
FEATURE_DIM = len(HYBRID_FEATURE_NAMES)   # 58
CACHE_DIR = ROOT / "training_data" / "features"
MODELS_DIR = ROOT / "models"
CKPT_STEM = "quantile_tcn_overlay"


# ───────────────────────────── dataset assembly ─────────────────────────────
def build_symbol_dataset(sym: str, start_year: int, start_month: int) -> dict:
    """{X (N,58) float32, targets (N,H), close/high/low (N,), timestamps} for one symbol,
    hybrid-contract features + vol-normalized forward-return targets. Disk-cached."""
    import pretrain as pre                       # data loaders (cached parquet)
    cache = CACHE_DIR / f"hybrid_{sym}_{start_year}{start_month:02d}_h{'-'.join(map(str, HORIZONS))}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if int(z["feature_dim"]) == FEATURE_DIM:
            return {k: z[k] for k in ("X", "y", "close", "high", "low", "timestamps")}
    dfs = pre.load_full_history(sym, start_year, start_month, skip_download=True)
    df5 = dfs["5m"]
    X = build_hybrid_matrix(df5, dfs["1h"], dfs["4h"])                    # (N, 58)
    close = df5["close"].to_numpy(np.float64)
    y = np.stack([vol_normalized_forward_return(close, h) for h in HORIZONS], axis=1)
    out = dict(X=X.astype(np.float32), y=y.astype(np.float32),
               close=close,
               high=(df5["high"] if "high" in df5 else df5["close"]).to_numpy(np.float64),
               low=(df5["low"] if "low" in df5 else df5["close"]).to_numpy(np.float64),
               timestamps=df5["timestamp"].to_numpy())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, feature_dim=np.int64(FEATURE_DIM), **out)
    return out


def window_ends(n: int, y: np.ndarray, stride: int) -> np.ndarray:
    """Valid window END indices (window = X[e-SEQ_LEN:e], label row e-1), striding to
    control dataset size; drops rows whose target is NaN (tail mask)."""
    ends = np.arange(SEQ_LEN, n, max(1, stride))
    ok = np.all(np.isfinite(y[ends - 1]), axis=1)
    return ends[ok]


def split_ends(ends: np.ndarray, embargo: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological (train, val, test) split of window-end indices with a train-side
    embargo so a train label window can't overlap the val period."""
    n = len(ends)
    te = max(2, int(n * (1.0 - TEST_FRAC)))
    cut = max(1, min(te - 1, int(n * (1.0 - TEST_FRAC - VAL_FRAC))))
    tr_hi = max(1, cut - max(0, embargo))
    return ends[:tr_hi], ends[cut:te], ends[te:]


# ───────────────────────────── training core ─────────────────────────────
def _batches(F: np.ndarray, y: np.ndarray, sid: int, ends: np.ndarray,
             batch: int, shuffle: bool, rng: np.random.Generator):
    order = rng.permutation(len(ends)) if shuffle else np.arange(len(ends))
    for i in range(0, len(order), batch):
        sel = ends[order[i:i + batch]]
        xb = np.stack([F[e - SEQ_LEN:e] for e in sel]).astype(np.float32)
        yb = y[sel - 1]
        yield (torch.from_numpy(xb), torch.from_numpy(yb),
               torch.full((len(sel),), sid, dtype=torch.long))


def edge_score(edge: np.ndarray, unc: np.ndarray, fwd: np.ndarray,
               ir_min: float = 0.25, cost_norm: float = 0.05) -> float:
    """Cost-aware selection score: trade sign(edge) only when |edge|/unc clears ``ir_min``;
    per-sample pnl = sign·fwd − cost (all in vol-normalized units, so ``cost_norm`` is the
    round-trip cost as a fraction of a 1σ move). Sharpe-like ratio of the pnl stream."""
    unc = np.maximum(unc, 1e-9)
    act = (np.abs(edge) / unc) >= ir_min
    sig = np.where(act, np.sign(edge), 0.0)
    pnl = sig * fwd - cost_norm * np.abs(sig)
    sd = pnl.std()
    return float(pnl.mean() / sd * np.sqrt(252.0)) if sd > 1e-12 else 0.0


def train_one_seed(datasets: Dict[str, dict], seed: int, *, epochs: int, stride: int,
                   hidden: int, lr: float, batch: int, device: torch.device,
                   log=print) -> Tuple[QuantileTCN, float]:
    """Train one QuantileTCN seed on all symbols; keep the best val edge-score epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = QuantileTCN(input_size=FEATURE_DIM, hidden=hidden,
                        num_symbols=len(SYMBOL_TO_ID), num_horizons=len(HORIZONS),
                        quantiles=QUANTILES).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    embargo = max(HORIZONS) // max(1, stride) + 1

    splits = {}
    for s, d in datasets.items():
        ends = window_ends(len(d["X"]), d["y"], stride)
        splits[s] = split_ends(ends, embargo)

    best_state, best_score = None, -np.inf
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for s, d in datasets.items():
            sid = SYMBOL_TO_ID.get(s, 0)
            for xb, yb, sb in _batches(d["X"], d["y"], sid, splits[s][0], batch, True, rng):
                xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
                opt.zero_grad(set_to_none=True)
                q_list = model(xb, sb)
                loss = sum(pinball_loss(q_list[h], yb[:, h], QUANTILES)
                           for h in range(len(HORIZONS))) / len(HORIZONS)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss.item()); nb += 1

        # val: cost-aware edge score at the primary (1h) horizon, pooled over symbols
        edges, uncs, fwds = [], [], []
        model.eval()
        with torch.no_grad():
            for s, d in datasets.items():
                sid = SYMBOL_TO_ID.get(s, 0)
                for xb, yb, sb in _batches(d["X"], d["y"], sid, splits[s][1], batch, False, rng):
                    q = model(xb.to(device), sb.to(device))[0].cpu().numpy()   # (B, Q) @ H+12
                    edges.append(q[:, 1]); uncs.append(q[:, 2] - q[:, 0])
                    fwds.append(yb[:, 0].numpy())
        sc = edge_score(np.concatenate(edges), np.concatenate(uncs), np.concatenate(fwds))
        log(f"  seed {seed} epoch {ep:02d}/{epochs}  train_pinball={tot / max(nb, 1):.4f}  "
            f"val_edge_score={sc:+.3f}  ({time.time() - t0:.0f}s)")
        if sc > best_score:
            best_score = sc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


def _contiguous_blocks(ends: np.ndarray, block: int, rng: np.random.Generator):
    """Yield contiguous chunks of window-ends in random ORDER (the turnover term needs
    consecutive positions inside a block; Sharpe itself is order-free)."""
    starts = np.arange(0, len(ends), block)
    for s in rng.permutation(starts):
        sel = ends[s:s + block]
        if len(sel) >= 8:
            yield sel


def _sharpe_loss(p: torch.Tensor, vnr: torch.Tensor, cost_norm: float) -> torch.Tensor:
    """-net Sharpe of one contiguous block: pnl_t = p_t·vnr_t − c·|Δp| (vol-normalized
    units, so ``cost_norm`` is the round-trip cost as a fraction of a 1σ move)."""
    dp = torch.abs(p[1:] - p[:-1])
    turn = torch.cat([torch.zeros(1, device=p.device, dtype=p.dtype), dp])
    pnl = p * vnr - cost_norm * turn
    return -(pnl.mean() / (pnl.std() + 1e-6))


def train_one_seed_sharpe(datasets: Dict[str, dict], seed: int, *, epochs: int, stride: int,
                          hidden: int, lr: float, block: int, device: torch.device,
                          cost_norm: float = 0.05, log=print) -> Tuple[DMNPositionNet, float]:
    """DMN objective: train the position head end-to-end on the NET Sharpe of holding
    each emitted position for one decision interval (``stride`` bars). The target return
    is the stride-matched vol-normalized forward return, so sizing is vol-scaled by
    construction (the MOP/DMN convention)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = DMNPositionNet(input_size=FEATURE_DIM, hidden=hidden,
                           num_symbols=len(SYMBOL_TO_ID), num_horizons=len(HORIZONS),
                           quantiles=QUANTILES).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    prep = {}
    for s, d in datasets.items():
        vnr = vol_normalized_forward_return(d["close"], h=stride)        # hold-to-next-decision
        ends = np.arange(SEQ_LEN, len(d["X"]), max(1, stride))
        ends = ends[np.isfinite(vnr[ends - 1])]
        tr, va, te = split_ends(ends, embargo=2)
        prep[s] = (vnr, tr, va)

    best_state, best_score = None, -np.inf
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for s, d in datasets.items():
            vnr, tr, _ = prep[s]
            sid = SYMBOL_TO_ID.get(s, 0)
            for sel in _contiguous_blocks(tr, block, rng):
                xb = torch.from_numpy(np.stack([d["X"][e - SEQ_LEN:e] for e in sel]).astype(np.float32)).to(device)
                sb = torch.full((len(sel),), sid, dtype=torch.long, device=device)
                yb = torch.from_numpy(vnr[sel - 1].astype(np.float32)).to(device)
                opt.zero_grad(set_to_none=True)
                loss = _sharpe_loss(model.forward_position(xb, sb), yb, cost_norm)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss.item()); nb += 1

        # val: net Sharpe of the emitted positions (the same objective, held out)
        pnls = []
        model.eval()
        with torch.no_grad():
            for s, d in datasets.items():
                vnr, _, va = prep[s]
                sid = SYMBOL_TO_ID.get(s, 0)
                for i in range(0, len(va), 512):
                    sel = va[i:i + 512]
                    xb = torch.from_numpy(np.stack([d["X"][e - SEQ_LEN:e] for e in sel]).astype(np.float32)).to(device)
                    sb = torch.full((len(sel),), sid, dtype=torch.long, device=device)
                    p = model.forward_position(xb, sb).cpu().numpy()
                    turn = np.abs(np.diff(p, prepend=p[:1]))
                    pnls.append(p * vnr[sel - 1] - cost_norm * turn)
        pnl = np.concatenate(pnls) if pnls else np.zeros(1)
        sc = float(pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(252.0))
        log(f"  seed {seed} epoch {ep:02d}/{epochs}  train_negSharpe={tot / max(nb, 1):+.3f}  "
            f"val_net_sharpe={sc:+.3f}  ({time.time() - t0:.0f}s)")
        if sc > best_score:
            best_score = sc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


def save_seed(model: QuantileTCN, seed: int, score: float, symbols: List[str],
              objective: str = "pinball") -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = MODELS_DIR / f"{CKPT_STEM}_{objective}_seed{seed}.pt"
    torch.save({"model_state_dict": model.state_dict(), "seed": seed,
                "objective": objective, "val_edge_score": score,
                "feature_dim": FEATURE_DIM, "feature_contract": "hybrid58",
                "horizons": HORIZONS, "quantiles": QUANTILES, "seq_len": SEQ_LEN,
                "symbols": symbols, "hidden": model.tcn.proj.out_channels}, p)
    return p


def load_seeds(device, objective: str = "pinball") -> List[QuantileTCN]:
    models = []
    for p in sorted(MODELS_DIR.glob(f"{CKPT_STEM}_{objective}_seed*.pt")):
        ck = torch.load(p, map_location=device, weights_only=False)
        cls = DMNPositionNet if ck.get("objective") == "sharpe" else QuantileTCN
        m = cls(input_size=ck["feature_dim"], hidden=ck.get("hidden", 64),
                num_symbols=len(SYMBOL_TO_ID), num_horizons=len(ck["horizons"]),
                quantiles=tuple(ck["quantiles"])).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        models.append(m)
    return models


# ───────────────────────────── overlay A/B ─────────────────────────────
def ensemble_edge_unc(models: List[QuantileTCN], F: np.ndarray, ends: np.ndarray,
                      sid: int, device, horizon_idx: int = 1,
                      batch: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Seed-averaged (edge, uncertainty) at each window end. horizon_idx=1 → H+48 (4h),
    matching the 4h sleeve's decision cadence. DMN (Sharpe-objective) models emit a
    position directly → edge = position, uncertainty = 1 (the tanh magnitude already
    encodes conviction)."""
    E = np.zeros(len(ends)); U = np.zeros(len(ends))
    with torch.no_grad():
        for i in range(0, len(ends), batch):
            sel = ends[i:i + batch]
            xb = torch.from_numpy(np.stack([F[e - SEQ_LEN:e] for e in sel]).astype(np.float32)).to(device)
            sb = torch.full((len(sel),), sid, dtype=torch.long, device=device)
            es, us = [], []
            for m in models:
                if isinstance(m, DMNPositionNet):
                    es.append(m.forward_position(xb, sb).cpu().numpy())
                    us.append(np.ones(len(sel)))
                else:
                    q = m(xb, sb)[horizon_idx].cpu().numpy()                 # (B, Q)
                    es.append(q[:, 1]); us.append(q[:, 2] - q[:, 0])
            E[i:i + batch] = np.mean(es, axis=0)
            U[i:i + batch] = np.mean(us, axis=0)
    return E, U


def gated_positions(pos: np.ndarray, edge: np.ndarray, unc: np.ndarray,
                    ir_cap: float = 1.0, floor: float = 0.25) -> np.ndarray:
    """Scale the PRIMARY strategy's position by model agreement — never originate.
    Delegates to backend.models.overlay_gate.gate_weight (the SINGLE source of the gate
    math, shared with the live agent so backtest == live by construction)."""
    from backend.models.overlay_gate import gate_weight
    pos = np.asarray(pos, dtype=np.float64)
    return pos * np.asarray(gate_weight(pos, edge, unc, ir_cap=ir_cap, floor=floor))


def _net_returns(pos: np.ndarray, close: np.ndarray, cost: float) -> np.ndarray:
    ret = np.zeros(len(close)); ret[1:] = close[1:] / close[:-1] - 1.0
    m = min(len(pos), len(ret)); pos, ret = pos[:m], ret[:m]
    gross = np.zeros(m); gross[1:] = pos[:-1] * ret[1:]
    turn = np.zeros(m); turn[0] = abs(pos[0]); turn[1:] = np.abs(np.diff(pos))
    return gross - turn * cost


def _stats(r: np.ndarray, ppy: float) -> dict:
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    sd = r.std(); eq = np.cumprod(1 + r); peak = np.maximum.accumulate(eq)
    return dict(sharpe=(r.mean() / sd * np.sqrt(ppy)) if sd > 1e-12 else 0.0,
                cagr=(eq[-1] ** (ppy / max(len(r), 1)) - 1.0) if eq.size and eq[-1] > 0 else 0.0,
                maxdd=float(((eq - peak) / peak).min()) if eq.size else 0.0)


def run_ab(datasets: Dict[str, dict], models: List[QuantileTCN], device,
           cost_bps: float = 15.0, log=print) -> dict:
    """A/B on the untouched TEST tail: 4h TS-momentum sleeve with vs without the overlay."""
    params = TSMomentumParams(entry_channel=48, exit_channel=24, atr_mult=3.0,
                              ema_trend=50, adx_min=25, allow_short=True)
    strat = TSMomentumBreakout(params)
    cost = cost_bps / 1e4
    base_book, over_book, ctrl_book = [], [], []
    sat_medians = []
    for s, d in datasets.items():
        # resample the cached 5m arrays to 4h bars
        df5 = pd.DataFrame({"timestamp": pd.to_datetime(d["timestamps"]),
                            "open": d["close"], "high": d["high"],
                            "low": d["low"], "close": d["close"],
                            "volume": np.ones_like(d["close"])})
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        d4 = (df5.set_index("timestamp").resample("4h").agg(agg)
              .dropna(subset=["close"]).reset_index())
        pos4 = strat.generate_positions({s: d4})[s]

        # model edge at each 4h bar close: map 4h timestamps → 5m row indices
        ts5 = pd.to_datetime(d["timestamps"])
        # bar close time of 4h bar = open time + 4h; use searchsorted on the 5m opens
        bar_close = pd.to_datetime(d4["timestamp"]) + pd.Timedelta(hours=4)
        idx5 = np.searchsorted(ts5.values, bar_close.values, side="right") - 1
        ok = idx5 >= SEQ_LEN
        ends = (idx5[ok] + 1).astype(np.int64)               # window END = row index + 1
        edge = np.zeros(len(d4)); unc = np.full(len(d4), 1e9)
        e, u = ensemble_edge_unc(models, d["X"], ends, SYMBOL_TO_ID.get(s, 0), device)
        edge[ok] = e; unc[ok] = u

        # TEST tail only (same fraction the trainer never touched)
        n4 = len(d4)
        te = int(n4 * (1.0 - TEST_FRAC))
        close4 = d4["close"].to_numpy(np.float64)
        pos_base = pos4.copy(); pos_over = gated_positions(pos4, edge, unc)
        active = pos_base[te:] != 0
        w_mean = float(np.mean(np.abs(pos_over[te:][active]) / np.abs(pos_base[te:][active]))) \
            if active.any() else 1.0
        pos_ctrl = pos_base * w_mean               # constant de-lever at the SAME mean weight
        base_book.append(_net_returns(pos_base, close4, cost)[te:])
        over_book.append(_net_returns(pos_over, close4, cost)[te:])
        ctrl_book.append(_net_returns(pos_ctrl, close4, cost)[te:])
        sat_medians.append(float(np.median(np.abs(edge[te:]))))
        log(f"  {s}: test bars={n4 - te}  |edge| median={sat_medians[-1]:.3f}  "
            f"overlay mean weight={w_mean:.2f}")

    L = min(len(x) for x in base_book)
    base = np.vstack([x[-L:] for x in base_book]).mean(axis=0)
    over = np.vstack([x[-L:] for x in over_book]).mean(axis=0)
    ctrl = np.vstack([x[-L:] for x in ctrl_book]).mean(axis=0)
    ppy = 6 * 365
    sb, so, sc = _stats(base, ppy), _stats(over, ppy), _stats(ctrl, ppy)
    log("\n=== OOS A/B — 4h TS-momentum sleeve, TEST tail only ===")
    log(f"  baseline (no overlay):   Sharpe {sb['sharpe']:+.2f}  CAGR {sb['cagr']*100:+.1f}%  "
        f"maxDD {sb['maxdd']*100:+.1f}%")
    log(f"  CONTROL (const x mean w): Sharpe {sc['sharpe']:+.2f}  CAGR {sc['cagr']*100:+.1f}%  "
        f"maxDD {sc['maxdd']*100:+.1f}%   <- de-levering alone (constant scaling ~preserves Sharpe)")
    log(f"  with model gate:          Sharpe {so['sharpe']:+.2f}  CAGR {so['cagr']*100:+.1f}%  "
        f"maxDD {so['maxdd']*100:+.1f}%")
    timing = so["sharpe"] - sc["sharpe"]
    log(f"  TIMING SKILL (gate − control Sharpe): {timing:+.2f}   "
        f"<- the number that must clear +0.05; DD/CAGR gains shared with the control are "
        f"just size reduction, not intelligence")
    if np.median(sat_medians) > 0.95:
        log("  NOTE: |edge| median ≈ 1 — the tanh position head is SATURATED (acting as a "
            "binary agree/disagree filter). Consider conviction shrinkage / lower lr.")
    verdict = "LIFT — keep researching" if timing > 0.05 else \
              "NO LIFT — park the overlay (matches the project's meta-labeling result)"
    log(f"  VERDICT: {verdict}")
    return {"base": sb, "overlay": so, "control": sc, "timing_skill": timing}


# ───────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", nargs="+",
                    default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"])
    ap.add_argument("--start-year", type=int, default=2022)
    ap.add_argument("--start-month", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--stride", type=int, default=6, help="train-window stride in 5m bars (6 = every 30min)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds for the ensemble")
    ap.add_argument("--ab-only", action="store_true", help="skip training; A/B saved checkpoints")
    ap.add_argument("--objective", choices=["pinball", "sharpe"], default="pinball",
                    help="pinball = calibrated quantile forecast (edge + uncertainty band); "
                         "sharpe = DMN-style position head trained on NET Sharpe directly "
                         "(Lim/Zohren/Roberts 2019) with a turnover cost term")
    ap.add_argument("--block", type=int, default=64,
                    help="contiguous block length for the sharpe objective's turnover term")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  features={FEATURE_DIM}  horizons={HORIZONS}  quantiles={QUANTILES}")

    datasets = {}
    for s in args.symbols:
        try:
            print(f"building dataset {s} ...")
            datasets[s] = build_symbol_dataset(s, args.start_year, args.start_month)
        except Exception as e:
            print(f"  skip {s}: {str(e)[:100]}")
    if not datasets:
        raise SystemExit("no datasets — populate the raw parquet caches first")

    if not args.ab_only:
        for seed in [int(x) for x in args.seeds.split(",")]:
            print(f"training seed {seed} ({args.objective}) ...")
            if args.objective == "sharpe":
                # DMN decisions live on the sleeve's 4h cadence → stride = 48 bars
                model, score = train_one_seed_sharpe(datasets, seed, epochs=args.epochs,
                                                     stride=48, hidden=args.hidden,
                                                     lr=args.lr, block=args.block,
                                                     device=device)
            else:
                model, score = train_one_seed(datasets, seed, epochs=args.epochs,
                                              stride=args.stride, hidden=args.hidden,
                                              lr=args.lr, batch=args.batch, device=device)
            p = save_seed(model, seed, score, list(datasets), objective=args.objective)
            print(f"  saved {p}  (best val score {score:+.3f})")

    models = load_seeds(device, objective=args.objective)
    if not models:
        raise SystemExit("no saved checkpoints to A/B")
    print(f"\nA/B with {len(models)}-seed ensemble ...")
    run_ab(datasets, models, device)


if __name__ == "__main__":
    main()
