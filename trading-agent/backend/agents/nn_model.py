import os
import math
import random
import itertools
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Match the env-var caps set in backend/main.py so the torch thread pool size
# is also explicitly clamped (env vars are picked up by torch only on the first
# tensor operation, which has already happened by the time we import this).
try:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1") or "1"))
    torch.set_num_interop_threads(1)
except RuntimeError:
    # set_num_interop_threads must be called before any parallel work — ignore
    # the "already set" error harmlessly if another module beat us to it.
    pass

import structlog
from backend.core.config import settings

from backend.agents.improved_model import (
    ImprovedTradingLSTM, SYMBOL_TO_ID, HORIZONS,
    INPUT_SIZE, HIDDEN_SIZE, NUM_LSTM_LAYERS,
    SL_FRAC_RANGE, TP_FRAC_RANGE, TRAIL_FRAC_RANGE, _scale,
)
from backend.agents.model_io import load_checkpoint, save_checkpoint, CheckpointSchemaMismatch
from backend.signals import feature_spec as fs

logger = structlog.get_logger(__name__)


@dataclass
class TradeExperience:
    features_sequence: np.ndarray  # (sequence_length, INPUT_SIZE=70)
    direction_taken: int           # 0=long, 1=short, 2=hold
    actual_pnl_pct: float          # ALREADY net of fees (from close_position)
    symbol_id: int = 0
    size_taken: float = 0.1        # position size fraction actually used
    sl_taken: float = 0.0          # stop-loss fraction emitted at entry
    tp_taken: float = 0.0          # take-profit fraction emitted at entry
    bars_held: float = 0.0         # 5m bars the position was held
    is_hold: bool = False

    def shaped_reward(self, downside_dev: float = 0.0) -> float:
        """Fee-net, inaction/holding-penalized, risk-adjusted reward for RL.

        actual_pnl_pct already nets fees (close_position subtracts them), so the
        base term is fee-aware. A holding penalty discourages sitting in a
        position far beyond the primary prediction horizon.

        Sortino-style risk adjustment: the PnL term is divided by
        ``1 + NN_DOWNSIDE_WEIGHT * downside_dev`` where ``downside_dev`` is the
        recent downside deviation of realized returns (computed by the caller
        from a rolling window). With ``downside_dev == 0`` this reduces exactly
        to the original ``tanh(pnl*k) - hold_pen`` reward, so legacy behaviour
        (and the existing reward-shaping tests) is preserved. In volatile,
        drawdown-prone regimes the same PnL yields a smaller reward, nudging the
        policy toward smoother, more consistent equity curves.
        """
        k = float(getattr(settings, "NN_REWARD_K", 10.0))
        base = math.tanh(self.actual_pnl_pct * k)
        w = float(getattr(settings, "NN_DOWNSIDE_WEIGHT", 25.0))
        dd = max(0.0, float(downside_dev))
        base = base / (1.0 + w * dd)
        target_h = float(HORIZONS[0]) if HORIZONS else 3.0
        excess = max(0.0, self.bars_held - target_h)
        hold_pen = float(getattr(settings, "NN_HOLD_PENALTY", 0.05)) * (excess / max(target_h, 1.0))
        return float(base - hold_pen)

    @property
    def reward(self) -> float:
        """Backwards-compatible un-risk-adjusted reward (downside_dev = 0).

        Kept so existing callers/tests that read ``ex.reward`` as a property
        still work; the risk-adjusted path uses ``shaped_reward(downside_dev)``."""
        return self.shaped_reward(0.0)


@dataclass
class InferenceResult:
    direction: str          # "long" / "short" / "hold"
    size: float             # fraction of capital [0.02, 0.20]
    probs: dict             # primary-horizon {long, short, hold}
    sl: float               # stop-loss as a fraction of entry price
    tp: float               # take-profit as a fraction of entry price
    trail: float            # trailing distance as a fraction of price
    edge_mean: float        # E[p_long - p_short]
    edge_std: float         # MC-dropout std of the edge (0 if mc_samples<=1)
    horizon_probs: list     # list of {long, short, hold} per horizon


class ValueBaseline(nn.Module):
    """State-value baseline V(s) for Advantage-Weighted Regression.

    Operates on the policy's 64-dim shared trunk embedding. Trained to predict
    the shaped reward; its output is used (detached) to form advantages."""

    # CONSIDER MODIFYING NEURAL NETWORK TO MAKE AVOID OVERFITS YET PREDICT MORE ACCURATELY AT COST OF LONGER AND MORE COMPLEX TRAININGS
    # E.G.: DYNAMIC STRUCTURE AND DIFFERENT LEARNING ALGORITHMS

    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, experience: TradeExperience) -> None:
        self.buffer.append(experience)

    def sample(self, n: int) -> list:
        return random.sample(self.buffer, min(n, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class PersistentTradingModel:
    MODEL_PATH = Path("models/trading_lstm_latest.pt")
    CHECKPOINT_DIR = Path("models/checkpoints/")
    SEQUENCE_LENGTH = 60

    def __init__(self):
        self.idle_pressure = 0.0
        # Trade the SELECTED tradeable horizon (default H+12 = 1h), NOT the H+3
        # head. H+3 (15m) was validated as pure noise / negative expectancy, yet
        # every inference path used to default to horizon_idx=0. Single source of
        # truth so infer / infer_with_distribution / infer_batch all agree.
        try:
            _sel = int(getattr(settings, "NN_SELECT_HORIZON", 1))
        except Exception:
            _sel = 1
        self.primary_horizon_idx = max(0, min(_sel, len(HORIZONS) - 1))
        self._build_fresh()

        self.trade_count = 0
        self.cumulative_pnl = 0.0
        # Sortino-style risk adjustment: rolling window of realized per-trade
        # returns used to estimate downside deviation for reward shaping.
        _sortino_win = int(getattr(settings, "NN_SORTINO_WINDOW", 50))
        self.recent_returns: deque = deque(maxlen=max(2, _sortino_win))

        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        self._load_or_initialise()

    # ------------------------------------------------------------------ setup
    def _build_fresh(self):
        self.model = ImprovedTradingLSTM()
        self.value_baseline = ValueBaseline(in_dim=64)
        self._apply_ipex_if_available()
        params = list(self.model.parameters()) + list(self.value_baseline.parameters())
        _wd = float(getattr(settings, "NN_WEIGHT_DECAY", 1e-4))   # A4: tunable L2
        self.optimizer = optim.Adam(params, lr=1e-4, weight_decay=_wd)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500)
        self.replay_buffer = ReplayBuffer(max_size=10_000)

    def _apply_ipex_if_available(self) -> None:
        """Phase 7b: apply Intel Extension for PyTorch (CPU MKL-DNN speedup) when
        installed. Safe no-op otherwise — keeps the path compatible with a stock
        PyTorch install."""
        if not bool(getattr(settings, "NN_USE_IPEX", True)):
            return
        try:
            import intel_extension_for_pytorch as ipex  # type: ignore
        except ImportError:
            return
        try:
            self.model = ipex.optimize(self.model)
            self.value_baseline = ipex.optimize(self.value_baseline)
            logger.info("ipex_optimize_applied")
        except Exception as e:
            logger.warning("ipex_optimize_failed", error=str(e))

    def _cold_start(self):
        unsafe = os.environ.get("FORCE_UNSAFE_START", "").lower() in ("true", "1", "yes")
        if not unsafe:
            raise RuntimeError(
                "NO TRAINED CHECKPOINT FOUND. The model cannot trade with random weights. "
                "Run `python scripts/pretrain.py` first, or set FORCE_UNSAFE_START=true "
                "in your environment (development-only; never on a live account)."
            )
        logger.warning(
            "model_cold_start_untrained",
            note="No compatible v2 checkpoint. FORCE_UNSAFE_START=true — trading with "
                 "random weights. Run scripts/pretrain.py to produce trained weights.",
        )
        self._build_fresh()
        self.trade_count = 0
        self.cumulative_pnl = 0.0
        self.safe_checkpoint(label="initial")

    def _load_or_initialise(self):
        if not self.MODEL_PATH.exists():
            logger.info("first_run_initialising_model")
            self._cold_start()
            return

        try:
            ckpt = load_checkpoint(self.MODEL_PATH)
        except CheckpointSchemaMismatch as e:
            logger.warning("checkpoint_schema_mismatch_cold_start", error=str(e), path=str(self.MODEL_PATH))
            self._cold_start()
            return
        except Exception as e:
            logger.warning("checkpoint_load_failed_cold_start", error=str(e), path=str(self.MODEL_PATH))
            self._cold_start()
            return

        state = ckpt.get("model_state_dict", {})
        # Phase 7b: stocks were appended to the symbol registry (8 → 13 rows),
        # so an older checkpoint with `symbol_embedding.weight` shape (8,16)
        # can't be dropped straight into a (13,16) embedding. We splice it in:
        # copy the first 8 rows from the checkpoint, leave rows 8..12 freshly
        # initialised. Drops the conflicting tensor from the state dict so
        # load_state_dict treats it as a "missing" key (acceptable).
        ckpt_emb = state.get("symbol_embedding.weight")
        live_emb = getattr(self.model, "symbol_embedding", None)
        if ckpt_emb is not None and live_emb is not None:
            ck_rows = int(ckpt_emb.shape[0]); ck_dim = int(ckpt_emb.shape[1])
            live_rows = live_emb.weight.shape[0]; live_dim = live_emb.weight.shape[1]
            if ck_rows != live_rows and ck_dim == live_dim and ck_rows < live_rows:
                with torch.no_grad():
                    live_emb.weight[:ck_rows].copy_(ckpt_emb)
                logger.info("symbol_embedding_spliced",
                             checkpoint_rows=ck_rows, live_rows=live_rows,
                             fresh_rows=list(range(ck_rows, live_rows)))
                state = {k: v for k, v in state.items() if k != "symbol_embedding.weight"}

        try:
            result = self.model.load_state_dict(state, strict=False)
        except RuntimeError as e:
            logger.warning("checkpoint_shape_mismatch_cold_start", error=str(e)[:200])
            self._cold_start()
            return
        non_exit_missing = [k for k in result.missing_keys
                            if not k.startswith(self.model.EXIT_PARAM_PREFIXES)
                            and k != "symbol_embedding.weight"]
        if result.unexpected_keys or non_exit_missing:
            logger.warning(
                "checkpoint_core_mismatch_cold_start",
                missing=non_exit_missing[:8], unexpected=list(result.unexpected_keys)[:8],
            )
            self._cold_start()
            return

        if ckpt.get("value_baseline_state_dict"):
            try:
                self.value_baseline.load_state_dict(ckpt["value_baseline_state_dict"])
            except (ValueError, RuntimeError) as ve:
                logger.warning("value_baseline_state_incompatible_skipped", error=str(ve))
        if ckpt.get("optimizer_state_dict"):
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError) as opt_err:
                logger.warning("optimizer_state_incompatible_skipped", error=str(opt_err))
        if ckpt.get("scheduler_state_dict"):
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except (ValueError, KeyError) as sched_err:
                logger.warning("scheduler_state_incompatible_skipped", error=str(sched_err))

        self.trade_count = ckpt.get("trade_count", 0)
        self.cumulative_pnl = ckpt.get("cumulative_pnl", 0.0)

        # News-embedding backend consistency (Cycle 3): the 16 news dims [70:86] only
        # mean the same thing live as in training if produced by the SAME backend.
        # Non-fatal — we only warn, never block — so loading behaviour is unchanged.
        ckpt_news = ckpt.get("news_backend")
        if ckpt_news and ckpt_news != "disabled":
            try:
                from backend.signals.news_embedding import get_embedder
                live_news = get_embedder().effective_backend()
            except Exception:
                live_news = None
            if live_news and live_news != ckpt_news:
                logger.warning(
                    "news_embed_backend_mismatch",
                    trained_with=ckpt_news, live=live_news,
                    fix=f"pip install sentence-transformers and set NN_NEWS_EMBED_BACKEND={ckpt_news} "
                        f"so live news features match the trained model",
                )
        elif ckpt_news == "disabled":
            logger.info("checkpoint_trained_without_news_alignment")

        if result.missing_keys:
            logger.info("model_loaded_core_exit_heads_fresh", missing=len(result.missing_keys),
                        trade_count=self.trade_count)
            self.safe_checkpoint(label="exitheads_init")
        else:
            logger.info("model_loaded_strict_v2", trade_count=self.trade_count,
                        cumulative_pnl=self.cumulative_pnl, path=str(self.MODEL_PATH))

    def safe_checkpoint(self, label: str = ""):
        save_checkpoint(
            self.MODEL_PATH,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            value_baseline=self.value_baseline,
            trade_count=self.trade_count,
            cumulative_pnl=self.cumulative_pnl,
            extra_meta={
                "seq_len": self.SEQUENCE_LENGTH,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LSTM_LAYERS,
                "horizons": list(HORIZONS),
                "num_symbols": len(SYMBOL_TO_ID),
            },
        )
        ckpt_filename = f"ckpt_{self.trade_count}{'_' + label if label else ''}.pt"
        ckpt_path = self.CHECKPOINT_DIR / ckpt_filename
        import shutil
        shutil.copy(self.MODEL_PATH, ckpt_path)

        checkpoints = sorted(self.CHECKPOINT_DIR.glob("ckpt_*.pt"), key=os.path.getmtime)
        if len(checkpoints) > 10:
            for old_ckpt in checkpoints[:-10]:
                old_ckpt.unlink()
        logger.info("model_checkpoint_saved", path=str(self.MODEL_PATH), trade_count=self.trade_count)

    # ------------------------------------------------------------- anti-hold
    def set_idle_pressure(self, pressure: float) -> None:
        """Set how strongly to discourage holding (0..1). Grows with idle time so
        the agent doesn't get stuck never trading."""
        self.idle_pressure = float(max(0.0, min(1.0, pressure)))

    # -------------------------------------------------------------- inference
    def _batched_mc(self, seq_t: torch.Tensor, sid_t: torch.Tensor, K: int) -> list:
        """A1: one batched forward with dropout active = K independent MC samples.

        Tiling the single (1,T,F) input to (K,T,F) and forwarding once gives each
        batch row its own dropout mask, so it is mathematically identical to K
        separate batch-1 dropout forwards — but ~K× fewer model calls. Returns a
        list (len H) of (K, 3) softmax-probability arrays, one per horizon."""
        K = max(1, int(K))
        self.model.eval()
        if K > 1:
            self.model.enable_dropout()
        seq_k = seq_t.expand(K, -1, -1).contiguous()
        sid_k = sid_t.expand(K).contiguous()
        with torch.inference_mode():
            _, probs_list, _, _, _ = self.model(seq_k, sid_k)
        return [p.detach().cpu().numpy() for p in probs_list]  # each (K, 3)

    def infer(self, feature_sequence: np.ndarray, symbol_id: int = 0,
              mc_samples: int = 1, horizon_idx: int | None = None) -> InferenceResult:
        if horizon_idx is None:
            horizon_idx = self.primary_horizon_idx
        seq = torch.tensor(np.asarray(feature_sequence), dtype=torch.float32).unsqueeze(0)
        sid = torch.tensor([int(symbol_id)], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            _, probs_list, size, exits, _ = self.model(seq, sid)

            primary = probs_list[horizon_idx][0].numpy().copy()
            primary[2] *= float(max(0.0, settings.NN_HOLD_PROB_MULTIPLIER))
            # anti-hold: shrink hold probability the longer we've gone without trading
            if self.idle_pressure > 0:
                primary[2] *= max(0.0, 1.0 - self.idle_pressure * float(getattr(settings, "NN_IDLE_HOLD_DECAY", 0.5)))
            total = primary.sum()
            primary = primary / total if total > 0 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

            probs_dict = {"long": float(primary[0]), "short": float(primary[1]), "hold": float(primary[2])}
            decision = {0: "long", 1: "short", 2: "hold"}[int(np.argmax(primary))]
            if float(np.max(primary)) < float(settings.NN_MIN_ACTION_CONFIDENCE):
                decision = "hold"

            size_pct = float(np.clip(float(size[0].item()), 0.02, 0.20))
            # Phase 17: prefer ATR-multiple heads; convert via the volatility slot
            # in the most recent feature row. fs.VOLATILITY's first index is the
            # `atr_norm` slot (ATR/close clipped to 0..0.1 then *10). Inverse:
            # atr_pct = atr_norm / 10. Clamp to a minimum so tiny vol regimes
            # don't degenerate the stop.
            try:
                atr_norm_last = float(feature_sequence[-1, fs.VOLATILITY.start])
                atr_pct = max(0.001, atr_norm_last / 10.0)   # ≥ 0.1% floor
            except Exception:
                atr_pct = 0.01   # 1% safe fallback
            if "sl_mult" in exits:
                sl = float(exits["sl_mult"][0].item()) * atr_pct
                tp = float(exits["tp_mult"][0].item()) * atr_pct
                trail = float(exits["trail_mult"][0].item()) * atr_pct
            else:
                sl = float(exits["sl"][0].item())
                tp = float(exits["tp"][0].item())
                trail = float(exits["trail"][0].item())
            horizon_probs = [
                {"long": float(p[0][0]), "short": float(p[0][1]), "hold": float(p[0][2])}
                for p in probs_list
            ]
            edge_mean = float(primary[0] - primary[1])
            edge_std = 0.0

        if mc_samples and mc_samples > 1:
            # A1: single batched MC forward instead of a K-iteration Python loop.
            pl_k = self._batched_mc(seq, sid, int(mc_samples))
            ph = pl_k[horizon_idx]                       # (K, 3)
            edges = ph[:, 0] - ph[:, 1]                  # p_long - p_short per sample
            edge_mean = float(np.mean(edges))
            edge_std = float(np.std(edges))

        return InferenceResult(
            direction=decision, size=size_pct, probs=probs_dict,
            sl=sl, tp=tp, trail=trail,
            edge_mean=edge_mean, edge_std=edge_std, horizon_probs=horizon_probs,
        )

    def infer_with_distribution(self, feature_sequence: np.ndarray, symbol_id: int = 0,
                                 mc_samples: int = 16, horizon_idx: int | None = None):
        """A2: ONE batched MC forward yields BOTH the decision's edge stats and
        the full (K,H) predictive distribution, so the trading cycle and the
        visualization loop don't each pay for their own MC pass.

        Returns ``(InferenceResult, {"horizons", "edge_samples"})``. The
        InferenceResult's decision/size/exits match ``infer()``; its
        edge_mean/edge_std are refined from the same MC samples used for the bands."""
        if horizon_idx is None:
            horizon_idx = self.primary_horizon_idx
        res = self.infer(feature_sequence, symbol_id, mc_samples=1, horizon_idx=horizon_idx)
        seq = torch.tensor(np.asarray(feature_sequence), dtype=torch.float32).unsqueeze(0)
        sid = torch.tensor([int(symbol_id)], dtype=torch.long)
        K = max(1, int(mc_samples))
        H = len(HORIZONS)
        pl_k = self._batched_mc(seq, sid, K)             # list[H] of (K, 3)
        edge_samples = np.zeros((K, H), dtype=np.float32)
        for hi in range(H):
            ph = pl_k[hi]
            edge_samples[:, hi] = ph[:, 0] - ph[:, 1]
        e0 = edge_samples[:, horizon_idx]
        res.edge_mean = float(np.mean(e0))
        res.edge_std = float(np.std(e0))
        return res, {"horizons": list(HORIZONS), "edge_samples": edge_samples}

    def infer_batch(self, sequences: list, symbol_ids: list,
                    mc_samples: int = 16, horizon_idx: int | None = None) -> list:
        """A3: cross-symbol batched inference. Runs ONE deterministic forward over
        all N symbols and ONE batched MC forward over (N*K) rows, then does the
        cheap per-symbol decision post-processing. Returns a list of
        ``(InferenceResult, {"horizons","edge_samples"})`` aligned to the inputs.

        Equivalent to calling ``infer_with_distribution`` per symbol, but collapses
        2N model calls into 2 — the win scales with the number of symbols."""
        if horizon_idx is None:
            horizon_idx = self.primary_horizon_idx
        N = len(sequences)
        if N == 0:
            return []
        seq_t = torch.tensor(np.stack(sequences), dtype=torch.float32)        # (N, T, F)
        sid_t = torch.tensor([int(s) for s in symbol_ids], dtype=torch.long)  # (N,)
        H = len(HORIZONS)

        self.model.eval()
        with torch.inference_mode():
            _, probs_list, size, exits, _ = self.model(seq_t, sid_t)

        probs0 = probs_list[horizon_idx].detach().cpu().numpy()              # (N, 3)
        horizon_probs_all = [p.detach().cpu().numpy() for p in probs_list]   # list[H] (N,3)
        size_np = size.detach().cpu().numpy().reshape(-1)                    # (N,)
        has_mult = "sl_mult" in exits
        sl_np = (exits["sl_mult"] if has_mult else exits["sl"]).detach().cpu().numpy().reshape(-1)
        tp_np = (exits["tp_mult"] if has_mult else exits["tp"]).detach().cpu().numpy().reshape(-1)
        tr_np = (exits["trail_mult"] if has_mult else exits["trail"]).detach().cpu().numpy().reshape(-1)

        # Batched MC: repeat each row K times (grouped by symbol) → reshape (N,K).
        K = max(1, int(mc_samples))
        edge_by_h = []
        if K > 1:
            self.model.enable_dropout()
            seq_k = seq_t.repeat_interleave(K, dim=0)                        # (N*K, T, F)
            sid_k = sid_t.repeat_interleave(K, dim=0)
            with torch.inference_mode():
                _, probs_list_k, _, _, _ = self.model(seq_k, sid_k)
            for hi in range(H):
                ph = probs_list_k[hi].detach().cpu().numpy()                 # (N*K, 3)
                edge_by_h.append((ph[:, 0] - ph[:, 1]).reshape(N, K))        # (N, K)
        else:
            for hi in range(H):
                p = horizon_probs_all[hi]
                edge_by_h.append((p[:, 0] - p[:, 1]).reshape(N, 1))

        hold_mult = float(max(0.0, settings.NN_HOLD_PROB_MULTIPLIER))
        idle_decay = float(getattr(settings, "NN_IDLE_HOLD_DECAY", 0.5))
        min_action = float(settings.NN_MIN_ACTION_CONFIDENCE)
        results = []
        for n in range(N):
            primary = probs0[n].copy()
            primary[2] *= hold_mult
            if self.idle_pressure > 0:
                primary[2] *= max(0.0, 1.0 - self.idle_pressure * idle_decay)
            total = primary.sum()
            primary = primary / total if total > 0 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
            probs_dict = {"long": float(primary[0]), "short": float(primary[1]), "hold": float(primary[2])}
            decision = {0: "long", 1: "short", 2: "hold"}[int(np.argmax(primary))]
            if float(np.max(primary)) < min_action:
                decision = "hold"
            size_pct = float(np.clip(float(size_np[n]), 0.02, 0.20))
            try:
                atr_pct = max(0.001, float(sequences[n][-1, fs.VOLATILITY.start]) / 10.0)
            except Exception:
                atr_pct = 0.01
            if has_mult:
                sl, tp, trail = float(sl_np[n]) * atr_pct, float(tp_np[n]) * atr_pct, float(tr_np[n]) * atr_pct
            else:
                sl, tp, trail = float(sl_np[n]), float(tp_np[n]), float(tr_np[n])
            horizon_probs = [
                {"long": float(horizon_probs_all[hi][n][0]),
                 "short": float(horizon_probs_all[hi][n][1]),
                 "hold": float(horizon_probs_all[hi][n][2])}
                for hi in range(H)
            ]
            edge_samples = np.stack([edge_by_h[hi][n] for hi in range(H)], axis=1)  # (K, H)
            e0 = edge_samples[:, horizon_idx]
            res = InferenceResult(
                direction=decision, size=size_pct, probs=probs_dict,
                sl=sl, tp=tp, trail=trail,
                edge_mean=float(np.mean(e0)), edge_std=float(np.std(e0)),
                horizon_probs=horizon_probs,
            )
            results.append((res, {"horizons": list(HORIZONS), "edge_samples": edge_samples}))
        return results

    def infer_predictive_distribution(self, feature_sequence: np.ndarray,
                                       symbol_id: int = 0,
                                       mc_samples: int = 16) -> dict:
        """Phase 18 v1: K MC-dropout passes → per-horizon edge samples.

        Returns ``{"horizons": [h1, h2, h3], "edge_samples": np.ndarray(K, H)}``
        where each ``edge_samples[k, h]`` is the sampled ``p_long - p_short`` at
        horizon ``HORIZONS[h]`` for MC pass ``k``. The visualization loop turns
        this into median + p25/p75 price bands. Separate from ``infer()`` so the
        trading critical path isn't penalised by the extra K forwards.
        """
        seq = torch.tensor(np.asarray(feature_sequence), dtype=torch.float32).unsqueeze(0)
        sid = torch.tensor([int(symbol_id)], dtype=torch.long)
        K = max(1, int(mc_samples))
        H = len(HORIZONS)

        # A1: one batched MC forward (K rows) instead of K sequential forwards.
        pl_k = self._batched_mc(seq, sid, K)             # list[H] of (K, 3)
        edge_samples = np.zeros((K, H), dtype=np.float32)
        for hi in range(H):
            ph = pl_k[hi]                                 # (K, 3)
            edge_samples[:, hi] = ph[:, 0] - ph[:, 1]
        return {"horizons": list(HORIZONS), "edge_samples": edge_samples}

    # --------------------------------------------------------------- learning
    def online_update(self, experience: TradeExperience) -> None:
        self.replay_buffer.add(experience)
        self.trade_count += 1
        self.cumulative_pnl += experience.actual_pnl_pct
        # Track realized returns for the Sortino downside-deviation estimate.
        self.recent_returns.append(float(experience.actual_pnl_pct))

        if self.trade_count % 10 == 0 and len(self.replay_buffer) >= 32:
            self._awr_update(self.replay_buffer.sample(32))
        if self.trade_count % 50 == 0:
            self.safe_checkpoint()

    def _downside_deviation(self, mar: float = 0.0) -> float:
        """Downside deviation of recent realized returns vs a minimum-acceptable
        return (MAR, default 0). Only shortfalls (r < mar) contribute, so upside
        volatility isn't penalized — the Sortino, not Sharpe, philosophy.

        Returns 0.0 until enough samples have accrued, so early-life trading
        (and the unit tests) see the unadjusted reward."""
        rets = self.recent_returns
        if len(rets) < max(2, int(getattr(settings, "NN_SORTINO_WINDOW", 50)) // 5):
            return 0.0
        shortfalls = [min(0.0, float(r) - mar) for r in rets]
        msq = sum(s * s for s in shortfalls) / max(1, len(shortfalls))
        return float(math.sqrt(msq))

    def _awr_update(self, batch: list) -> None:
        """Advantage-Weighted Regression: reinforce the taken action (direction +
        size + SL/TP) in proportion to exp(advantage / beta). True off-policy RL —
        no more 'train toward the opposite action on a loss'."""
        seqs = np.stack([ex.features_sequence for ex in batch])
        seq_t = torch.tensor(seqs, dtype=torch.float32)
        # A4 anti-overfit: optionally jitter the input sequences with Gaussian
        # noise so the online learner doesn't memorise its small replay buffer.
        _noise = float(getattr(settings, "NN_AUGMENT_NOISE_STD", 0.0))
        if _noise > 0:
            seq_t = seq_t + torch.randn_like(seq_t) * _noise
        sid_t = torch.tensor([int(getattr(ex, "symbol_id", 0)) for ex in batch], dtype=torch.long)
        taken_dir = torch.tensor([int(ex.direction_taken) for ex in batch], dtype=torch.long)
        size_taken = torch.tensor([float(getattr(ex, "size_taken", 0.1)) for ex in batch], dtype=torch.float32)
        sl_taken = torch.tensor([float(getattr(ex, "sl_taken", 0.0)) for ex in batch], dtype=torch.float32)
        tp_taken = torch.tensor([float(getattr(ex, "tp_taken", 0.0)) for ex in batch], dtype=torch.float32)
        # Sortino-style: dampen rewards by the recent downside deviation so the
        # policy is steered toward smoother equity curves, not just raw PnL.
        downside_dev = self._downside_deviation()
        rewards = torch.tensor([float(ex.shaped_reward(downside_dev)) for ex in batch], dtype=torch.float32)

        self.model.train()
        self.value_baseline.train()
        self.optimizer.zero_grad()

        shared, _ = self.model._trunk(seq_t, sid_t)                 # (B, 64)
        logits0 = self.model.direction_heads[0](shared) / self.model.temperature
        probs0 = F.softmax(logits0, dim=-1)
        sizes = self.model.size_head(shared).squeeze(-1)            # (B,)
        scaled_sl = _scale(self.model.sl_head(shared), *SL_FRAC_RANGE).squeeze(-1)
        scaled_tp = _scale(self.model.tp_head(shared), *TP_FRAC_RANGE).squeeze(-1)

        values = self.value_baseline(shared.detach()).squeeze(-1)  # (B,)
        beta = max(float(getattr(settings, "NN_AWR_BETA", 1.0)), 1e-3)
        w_cap = float(getattr(settings, "NN_AWR_WEIGHT_CAP", 20.0))
        advantages = (rewards - values).detach()
        weights = torch.clamp(torch.exp(advantages / beta), 0.0, w_cap)

        # A4 anti-overfit: label-smoothed cross-entropy. Mixes the hard target
        # with a uniform prior so the policy stays less overconfident (which
        # generalises better and curbs runaway conviction on noisy edges).
        log_probs0 = torch.log(probs0 + 1e-8)
        eps_ls = float(getattr(settings, "NN_LABEL_SMOOTHING", 0.0))
        nll = -log_probs0.gather(1, taken_dir.unsqueeze(1)).squeeze(1)   # (B,)
        if eps_ls > 0:
            smooth = -log_probs0.mean(dim=1)                            # uniform part
            ce = (1.0 - eps_ls) * nll + eps_ls * smooth
        else:
            ce = nll
        dir_loss = (weights * ce).mean()
        size_loss = (weights * (sizes - size_taken) ** 2).mean()
        sl_loss = (weights * (scaled_sl - sl_taken) ** 2).mean()
        tp_loss = (weights * (scaled_tp - tp_taken) ** 2).mean()
        policy_loss = dir_loss + size_loss + sl_loss + tp_loss

        value_loss = F.mse_loss(values, rewards)
        loss = policy_loss + value_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(self.model.parameters(), self.value_baseline.parameters()), 1.0
        )
        self.optimizer.step()
        self.scheduler.step()
        logger.debug("awr_update", loss=float(loss.item()), value_loss=float(value_loss.item()),
                     mean_weight=float(weights.mean().item()))

    def check_and_rollback(self, recent_pnl_pct: float, threshold: float = -0.05) -> bool:
        if recent_pnl_pct >= threshold:
            return False
        checkpoints = sorted(self.CHECKPOINT_DIR.glob("ckpt_*.pt"), key=os.path.getmtime)
        if len(checkpoints) < 2:
            return False
        target_ckpt = checkpoints[-2]
        logger.warning("pnl_threshold_breached_triggering_rollback", target_checkpoint=str(target_ckpt))
        try:
            checkpoint = torch.load(target_ckpt, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if checkpoint.get("value_baseline_state_dict"):
                self.value_baseline.load_state_dict(checkpoint["value_baseline_state_dict"])
            if checkpoint.get("optimizer_state_dict"):
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.trade_count = checkpoint.get("trade_count", self.trade_count)
            self.cumulative_pnl = checkpoint.get("cumulative_pnl", self.cumulative_pnl)
            self.safe_checkpoint(label="rollback")
            return True
        except Exception as e:
            logger.error("rollback_failed", error=str(e))
            return False
