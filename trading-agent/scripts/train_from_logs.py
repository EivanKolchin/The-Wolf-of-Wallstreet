import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from structlog import get_logger

from backend.agents.nn_model import PersistentTradingModel

log = get_logger("scripts.train_from_logs")


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _direction_to_index(direction: str) -> int:
    d = (direction or "hold").lower()
    return {"long": 0, "short": 1, "hold": 2}.get(d, 2)


def _build_training_set(
    decisions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    min_abs_pnl_pct: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    # Match each decision to the next outcome of same symbol by timestamp.
    outcomes_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        sym = row.get("symbol")
        if not sym:
            continue
        outcomes_by_symbol[sym].append(row)
    for sym in outcomes_by_symbol:
        outcomes_by_symbol[sym].sort(key=lambda r: _parse_ts(r["timestamp"]))

    out_idx: dict[str, int] = defaultdict(int)
    X: list[np.ndarray] = []
    y: list[int] = []

    decisions_sorted = sorted(decisions, key=lambda r: _parse_ts(r.get("timestamp", "1970-01-01T00:00:00+00:00")))
    for row in decisions_sorted:
        if not row.get("approved", False):
            continue

        symbol = row.get("symbol")
        seq = row.get("features")
        if not symbol or seq is None:
            continue

        try:
            seq_np = np.asarray(seq, dtype=np.float32)
        except Exception:
            continue
        if seq_np.ndim != 2 or seq_np.shape[0] <= 0:
            continue

        ts = _parse_ts(row["timestamp"])
        outs = outcomes_by_symbol.get(symbol, [])
        j = out_idx[symbol]
        matched = None
        while j < len(outs):
            cand = outs[j]
            cand_ts = _parse_ts(cand["timestamp"])
            if cand_ts >= ts:
                matched = cand
                j += 1
                break
            j += 1
        out_idx[symbol] = j
        if not matched:
            continue

        pnl_pct = float(matched.get("pnl_pct", 0.0))
        if abs(pnl_pct) < min_abs_pnl_pct:
            continue

        direction = _direction_to_index(row.get("decision", "hold"))
        if pnl_pct > 0:
            target = direction
        elif pnl_pct < 0:
            target = 1 if direction == 0 else 0 if direction == 1 else 2
        else:
            target = 2

        X.append(seq_np)
        y.append(target)

    if not X:
        return np.empty((0, 0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64)


def train_from_logs(
    data_dir: str = "training_data",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 5e-5,
    min_abs_pnl_pct: float = 0.0,
) -> None:
    data_path = Path(data_dir)
    decisions = _load_jsonl(data_path / "decision_log.jsonl")
    outcomes = _load_jsonl(data_path / "outcome_log.jsonl")

    if not decisions:
        log.error("no_decision_logs_found", path=str(data_path / "decision_log.jsonl"))
        return
    if not outcomes:
        log.error("no_outcome_logs_found", path=str(data_path / "outcome_log.jsonl"))
        return

    X, y = _build_training_set(decisions, outcomes, min_abs_pnl_pct=min_abs_pnl_pct)
    if X.size == 0:
        log.error("no_aligned_training_rows", decisions=len(decisions), outcomes=len(outcomes))
        return

    model = PersistentTradingModel()
    model.optimizer = torch.optim.Adam(model.model.parameters(), lr=lr, weight_decay=1e-5)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    log.info("offline_train_start", rows=len(dataset), epochs=epochs, batch_size=batch_size)
    for epoch in range(epochs):
        model.model.train()
        epoch_loss = 0.0
        total = 0
        correct = 0
        for bx, by in loader:
            model.optimizer.zero_grad()
            probs, _ = model.model(bx)
            loss = F.nll_loss(torch.log(probs + 1e-8), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            model.optimizer.step()

            epoch_loss += loss.item() * bx.size(0)
            pred = probs.argmax(dim=1)
            correct += (pred == by).sum().item()
            total += bx.size(0)

        avg_loss = epoch_loss / max(1, total)
        acc = correct / max(1, total)
        log.info("offline_train_epoch", epoch=epoch + 1, avg_loss=avg_loss, acc=acc)

    model.safe_checkpoint(label="offline_logs")
    log.info("offline_train_done", checkpoint=str(model.MODEL_PATH))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline fine-tune from decision/outcome JSONL logs.")
    parser.add_argument("--data-dir", type=str, default="training_data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-abs-pnl-pct", type=float, default=0.0)
    args = parser.parse_args()

    train_from_logs(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_abs_pnl_pct=args.min_abs_pnl_pct,
    )