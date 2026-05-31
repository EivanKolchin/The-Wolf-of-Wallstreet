"""Phase 7b finishing tests + Alpaca smoothing.

Locks the two things that just changed:
  - Symbol registry now contains the 5 US stock underlyings appended after
    the 8 cryptos (live + offline must agree).
  - Live agent's checkpoint loader splices an old 8-row embedding into the new
    13-row one instead of crashing on shape mismatch.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))


def test_symbol_registry_contains_stocks_after_crypto():
    from agents.improved_model import SYMBOLS, SYMBOL_TO_ID
    assert SYMBOLS[0] == "BTCUSDT"
    assert SYMBOLS[7] == "DOGEUSDT"
    for sym in ("SNDK", "AMD", "MU", "AXTI", "BE"):
        assert sym in SYMBOL_TO_ID
        assert SYMBOL_TO_ID[sym] >= 8


def test_offline_symbol_registry_matches_live_order():
    """Offline pretrain.py SYMBOLS must agree on ordering — embedding row IDs
    must mean the same thing in both places or trained weights mis-load."""
    from agents.improved_model import SYMBOLS as live_syms
    from scripts.pretrain import SYMBOLS as off_syms
    assert list(live_syms) == list(off_syms)


def test_checkpoint_loader_splices_smaller_embedding(tmp_path, monkeypatch):
    """Save a checkpoint with 8 embedding rows; load it into a model expecting 13.
    The first 8 rows must match exactly; rows 8..12 should be freshly initialised."""
    from agents.nn_model import PersistentTradingModel
    from agents.improved_model import ImprovedTradingLSTM
    from agents.model_io import save_checkpoint

    # Build a small ImprovedTradingLSTM with only 8 symbols.
    legacy = ImprovedTradingLSTM(num_symbols=8)
    legacy.symbol_embedding.weight.data.fill_(0.5)   # easy-to-detect signature

    ckpt_path = tmp_path / "trading_lstm_latest.pt"
    save_checkpoint(ckpt_path, model=legacy)

    monkeypatch.setattr(PersistentTradingModel, "MODEL_PATH", ckpt_path)
    monkeypatch.setattr(PersistentTradingModel, "CHECKPOINT_DIR", tmp_path / "ckpts")
    pm = PersistentTradingModel()

    # Live model is the new 13-row one.
    live_emb = pm.model.symbol_embedding.weight
    assert live_emb.shape[0] == 13
    # First 8 rows should be the spliced-in 0.5 sentinel...
    assert torch.allclose(live_emb[:8], torch.full_like(live_emb[:8], 0.5))
    # ...and rows 8..12 should NOT be the sentinel (freshly init'd by xavier).
    assert not torch.allclose(live_emb[8:], torch.full_like(live_emb[8:], 0.5))


def test_pretrain_alpaca_helpers_exposed():
    from scripts.pretrain import _is_stock_symbol, load_alpaca_history
    assert _is_stock_symbol("AMD") is True
    assert _is_stock_symbol("BTCUSDT") is False
    # The function must exist + take the standard (sym, year, month) signature.
    import inspect
    sig = inspect.signature(load_alpaca_history)
    assert list(sig.parameters)[:3] == ["symbol", "start_year", "start_month"]


def test_load_full_history_dispatches_to_alpaca_for_stocks(monkeypatch):
    """Calling load_full_history with a stock symbol should hit the Alpaca
    branch (not try to download Binance CSVs)."""
    from scripts import pretrain as p

    called = {"alpaca": 0, "binance": 0}
    monkeypatch.setattr(p, "load_alpaca_history",
                         lambda s, y, m: (called.__setitem__("alpaca", 1) or {"5m": None, "1h": None, "4h": None}))
    monkeypatch.setattr(p, "load_or_download",
                         lambda *a, **kw: called.__setitem__("binance", 1) or None)

    p.load_full_history("AMD", 2024, 1)
    assert called["alpaca"] == 1
    assert called["binance"] == 0
