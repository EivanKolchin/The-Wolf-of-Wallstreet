"""G7 test: prediction-viz data routing (crypto deque vs Alpaca stock path)."""
import asyncio
import sys
import types
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from signals import feature_spec as fs  # noqa: E402


def test_acquire_viz_routes_stock_to_alpaca_path():
    from backend.agents.nn_agent import NNTradingAgent
    stub = types.SimpleNamespace()
    df = pd.DataFrame({"open": [1, 2, 3], "close": [1.0, 2.0, 3.0]})
    seq = np.zeros((60, fs.INPUT), dtype=np.float32)
    stub._get_stock_df_cached = AsyncMock(return_value=df)
    stub._build_stock_sequence = AsyncMock(return_value=seq)

    out = asyncio.run(NNTradingAgent._acquire_viz_data(stub, "AMD"))
    assert out is not None
    rdf, price, rseq = out
    assert price == 3.0
    assert rseq.shape == (60, fs.INPUT)
    stub._get_stock_df_cached.assert_awaited_once()


def test_acquire_viz_crypto_not_ready_returns_none():
    from backend.agents.nn_agent import NNTradingAgent
    stub = types.SimpleNamespace()
    stub.feature_sequences = {"BTCUSDT": deque(maxlen=60)}  # empty -> warming up
    stub.model = types.SimpleNamespace(SEQUENCE_LENGTH=60)
    out = asyncio.run(NNTradingAgent._acquire_viz_data(stub, "BTCUSDT"))
    assert out is None


def test_stock_symbols_included_in_universe():
    # The viz loop iterates crypto + stocks; stocks must classify as us_stock.
    from backend.core import universe as u
    for s in u.STOCK_UNDERLYINGS:
        assert u.asset_class_of(s) == "us_stock"
