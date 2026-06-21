"""Data caching: a cached parquet must be reused on EVERY run (the bug was that
--skip-download forced a full re-download instead of reading the cache). No network."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location("pretrain_cache_mod", str(ROOT / "scripts" / "pretrain.py"))
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)


def _ohlcv(n=3):
    ts = pd.to_datetime([f"2022-01-0{i+1}" for i in range(n)])
    return pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0,
                         "close": 1.0, "volume": 1.0})


def test_cache_is_used_for_both_skip_flags(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")   # parquet engine — present on Colab/CI, not the slim local venv
    monkeypatch.setattr(pt, "DATA_DIR", tmp_path)
    cache = tmp_path / "BTCUSDT_5m_202201_202605.parquet"
    _ohlcv(3).to_parquet(cache)
    # Both skip_download True and False must read the cache (and never download).
    for sk in (True, False):
        got = pt.load_or_download("BTCUSDT", "5m", 2022, 1, 2026, 5, skip_download=sk)
        assert len(got) == 3


def test_skip_download_without_cache_raises_instead_of_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(pt, "DATA_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="skip-download"):
        pt.load_or_download("NOPE", "5m", 2022, 1, 2026, 5, skip_download=True)
