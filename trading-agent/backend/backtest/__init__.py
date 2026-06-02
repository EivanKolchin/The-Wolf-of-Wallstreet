"""Cost-aware, event-driven backtest harness — the measuring stick for model
changes. ``engine`` is a pure, dependency-light simulator + metrics; ``scripts/
backtest.py`` drives a trained checkpoint through it on out-of-sample data."""
from backend.backtest.engine import (  # noqa: F401
    run_backtest, run_exec_backtest, compute_metrics, directional_signal,
    atr_from_ohlc, BacktestResult,
)
