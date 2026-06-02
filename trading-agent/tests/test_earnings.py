"""Cycle 7: leakage-safe earnings-calendar features + the feature-spec EARNINGS block."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.signals import feature_spec as fs  # noqa: E402
from backend.signals.earnings import (  # noqa: E402
    earnings_feature_matrix, normalize_events, EarningsProvider,
)


def _events():
    # one already-reported beat + one future report (estimate only)
    return [
        {"dt": pd.Timestamp("2024-01-15 16:00"), "eps_actual": 1.2, "eps_estimate": 1.0},
        {"dt": pd.Timestamp("2024-04-15 16:00"), "eps_actual": None, "eps_estimate": 1.1},
    ]


def test_anticipatory_proximity_and_pre_window():
    ev = _events()
    far = earnings_feature_matrix(ev, [pd.Timestamp("2024-03-01")])[0]
    near = earnings_feature_matrix(ev, [pd.Timestamp("2024-04-14")])[0]
    assert near[0] > far[0]                    # proximity to next report rises as it nears
    assert near[1] == 1.0 and far[1] == 0.0    # pre-earnings flag only within PRE_DAYS


def test_post_earnings_drift_and_surprise_sign():
    just_after = earnings_feature_matrix(_events(), [pd.Timestamp("2024-01-16")])[0]
    assert just_after[2] > 0.5                  # drift proximity high right after the report
    assert just_after[3] > 0                    # positive surprise (1.2 vs 1.0 estimate)


def test_no_leakage_before_report():
    before = earnings_feature_matrix(_events(), [pd.Timestamp("2024-01-01")])[0]
    # the Jan-15 actual must NOT influence a Jan-1 bar
    assert before[2] == 0.0 and before[3] == 0.0


def test_negative_surprise_is_negative():
    ev = [{"dt": pd.Timestamp("2024-01-15 16:00"), "eps_actual": 0.7, "eps_estimate": 1.0}]
    row = earnings_feature_matrix(ev, [pd.Timestamp("2024-01-16")])[0]
    assert row[3] < 0                           # miss → negative surprise


def test_empty_events_are_zeros():
    out = earnings_feature_matrix([], [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")])
    assert out.shape == (2, fs.EARNINGS_DIM) and np.allclose(out, 0.0)


def test_normalize_finnhub_rows():
    ev = normalize_events([{"date": "2024-01-15", "hour": "amc", "epsActual": 1.2, "epsEstimate": 1.0}])
    assert len(ev) == 1 and ev[0]["eps_actual"] == 1.2 and ev[0]["dt"].hour == 16


def test_provider_no_token_returns_empty():
    assert EarningsProvider("").events("NVDA", "2024-01-01", "2024-12-31") == []


# ───────────────────────── feature-spec EARNINGS block ───────────────────────
def test_feature_spec_earnings_layout():
    assert fs.INPUT == 90 and fs.EARNINGS_DIM == 4
    assert (fs.EARNINGS.start, fs.EARNINGS.stop) == (86, 90)
    assert (fs.NEWS_EMBED.start, fs.NEWS_EMBED.stop) == (70, 86)   # unchanged, not absorbing earnings
    assert (fs.HTF_START, fs.HTF_END) == (62, 70)
    assert fs.checkpoint_meta()["earnings_dim"] == 4
