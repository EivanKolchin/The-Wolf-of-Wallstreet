"""Phase 10 standalone tests: correlation matrix + lead-lag detector + append-only persistence."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.agents.relational import (
    correlation_matrix_from_prices, lead_lag, CorrelationMatrix,
)
from backend.memory.database import Base, CorrelationSnapshot


def test_correlation_matrix_recovers_planted_high_correlation():
    rng = np.random.default_rng(0)
    n = 200
    # Start at 100 + small steps so prices stay positive (returns = diff/price stable).
    base = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    a = base + rng.normal(0, 0.05, n)        # strongly correlated to base
    b = base + rng.normal(0, 0.05, n)        # strongly correlated to base
    c = 100.0 + np.cumsum(rng.normal(0, 0.5, n))  # independent random walk

    symbols, M = correlation_matrix_from_prices({"A": a.tolist(), "B": b.tolist(), "C": c.tolist()})
    assert symbols == ["A", "B", "C"]
    assert M.shape == (3, 3)
    ia, ib, ic = symbols.index("A"), symbols.index("B"), symbols.index("C")
    assert M[ia, ib] > 0.7                                  # A,B move together
    assert abs(M[ia, ic]) < 0.5 and abs(M[ib, ic]) < 0.5    # C is independent
    assert np.allclose(np.diag(M), 1.0)


def test_correlation_matrix_too_short_returns_identity():
    syms, M = correlation_matrix_from_prices({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    assert M.shape == (2, 2)
    assert np.allclose(M, np.eye(2))


def test_lead_lag_detects_planted_lead():
    rng = np.random.default_rng(1)
    n = 300
    leader = rng.normal(0, 1.0, n).cumsum()    # a leads
    LAG = 3
    follower = np.r_[np.zeros(LAG), leader[:-LAG]] + rng.normal(0, 0.05, n)
    res = lead_lag(leader, follower, max_lag=6)
    # a leads b by LAG -> our convention: positive lag means a leads b
    assert res["lag"] == LAG
    assert res["abs_corr"] > 0.5


def test_lead_lag_low_data_safe():
    res = lead_lag([1.0, 2.0], [1.0, 2.0], max_lag=3)
    assert res == {"lag": 0, "corr": 0.0, "abs_corr": 0.0}


@pytest.mark.asyncio
async def test_correlation_matrix_persists_append_only(tmp_path):
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path.as_posix()}/r.db")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SL = async_sessionmaker(eng, expire_on_commit=False)

    cm = CorrelationMatrix(db_session_factory=SL)
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 100).cumsum().tolist()
    b = rng.normal(0, 1, 100).cumsum().tolist()

    assert await cm.update_and_persist({"A": a, "B": b}) is True
    assert await cm.update_and_persist({"A": a, "B": b}) is True  # second snapshot
    assert cm.version == 2

    async with SL() as s:
        rows = (await s.execute(select(CorrelationSnapshot))).scalars().all()
    assert len(rows) == 2                              # append-only: BOTH snapshots persist
    assert {r.version for r in rows} == {1, 2}


def test_correlation_matrix_pair_lookup():
    cm = CorrelationMatrix()
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 100).cumsum().tolist()
    b = rng.normal(0, 1, 100).cumsum().tolist()
    cm.compute({"A": a, "B": b})
    assert cm.pair("A", "B") is not None
    assert cm.pair("A", "X") is None
