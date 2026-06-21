"""Systematic strategy library — the core alpha layer of the multi-strategy book."""
from backend.strategies.base import (  # noqa: F401
    Strategy, Signal, StrategySpec, positions_to_signals,
)
from backend.strategies.ts_momentum import TSMomentumBreakout, TSMomentumParams  # noqa: F401
from backend.strategies.cross_sectional import (  # noqa: F401
    CrossSectionalMomentum, XSectionalMomentumParams,
)
from backend.strategies.stat_arb import (  # noqa: F401
    StatArbPairs, StatArbParams, find_cointegrated_pairs, ou_half_life,
)
from backend.strategies.mean_reversion import (  # noqa: F401
    MeanReversion, MeanReversionParams,
)
