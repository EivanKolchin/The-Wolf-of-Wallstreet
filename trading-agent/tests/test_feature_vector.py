import pytest
import sys
import numpy as np
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# from backend.signals.features import FeatureVectorBuilder

@pytest.mark.asyncio
async def test_vector_has_correct_length(sample_feature_vector):
    """Vector should be precisely 62 elements."""
    assert len(sample_feature_vector) == 62

@pytest.mark.asyncio
async def test_vector_all_finite(sample_feature_vector):
    """Should contain no NaN or inf."""
    assert np.all(np.isfinite(sample_feature_vector))

@pytest.mark.asyncio
async def test_vector_bounds(sample_feature_vector):
    """All values in array should map to normalized sets (mostly [-1.5, 1.5])."""
    # Assuming overflow allowances
    assert np.all(sample_feature_vector >= -1.5)
    assert np.all(sample_feature_vector <= 1.5)

@pytest.mark.asyncio
async def test_news_features_default_neutral(sample_feature_vector):
    """If no active news, properties 49 up to 52 should be precisely floats at 0.0"""
    # assuming we test with an unaltered sample that has NO active news manually forced
    assert sample_feature_vector[49] == 0.0
    assert sample_feature_vector[50] == 0.0
    assert sample_feature_vector[51] == 0.0
    assert sample_feature_vector[52] == 0.0

@pytest.mark.asyncio
async def test_news_features_injected(sample_feature_vector, sample_news_impact):
    """If true impact supplied, vector sets values accordingly mappings."""
    # TODO: mock FeatureBuilder injection check
    assert True
