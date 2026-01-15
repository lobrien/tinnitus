import numpy as np
import pytest
from tinnitus.generator import NOISE_DISPATCH, NoiseType, AudioProcessor

@pytest.fixture(autouse=True)
def stable_seed() -> None:
    """Ensure reproducible randomness for all tests."""
    np.random.seed(42)

def test_dispatch_registry() -> None:
    """Verify that every NoiseType Enum has a corresponding registered strategy."""
    for noise_type in NoiseType:
        assert noise_type in NOISE_DISPATCH
        assert callable(NOISE_DISPATCH[noise_type])

def test_white_noise_properties() -> None:
    """White noise should have mean ~0 and std ~1."""
    strategy = NOISE_DISPATCH[NoiseType.WHITE]
    chunk = strategy(100_000)
    
    assert chunk.dtype == np.float32
    assert len(chunk) == 100_000
    assert np.abs(np.mean(chunk)) < 0.05  # Allow small variance
    assert np.abs(np.std(chunk) - 1.0) < 0.05

def test_brown_noise_properties() -> None:
    """
    Brown noise is the integration of white noise.
    Therefore, the discrete difference of Brown noise should be White noise.
    """
    strategy = NOISE_DISPATCH[NoiseType.BROWN]
    chunk = strategy(100_000)
    
    # Calculate the discrete difference (inverse of cumsum)
    diff = np.diff(chunk)
    
    assert chunk.dtype == np.float32
    assert np.abs(np.mean(diff)) < 0.05
    assert np.abs(np.std(diff) - 1.0) < 0.05

def test_pink_noise_sanity() -> None:
    """Verify Pink noise generates valid float32 audio data."""
    strategy = NOISE_DISPATCH[NoiseType.PINK]
    chunk = strategy(44100)
    
    assert chunk.dtype == np.float32
    assert len(chunk) == 44100