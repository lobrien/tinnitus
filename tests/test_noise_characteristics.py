import numpy as np
import pytest
from scipy.signal import welch
from tinnitus.generator import NOISE_DISPATCH, NoiseType

def calculate_spectral_slope(data: np.ndarray, fs: int) -> float:
    """
    Computes the slope of the Power Spectral Density (PSD) on a log-log scale.
    
    Returns:
        The slope coefficient.
        0.0  = White Noise
        -1.0 = Pink Noise
        -2.0 = Brown Noise
    """
    # Compute PSD using Welch's method
    freqs, psd = welch(data, fs=fs, nperseg=4096)
    
    # Filter to valid range to avoid DC offset (0Hz) and Nyquist roll-off
    # We analyze the "meaty" part of the spectrum: 100 Hz to 10 kHz
    mask = (freqs >= 100) & (freqs <= 10000)
    valid_freqs = freqs[mask]
    valid_psd = psd[mask]
    
    # Avoid log(0) errors
    valid_psd = np.maximum(valid_psd, 1e-10)
    
    # Linear regression on log-log data
    # log(Power) = slope * log(Freq) + intercept
    log_freqs = np.log10(valid_freqs)
    log_psd = np.log10(valid_psd)
    
    slope, _ = np.polyfit(log_freqs, log_psd, 1)
    return slope

@pytest.fixture(scope="module")
def generated_samples():
    """
    Generates a sufficiently long sample of each noise type for spectral analysis.
    Scope is module to generate once and reuse.
    """
    fs = 44100
    duration_sec = 2
    n_samples = fs * duration_sec
    
    # Generate samples using the strategies directly
    samples = {
        NoiseType.WHITE: NOISE_DISPATCH[NoiseType.WHITE](n_samples),
        NoiseType.PINK: NOISE_DISPATCH[NoiseType.PINK](n_samples),
        NoiseType.BROWN: NOISE_DISPATCH[NoiseType.BROWN](n_samples),
    }
    return samples

def test_white_noise_physics(generated_samples):
    """
    White noise should have equal power per frequency bin.
    Expected Slope: 0
    """
    data = generated_samples[NoiseType.WHITE]
    slope = calculate_spectral_slope(data, fs=44100)
    
    # Allow some stochastic variance (±0.15)
    assert slope == pytest.approx(0.0, abs=0.15), \
        f"White noise spectral slope {slope:.2f} is not flat (expected 0.0)"

def test_pink_noise_physics(generated_samples):
    """
    Pink noise should decrease by 3dB per octave (1/f power).
    Expected Slope: -1
    """
    data = generated_samples[NoiseType.PINK]
    slope = calculate_spectral_slope(data, fs=44100)
    
    assert slope == pytest.approx(-1.0, abs=0.15), \
        f"Pink noise spectral slope {slope:.2f} incorrect (expected -1.0)"

def test_brown_noise_physics(generated_samples):
    """
    Brown noise should decrease by 6dB per octave (1/f^2 power).
    Expected Slope: -2
    """
    data = generated_samples[NoiseType.BROWN]
    slope = calculate_spectral_slope(data, fs=44100)
    
    assert slope == pytest.approx(-2.0, abs=0.15), \
        f"Brown noise spectral slope {slope:.2f} incorrect (expected -2.0)"