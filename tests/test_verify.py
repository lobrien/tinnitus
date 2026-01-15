import pytest
import numpy as np
from scipy import signal
from pathlib import Path
from tinnitus.verify_notch import analyze_notch_depth, compute_spectrum, parse_arguments

def test_analyze_notch_depth_synthetic() -> None:
    """
    Create synthetic white noise, apply a known notch, 
    and verify the analyzer detects the attenuation.
    """
    # 1. Generate White Noise
    fs = 44100
    duration = 2
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    noise = np.random.randn(len(t))
    
    # 2. Apply a strong notch at 1000 Hz
    target_freq = 1000.0
    b, a = signal.iirnotch(w0=target_freq, Q=10, fs=fs)
    filtered_noise = signal.filtfilt(b, a, noise)
    
    # 3. Analyze
    freqs, power_db = compute_spectrum(filtered_noise, fs)
    depth = analyze_notch_depth(freqs, power_db, target_freq)
    
    # 4. Assert
    # A generic IIR notch should define a significant drop. 
    # We expect > 10dB depth typically.
    assert depth > 10.0, f"Expected notch depth > 10dB, got {depth:.2f}dB"

def test_analyze_no_notch_synthetic() -> None:
    """
    Verify that analyzing white noise (flat spectrum) returns 
    near-zero notch depth.
    """
    fs = 44100
    noise = np.random.randn(fs * 2) # 2 seconds
    
    freqs, power_db = compute_spectrum(noise, fs)
    depth = analyze_notch_depth(freqs, power_db, 1000.0)
    
    # The variance of white noise PSD might cause small fluctuations,
    # but it shouldn't look like a deep notch.
    assert abs(depth) < 3.0, f"False positive notch detected: {depth}dB"

def test_verify_arguments() -> None:
    """Test CLI argument parsing."""
    argv = ["input.wav", "--freq", "4000"]
    config = parse_arguments(argv)
    
    assert config.input_file == Path("input.wav")
    assert config.target_freq == 4000.0
    # Default plot behavior
    assert config.output_plot == Path("input.png")

def test_verify_arguments_explicit_plot() -> None:
    argv = ["input.wav", "--freq", "4000", "--plot", "analysis.png"]
    config = parse_arguments(argv)
    assert config.output_plot == Path("analysis.png")