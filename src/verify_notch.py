# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "scipy",
#     "matplotlib"
# ]
# ///

import sys
import logging
import subprocess
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_audio_properties(file_path: Path) -> int:
    """
    Uses ffprobe to determine the sample rate of the audio file.
    """
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "a:0", 
            "-show_entries", "stream=sample_rate", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        sample_rate = int(result.stdout.strip())
        logger.info(f"Detected sample rate: {sample_rate} Hz")
        return sample_rate
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Failed to probe audio file: {e}")
        sys.exit(1)

def decode_audio_stream(file_path: Path, sample_rate: int) -> np.ndarray:
    """
    Decodes audio to raw PCM (float32, mono) using ffmpeg pipe.
    """
    try:
        cmd = [
            "ffmpeg", 
            "-i", str(file_path), 
            "-f", "f32le",     # Force 32-bit float raw output
            "-ac", "1",        # Mix to mono for spectral analysis
            "-ar", str(sample_rate), 
            "-"                # Output to stdout
        ]
        
        # Increase buffer size for large files to prevent pipe blocking
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.DEVNULL, 
            bufsize=10**7
        )
        
        raw_data, _ = process.communicate()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        # Convert bytes to numpy array
        audio_array = np.frombuffer(raw_data, dtype=np.float32)
        return audio_array
        
    except Exception as e:
        logger.error(f"Failed to decode audio stream: {e}")
        sys.exit(1)

def compute_spectrum(data: np.ndarray, rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Power Spectral Density (PSD) using Welch's method.
    """
    # nperseg=4096 gives ~10-12 Hz resolution
    freqs, power = welch(data, fs=rate, nperseg=4096)
    
    # Convert to dB, adding epsilon to avoid log(0)
    power_db = 10 * np.log10(power + 1e-10)
    return freqs, power_db

def analyze_notch_depth(freqs: np.ndarray, power_db: np.ndarray, target_freq: float) -> float:
    """
    Calculates the attenuation at the target frequency relative to the baseline.
    """
    idx_target = np.abs(freqs - target_freq).argmin()
    power_at_target = power_db[idx_target]
    
    # Baseline: Median power to reject outliers (like the notch itself)
    baseline_power = np.median(power_db)
    
    # Return positive depth value
    return baseline_power - power_at_target

def generate_plot(freqs: np.ndarray, power_db: np.ndarray, target_freq: float, output_path: Path) -> None:
    """
    Generates a spectral analysis plot highlighting the notch.
    """
    plt.figure(figsize=(12, 7))
    plt.semilogx(freqs, power_db, label='Spectral Density', color='#2c3e50', alpha=0.9, linewidth=1)
    
    # Visual Guides
    plt.axvline(x=target_freq, color='#e74c3c', linestyle='--', label=f'Target Center: {target_freq} Hz')
    
    # Theoretical 1-octave bounds
    lower_bound = target_freq * (2 ** -0.5)
    upper_bound = target_freq * (2 ** 0.5)
    plt.axvspan(lower_bound, upper_bound, color='#e74c3c', alpha=0.15, label='1 Octave Bandwidth')

    plt.title(f'Notch Filter Analysis (Target: {target_freq} Hz)')
    plt.xlabel('Frequency (Hz) [Log Scale]')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    # Set x-axis limits to relevant hearing range
    plt.xlim(100, 20000)
    
    try:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Verification plot saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python verify_notch_m4a.py <file.m4a>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    target_freq = 5800

    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        sys.exit(1)

    logger.info(f"Processing {input_path.name}...")

    # Pipeline
    rate = get_audio_properties(input_path)
    data = decode_audio_stream(input_path, rate)
    freqs, power_db = compute_spectrum(data, rate)
    depth = analyze_notch_depth(freqs, power_db, target_freq)
    
    logger.info(f"Measured Notch Depth: {depth:.2f} dB")
    
    if depth < 10:
        logger.warning("WARNING: Notch depth is shallow. Check encoding parameters.")
    else:
        logger.info("SUCCESS: Deep notch detected.")

    output_plot = input_path.with_suffix('.png')
    generate_plot(freqs, power_db, target_freq, output_plot)

if __name__ == "__main__":
    main()