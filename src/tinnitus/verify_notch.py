import sys
import logging
import subprocess
import argparse
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, NamedTuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class VerificationConfig(NamedTuple):
    input_file: Path
    target_freq: float
    output_plot: Optional[Path]

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
        # In a library context, we would raise, but for a CLI tool, exiting is acceptable
        # strictly if caught by main, but raising is better for testing.
        raise RuntimeError(f"FFprobe failed: {e}")

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
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.DEVNULL, 
            bufsize=10**7
        )
        
        raw_data, _ = process.communicate()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        audio_array = np.frombuffer(raw_data, dtype=np.float32)
        return audio_array
        
    except Exception as e:
        logger.error(f"Failed to decode audio stream: {e}")
        raise RuntimeError(f"FFmpeg failed: {e}")

def compute_spectrum(data: np.ndarray, rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Power Spectral Density (PSD) using Welch's method.
    """
    freqs, power = welch(data, fs=rate, nperseg=4096)
    power_db = 10 * np.log10(power + 1e-10)
    return freqs, power_db

def analyze_notch_depth(freqs: np.ndarray, power_db: np.ndarray, target_freq: float) -> float:
    """
    Calculates the attenuation at the target frequency relative to the baseline.
    Returns depth in dB (positive value = attenuation).
    """
    idx_target = np.abs(freqs - target_freq).argmin()
    power_at_target = power_db[idx_target]
    
    # Baseline: Median power to reject outliers
    baseline_power = np.median(power_db)
    
    return baseline_power - power_at_target

def generate_plot(freqs: np.ndarray, power_db: np.ndarray, target_freq: float, output_path: Path) -> None:
    """
    Generates a spectral analysis plot highlighting the notch.
    """
    plt.figure(figsize=(12, 7))
    plt.semilogx(freqs, power_db, label='Spectral Density', color='#2c3e50', alpha=0.9, linewidth=1)
    
    plt.axvline(x=target_freq, color='#e74c3c', linestyle='--', label=f'Target: {target_freq} Hz')
    
    lower_bound = target_freq * (2 ** -0.5)
    upper_bound = target_freq * (2 ** 0.5)
    plt.axvspan(lower_bound, upper_bound, color='#e74c3c', alpha=0.15, label='1 Octave Bandwidth')

    plt.title(f'Notch Filter Analysis (Target: {target_freq} Hz)')
    plt.xlabel('Frequency (Hz) [Log Scale]')
    plt.ylabel('PSD (dB/Hz)')
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xlim(100, 20000)
    
    try:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Verification plot saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
    finally:
        plt.close()

def parse_arguments(argv: Optional[list[str]] = None) -> VerificationConfig:
    parser = argparse.ArgumentParser(description="Verify notch filter depth in audio files.")
    parser.add_argument("file", type=Path, help="Input audio file (m4a, wav, etc)")
    parser.add_argument("--freq", type=float, required=True, help="Target notch frequency (Hz)")
    parser.add_argument("--plot", type=Path, default=None, help="Optional path to save analysis plot")
    
    args = parser.parse_args(argv)
    
    # Default plot name if not provided but useful to have
    if args.plot is None:
        args.plot = args.file.with_suffix('.png')

    return VerificationConfig(args.file, args.freq, args.plot)

def main():
    try:
        config = parse_arguments()
    except SystemExit:
        sys.exit(1)

    if not config.input_file.exists():
        logger.error(f"File not found: {config.input_file}")
        sys.exit(1)

    try:
        rate = get_audio_properties(config.input_file)
        data = decode_audio_stream(config.input_file, rate)
        freqs, power_db = compute_spectrum(data, rate)
        depth = analyze_notch_depth(freqs, power_db, config.target_freq)
        
        logger.info(f"Measured Notch Depth: {depth:.2f} dB")
        
        if depth < 10:
            logger.warning("WARNING: Notch depth is shallow (< 10dB).")
        else:
            logger.info("SUCCESS: Deep notch detected.")

        if config.output_plot:
            generate_plot(freqs, power_db, config.target_freq, config.output_plot)
            
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()