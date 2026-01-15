import sys
import logging
import subprocess
import argparse
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, NamedTuple, Optional

logger = logging.getLogger(__name__)

class VerificationConfig(NamedTuple):
    input_file: Path
    target_freq: float
    output_plot: Optional[Path]

def get_audio_properties(file_path: Path) -> int:
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0", 
            "-show_entries", "stream=sample_rate", 
            "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except Exception as e:
        raise RuntimeError(f"FFprobe failed: {e}")

def decode_audio_stream(file_path: Path, sample_rate: int) -> np.ndarray:
    try:
        cmd = [
            "ffmpeg", "-i", str(file_path), "-f", "f32le", 
            "-ac", "1", "-ar", str(sample_rate), "-"
        ]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7
        )
        raw_data, _ = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        return np.frombuffer(raw_data, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"FFmpeg failed: {e}")

def compute_spectrum(data: np.ndarray, rate: int) -> Tuple[np.ndarray, np.ndarray]:
    freqs, power = welch(data, fs=rate, nperseg=4096)
    power_db = 10 * np.log10(power + 1e-10)
    return freqs, power_db

def analyze_notch_depth(freqs: np.ndarray, power_db: np.ndarray, target_freq: float) -> float:
    idx_target = np.abs(freqs - target_freq).argmin()
    baseline_power = np.median(power_db)
    return baseline_power - power_db[idx_target]

def generate_plot(freqs: np.ndarray, power_db: np.ndarray, target_freq: float, output_path: Path) -> None:
    plt.figure(figsize=(12, 7))
    plt.semilogx(freqs, power_db, label='Spectral Density', color='#2c3e50', alpha=0.9)
    plt.axvline(x=target_freq, color='#e74c3c', linestyle='--', label=f'Target: {target_freq} Hz')
    
    lower = target_freq * (2 ** -0.5)
    upper = target_freq * (2 ** 0.5)
    plt.axvspan(lower, upper, color='#e74c3c', alpha=0.15, label='1 Octave Bandwidth')

    plt.title(f'Notch Filter Analysis (Target: {target_freq} Hz)')
    plt.grid(True, which="both", alpha=0.3)
    plt.xlim(100, 20000)
    plt.legend()
    
    try:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Plot saved: {output_path}")
    finally:
        plt.close()

# --- CLI Helpers ---

def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("file", type=Path, help="Input audio file")
    parser.add_argument("--freq", type=float, required=True, help="Target notch frequency (Hz)")
    parser.add_argument("--plot", type=Path, default=None, help="Path to save analysis plot")

def config_from_args(args: argparse.Namespace) -> VerificationConfig:
    plot_path = args.plot if args.plot else args.file.with_suffix('.png')
    return VerificationConfig(args.file, args.freq, plot_path)

def run(config: VerificationConfig) -> None:
    if not config.input_file.exists():
        logger.error(f"File not found: {config.input_file}")
        sys.exit(1)

    try:
        rate = get_audio_properties(config.input_file)
        data = decode_audio_stream(config.input_file, rate)
        freqs, power_db = compute_spectrum(data, rate)
        depth = analyze_notch_depth(freqs, power_db, config.target_freq)
        
        logger.info(f"Notch Depth: {depth:.2f} dB")
        if depth < 10:
            logger.warning("WARNING: Shallow notch (< 10dB).")
        else:
            logger.info("SUCCESS: Deep notch detected.")

        if config.output_plot:
            generate_plot(freqs, power_db, config.target_freq, config.output_plot)
            
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

def parse_arguments(argv: Optional[list[str]] = None) -> VerificationConfig:
    parser = argparse.ArgumentParser()
    setup_parser(parser)
    args = parser.parse_args(argv)
    return config_from_args(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(parse_arguments())