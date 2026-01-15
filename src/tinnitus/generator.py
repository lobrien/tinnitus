import argparse
import sys
import logging
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Callable, Dict, Optional

import numpy as np
import soundfile as sf
from scipy import signal

logger = logging.getLogger(__name__)

class NoiseType(str, Enum):
    WHITE = "white"
    PINK = "pink"
    BROWN = "brown"

NoiseStrategy = Callable[[int], np.ndarray]

class AudioConfig(NamedTuple):
    noise_type: NoiseType
    center_freq: float
    notch_width_q: float
    duration_sec: int
    sample_rate: int
    chunk_duration: int
    output_file: Path

# --- Strategies ---

def _strategy_white(num_samples: int) -> np.ndarray:
    return np.random.randn(num_samples).astype(np.float32)

def _strategy_brown(num_samples: int) -> np.ndarray:
    white = np.random.randn(num_samples)
    return np.cumsum(white).astype(np.float32)

def _strategy_pink(num_samples: int) -> np.ndarray:
    white = np.random.randn(num_samples)
    spectrum = np.fft.rfft(white)
    indices = np.arange(1, len(spectrum) + 1)
    scale = 1 / np.sqrt(indices)
    spectrum = spectrum * scale
    pink = np.fft.irfft(spectrum)
    max_val = np.max(np.abs(pink))
    if max_val > 0:
        pink /= max_val
    return pink.astype(np.float32)

NOISE_DISPATCH: Dict[NoiseType, NoiseStrategy] = {
    NoiseType.WHITE: _strategy_white,
    NoiseType.BROWN: _strategy_brown,
    NoiseType.PINK: _strategy_pink,
}

# --- Core Logic ---

class AudioProcessor:
    @staticmethod
    def generate_noise_chunk(strategy: NoiseStrategy, num_samples: int) -> np.ndarray:
        return strategy(num_samples)

    @staticmethod
    def apply_notch_filter(audio_data: np.ndarray, sample_rate: int, center_freq: float, q_factor: float) -> np.ndarray:
        if center_freq <= 0:
            return audio_data
        b, a = signal.iirnotch(w0=center_freq, Q=q_factor, fs=sample_rate)
        # zero-phase filtering (filtfilt) is preferred for static noise masking
        filtered = signal.filtfilt(b, a, audio_data)
        return filtered.astype(np.float32)

# --- CLI Helpers ---

def setup_parser(parser: argparse.ArgumentParser) -> None:
    """Configures arguments for the generator subcommand."""
    parser.add_argument("--type", type=str, choices=[t.value for t in NoiseType], default="pink",
                        help="Type of spectral noise to generate.")
    parser.add_argument("--freq", type=float, required=True,
                        help="Center frequency of the notch filter (Hz).")
    parser.add_argument("--width", type=float, default=1.0,
                        help="Notch width in octaves (default: 1.0).")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration of audio to generate in seconds.")
    parser.add_argument("--output", type=Path, default=Path("output.flac"),
                        help="Output filename (supports .wav, .flac, .ogg).")
    parser.add_argument("--rate", type=int, default=44100,
                        help="Sample rate (Hz).")

def config_from_args(args: argparse.Namespace) -> AudioConfig:
    # Convert width (octaves) to Q-factor
    # Q = sqrt(2&bw) / (2^bw-1)
    bw = args.width
    q_factor = np.sqrt(2**bw) / (2**bw-1)

    return AudioConfig(
        noise_type=NoiseType(args.type),
        center_freq=args.freq,
        notch_width_q=q_factor,
        duration_sec=args.duration,
        sample_rate=args.rate,
        chunk_duration=10,
        output_file=args.output
    )

def print_progress(current_chunk: int, total_chunks: int) -> None:
    percent = (current_chunk / total_chunks) * 100
    sys.stdout.write(f"\rProgress: [{percent:6.2f}%] processing chunk {current_chunk}/{total_chunks}")
    sys.stdout.flush()

def run(config: AudioConfig) -> None:
    """Main execution entry point."""
    total_samples = config.sample_rate * config.duration_sec
    chunk_samples = config.sample_rate * config.chunk_duration
    total_chunks = int(np.ceil(total_samples / chunk_samples))
    
    strategy = NOISE_DISPATCH[config.noise_type]
    full_buffer = np.zeros(total_samples, dtype=np.float32)

    logger.info(f"Generating {config.noise_type.value.upper()} noise")
    logger.info(f"Notch: {config.center_freq} Hz (Q={config.notch_width_q})")

    for i in range(total_chunks):
        start_idx = i * chunk_samples
        end_idx = min(start_idx + chunk_samples, total_samples)
        current_len = end_idx - start_idx
        
        if current_len <= 0: break

        raw_chunk = AudioProcessor.generate_noise_chunk(strategy, current_len)
        filtered_chunk = AudioProcessor.apply_notch_filter(
            raw_chunk, config.sample_rate, config.center_freq, config.notch_width_q
        )
        full_buffer[start_idx:end_idx] = filtered_chunk
        print_progress(i + 1, total_chunks)

    print() 

    logger.info("Normalizing...")
    max_amp = np.max(np.abs(full_buffer))
    if max_amp > 0:
        full_buffer = full_buffer / max_amp * 0.707
    
    logger.info(f"Saving to {config.output_file}...")
    try:
        sf.write(config.output_file, full_buffer, config.sample_rate)
        logger.info("Success.")
    except Exception as e:
        logger.error(f"Write failed: {e}")
        sys.exit(1)

# Backward compatibility / Direct script usage
def parse_arguments(argv: Optional[list[str]] = None) -> AudioConfig:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    setup_parser(parser)
    args = parser.parse_args(argv)
    return config_from_args(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(parse_arguments())