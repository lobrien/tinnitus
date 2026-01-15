# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "scipy",
# ]
# ///

import argparse
import sys
import logging
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.io.wavfile as wav
from scipy import signal

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class NoiseType(str, Enum):
    WHITE = "white"
    PINK = "pink"
    BROWN = "brown"

class AudioConfig(NamedTuple):
    """Immutable configuration container."""
    noise_type: NoiseType
    center_freq: float
    notch_width_q: float
    duration_sec: int
    sample_rate: int
    chunk_duration: int
    output_file: Path

class AudioProcessor:
    """
    Handles the computational logic for generating and filtering audio.
    Stateless functional core.
    """
    
    @staticmethod
    def generate_noise_chunk(noise_type: NoiseType, num_samples: int) -> np.ndarray:
        """
        Generates a chunk of noise based on the specified type.
        """
        # Phase 2 will effectively abstract this into a Strategy pattern.
        # For Phase 1, we map the enum to the implementation here.
        if noise_type == NoiseType.PINK:
            return AudioProcessor._pink_noise(num_samples)
        elif noise_type == NoiseType.WHITE:
             return np.random.randn(num_samples).astype(np.float32)
        elif noise_type == NoiseType.BROWN:
            # Integration of white noise (1/f^2)
            white = np.random.randn(num_samples)
            return np.cumsum(white).astype(np.float32)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

    @staticmethod
    def _pink_noise(num_samples: int) -> np.ndarray:
        """Internal 1/f spectral synthesis."""
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

    @staticmethod
    def apply_notch_filter(audio_data: np.ndarray, sample_rate: int, center_freq: float, q_factor: float) -> np.ndarray:
        """
        Applies the notch filter to a specific data chunk using zero-phase filtering.
        """
        if center_freq <= 0:
            return audio_data

        b, a = signal.iirnotch(w0=center_freq, Q=q_factor, fs=sample_rate)
        filtered = signal.filtfilt(b, a, audio_data)
        return filtered.astype(np.float32)

def parse_arguments() -> AudioConfig:
    parser = argparse.ArgumentParser(
        description="Generate spectral noise with a specific frequency notch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--type", type=str, choices=[t.value for t in NoiseType], default="pink",
                        help="Type of spectral noise to generate.")
    parser.add_argument("--freq", type=float, required=True,
                        help="Center frequency of the notch filter (Hz).")
    parser.add_argument("--width", type=float, default=1.414,
                        help="Q-factor of the notch (approx 1.414 for 1 octave).")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration of audio to generate in seconds.")
    parser.add_argument("--output", type=Path, default=Path("output.wav"),
                        help="Output filename.")
    parser.add_argument("--rate", type=int, default=44100,
                        help="Sample rate (Hz).")

    args = parser.parse_args()

    return AudioConfig(
        noise_type=NoiseType(args.type),
        center_freq=args.freq,
        notch_width_q=args.width,
        duration_sec=args.duration,
        sample_rate=args.rate,
        chunk_duration=10, # Fixed chunk size for memory management
        output_file=args.output
    )

def print_progress(current_chunk: int, total_chunks: int):
    percent = (current_chunk / total_chunks) * 100
    sys.stdout.write(f"\rProgress: [{percent:6.2f}%] processing chunk {current_chunk}/{total_chunks}")
    sys.stdout.flush()

def main():
    config = parse_arguments()
    
    # Initialization
    total_samples = config.sample_rate * config.duration_sec
    chunk_samples = config.sample_rate * config.chunk_duration
    total_chunks = int(np.ceil(total_samples / chunk_samples))
    
    # Pre-allocate buffer (Float32)
    full_buffer = np.zeros(total_samples, dtype=np.float32)

    logging.info(f"Generating {config.noise_type.value.upper()} noise")
    logging.info(f"Notch: {config.center_freq} Hz (Q={config.notch_width_q})")
    logging.info(f"Output: {config.output_file}")

    # Processing Loop
    processor = AudioProcessor()
    
    for i in range(total_chunks):
        start_idx = i * chunk_samples
        end_idx = min(start_idx + chunk_samples, total_samples)
        current_len = end_idx - start_idx
        
        if current_len <= 0: break

        # 1. Generate
        raw_chunk = processor.generate_noise_chunk(config.noise_type, current_len)
        
        # 2. Filter
        filtered_chunk = processor.apply_notch_filter(
            raw_chunk, config.sample_rate, config.center_freq, config.notch_width_q
        )
        
        # 3. Store
        full_buffer[start_idx:end_idx] = filtered_chunk
        
        print_progress(i + 1, total_chunks)

    print() # Newline

    # Final IO Phase
    logging.info("Normalizing and Saving to disk...")
    
    max_amp = np.max(np.abs(full_buffer))
    if max_amp > 0:
        # Normalize to -3dB
        full_buffer = full_buffer / max_amp * 0.707
    
    try:
        wav.write(config.output_file, config.sample_rate, full_buffer)
        logging.info("Success.")
    except IOError as e:
        logging.error(f"Write failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()