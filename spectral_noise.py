# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "scipy",
# ]
# ///

import numpy as np
import scipy.io.wavfile as wav

import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import sys
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class AudioProcessor:
    """
    Handles the computational logic for generating and filtering audio.
    Stateless functional core.
    """
    
    @staticmethod
    def generate_pink_noise_chunk(num_samples: int) -> np.ndarray:
        """
        Generates a chunk of pink noise using spectral synthesis.
        """
        # White noise
        white = np.random.randn(num_samples)
        
        # FFT
        spectrum = np.fft.rfft(white)
        
        # 1/sqrt(f) scaling
        indices = np.arange(1, len(spectrum) + 1)
        scale = 1 / np.sqrt(indices)
        spectrum = spectrum * scale
        
        # Inverse FFT
        pink = np.fft.irfft(spectrum)
        
        # Normalize chunk locally to ensure consistency
        max_val = np.max(np.abs(pink))
        if max_val > 0:
            pink /= max_val
            
        return pink.astype(np.float32)

    @staticmethod
    def apply_notch_filter(audio_data: np.ndarray, sample_rate: int, center_freq: float, q_factor: float) -> np.ndarray:
        """
        Applies the notch filter to a specific data chunk.
        """
        # Using lfilter (or filtfilt) on chunks requires handling filter state (zi)
        # for perfect continuity. However, for noise masking, processing chunks 
        # independently with filtfilt (zero-phase) is statistically acceptable 
        # and avoids transient "clicks" better than unmanaged IIR states.
        
        b, a = signal.iirnotch(w0=center_freq, Q=q_factor, fs=sample_rate)
        
        # We use filtfilt to prevent phase shift, processing the chunk as a discrete block.
        # For continuous tones this would cause discontinuities, but for stochastic noise
        # it is acoustically transparent.
        filtered = signal.filtfilt(b, a, audio_data)
        
        return filtered.astype(np.float32)

def print_progress(current_chunk: int, total_chunks: int):
    """
    Prints an in-place progress percentage counter to the CLI.
    """
    percent = (current_chunk / total_chunks) * 100
    sys.stdout.write(f"\rProgress: [{percent:6.2f}%] processing chunk {current_chunk}/{total_chunks}")
    sys.stdout.flush()

def main():
    # --- Configuration ---
    SAMPLE_RATE = 44100
    DURATION_SEC = 60      # Adjustable duration
    CHUNK_DURATION = 10    # Process in 10-second blocks
    CENTER_FREQ = 5588.0   # F8
    NOTCH_WIDTH_Q = 1.414  # ~1 Octave
    OUTPUT_FILE = "pink_noise_notch_F8_float32.wav"

    # --- Initialization ---
    total_samples = SAMPLE_RATE * DURATION_SEC
    chunk_samples = SAMPLE_RATE * CHUNK_DURATION
    
    # Calculate total chunks (rounding up)
    total_chunks = int(np.ceil(total_samples / chunk_samples))
    
    # Pre-allocate buffer for the full file (Float32)
    # This prevents memory fragmentation during the loop.
    full_buffer = np.zeros(total_samples, dtype=np.float32)

    logging.info(f"Starting Generation: {DURATION_SEC}s / {total_chunks} chunks")

    # --- Processing Loop ---
    processor = AudioProcessor()
    
    for i in range(total_chunks):
        # Calculate indices
        start_idx = i * chunk_samples
        end_idx = min(start_idx + chunk_samples, total_samples)
        current_len = end_idx - start_idx
        
        if current_len <= 0: break

        # 1. Generate Pink Noise
        raw_chunk = processor.generate_pink_noise_chunk(current_len)
        
        # 2. Apply Notch
        filtered_chunk = processor.apply_notch_filter(
            raw_chunk, SAMPLE_RATE, CENTER_FREQ, NOTCH_WIDTH_Q
        )
        
        # 3. Store in Buffer
        full_buffer[start_idx:end_idx] = filtered_chunk
        
        # 4. Update UI
        print_progress(i + 1, total_chunks)

    # Newline after progress bar completes
    print() 

    # --- Final IO Phase ---
    logging.info("Normalizing and Saving to disk...")
    
    # Global Normalization (Safety)
    max_amp = np.max(np.abs(full_buffer))
    if max_amp > 0:
        full_buffer = full_buffer / max_amp * 0.707
    
    try:
        wav.write(OUTPUT_FILE, SAMPLE_RATE, full_buffer)
        logging.info(f"Done. Saved to {OUTPUT_FILE}")
    except IOError as e:
        logging.error(f"Write failed: {e}")

if __name__ == "__main__":
    main()