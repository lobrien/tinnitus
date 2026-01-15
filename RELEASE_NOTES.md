# Release Notes

## [0.1.0] - 2026-01-15

### Initial Dev
First release of notched noise generator. This version consolidates previous scripts into a unified Python package with a dedicated CLI.

### 🚀 Features
* **Unified CLI:** Single entry point `tinnitus` with subcommands `generate` and `verify`.
* **Spectral Strategies:** Validated generation of White, Pink ($1/f$), and Brown ($1/f^2$) noise.
* **Notch Filtering:** Configurable center frequency and Q-factor for targeted tinnitus investigation.
* **Analysis Tool:** Built-in `verify` command to measure notch depth and generate spectral plots.
* **Modern Audio IO:** Defaults to high-quality FLAC output (float32), with support for WAV and OGG.

### 🛠 Improvements
* **Physics Verification:** Added unit tests to verify the spectral slope of generated noise colors.
* **Zero-Phase Filtering:** Implementation now uses `filtfilt` to prevent phase distortion in the generated audio.

### ⚠️ System Requirements
* **FFmpeg:** The `verify` module requires `ffmpeg` and `ffprobe` binaries to be available in the system PATH.

### 📦 Installation
```bash
uv pip install -e .
```
