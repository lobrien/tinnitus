# Installation Guide

This project requires a Python environment (>=3.13) and the `ffmpeg` system binaries.

## 1. Install System Dependencies (FFmpeg)

The `tinnitus verify` command uses `ffmpeg` to pipe raw float32 audio data for spectral analysis. You must install it before using the verification tools. You may still generate soundfiles in the absence of FFMpeg.

### macOS
Use Homebrew:

```bash
brew install ffmpeg
```

Ubuntu / Debian
```bash
sudo apt update
sudo apt install ffmpeg
```

Windows

* Option A: Winget (Recommended)

PowerShell
`winget install Gyan.FFmpeg`

Note: You may need to restart your terminal to update the PATH.

* Option B: Manual

Download the binaries from ffmpeg.org.

Extract the archive.

Add the bin folder to your System Environment Variables PATH.

### Verify FFmpeg Installation

Run this in your terminal to ensure the system can find the binaries:

```bash
ffmpeg -version
ffprobe -version
```

## 2. Install the Package
I recommend using uv for fast, strictly managed Python environments, though standard pip is supported.

### Using uv (Recommended)

1. Clone
```bash
git clone [https://github.com/yourusername/tinnitus.git](https://github.com/yourusername/tinnitus.git)
cd tinnitus
```

2. Initialize Virtual Environment (if not already active)
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install in Editable Mode
```bash
uv pip install -e .
```

## Using Standard pip

```bash
cd tinnitus
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 3. Verification
Run the help command to ensure the CLI is linked correctly:

```bash
tinnitus --help
```