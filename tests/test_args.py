import pytest
from pathlib import Path
from tinnitus.generator import parse_arguments, AudioConfig, NoiseType

def test_defaults() -> None:
    """Verify default values when only required args are provided."""
    argv = ["--freq", "4000"]
    config = parse_arguments(argv)
    
    assert config.center_freq == 4000.0
    assert config.noise_type == NoiseType.PINK
    assert config.duration_sec == 60
    assert config.sample_rate == 44100
    # CHANGE: Updated expectation to match the new .flac default
    assert config.output_file == Path("output.flac")

def test_custom_configuration() -> None:
    """Verify all flags override defaults correctly."""
    argv = [
        "--type", "brown",
        "--freq", "1000",
        "--width", "2.0",
        "--duration", "10",
        "--rate", "48000",
        "--output", "test.wav"
    ]
    config = parse_arguments(argv)
    
    assert config.noise_type == NoiseType.BROWN
    assert config.center_freq == 1000.0
    # 2 octaves -> Q = sqrt(4)/(4-1) = 2/3
    assert config.notch_width_q == pytest.approx(0.6666, rel=1e-3)
    assert config.duration_sec == 10
    assert config.sample_rate == 48000
    assert config.output_file == Path("test.wav")

def test_enum_validation() -> None:
    """Argparse should exit/fail if an invalid noise type is provided."""
    with pytest.raises(SystemExit):
        parse_arguments(["--freq", "1000", "--type", "invalid_color"])

def test_missing_required_freq() -> None:
    """The --freq argument is mandatory."""
    with pytest.raises(SystemExit):
        parse_arguments([])

def test_path_handling(tmp_path : str) -> None:
    """Ensure output path is correctly converted to a Path object."""
    output_path = tmp_path / "subdir" / "out.wav"
    argv = ["--freq", "500", "--output", str(output_path)]
    config = parse_arguments(argv)
    
    assert isinstance(config.output_file, Path)
    assert config.output_file == output_path