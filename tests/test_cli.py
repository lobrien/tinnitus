import sys
import pytest
from unittest.mock import patch, ANY
from tinnitus.cli import main
from tinnitus.generator import NoiseType

def test_cli_generate_dispatch() -> None:
    """
    Verify 'tinnitus generate' calls generator.run with correct config.
    """
    test_args = ["tinnitus", "generate", "--freq", "1000", "--duration", "5"]
    
    with patch("tinnitus.generator.run") as mock_run:
        with patch.object(sys, 'argv', test_args):
            main()
            
            # Verify dispatch happened
            mock_run.assert_called_once()
            
            # Verify argument parsing correctly populated the config
            config_arg = mock_run.call_args[0][0]
            assert config_arg.center_freq == 1000.0
            assert config_arg.duration_sec == 5
            assert config_arg.noise_type == NoiseType.PINK  # Default

def test_cli_verify_dispatch() -> None:
    """
    Verify 'tinnitus verify' calls verify_notch.run with correct config.
    """
    test_args = ["tinnitus", "verify", "my_audio.wav", "--freq", "4000"]
    
    with patch("tinnitus.verify_notch.run") as mock_run:
        with patch("tinnitus.verify_notch.Path.exists", return_value=True): # Mock file check
            with patch.object(sys, 'argv', test_args):
                main()
                
                mock_run.assert_called_once()
                
                config_arg = mock_run.call_args[0][0]
                assert str(config_arg.input_file) == "my_audio.wav"
                assert config_arg.target_freq == 4000.0

def test_cli_help() -> None:
    """
    Verify the help command exits cleanly (SystemExit 0).
    """
    with patch.object(sys, 'argv', ["tinnitus", "--help"]):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0

def test_cli_invalid_command() -> None:
    """
    Verify invalid subcommands exit with error (SystemExit 2).
    """
    with patch.object(sys, 'argv', ["tinnitus", "make-noise"]):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code != 0