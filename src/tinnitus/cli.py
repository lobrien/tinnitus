import argparse
import sys
import logging
from tinnitus import generator, verify_notch

def main() -> None:
    # Centralized Logging Config
    logging.basicConfig(
        level=logging.INFO, 
        format='%(levelname)s: %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        prog="tinnitus",
        description="Tinnitus Notch Filter Toolset"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: Generate
    gen_parser = subparsers.add_parser(
        "generate", 
        help="Generate notched spectral noise",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    generator.setup_parser(gen_parser)

    # Subcommand: Verify
    ver_parser = subparsers.add_parser(
        "verify", 
        help="Analyze notch depth in audio file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    verify_notch.setup_parser(ver_parser)

    # Parse and Dispatch
    args = parser.parse_args()

    if args.command == "generate":
        gen_config = generator.config_from_args(args)
        generator.run(gen_config)
    elif args.command == "verify":
        ver_config = verify_notch.config_from_args(args)
        verify_notch.run(ver_config)

if __name__ == "__main__":
    main()