"""Command-line interface for xaimed."""

from __future__ import annotations

import argparse
from typing import Sequence

from xaimed.utils.io import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="xaimed",
        description="Tools for training and explaining a medical image classifier.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a YAML config file to load.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the xaimed CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config:
        load_yaml_config(args.config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())