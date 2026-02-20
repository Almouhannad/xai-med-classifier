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

    subparsers = parser.add_subparsers(dest="command")

    download_parser = subparsers.add_parser(
        "download-data",
        help="Download MedMNIST splits into a local data directory.",
    )
    download_parser.add_argument("--dataset", default="pathmnist", help="MedMNIST dataset name")
    download_parser.add_argument("--data-dir", default="data", help="Output directory")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the xaimed CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config:
        load_yaml_config(args.config)

    if args.command == "download-data":
        from xaimed.data.medmnist import download_medmnist

        counts = download_medmnist(dataset_name=args.dataset, data_dir=args.data_dir)
        print(f"Downloaded '{args.dataset}' into {args.data_dir}")
        for split, count in counts.items():
            print(f"  {split}: {count} samples")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())