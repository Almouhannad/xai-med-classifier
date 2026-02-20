"""Command-line interface for xaimed."""

from __future__ import annotations

import argparse
from typing import Any, Sequence

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

    subparsers.add_parser("train", help="Train model and save checkpoints.")
    subparsers.add_parser("explain", help="Generate explanation artifacts.")
    subparsers.add_parser("report", help="Build report artifacts.")

    return parser


def _load_config_if_needed(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    return load_yaml_config(config_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the xaimed CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    config = _load_config_if_needed(args.config)

    if args.command == "download-data":
        from xaimed.data.medmnist import download_medmnist

        counts = download_medmnist(dataset_name=args.dataset, data_dir=args.data_dir)
        print(f"Downloaded '{args.dataset}' into {args.data_dir}")
        for split, count in counts.items():
            print(f"  {split}: {count} samples")
        return 0

    if args.command == "train":
        from xaimed.train import run_training

        result = run_training(config)
        print(f"Best checkpoint: {result.best_checkpoint_path}")
        print(f"Last checkpoint: {result.last_checkpoint_path}")
        return 0

    if args.command == "explain":
        print("Explain command is not implemented yet.")
        return 0

    if args.command == "report":
        print("Report command is not implemented yet.")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())