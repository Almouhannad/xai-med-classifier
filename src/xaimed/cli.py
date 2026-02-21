"""Command-line interface for xaimed."""

from __future__ import annotations

import argparse
from pathlib import Path
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
    subparsers.add_parser("eval", help="Evaluate a checkpoint and create confusion matrix artifacts.")
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

        train_result = run_training(config)
        print(f"Best checkpoint: {train_result.best_checkpoint_path}")
        print(f"Last checkpoint: {train_result.last_checkpoint_path}")
        return 0

    if args.command == "explain":
        from xaimed.xai.explain import run_explain

        explain_result = run_explain(config)
        print(f"Explain artifacts directory: {explain_result.output_dir}")
        for path in explain_result.overlay_paths:
            print(f"Grad-CAM overlay saved: {path}")
        return 0

    if args.command == "eval":
        from xaimed.eval import run_evaluation

        eval_result = run_evaluation(config)
        print(f"Metrics saved: {eval_result.metrics_path}")
        print(f"Confusion matrix saved: {eval_result.confusion_matrix_path}")
        print(f"Accuracy: {float(eval_result.metrics['accuracy']):.4f}")
        print(f"Macro F1: {float(eval_result.metrics['macro_f1']):.4f}")
        print(f"Failure gallery CSV saved: {eval_result.failure_gallery.csv_path}")
        print(f"High-confidence wrong grid saved: {eval_result.failure_gallery.high_conf_wrong_grid_path}")
        print(f"Low-confidence correct grid saved: {eval_result.failure_gallery.low_conf_correct_grid_path}")
        training_curves_paths = getattr(eval_result, "training_curves_paths", {})
        for split_name, chart_path in sorted(training_curves_paths.items()):
            print(f"Training curves ({split_name}) saved: {chart_path}")
        return 0

    if args.command == "report":
        from xaimed.reporting import build_report

        output_dir = Path(str(config.get("report", {}).get("output_dir", "artifacts/report")))
        report_path = build_report(output_dir)
        print(f"Report scaffold saved: {report_path}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())