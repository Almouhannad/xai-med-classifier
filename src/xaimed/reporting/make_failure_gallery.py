"""Failure gallery generation placeholder."""

from __future__ import annotations

from pathlib import Path


def build_failure_gallery(output_dir: str | Path) -> Path:
    """Create a placeholder failure gallery artifact path."""
    path = Path(output_dir) / "failure_gallery.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Failure Gallery\n\nTo be generated in a later task.\n", encoding="utf-8")
    return path
