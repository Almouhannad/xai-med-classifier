"""Report generation placeholder."""

from __future__ import annotations

from pathlib import Path


def build_report(output_dir: str | Path) -> Path:
    """Create a placeholder report artifact."""
    path = Path(output_dir) / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Report\n\nReport generation is scaffolded for later tasks.\n", encoding="utf-8")
    return path
