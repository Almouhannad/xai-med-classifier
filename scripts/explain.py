"""Script entrypoint for explainability generation."""

from __future__ import annotations

import sys

from xaimed.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["explain", *sys.argv[1:]]))
