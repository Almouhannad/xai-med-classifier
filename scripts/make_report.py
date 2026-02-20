"""Script entrypoint for report generation."""

from __future__ import annotations

import sys

from xaimed.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["report", *sys.argv[1:]]))
