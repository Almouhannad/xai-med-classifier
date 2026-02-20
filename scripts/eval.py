"""Script entrypoint for model evaluation."""

from __future__ import annotations

import sys

from xaimed.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["eval", *sys.argv[1:]]))
