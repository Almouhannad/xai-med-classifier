"""Script entrypoint for model training."""

from __future__ import annotations

import sys

from xaimed.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
