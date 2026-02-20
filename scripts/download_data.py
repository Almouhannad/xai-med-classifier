"""Download MedMNIST data splits to a local directory."""

from __future__ import annotations

import sys

from xaimed.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["download-data", *sys.argv[1:]]))