"""Host-side benchmark entry. The cross-vendor GPU script is bench/gpu_bench.py."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ntc-bench")
    p.add_argument("script", nargs="?", default="gpu", choices=["gpu", "blocks", "download"])
    args, rest = p.parse_known_args(argv)
    root = Path(__file__).resolve().parents[2]
    mapping = {
        "gpu": root / "bench" / "gpu_bench.py",
        "blocks": root / "bench" / "block_baseline.py",
        "download": root / "bench" / "download_materials.py",
    }
    sys.argv = [str(mapping[args.script]), *rest]
    runpy.run_path(str(mapping[args.script]), run_name="__main__")
