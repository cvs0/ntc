"""Print two gpu_bench result files side by side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    args = p.parse_args()
    A = _load(args.a)
    B = _load(args.b)
    print("| field | A | B |")
    print("| --- | --- | --- |")
    print(f"| platform | {A.get('platform')} | {B.get('platform')} |")
    print(f"| rtxntc | {A.get('rtxntc', {}).get('reason', '')} | {B.get('rtxntc', {}).get('reason', '')} |")
    ra = {r["api"]: r for r in A.get("runs", [])}
    rb = {r["api"]: r for r in B.get("runs", [])}
    for api in sorted(set(ra) | set(rb)):
        ta = ra.get(api, {}).get("text", "").replace("\n", " ")
        tb = rb.get(api, {}).get("text", "").replace("\n", " ")
        print(f"| {api} | {ta} | {tb} |")


if __name__ == "__main__":
    main()
