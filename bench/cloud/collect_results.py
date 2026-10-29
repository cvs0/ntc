"""Merge cloud shard jsons into one directory and a short table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect(inputs: list[Path], dest: Path) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in inputs:
        for path in sorted(folder.glob("*.json")):
            if path.name in {"device.json", "precision.json", "cloud_split.json"}:
                continue
            try:
                row = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(row, dict) or "name" not in row:
                continue
            target = dest / path.name
            if target.is_file():
                prev = json.loads(target.read_text())
                if prev.get("done") and not row.get("done"):
                    continue
            target.write_text(json.dumps(row, indent=2))
            if row.get("done"):
                rows.append(row)
    rows.sort(key=lambda item: str(item.get("name")))
    summary = []
    for row in rows:
        summary.append({
            "name": row.get("name"),
            "psnr_u8": row.get("psnr_u8"),
            "total_bpp": row.get("total_bpp"),
            "psnr_u8_init": row.get("psnr_u8_init"),
            "steps_ran": row.get("steps_ran"),
        })
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = collect(args.inputs, args.out)
    print(f"done {len(rows)} -> {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
