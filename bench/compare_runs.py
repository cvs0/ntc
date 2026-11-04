"""Print two hardware reports side by side.

Each argument is a ``results/hw/<gpu-slug>`` directory or a ``report.json`` file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    if path.is_dir():
        path = path / "report.json"
    return json.loads(path.read_text())


def _index(report: dict) -> dict[tuple, dict]:
    out = {}
    for run in report.get("runs", []):
        for t in run.get("timings", []):
            out[(run.get("material"), run.get("api"), t.get("name"))] = t
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    args = p.parse_args()
    A = _load(args.a)
    B = _load(args.b)
    ga = (A.get("gpu") or {}).get("name") or (A.get("opengl") or {}).get("renderer")
    gb = (B.get("gpu") or {}).get("name") or (B.get("opengl") or {}).get("renderer")
    print("| field | A | B |")
    print("| --- | --- | --- |")
    print(f"| gpu | {ga} | {gb} |")
    print(f"| os | {A.get('os')} | {B.get('os')} |")
    print(f"| cpu | {A.get('cpu')} | {B.get('cpu')} |")
    print(f"| driver | {(A.get('gpu') or {}).get('driver')} | {(B.get('gpu') or {}).get('driver')} |")
    print(f"| vulkan | {(A.get('gpu') or {}).get('api')} | {(B.get('gpu') or {}).get('api')} |")
    print(f"| subgroup | {(A.get('gpu') or {}).get('subgroup_size')} | {(B.get('gpu') or {}).get('subgroup_size')} |")
    print(f"| rtxntc | {A.get('rtxntc', {}).get('reason', '')} | {B.get('rtxntc', {}).get('reason', '')} |")
    ia = _index(A)
    ib = _index(B)
    keys = sorted(set(ia) | set(ib))
    if not keys:
        return
    print("| material | api | pass | A median_ms | B median_ms | A texels/s | B texels/s |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for mat, api, name in keys:
        ta = ia.get((mat, api, name), {})
        tb = ib.get((mat, api, name), {})
        print(
            f"| {mat} | {api} | {name} | {ta.get('median_ms', ta.get('skipped', ''))} | "
            f"{tb.get('median_ms', tb.get('skipped', ''))} | {ta.get('texels_s', '')} | {tb.get('texels_s', '')} |"
        )


if __name__ == "__main__":
    main()
