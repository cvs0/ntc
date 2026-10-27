"""Download CC0 materials used by the benchmark. Nothing here is committed."""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

POLY = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/{name}/{name}_{map}_1k.jpg"
AMBIENT = "https://ambientcg.com/get?file={name}_1K-JPG.zip"


def _get(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1000:
        return
    urllib.request.urlretrieve(url, dest)


def polyhaven(root: Path, name: str, maps: dict[str, str]) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for role, token in maps.items():
        fn = f"{role}.jpg"
        _get(POLY.format(name=name, map=token), d / fn)
        manifest[role] = fn
    manifest["source"] = f"https://polyhaven.com/a/{name}"
    manifest["license"] = "CC0"
    manifest["normal_convention"] = "opengl"
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))


def ambient(root: Path, name: str) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    zpath = d / f"{name}.zip"
    _get(AMBIENT.format(name=name), zpath)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(d)
    # ambientCG names vary. Point the manifest at the extracted JPGs.
    color = next(d.glob("*Color.jpg"))
    normal = next(d.glob("*NormalGL.jpg"))
    rough = next(d.glob("*Roughness.jpg"))
    ao = next(d.glob("*AmbientOcclusion.jpg"))
    height = next(d.glob("*Displacement.jpg"), None)
    manifest = {
        "albedo": color.name,
        "normal": normal.name,
        "roughness": rough.name,
        "ao": ao.name,
        "source": f"https://ambientcg.com/view?id={name}",
        "license": "CC0",
        "normal_convention": "opengl",
    }
    if height:
        manifest["height"] = height.name
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, default=Path("data/materials"))
    args = p.parse_args()
    polyhaven(
        args.dest,
        "metal_plate",
        {"albedo": "diff", "normal": "nor_gl", "roughness": "rough", "metalness": "metal", "ao": "ao"},
    )
    ambient(args.dest, "PavingStones070")
    print(args.dest)


if __name__ == "__main__":
    main()
