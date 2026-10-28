"""Download CC0 materials used by the benchmark. Nothing here is committed."""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

POLY = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/{size}/{name}/{name}_{map}_{size}.jpg"
AMBIENT = "https://ambientcg.com/get?file={name}_{size}-JPG.zip"
# ambientCG returns 403 when the client sends no User-Agent.
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"


def _get(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1000:
        return
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())


def polyhaven(root: Path, name: str, maps: dict[str, str], size: str = "1k") -> None:
    d = root / name if size == "1k" else root / f"{name}_{size}"
    d.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for role, token in maps.items():
        fn = f"{role}.jpg"
        _get(POLY.format(size=size, name=name, map=token), d / fn)
        manifest[role] = fn
    manifest["source"] = f"https://polyhaven.com/a/{name}"
    manifest["license"] = "CC0"
    manifest["normal_convention"] = "opengl"
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))


def ambient(root: Path, name: str, size: str = "1K") -> None:
    d = root / name if size == "1K" else root / f"{name}_{size}"
    d.mkdir(parents=True, exist_ok=True)
    zpath = d / f"{name}.zip"
    _get(AMBIENT.format(name=name, size=size), zpath)
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


_MAPS_METAL = {"albedo": "diff", "normal": "nor_gl", "roughness": "rough", "metalness": "metal", "ao": "ao"}
_MAPS = {"albedo": "diff", "normal": "nor_gl", "roughness": "rough", "ao": "ao"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, default=Path("data/materials"))
    p.add_argument("--preset", choices=("base", "max"), default="base")
    args = p.parse_args()
    polyhaven(args.dest, "metal_plate", _MAPS_METAL)
    ambient(args.dest, "PavingStones070")
    if args.preset == "max":
        for name in (
            "wood_floor",
            "denim_fabric",
            "aerial_rocks_01",
            "forest_ground_04",
            "plastered_wall",
            "concrete_floor_02",
        ):
            polyhaven(args.dest, name, _MAPS, size="1k")
        # Two 2K sets. The directory name is ``{asset}_2k``.
        polyhaven(args.dest, "wood_floor", _MAPS, size="2k")
        polyhaven(args.dest, "aerial_rocks_01", _MAPS, size="2k")
    print(args.dest)


if __name__ == "__main__":
    main()
