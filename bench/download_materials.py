"""Download CC0 materials used by the benchmark. Nothing here is committed."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

# Running ``python bench/download_materials.py`` puts bench/ on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

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


def _zip_opens(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            zf.namelist()
    except zipfile.BadZipFile:
        return False
    return True


def _one_map(directory: Path, pattern: str, material: str, role: str) -> Path:
    found = next(directory.glob(pattern), None)
    if found is None:
        raise RuntimeError(f"{material}: missing map {role}")
    return found


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


def ambient(root: Path, name: str, size: str = "1K") -> Path:
    d = root / name if size == "1K" else root / f"{name}_{size}"
    if (d / "manifest.json").is_file():
        return d
    d.mkdir(parents=True, exist_ok=True)
    zpath = d / f"{name}.zip"
    url = AMBIENT.format(name=name, size=size)
    opened = False
    for _attempt in (1, 2):
        if not _zip_opens(zpath):
            zpath.unlink(missing_ok=True)
            _get(url, zpath)
        try:
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(d)
            opened = True
            break
        except zipfile.BadZipFile:
            zpath.unlink(missing_ok=True)
    if not opened:
        raise RuntimeError(f"{name}: zip failed to open after retry")
    # ambientCG names vary. Point the manifest at the extracted JPGs.
    color = _one_map(d, "*Color.jpg", name, "Color")
    normal = _one_map(d, "*NormalGL.jpg", name, "NormalGL")
    rough = _one_map(d, "*Roughness.jpg", name, "Roughness")
    ao = next(d.glob("*AmbientOcclusion.jpg"), None)
    metal = next(d.glob("*Metalness.jpg"), None)
    if metal is None:
        metal = next(d.glob("*Metallic.jpg"), None)
    height = next(d.glob("*Displacement.jpg"), None)
    opacity = next(d.glob("*Opacity.jpg"), None)
    manifest = {
        "albedo": color.name,
        "normal": normal.name,
        "roughness": rough.name,
        "source": f"https://ambientcg.com/view?id={name}",
        "license": "CC0",
        "normal_convention": "opengl",
    }
    if ao:
        manifest["ao"] = ao.name
    if metal:
        manifest["metalness"] = metal.name
    if height:
        manifest["height"] = height.name
    if opacity:
        manifest["opacity"] = opacity.name
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return d


_MAPS_METAL = {"albedo": "diff", "normal": "nor_gl", "roughness": "rough", "metalness": "metal", "ao": "ao"}
_MAPS = {"albedo": "diff", "normal": "nor_gl", "roughness": "rough", "ao": "ao"}


def cloud_materials(root: Path) -> None:
    """Train and held-out ambientCG 2K sets for the 5090 queue.

    Writes ``cloud_split.json``. Held-out directories are downloaded and
    are not training jobs.
    """
    from bench.train_max_v2 import CLOUD_HOLDOUT, CLOUD_TRAIN
    from ntc.materials import export_reference_mips

    names = list(CLOUD_TRAIN) + list(CLOUD_HOLDOUT)
    for name in names:
        src = ambient(root, name, size="2K")
        export_reference_mips(src, src / "reference")
    split = {
        "train": [f"{name}_2K" for name in CLOUD_TRAIN],
        "heldout": [f"{name}_2K" for name in CLOUD_HOLDOUT],
    }
    (root / "cloud_split.json").write_text(json.dumps(split, indent=2))


def sdk_materials(root: Path) -> None:
    """ambientCG 2048 sets used by the RTXNTC SDK samples, with a full mip chain.

    The zip is ``https://ambientcg.com/get?file=<Name>_2K-JPG.zip``. The
    request sends a browser User-Agent. Mips are a 2x2 box filter, the same
    crop the trainer uses, written for ``ntc-cli --loadMips``.
    """
    from ntc.materials import export_reference_mips

    for name in ("MetalPlates013", "PavingStones070", "Tiles135D"):
        src = ambient(root, name, size="2K")
        export_reference_mips(src, src / "reference")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, default=Path("data/materials"))
    p.add_argument("--preset", choices=("base", "max", "rtxntc", "cloud"), default="base")
    p.add_argument("--dry-run", action="store_true", help="Resolve the cloud name lists and do not download.")
    args = p.parse_args()
    if args.preset == "cloud" and args.dry_run:
        from bench.train_max_v2 import CLOUD_HOLDOUT, CLOUD_TRAIN

        print(f"cloud train {len(CLOUD_TRAIN)} holdout {len(CLOUD_HOLDOUT)}")
        return
    if args.preset == "cloud":
        cloud_materials(args.dest)
        print(args.dest)
        return
    if args.preset == "rtxntc":
        sdk_materials(args.dest)
        print(args.dest)
        return
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
        sdk_materials(args.dest)
    print(args.dest)


if __name__ == "__main__":
    main()
