"""Compare this encoder with RTXNTC ``ntc-cli`` on three 2048 ambientCG sets.

RTXNTC is compressed at ``--bitsPerPixel`` 2.5, 5, 10, 15, and 20. That
flag is a latent-grid rate for mip 0 and does not include the MLP. Ours is
trained at the same numbers with ``rate="latent"`` (codes + fp16 latent
scales, per mip 0 texel, no MLP weights). Both decoded chains are scored by
``bench/score_mips.py``. The report also stores ``file_bpp_chain`` and
``bppc`` for each file.

Published points from Vaidyanathan et al. 2023, Table 3, are drawn on the
bppc axis. They are a 20-material average, bits/pixel/channel including
mips and weights, on a different material set. They are not a matched
comparison.

Decode timing is the wall time of ``ntc-cli --vk --decompress`` with
cooperative vectors left on, and again with ``--no-coopVec``. One warmup
run is dropped. The median of the timed runs is stored. Stdout is kept.
This is not a GPU timestamp query.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import time

import numpy as np
from PIL import Image

from ntc.container import (
    SEM_ALBEDO_B,
    SEM_ALBEDO_G,
    SEM_ALBEDO_R,
    SEM_AO,
    SEM_HEIGHT,
    SEM_METALNESS,
    SEM_NORMAL_X,
    SEM_NORMAL_Y,
    SEM_NORMAL_Z,
    SEM_ROUGHNESS,
)
from ntc.decode_ref import decode_mip
from ntc.encode import TrainConfig, encode_arrays
from ntc.materials import export_reference_mips, load_material

# Vaidyanathan et al. 2023, Table 3. 20-material average.
# bits/pixel/channel including mips and weights. Different materials.
PAPER_TABLE3 = (
    (0.23, 32.71),
    (0.56, 36.12),
    (1.13, 39.92),
    (2.51, 45.30),
)
PAPER_NOTE = (
    "Vaidyanathan et al. 2023 Table 3, 20-material average, "
    "bits/pixel/channel including mips and weights. "
    "A different material set, not a matched comparison."
)
SDK_NAMES = ("MetalPlates013_2K", "PavingStones070_2K", "Tiles135D_2K")
SDK_RATES = (2.5, 5.0, 10.0, 15.0, 20.0)
_ROLE = {
    SEM_ALBEDO_R: ("albedo", 0),
    SEM_ALBEDO_G: ("albedo", 1),
    SEM_ALBEDO_B: ("albedo", 2),
    SEM_NORMAL_X: ("normal", 0),
    SEM_NORMAL_Y: ("normal", 1),
    SEM_NORMAL_Z: ("normal", 2),
    SEM_ROUGHNESS: ("roughness", 0),
    SEM_METALNESS: ("metalness", 0),
    SEM_AO: ("ao", 0),
    SEM_HEIGHT: ("height", 0),
}


def slug_gpu() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=30,
        )
        name = out.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        name = os.environ.get("GPU_NAME", "unknown")
    text = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return text[:80] or "unknown"


def _svg(path: Path, ours: list[dict], rtx: list[dict]) -> None:
    def panel(x0, title, key, paper):
        pts_o = [(r[key], r["psnr_u8"]) for r in ours if r.get(key) and r.get("psnr_u8")]
        pts_r = [(r[key], r["psnr_u8"]) for r in rtx if r.get(key) and r.get("psnr_u8")]
        xs = [p[0] for p in pts_o + pts_r] + ([p[0] for p in paper] if paper else [])
        ys = [p[1] for p in pts_o + pts_r] + ([p[1] for p in paper] if paper else [])
        if not xs:
            xs, ys = [0.0, 1.0], [30.0, 50.0]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        pad_x = (xmax - xmin) * 0.08
        pad_y = (ymax - ymin) * 0.08
        xmin -= pad_x
        xmax += pad_x
        ymin -= pad_y
        ymax += pad_y

        def xy(x, y):
            px = x0 + 50 + (x - xmin) / (xmax - xmin) * 320
            py = 40 + (1.0 - (y - ymin) / (ymax - ymin)) * 220
            return px, py

        parts = [f'<text x="{x0 + 50}" y="24" font-size="13">{title}</text>']
        for series, color in ((pts_o, "#1d4ed8"), (pts_r, "#b45309")):
            if len(series) >= 2:
                d = " ".join(f"{xy(x, y)[0]:.1f},{xy(x, y)[1]:.1f}" for x, y in sorted(series))
                parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{d}"/>')
            for x, y in series:
                px, py = xy(x, y)
                parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>')
        if paper:
            for x, y in paper:
                px, py = xy(x, y)
                parts.append(f'<rect x="{px - 3:.1f}" y="{py - 3:.1f}" width="6" height="6" fill="#111827"/>')
        parts.append(
            f'<text x="{x0 + 50}" y="285" font-size="11">x {xmin + pad_x:.2f}..{xmax - pad_x:.2f}  '
            f'y {ymin + pad_y:.1f}..{ymax - pad_y:.1f} dB</text>'
        )
        return "".join(parts)

    body = panel(0, "bppc (paper axis)", "bppc", PAPER_TABLE3)
    body += panel(400, "latent bpp, mip 0 (SDK axis)", "latent_bpp_mip0", ())
    pairs = " ".join(f"{b:.2f} bppc {db:.2f} dB" for b, db in PAPER_TABLE3)
    note = (
        "blue ours, orange RTXNTC, black squares Table 3 (different material set): " + pairs
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="820" height="320" font-family="sans-serif">'
        f"{body}<text x=\"20\" y=\"308\" font-size=\"11\">{note}</text></svg>"
    )
    path.write_text(svg)


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    wall = time.perf_counter() - t0
    text = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, text, wall


def _median_decode(cli: str, ntc: Path, dest: Path, coop: bool, repeats: int) -> dict:
    cmd = [cli, "--vk", "--loadCompressed", str(ntc), "--decompress", "--saveImages", str(dest),
           "--saveMips", "--imageFormat", "png", "--bcFormat", "none"]
    if not coop:
        cmd.append("--no-coopVec")
    if dest.exists():
        shutil.rmtree(dest)
    code, text, _warm = _run(cmd)
    samples = []
    texts = [text]
    if code == 0:
        for _ in range(repeats):
            if dest.exists():
                shutil.rmtree(dest)
            code, text, wall = _run(cmd)
            texts.append(text)
            samples.append(wall)
            if code != 0:
                break
    return {
        "coopvec": coop,
        "returncode": code,
        "median_wall_s": statistics.median(samples) if samples else None,
        "samples_s": samples,
        "log_tail": texts[-1][-4000:],
    }


def _write_decoded(blob: bytes, semantics: list[int], dest: Path) -> None:
    from ntc.container import parse_texture_set

    ts = parse_texture_set(blob)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for mip in range(ts.num_mips):
        img = decode_mip(ts, mip)
        planes: dict[str, list[np.ndarray | None]] = {}
        for channel, sem in enumerate(semantics):
            role = _ROLE.get(sem)
            if role is None:
                continue
            name, slot = role
            bucket = planes.setdefault(name, [None, None, None])
            bucket[slot] = img[:, :, channel]
        for name, slots in planes.items():
            used = [s for s in slots if s is not None]
            if not used:
                continue
            stack = np.stack(used, axis=2)
            q = np.clip(np.round(np.clip(stack, 0, 1) * 255), 0, 255).astype(np.uint8)
            if q.shape[2] == 1:
                image = Image.fromarray(q[:, :, 0], mode="L")
            else:
                image = Image.fromarray(q, mode="RGB")
            if mip == 0:
                path = dest / f"{name}.png"
            else:
                path = dest / "mips" / f"{name}.{mip:02d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)


def bd_rate(
    anchor_rate: list[float],
    anchor_psnr: list[float],
    test_rate: list[float],
    test_psnr: list[float],
) -> float:
    """Bjontegaard rate difference of ``test`` against ``anchor``, in percent.

    Positive means ``test`` spends more rate for the same PSNR. Identical
    curves are 0. The integral is over the overlapping PSNR range.
    """
    anchor_r = np.log(np.asarray(anchor_rate, dtype=np.float64))
    anchor_p = np.asarray(anchor_psnr, dtype=np.float64)
    test_r = np.log(np.asarray(test_rate, dtype=np.float64))
    test_p = np.asarray(test_psnr, dtype=np.float64)
    order_a = np.argsort(anchor_p)
    order_t = np.argsort(test_p)
    anchor_r, anchor_p = anchor_r[order_a], anchor_p[order_a]
    test_r, test_p = test_r[order_t], test_p[order_t]
    low = float(max(anchor_p[0], test_p[0]))
    high = float(min(anchor_p[-1], test_p[-1]))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-6:
        return float("nan")
    degree = int(min(3, len(anchor_p) - 1, len(test_p) - 1))
    poly_a = np.polyfit(anchor_p, anchor_r, degree)
    poly_t = np.polyfit(test_p, test_r, degree)
    int_a = np.polyint(poly_a)
    int_t = np.polyint(poly_t)
    area = (np.polyval(int_t, high) - np.polyval(int_t, low)) - (
        np.polyval(int_a, high) - np.polyval(int_a, low)
    )
    return float((np.exp(area / (high - low)) - 1.0) * 100.0)


def summary_table(ours: list[dict], rtx: list[dict]) -> dict:
    """One row per material and a BD-rate of ntc against RTXNTC on BPPC.

    RTXNTC rows come from the published ``ntc-cli`` only. Missing rows are
    omitted. ``bppc`` includes the MLP. ``latent_bpp_mip0`` does not.
    """
    by_material: dict[str, dict[str, list]] = {}
    for row in ours:
        if row.get("error") or row.get("bppc") is None or row.get("psnr_u8") is None:
            continue
        bucket = by_material.setdefault(str(row["material"]), {"ours_bppc": [], "ours_psnr": [], "rtx_bppc": [], "rtx_psnr": []})
        bucket["ours_bppc"].append(float(row["bppc"]))
        bucket["ours_psnr"].append(float(row["psnr_u8"]))
    for row in rtx:
        if row.get("error") or row.get("bppc") is None or row.get("psnr_u8") is None:
            continue
        bucket = by_material.setdefault(str(row["material"]), {"ours_bppc": [], "ours_psnr": [], "rtx_bppc": [], "rtx_psnr": []})
        bucket["rtx_bppc"].append(float(row["bppc"]))
        bucket["rtx_psnr"].append(float(row["psnr_u8"]))
    materials = []
    for name, cols in sorted(by_material.items()):
        rate = None
        if len(cols["ours_bppc"]) >= 2 and len(cols["rtx_bppc"]) >= 2:
            rate = bd_rate(cols["rtx_bppc"], cols["rtx_psnr"], cols["ours_bppc"], cols["ours_psnr"])
        materials.append({"material": name, "bd_rate_percent": rate, "ours": len(cols["ours_bppc"]), "rtx": len(cols["rtx_bppc"])})
    return {"materials": materials, "note": "bd_rate_percent is ntc versus ntc-cli on bppc. Positive means ntc uses more rate."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--materials", type=Path, default=Path("data/materials"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--ntc-cli", type=Path, default=None)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--skip-rtx", action="store_true")
    p.add_argument("--skip-ours", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    slug = slug_gpu()
    out = args.out or Path("results/rtxntc_compare") / slug
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        _svg(out / "rd.svg", [], [])
        (out / "paper.json").write_text(json.dumps({"note": PAPER_NOTE, "table3": PAPER_TABLE3}, indent=2))
        print("wrote", out / "rd.svg")
        return

    from bench.score_mips import score_tree

    rows_ours = []
    rows_rtx = []
    for name in SDK_NAMES:
        src = args.materials / name
        ref = src / "reference"
        if not (src / "manifest.json").is_file():
            rows_ours.append({"material": name, "error": f"missing {src}. run download_materials.py --preset rtxntc"})
            continue
        if not (ref / "albedo.png").is_file():
            export_reference_mips(src, ref, 2048)
        images, sems, _info = load_material(src, 2048)
        for rate in SDK_RATES:
            tag = f"{name}_b{rate}"
            if not args.skip_ours:
                cfg = TrainConfig(
                    steps=args.steps,
                    target_bpp=rate,
                    rate="latent",
                    seed=0,
                    latent_coder="best",
                    latent_deflate=True,
                    mip_levels=12,
                    mip_loss="footprint",
                    warmup_steps=400,
                    schedule="cosine",
                    eval_every=500,
                    early_stop=6,
                    batch_pixels=1024 * 1024,
                    device="auto",
                    with_ssim=False,
                )
                t0 = time.perf_counter()
                blob, stats = encode_arrays(images, sems, cfg)
                wall = time.perf_counter() - t0
                ntc_path = out / "ours" / f"{tag}.ntc"
                ntc_path.parent.mkdir(parents=True, exist_ok=True)
                ntc_path.write_bytes(blob)
                decoded = out / "ours" / tag
                _write_decoded(blob, sems, decoded)
                scored = score_tree(ref, decoded)
                row = {
                    "tool": "ours",
                    "material": name,
                    "requested_latent_bpp": rate,
                    "psnr_u8": scored["psnr_u8"],
                    "groups": scored["groups"],
                    "mips": scored["mips"],
                    "latent_bpp_mip0": stats["latent_bpp_mip0"],
                    "file_bpp_chain": stats["file_bpp_chain"],
                    "bppc": stats["bppc"],
                    "total_bpp": stats["total_bpp"],
                    "train_s": wall,
                    "ntc": str(ntc_path),
                }
                rows_ours.append(row)
                print(
                    f"ours {tag} psnr {row['psnr_u8']} latent {row['latent_bpp_mip0']:.3f} "
                    f"bppc {row['bppc']:.3f}",
                    flush=True,
                )
            if args.skip_rtx:
                continue
            cli = str(args.ntc_cli) if args.ntc_cli else shutil.which("ntc-cli")
            if not cli:
                rows_rtx.append({"tool": "rtxntc", "material": name, "error": "ntc-cli not found"})
                continue
            comp = out / "rtxntc" / f"{tag}.ntc"
            comp.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                cli, "--loadImages", str(ref), "--loadMips", "--compress",
                "--bitsPerPixel", str(rate), "--saveCompressed", str(comp),
            ]
            code, text, wall = _run(cmd)
            if code != 0 or not comp.is_file():
                rows_rtx.append({"tool": "rtxntc", "material": name, "rate": rate, "error": text[-2000:], "returncode": code})
                continue
            dec_dir = out / "rtxntc" / tag
            timing_on = _median_decode(cli, comp, dec_dir, True, args.repeats)
            timing_off = _median_decode(cli, comp, dec_dir, False, args.repeats)
            # The off run overwrites the images. Score that tree if it exists,
            # otherwise the coopvec tree. The pixels should match.
            scored = score_tree(ref, dec_dir) if (dec_dir / "albedo.png").is_file() else {"psnr_u8": None, "groups": {}, "mips": []}
            file_bytes = comp.stat().st_size
            chain = sum((max(2048 >> m, 1) ** 2) for m in range(12))
            channels = images.shape[0]
            rows_rtx.append({
                "tool": "rtxntc",
                "material": name,
                "requested_latent_bpp": rate,
                "latent_bpp_mip0": rate,
                "file_bytes": file_bytes,
                "file_bpp_chain": 8.0 * file_bytes / float(chain),
                "bppc": 8.0 * file_bytes / float(chain * channels),
                "psnr_u8": scored.get("psnr_u8"),
                "groups": scored.get("groups"),
                "mips": scored.get("mips"),
                "compress_wall_s": wall,
                "decode_coopvec": timing_on,
                "decode_no_coopvec": timing_off,
                "compress_log_tail": text[-2000:],
                "note": "latent_bpp_mip0 is the --bitsPerPixel request. file_bpp_chain and bppc use the whole .ntc file.",
            })
            print(f"rtx {tag} psnr {scored.get('psnr_u8')} file_bpp_chain {rows_rtx[-1]['file_bpp_chain']:.3f}", flush=True)

    report = {
        "gpu": slug,
        "paper": {"note": PAPER_NOTE, "table3_bppc_db": PAPER_TABLE3},
        "ours": rows_ours,
        "rtxntc": rows_rtx,
        "definitions": {
            "latent_bpp_mip0": "stored latent codes + fp16 latent scales, per mip 0 texel. no MLP, no header",
            "file_bpp_chain": "whole file bits / texels in the declared mip chain",
            "bppc": "payload bits / (mip-chain texels * channels). paper definition. RTXNTC bppc here uses the whole file because the CLI does not split the payload",
            "total_bpp": "historical payload bits / mip 0 texels, weights included, header excluded, not per channel",
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    _svg(out / "rd.svg", rows_ours, rows_rtx)
    print("wrote", out / "report.json")


if __name__ == "__main__":
    main()
