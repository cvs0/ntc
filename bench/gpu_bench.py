"""Decode benchmark and screenshot driver.

``--software`` is the only path that sets ``VK_DRIVER_FILES`` to lavapipe and
``LIBGL_ALWAYS_SOFTWARE``. A normal run uses the driver the loader picks.
Rows tagged ``not_a_performance_claim`` are CPU devices. 1920x1080 and
3840x2160 are skipped there.

NTCB block files are written by ``bench/bc_pack.py`` (BC7 mode 6, BC1, BC5).
That is not the ``bc7enc`` encoder used for the rate-distortion table.
``astcenc`` is recorded and skipped unless it is on ``PATH``. RTXNTC is probed
and not required.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from bc_pack import write_material_blocks
from make_shots import write_material

from ntc.encode import TrainConfig, encode_arrays
from ntc.materials import load_material


def _run(cmd: list[str], env: dict | None = None) -> tuple[int, str]:
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, capture_output=True, text=True, env=e)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _slug(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    text = "".join(out).strip("-")
    return (text or "gpu")[:80]


def _kv_line(line: str) -> tuple[str, str] | None:
    if " " not in line:
        return None
    key, value = line.split(" ", 1)
    if key in {
        "device_name",
        "vendor_id",
        "device_id",
        "driver_raw",
        "api_version",
        "subgroup_size",
        "timestamp_period_ns",
        "device_index",
        "device_count",
    }:
        return key, value.strip()
    return None


def parse_bench(text: str) -> dict:
    meta = {}
    timings = []
    wrote = []
    for line in text.splitlines():
        kv = _kv_line(line)
        if kv:
            meta[kv[0]] = kv[1]
        if line.startswith("gl vendor "):
            meta["gl_line"] = line
            m = re.search(r"vendor (.*) renderer (.*) version (.*)", line)
            if m:
                meta["gl_vendor"] = m.group(1).strip()
                meta["gl_renderer"] = m.group(2).strip()
                meta["gl_version"] = m.group(3).strip()
        if line.startswith("gl_subgroup "):
            meta["gl_subgroup"] = line.split()[-1]
        if line.startswith("gl_software "):
            meta["gl_software"] = line.split()[-1]
        if line.startswith("bench "):
            parts = line.split()
            rec = {"name": parts[1], "raw": line}
            if "skipped" in parts:
                rec["skipped"] = " ".join(parts[parts.index("skipped") + 1 :])
            else:
                tokens = parts[2:]
                if tokens and tokens[-1] == "not_a_performance_claim":
                    rec["not_a_performance_claim"] = True
                    tokens = tokens[:-1]
                kv = {}
                for i in range(0, len(tokens) - 1, 2):
                    kv[tokens[i]] = tokens[i + 1]
                for key in ("median_ms", "p95_ms", "texels_s", "gb_s"):
                    if key in kv:
                        rec[key] = float(kv[key])
            timings.append(rec)
        if line.startswith("wrote "):
            wrote.append(line[len("wrote ") :])
    return {"meta": meta, "timings": timings, "wrote": wrote}


def probe_rtxntc() -> dict:
    exe = shutil.which("ntc-cli")
    if not exe:
        return {
            "available": False,
            "reason": (
                "ntc-cli is not on PATH. Compression in NVIDIA-RTX/RTXNTC requires an "
                "NVIDIA Turing-or-newer GPU and CUDA. Decompression of an existing file "
                "can pass --no-coopVec. This run did not invoke it."
            ),
        }
    return {"available": True, "path": exe, "reason": "ntc-cli is on PATH. This script does not invoke it."}


def probe_astc() -> dict:
    exe = shutil.which("astcenc") or shutil.which("astcenc-avx2") or shutil.which("astcenc-native")
    if not exe:
        return {
            "available": False,
            "reason": "astcenc is not on PATH. ASTC screenshot panels are skipped. The rate table still uses the earlier astcenc 5.7.0 numbers.",
        }
    return {"available": True, "path": exe, "reason": "astcenc is on PATH. Screenshot panels still use BC7 mode 6 from bc_pack.py. The rate table is unchanged."}


def _encode_material(manifest: Path, dest: Path, bpp: float) -> dict:
    images, sems, info = load_material(manifest, 256)
    cfg = TrainConfig(
        steps=40,
        target_bpp=bpp,
        log_every=10**9,
        latent_deflate=True,
        mip_levels=4,
        mip_loss="footprint",
    )
    blob, stats = encode_arrays(images, sems, cfg)
    dest.write_bytes(blob)
    ref = images.transpose(1, 2, 0)
    return {"stats": stats, "semantics": sems, "reference": ref, "info": info}


def _md(report: dict) -> str:
    g = report.get("gpu") or {}
    lines = [
        "| field | value |",
        "| --- | --- |",
        f"| os | {report.get('os')} |",
        f"| cpu | {report.get('cpu')} |",
        f"| gpu | {g.get('name')} |",
        f"| vendor_id | {g.get('vendor_id')} |",
        f"| device_id | {g.get('device_id')} |",
        f"| driver | {g.get('driver')} |",
        f"| vulkan_api | {g.get('api')} |",
        f"| subgroup_size | {g.get('subgroup_size')} |",
        f"| opengl | {report.get('opengl', {}).get('renderer')} |",
        f"| opengl_version | {report.get('opengl', {}).get('version')} |",
        f"| rtxntc | {report.get('rtxntc', {}).get('reason')} |",
        f"| astc | {report.get('astc', {}).get('reason')} |",
        f"| bc_encoder | {report.get('bc_encoder')} |",
        "",
        "| material | api | pass | median_ms | p95_ms | texels/s | GB/s | note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in report.get("runs", []):
        for t in run.get("timings", []):
            note = t.get("skipped") or ("not_a_performance_claim" if t.get("not_a_performance_claim") else "")
            lines.append(
                "| {mat} | {api} | {name} | {med} | {p95} | {tps} | {gbs} | {note} |".format(
                    mat=run.get("material"),
                    api=run.get("api"),
                    name=t.get("name"),
                    med=t.get("median_ms", ""),
                    p95=t.get("p95_ms", ""),
                    tps=t.get("texels_s", ""),
                    gbs=t.get("gb_s", ""),
                    note=note,
                )
            )
    if report.get("software"):
        lines.append("")
        lines.append("Software-renderer rows are not a performance result.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--software", action="store_true")
    p.add_argument("--encode", type=Path, default=None, help="materials directory with metal_plate and PavingStones070")
    p.add_argument("--ntc", type=Path, action="append", default=[])
    p.add_argument("--vk-bin", type=Path, default=Path("build/ntc_vk"))
    p.add_argument("--gl-bin", type=Path, default=Path("build/ntc_gl"))
    p.add_argument("--shader-dir", type=Path, default=Path("build/shaders"))
    p.add_argument("--gl-shader-dir", type=Path, default=Path("shaders"))
    p.add_argument("--out", type=Path, default=Path("results/hw"))
    p.add_argument("--shots", type=Path, default=Path("screenshots"))
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--bpp", type=float, default=16.0)
    args = p.parse_args()

    sw_env = {}
    if args.software:
        sw_env = {
            "DISPLAY": "",
            "VK_DRIVER_FILES": os.environ.get(
                "NTC_LAVAPIPE_ICD", "/usr/share/vulkan/icd.d/lvp_icd.json"
            ),
            "LIBGL_ALWAYS_SOFTWARE": "1",
        }

    jobs = []
    if args.encode:
        for name in ("metal_plate", "PavingStones070"):
            manifest = args.encode / name
            if not (manifest / "manifest.json").is_file() and not manifest.is_file():
                jobs.append({"material": name, "error": f"missing {manifest}"})
                continue
            ntc_path = Path("results") / "hw_encode" / f"{name}.ntc"
            ntc_path.parent.mkdir(parents=True, exist_ok=True)
            packed = _encode_material(manifest if (manifest / "manifest.json").is_file() else manifest, ntc_path, args.bpp)
            bc_dir = ntc_path.parent / name
            bc_info = write_material_blocks(packed["reference"], bc_dir, packed["semantics"])
            jobs.append(
                {
                    "material": name,
                    "ntc": ntc_path,
                    "semantics": packed["semantics"],
                    "reference": packed["reference"],
                    "bc_dir": bc_dir,
                    "bc_info": bc_info,
                    "psnr_u8": packed["stats"]["metrics"]["psnr_u8"],
                    "total_bpp": packed["stats"]["total_bpp"],
                }
            )
    for path in args.ntc:
        jobs.append({"material": path.stem, "ntc": path, "semantics": [], "reference": None, "bc_dir": None})

    runs = []
    gpu = {}
    gl_info = {}
    shot_dir = None
    hw_dir = None
    for job in jobs:
        if "error" in job or "ntc" not in job:
            runs.append(job)
            continue
        bc_dir = job.get("bc_dir")
        bc7 = str(bc_dir / "albedo_bc7.ntcb") if bc_dir else ""
        bc1 = str(bc_dir / "albedo_bc1.ntcb") if bc_dir else ""
        bc5 = str(bc_dir / "normal_bc5.ntcb") if bc_dir else ""
        for api, binary, shader, extra in (
            ("vulkan", args.vk_bin, args.shader_dir, []),
            ("opengl", args.gl_bin, args.gl_shader_dir, []),
        ):
            if not binary.is_file():
                runs.append({"material": job["material"], "api": api, "returncode": None, "error": f"missing {binary}"})
                continue
            read_dir = Path("results") / "hw_encode" / f"{job['material']}_{api}"
            cmd = [
                str(binary),
                str(job["ntc"]),
                str(shader),
                "--bench",
                str(read_dir),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
            ]
            if bc7:
                cmd += ["--bc7", bc7, "--bc1", bc1, "--bc5", bc5]
            code, text = _run(cmd, env=sw_env or None)
            parsed = parse_bench(text)
            if api == "vulkan" and not gpu and parsed["meta"].get("device_name"):
                gpu = {
                    "name": parsed["meta"].get("device_name"),
                    "vendor_id": parsed["meta"].get("vendor_id"),
                    "device_id": parsed["meta"].get("device_id"),
                    "driver": parsed["meta"].get("driver_raw"),
                    "api": parsed["meta"].get("api_version"),
                    "subgroup_size": parsed["meta"].get("subgroup_size"),
                    "timestamp_period_ns": parsed["meta"].get("timestamp_period_ns"),
                }
                slug = _slug(gpu["name"])
                hw_dir = args.out / slug
                shot_dir = args.shots / slug
                hw_dir.mkdir(parents=True, exist_ok=True)
                shot_dir.mkdir(parents=True, exist_ok=True)
            if api == "opengl" and parsed["meta"].get("gl_renderer"):
                gl_info = {
                    "vendor": parsed["meta"].get("gl_vendor"),
                    "renderer": parsed["meta"].get("gl_renderer"),
                    "version": parsed["meta"].get("gl_version"),
                    "subgroup": parsed["meta"].get("gl_subgroup"),
                }
                if hw_dir is None:
                    slug = _slug(parsed["meta"]["gl_renderer"])
                    hw_dir = args.out / slug
                    shot_dir = args.shots / slug
                    hw_dir.mkdir(parents=True, exist_ok=True)
                    shot_dir.mkdir(parents=True, exist_ok=True)
            if hw_dir is not None:
                (hw_dir / f"{job['material']}_{api}.txt").write_text(text)
            runs.append(
                {
                    "material": job["material"],
                    "api": api,
                    "returncode": code,
                    "timings": parsed["timings"],
                    "wrote": parsed["wrote"],
                    "psnr_u8": job.get("psnr_u8"),
                    "total_bpp": job.get("total_bpp"),
                    "bc_encoder": (job.get("bc_info") or {}).get("encoder"),
                }
            )
            if shot_dir is not None and job.get("reference") is not None and code == 0:
                write_material(shot_dir, api, job["material"], job["reference"], job["semantics"], read_dir)

    if hw_dir is None:
        hw_dir = args.out / ("software" if args.software else "unknown")
        hw_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "software": bool(args.software),
        "gpu": gpu,
        "opengl": gl_info,
        "rtxntc": probe_rtxntc(),
        "astc": probe_astc(),
        "bc_encoder": "bc_pack.py BC7 mode 6, BC1, BC5. Not bc7enc.",
        "warmup": args.warmup,
        "iters": args.iters,
        "runs": runs,
    }
    (hw_dir / "report.json").write_text(json.dumps(report, indent=2))
    (hw_dir / "report.md").write_text(_md(report))
    print(hw_dir / "report.md")


if __name__ == "__main__":
    main()
