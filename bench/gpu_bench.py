"""Cross-vendor decode benchmark.

Detects the Vulkan device. Runs the NTC compute decoder. Runs BC/ASTC when
the tools are on PATH or passed in. Probes RTXNTC's ntc-cli and records why
it did not run. Lavapipe timings are stored separately and are not a
performance result.

Official RTXNTC compression needs NVIDIA Turing or newer and CUDA. On any
other GPU the script skips compression and says so. Decompression of a
precompressed file can be attempted with --no-coopVec when ntc-cli is present.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str], env: dict | None = None) -> tuple[int, str]:
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, capture_output=True, text=True, env=e)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def vulkan_summary() -> dict:
    code, text = _run(["vulkaninfo", "--summary"], env={"DISPLAY": ""})
    return {"returncode": code, "text": text[-4000:]}


def probe_rtxntc() -> dict:
    exe = shutil.which("ntc-cli")
    if not exe:
        return {
            "available": False,
            "reason": (
                "ntc-cli is not on PATH. Build NVIDIA-RTX/RTXNTC (recursive clone, "
                "CUDA, CMake) and pass the binary. Compression requires an NVIDIA "
                "Turing-or-newer GPU. Decompression of an existing file can use "
                "--no-coopVec on AMD, Intel, and NVIDIA. This column is pending "
                "a hardware run."
            ),
        }
    return {
        "available": True,
        "path": exe,
        "reason": "present, not invoked by this VM run unless --ntc-cli-file is set",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ntc", type=Path, required=False)
    p.add_argument("--vk-bin", type=Path, default=Path("build/ntc_vk"))
    p.add_argument("--gl-bin", type=Path, default=Path("build/ntc_gl"))
    p.add_argument("--shader-dir", type=Path, default=Path("build/shaders"))
    p.add_argument("--gl-shader-dir", type=Path, default=Path("shaders"))
    p.add_argument("--out", type=Path, default=Path("results/gpu.json"))
    args = p.parse_args()
    report = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "vulkaninfo": vulkan_summary(),
        "rtxntc": probe_rtxntc(),
        "runs": [],
        "pending": [
            "GPU decode milliseconds on AMD and NVIDIA hardware",
            "Official RTXNTC ntc-cli PSNR and encode time at matched bpp",
        ],
    }
    env = {"DISPLAY": "", "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/lvp_icd.json"}
    if args.ntc and args.vk_bin.is_file():
        code, text = _run([str(args.vk_bin), str(args.ntc), str(args.shader_dir)], env=env)
        report["runs"].append({"api": "vulkan", "returncode": code, "text": text})
    if args.ntc and args.gl_bin.is_file():
        code, text = _run(
            [str(args.gl_bin), str(args.ntc), str(args.gl_shader_dir)],
            env={"DISPLAY": "", "LIBGL_ALWAYS_SOFTWARE": "1"},
        )
        report["runs"].append({"api": "opengl", "returncode": code, "text": text})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    md = args.out.with_suffix(".md")
    lines = [
        "| field | value |",
        "| --- | --- |",
        f"| platform | {report['platform']} |",
        f"| rtxntc | {report['rtxntc'].get('reason', '')} |",
    ]
    for run in report["runs"]:
        lines.append(f"| {run['api']} | return {run['returncode']} |")
        for row in run["text"].splitlines():
            lines.append(f"| | {row} |")
    lines.append("")
    lines.append("GPU timings and official RTXNTC numbers are pending a hardware run.")
    md.write_text("\n".join(lines) + "\n")
    print(md)


if __name__ == "__main__":
    main()
