"""Plot the measured metal_plate 256 rate-distortion points. No fitted curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    auto = json.loads(Path("results/rd_auto_256.json").read_text())["rows"]
    old = json.loads(Path("results/rd_metal_plate_256.json").read_text())["rows"]
    blocks = json.loads(Path("results/blocks_metal_plate_256.json").read_text())["codecs"]
    ntc = [r for r in auto if r["material"] == "metal_plate" and r["tag"].startswith("auto_h32")]
    ntc.sort(key=lambda r: r["total_bpp"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=140)
    ax.plot(
        [r["total_bpp"] for r in ntc],
        [r["psnr_u8"] for r in ntc],
        "o-",
        color="#1f4e79",
        label="NTC1 shipped encoder",
    )
    ax.plot(
        [r["total_bpp"] for r in old],
        [r["psnr_u8"] for r in old],
        "s--",
        color="#9aa0a6",
        label="half-resolution grid family",
    )
    markers = {
        "bc7_per_texture": ("D", "BC7 per texture"),
        "bc1_albedo_bc5_normal_bc1_orm": ("^", "BC1+BC5+BC1"),
        "astc_4x4_per_texture": ("P", "ASTC 4x4"),
        "astc_6x6_per_texture": ("X", "ASTC 6x6"),
        "astc_8x8_per_texture": ("v", "ASTC 8x8"),
    }
    for row in blocks:
        marker, label = markers[row["name"]]
        ax.scatter([row["bpp"]], [row["psnr_u8"]], marker=marker, s=36, label=label)
    ax.set_xlabel("bits per material texel")
    ax.set_ylabel("PSNR, 8-bit unorm (dB)")
    ax.set_title("metal_plate 256, CPU, torch 2.14.1+cpu")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = Path("results/rd_metal_plate_256.png")
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
