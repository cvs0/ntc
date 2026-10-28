import json
from pathlib import Path

import numpy as np
from PIL import Image

from bench.score_mips import score_tree
from ntc.grids import latent_payload_bits, select_grids


def _png(path: Path, value: int, size: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size, size, 3), value, np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def test_score_identical_chain(tmp_path: Path):
    ref = tmp_path / "ref"
    dec = tmp_path / "dec"
    for root in (ref, dec):
        _png(root / "albedo.png", 10, 4)
        _png(root / "mips" / "albedo.01.png", 20, 2)
        _png(root / "roughness.png", 30, 4)
    report = score_tree(ref, dec)
    assert report["psnr_u8"] == 100.0
    assert report["groups"]["albedo"]["psnr_u8"] == 100.0
    assert report["groups"]["roughness"]["psnr_u8"] == 100.0
    assert report["mips"] == [0, 1]
    assert report["texels"] == (4 * 4 * 3 + 2 * 2 * 3) + 4 * 4 * 3


def test_score_known_error(tmp_path: Path):
    ref = tmp_path / "ref"
    dec = tmp_path / "dec"
    _png(ref / "albedo.png", 0, 2)
    _png(dec / "albedo.png", 255, 2)
    report = score_tree(ref, dec)
    assert report["psnr_u8"] == 0.0


def test_latent_target_orders_shapes():
    low = select_grids(64, 64, 1.0, [16], 4, 0, 0, 8, 0, rate="latent")[0]
    high = select_grids(64, 64, 12.0, [16], 4, 0, 0, 8, 0, rate="latent")[0]
    assert latent_payload_bits(high) > latent_payload_bits(low)


def test_latent_min_bits_avoids_a_two_bit_tie():
    raw = select_grids(2048, 2048, 5.0, [32], 9, 0, 0, 8, 0, rate="latent")[0]
    assert min(g[3] for g in raw) == 2
    held = select_grids(2048, 2048, 5.0, [32], 9, 0, 0, 8, 0, rate="latent", min_bits=4)[0]
    assert min(g[3] for g in held) >= 4
    assert max(g[3] for g in held) == 4
    bpp = latent_payload_bits(held) / (2048 * 2048)
    assert abs(bpp - 5.0) < 0.05


def test_rd_svg_labels_table3(tmp_path: Path):
    from bench.rtxntc_compare import PAPER_NOTE, _svg

    dest = tmp_path / "rd.svg"
    _svg(
        dest,
        [{"bppc": 0.4, "latent_bpp_mip0": 4.0, "psnr_u8": 36.0}],
        [{"bppc": 0.5, "latent_bpp_mip0": 5.0, "psnr_u8": 37.0}],
    )
    text = dest.read_text()
    assert "different material set" in text
    assert "32.71" in text
    assert "45.30" in text
    assert PAPER_NOTE.startswith("Vaidyanathan")
    json.dumps({"note": PAPER_NOTE})
