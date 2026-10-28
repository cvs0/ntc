import json
from pathlib import Path

import numpy as np
from PIL import Image

from bench.score_mips import _psnr_from_mse, score_tree
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


def test_nonfinite_mse_is_not_a_perfect_score():
    assert _psnr_from_mse(float("nan")) == float("-inf")
    assert _psnr_from_mse(float("inf")) == float("-inf")


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
    assert max(g[0] for g in held) == 1024


def test_quality_prefers_halfres_features_and_four_bit():
    # Raw 10 is the b5 search (label 5 divided by the 0.5 coder scale).
    # A 2-channel full-resolution grid is an exact raw match and lost to a
    # half-resolution grid that has more 4-bit features.
    b5 = select_grids(2048, 2048, 10.0, [32], 9, 0, 0, 8, 0, rate="latent", min_bits=4)[0]
    assert max(g[3] for g in b5) == 4
    assert max(g[0] for g in b5) == 1024
    assert sum(g[2] for g in b5) >= 8
    # Raw 30 is the b15 search. The exact match is 12 channels at 8 bits.
    b15 = select_grids(2048, 2048, 30.0, [32], 9, 0, 0, 8, 0, rate="latent", min_bits=4)[0]
    assert max(g[3] for g in b15) == 4
    assert min(g[3] for g in b15) == 4


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
