"""BD-rate on synthetic curves. No encode."""

from bench.rd_curve import bd_rate_rows, bits_saved_percent


def test_half_rate_saves_about_50_percent():
    base = [(30.0, 2.0), (40.0, 4.0)]
    cand = [(30.0, 1.0), (40.0, 2.0)]
    saved, method = bits_saved_percent(base, cand)
    assert method == "bjontegaard_linear_log_rate"
    assert abs(saved - 50.0) < 1e-6


def test_single_baseline_point_uses_interpolation():
    base = [(35.0, 8.0)]
    cand = [(30.0, 4.0), (40.0, 12.0)]
    saved, method = bits_saved_percent(base, cand)
    assert method == "equal_psnr_interpolation"
    # Log-rate is linear in PSNR, so 35 dB is the geometric mean of 4 and 12.
    expect = (1.0 - (4.0 * 12.0) ** 0.5 / 8.0) * 100.0
    assert abs(saved - expect) < 1e-6


def test_bc_roundtrip_shapes():
    import numpy as np

    from bench.rd_curve import _transcode_role

    img = np.full((8, 8, 3), 0.5, np.float32)
    rec = _transcode_role(img, "bc1")
    assert rec.shape == (8, 8, 3)
    assert abs(float(rec.mean()) - 0.5) < 0.05
    tiny = np.full((1, 1, 3), 0.25, np.float32)
    assert _transcode_role(tiny, "bc7").shape == (1, 1, 3)


def test_paper_table_has_no_bd_rate():
    ours = [
        {"series": "ours", "bppc": 1.0, "psnr_u8": 30.0, "groups": ["albedo"], "file_bpp_chain": 4.0},
        {"series": "ours", "bppc": 2.0, "psnr_u8": 40.0, "groups": ["albedo"], "file_bpp_chain": 8.0},
    ]
    rows = bd_rate_rows(ours, [], [])
    paper = [r for r in rows if r["baseline"] == "paper_table3"]
    assert paper
    assert paper[0]["bits_saved_percent"] is None
    assert paper[0]["reason"] == "different material set"
