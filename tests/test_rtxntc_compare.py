"""BD-rate for the RTXNTC baseline table. The tool itself is not invoked."""

from bench.rtxntc_compare import bd_rate, summary_table


def test_identical_curves_have_zero_bd_rate():
    rate = [0.5, 1.0, 2.0, 4.0]
    psnr = [30.0, 34.0, 38.0, 42.0]
    assert abs(bd_rate(rate, psnr, rate, psnr)) < 1e-6


def test_double_rate_is_about_100_percent():
    anchor_r = [0.5, 1.0, 2.0, 4.0]
    psnr = [30.0, 34.0, 38.0, 42.0]
    doubled = [r * 2.0 for r in anchor_r]
    gap = bd_rate(anchor_r, psnr, doubled, psnr)
    assert 90.0 < gap < 110.0


def test_summary_skips_a_missing_tool_and_reports_bd_rate():
    ours = [
        {"material": "MetalPlates013_2K", "bppc": 0.5, "psnr_u8": 32.0},
        {"material": "MetalPlates013_2K", "bppc": 1.0, "psnr_u8": 36.0},
        {"material": "PavingStones070_2K", "error": "missing"},
    ]
    rtx = [
        {"tool": "rtxntc", "material": "MetalPlates013_2K", "bppc": 0.5, "psnr_u8": 32.0},
        {"tool": "rtxntc", "material": "MetalPlates013_2K", "bppc": 1.0, "psnr_u8": 36.0},
        {"tool": "rtxntc", "material": "Tiles135D_2K", "error": "ntc-cli not found"},
    ]
    table = summary_table(ours, rtx)
    metal = next(row for row in table["materials"] if row["material"] == "MetalPlates013_2K")
    assert abs(metal["bd_rate_percent"]) < 1e-6
    names = {row["material"] for row in table["materials"]}
    assert "PavingStones070_2K" not in names
    assert "Tiles135D_2K" not in names
