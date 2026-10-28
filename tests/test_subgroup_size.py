"""Subgroup 32 is requested only when the device can require it."""

import shutil
import subprocess
from pathlib import Path

from ntc.gpu_policy import required_subgroup_size

ROOT = Path(__file__).resolve().parents[1]
COMPUTE = 0x20


def test_subgroup_32_on_both_vendor_ranges():
    # NVIDIA often reports 32. The RX 9070 often reports 32..64 or a fixed 64.
    assert required_subgroup_size(32, 32, COMPUTE, COMPUTE, True) == 32
    assert required_subgroup_size(32, 64, COMPUTE, COMPUTE, True) == 32
    assert required_subgroup_size(64, 64, COMPUTE, COMPUTE, True) is None
    assert required_subgroup_size(32, 64, COMPUTE, COMPUTE, False) is None
    assert required_subgroup_size(32, 64, 0, COMPUTE, True) is None


def test_pipeline_helper_and_precise_mad_are_in_the_sources(tmp_path: Path):
    header = (ROOT / "examples/vulkan/stage.hpp").read_text()
    assert "requiredSubgroupSize = 32" in header
    assert "minSubgroupSize > 32" in header
    decode = (ROOT / "shaders/ntc_decode.glsl").read_text()
    assert "precise float prod = acc * w;" in decode
    glslang = shutil.which("glslangValidator")
    spirv_dis = shutil.which("spirv-dis")
    if glslang and spirv_dis:
        spv = tmp_path / "ntc_precise_check.spv"
        subprocess.check_call(
            [
                glslang,
                "-V",
                "--target-env",
                "vulkan1.3",
                f"-I{ROOT / 'shaders'}",
                str(ROOT / "shaders" / "ntc_decompress.comp"),
                "-o",
                str(spv),
            ]
        )
        text = subprocess.check_output([spirv_dis, str(spv)], text=True)
        assert "NoContraction" in text
