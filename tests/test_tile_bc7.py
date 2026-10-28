"""BC7 feedback uses the same mode-6 weights as the CPU packer."""

import re
import shutil
import subprocess
from pathlib import Path

from bench.bc_pack import _W4

ROOT = Path(__file__).resolve().parents[1]


def test_tile_shader_uses_the_mode6_weights():
    text = (ROOT / "shaders" / "ntc_tile_bc7.comp").read_text()
    found = re.search(r"const uint W4\[16\] = uint\[\]\(([^)]*)\);", text)
    assert found
    weights = [int(part.strip().rstrip("u")) for part in found.group(1).split(",")]
    assert tuple(weights) == _W4


def test_on_sample_fragment_is_stochastic():
    frag = (ROOT / "shaders" / "ntc_fs_stoch.frag").read_text()
    assert "ntc_eval_filtered" in frag
    assert ", 3u," in frag
    viewer = (ROOT / "shaders" / "ntc_pbr.frag").read_text()
    assert "g_desc[2].w, 3u," in viewer


def test_bc7_and_stochastic_shaders_compile(tmp_path: Path):
    glslang = shutil.which("glslangValidator")
    if not glslang:
        return
    for name in ("ntc_tile_bc7.comp", "ntc_fs_stoch.frag"):
        subprocess.check_call(
            [
                glslang,
                "-V",
                "--target-env",
                "vulkan1.3",
                f"-I{ROOT / 'shaders'}",
                str(ROOT / "shaders" / name),
                "-o",
                str(tmp_path / f"{name}.spv"),
            ]
        )
