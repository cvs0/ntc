"""Static checks that shaders/ntc_fast.glsl and cpp/include/ntc/fast.hpp agree."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _spec_ids_glsl() -> dict[str, int]:
    src = (ROOT / "shaders" / "ntc_fast.glsl").read_text()
    return {m.group(2): int(m.group(1))
            for m in re.finditer(r"constant_id = (\d+)\) const int (NTCF_\w+)", src)}


def _spec_ids_hpp() -> dict[str, int]:
    src = (ROOT / "cpp" / "include" / "ntc" / "fast.hpp").read_text()
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"(kSpec\w+) = (\d+)", src)}


def test_spec_constant_ids_match():
    g = _spec_ids_glsl()
    h = _spec_ids_hpp()
    pairs = {
        "NTCF_GRIDS": "kSpecGrids", "NTCF_LAYERS": "kSpecLayers", "NTCF_OUTCH": "kSpecOutCh",
        "NTCF_PE": "kSpecPeMode", "NTCF_PEOCT": "kSpecPeOct", "NTCF_ACT": "kSpecAct",
        "NTCF_REQUANT": "kSpecRequant", "NTCF_ACTI8": "kSpecActI8", "NTCF_NUMMIPS": "kSpecNumMips",
        "NTCF_LSOFF": "kSpecLatentScaleOff", "NTCF_ASOFF": "kSpecActScaleOff",
        "NTCF_W0": "kSpecWidth0", "NTCF_GC0": "kSpecGridCh0", "NTCF_GB0": "kSpecGridBits0",
        "NTCF_GS0": "kSpecGroupSize0", "NTCF_NG0": "kSpecNumGroups0", "NTCF_WO0": "kSpecWOff0",
        "NTCF_BO0": "kSpecBOff0", "NTCF_SO0": "kSpecSOff0", "NTCF_QO0": "kSpecQOff0",
        "NTCF_HO0": "kSpecHOff0",
    }
    for gname, hname in pairs.items():
        assert g[gname] == h[hname], (gname, hname)
    # ids are dense 0..kSpecCount-1 and the bench mode follows them
    assert sorted(g.values()) == list(range(h["kSpecCount"]))
    bench = (ROOT / "shaders" / "ntc_fast_bench.comp").read_text()
    assert f"constant_id = {h['kSpecCount']}) const int NTCF_MODE" in bench


def test_signed_packed_dot():
    # The uint overload of dotPacked4x8EXT is the unsigned dot. The weights
    # are signed, so every call must pass them as int. The activation operand
    # is int for signed codes and uint for unsigned codes (signed x unsigned).
    for name in ("ntc_fast.glsl", "ntc_sample_i8.comp"):
        src = (ROOT / "shaders" / name).read_text()
        for call in re.findall(r"dotPacked4x8EXT\(([^;]*)\);", src):
            assert call.startswith("int("), (name, call)


def test_make_bench_ntc_shapes(tmp_path):
    spec = importlib.util.spec_from_file_location("make_bench_ntc", ROOT / "bench" / "make_bench_ntc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from ntc.container import parse_texture_set
    from ntc.cost import decode_ops, fast_latent_bytes

    ts = parse_texture_set(mod.make(64, [(64, 64, 4, 4), (32, 32, 12, 4)], [16]))
    assert [layer.in_features for layer in ts.layers] == [16, 16]
    assert ts.layers[-1].out_features == 9
    ops = decode_ops(ts)
    # 4 channels at 4 bits: one 16-bit plane. 12 channels: 8 + 4 -> two planes.
    assert ops["fast"]["latent_loads"] == 4 * 3
    assert ops["legacy"]["latent_loads"] == 4 * 16
    assert fast_latent_bytes(ts) > 0
    for name, (size, grids, hidden) in mod.PRESETS.items():
        assert size & (size - 1) == 0, name
        assert sum(g[2] for g in grids) <= 160
        assert all(h <= 64 for h in hidden)


def test_fast_shader_generated():
    # ntc_fast.glsl is written by shaders/gen_ntc_fast.py.
    r = subprocess.run([sys.executable, str(ROOT / "shaders" / "gen_ntc_fast.py"), "--check"], capture_output=True)
    assert r.returncode == 0, r.stdout + r.stderr
    # No loop over a private array is left for the driver to unroll.
    assert "[[unroll]]" not in (ROOT / "shaders" / "ntc_fast.glsl").read_text()
