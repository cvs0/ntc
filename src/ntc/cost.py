"""Analytical decode cost. No timings.

Counts are for one reconstructed texel at one mip, feature-bilinear
(one MLP) unless noted.

Latent bytes are the packed bytes touched by the four bilinear taps.
A tap's channels live in separate planes, so a tap reads ``channels``
bytes at 8-bit, or the packed byte that holds that texel at 4-bit and
2-bit (one byte can serve two or four horizontal texels; the four taps
still miss that packing whenever the two x coordinates fall in different
bytes, which is the common case). The count below charges one packed
byte per channel per tap, which is an upper bound, and also reports the
unique-byte lower bound when the two x taps share a byte.

Weight bytes are the MLP parameters read once per texel if the working
set misses. A workgroup that keeps weights in shared memory or a uniform
buffer amortizes that read; ``weight_bytes_amortized(n)`` divides by
``n`` invocations that share the load.

MAC count is ``sum(in_features * out_features)`` over layers. A MAC is
one multiply and one add of the weight dot. Scale and bias add two
flops per output. The code lerp is three fused multiply-adds per latent
channel (lerp x, lerp x, lerp y) plus one dequant multiply-add, because
the affine dequant runs once after the code lerp.
"""

from __future__ import annotations

import math

from ntc.container import TextureSet, pe_feature_count


def _latent_tap_bytes(ts: TextureSet) -> dict:
    upper = 0
    shared_x = 0
    for g in ts.grids:
        # upper: 4 taps * channels * bytes_per_code, bytes_per_code = 1
        # even when several codes share a byte, because the taps are not
        # assumed to be neighbors in x. 4-bit and 2-bit still touch one
        # byte per tap per channel.
        upper += 4 * g.channels
        # if the two x samples land in the same packed byte, 4 taps collapse
        # toward 2 bytes per channel (the two y positions).
        if g.bits == 8:
            shared_x += 4 * g.channels
        elif g.bits == 4:
            shared_x += 2 * g.channels
        elif g.bits == 2:
            shared_x += 2 * g.channels
        else:
            shared_x += 4 * g.channels
    return {"latent_bytes_upper": upper, "latent_bytes_shared_x": shared_x}


def weight_bytes(ts: TextureSet) -> int:
    n = 0
    for layer in ts.layers:
        n += layer.weight_bytes
        n += layer.out_features * 2
        n += layer.out_features * layer.num_groups * 2
    return n


def macs(ts: TextureSet) -> int:
    return sum(layer.in_features * layer.out_features for layer in ts.layers)


def sample_cost(ts: TextureSet, taps: int = 1) -> dict:
    """``taps`` is the number of MLP evaluations (1 feature, 4 color bilinear,
    2 feature trilinear, N anisotropic).
    """
    lat = _latent_tap_bytes(ts)
    m = macs(ts)
    outputs = sum(layer.out_features for layer in ts.layers)
    hidden_units = sum(layer.out_features for layer in ts.layers[:-1])
    latent_ch = sum(g.channels for g in ts.grids)
    pe = pe_feature_count(ts.pe_mode, ts.pe_octaves)
    # per MLP:
    #   code lerp: 3 FMA = 6 flops per latent channel, times 1 (the four
    #   taps are combined by those 3 FMAs, not by 4 dequants)
    #   dequant: 1 FMA per latent channel (sub mid, mul scale) = 2 flops
    #   dot: 2 flops per MAC
    #   scale+bias: 2 flops per layer output
    #   relu: 1 flop per hidden unit (hgelu is more; counted separately)
    lerp_flops = latent_ch * 6
    dequant_flops = latent_ch * 2
    dot_flops = m * 2
    affine_flops = outputs * 2
    act_flops = hidden_units if ts.activation == 1 else (hidden_units * 4 if ts.activation == 2 else 0)
    pe_flops = 0
    if ts.pe_mode == 1:
        pe_flops = pe * 8  # sin/cos and the multiply by the frequency, rough
    elif ts.pe_mode == 2:
        pe_flops = pe * 4
    flops = lerp_flops + dequant_flops + dot_flops + affine_flops + act_flops + pe_flops
    wb = weight_bytes(ts)
    return {
        "mlp_evals": taps,
        "macs": m * taps,
        "flops": flops * taps,
        "latent_bytes_upper": lat["latent_bytes_upper"] * taps,
        "latent_bytes_shared_x": lat["latent_bytes_shared_x"] * taps,
        "weight_bytes_cold": wb * taps,
        "weight_bytes_per_invocation_if_shared_64": wb / 64.0,
        "latent_channels": latent_ch,
        "pe_features": pe,
        "input_features": ts.layers[0].in_features,
        "hidden_units": hidden_units,
        "activation_flops_per_hidden": 1 if ts.activation == 1 else (4 if ts.activation == 2 else 0),
    }


def _wrap_int(i: int, n: int) -> int:
    r = i % n
    if r < 0:
        r += n
    return r


def workgroup_latent_bytes(ts: TextureSet, group: int = 8, mip: int = 0) -> dict:
    """Unique packed latent bytes touched by a ``group`` x ``group`` color tile.

    One thread per texel, feature-bilinear, fetches 4 taps per latent channel.
    A tile that stages those texels in shared memory fetches each packed byte
    once. ``bytes_per_texel_unique`` is that staged cost. ``bytes_per_texel_upper``
    is the per-thread upper bound with no reuse.
    """
    color_w = max(1, ts.width >> mip)
    color_h = max(1, ts.height >> mip)
    gw = min(group, color_w)
    gh = min(group, color_h)
    addrs: set[tuple[int, int, int, int]] = set()
    tap_bytes = 0
    for y in range(gh):
        for x in range(gw):
            u = (x + 0.5) / float(color_w)
            v = (y + 0.5) / float(color_h)
            for gi, g in enumerate(ts.grids):
                tap_bytes += 4 * g.channels
                xl = u * g.width - 0.5
                yl = v * g.height - 0.5
                x0 = math.floor(xl)
                y0 = math.floor(yl)
                for yy in (y0, y0 + 1):
                    for xx in (x0, x0 + 1):
                        xxw = _wrap_int(xx, g.width)
                        yyw = _wrap_int(yy, g.height)
                        if g.bits == 8:
                            byte_x = xxw
                        elif g.bits == 4:
                            byte_x = xxw // 2
                        elif g.bits == 2:
                            byte_x = xxw // 4
                        else:
                            byte_x = xxw
                        for c in range(g.channels):
                            addrs.add((gi, c, yyw, byte_x))
    texels = gw * gh
    return {
        "group": gw,
        "texels": texels,
        "unique_bytes": len(addrs),
        "tap_bytes_upper": tap_bytes,
        "bytes_per_texel_unique": len(addrs) / float(texels),
        "bytes_per_texel_upper": tap_bytes / float(texels),
    }


# Instruction and load model of the two GPU decoders, one feature-bilinear
# sample. Mirrors ntc::legacy_sample_cost and ntc::fast_sample_cost in
# cpp/src/fast.cpp; tests/test_cost.py pins both to the same numbers.
#
# legacy  shaders/ntc_decode.glsl. Every latent code, weight byte, and fp16
#         half is its own 32-bit storage load plus shift and mask.
# fast    shaders/ntc_fast.glsl. One 32-bit load per tap per channel plane;
#         weights, bias, scales, and latent scales are uniform-buffer words
#         at constant offsets (constant-bank operands on NVIDIA, scalar loads
#         on AMD, shared by the whole wave).


def fast_planes(channels: int, bits: int) -> list[tuple[int, int, int]]:
    """``(first_channel, channels, record_bits)`` per plane, as ntc::fast_planes."""
    out = []
    cap = 32 // bits
    c = 0
    while c < channels:
        rem = channels - c
        n = 1
        while n * 2 <= rem:
            n *= 2
        n = min(n, cap)
        out.append((c, n, n * bits))
        c += n
    return out


def _groups(layer) -> int:
    one = layer.group_size <= 0 or layer.group_size >= layer.in_features
    return 1 if one else max(1, layer.num_groups)


def _mlp_alu(ts: TextureSet) -> int:
    alu = 0
    for layer in ts.layers:
        alu += layer.out_features * (_groups(layer) + 2)
        if ts.act_int8:
            alu += layer.in_features * 4
    return alu


def decode_ops(ts: TextureSet) -> dict:
    """Per-sample ops and bytes for the legacy and fast decoders."""
    pe = pe_feature_count(ts.pe_mode, ts.pe_octaves)
    m = macs(ts)
    leg = {"macs": m, "latent_loads": 0, "weight_loads": 0, "scale_loads": 0, "uniform_words": 0}
    alu = 0
    for g in ts.grids:
        leg["latent_loads"] += 4 * g.channels
        leg["scale_loads"] += 2 * g.channels
        alu += g.channels * (4 * 3 + 6 + 3) + 20
    alu += pe * 4 + _mlp_alu(ts)
    for layer in ts.layers:
        leg["weight_loads"] += layer.in_features * layer.out_features
        leg["weight_loads"] += layer.out_features * (2 + 2 * _groups(layer))
        alu += layer.in_features * layer.out_features * 4
    leg["alu"] = alu
    leg["loads"] = leg["latent_loads"] + leg["weight_loads"] + leg["scale_loads"]
    leg["load_bytes"] = 4 * leg["loads"]

    fast = {"macs": m, "latent_loads": 0, "weight_loads": 0, "scale_loads": 0, "uniform_words": 0}
    alu = 0
    for g in ts.grids:
        planes = fast_planes(g.channels, g.bits)
        fast["latent_loads"] += 4 * len(planes)
        fast["uniform_words"] += g.channels + 4
        alu += g.channels * (4 + 6 + 2) + len(planes) * 8 + 16
    alu += pe * 4 + _mlp_alu(ts)
    for layer in ts.layers:
        fast["uniform_words"] += layer.in_features * layer.out_features
        fast["uniform_words"] += layer.out_features * (1 + _groups(layer))
    fast["alu"] = alu
    fast["loads"] = fast["latent_loads"]
    fast["load_bytes"] = 4 * fast["loads"]
    return {"legacy": leg, "fast": fast}


def fast_latent_bytes(ts: TextureSet) -> int:
    """Resident bytes of the fast latent layout: each grid's chain of up to 8
    box-filtered levels, down to 1x1, as ntc::build_mips."""
    n = 0
    for g in ts.grids:
        w, h = g.width, g.height
        for _ in range(8):
            for _, _, rbits in fast_planes(g.channels, g.bits):
                n += ((w * rbits + 31) // 32) * 4 * h
            if w == 1 and h == 1:
                break
            w, h = max(1, w // 2), max(1, h // 2)
    return n


def main(argv: list[str] | None = None) -> int:
    """``python -m ntc.cost FILE.ntc ...``: print the per-sample cost model."""
    import argparse
    import json
    from pathlib import Path

    from ntc.container import parse_texture_set

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    rows = []
    for f in args.files:
        ts = parse_texture_set(Path(f).read_bytes())
        ops = decode_ops(ts)
        sc = sample_cost(ts)
        row = {"file": f, "width": ts.width, "height": ts.height, "channels": ts.num_channels,
               "grids": [(g.width, g.height, g.channels, g.bits) for g in ts.grids],
               "layers": [(lay.in_features, lay.out_features) for lay in ts.layers],
               "flops": sc["flops"], "weight_bytes": weight_bytes(ts),
               "fast_latent_bytes": fast_latent_bytes(ts), **{k: v for k, v in ops.items()}}
        rows.append(row)
        if not args.json:
            leg, fast = ops["legacy"], ops["fast"]
            print(f"{f}: {ts.width}x{ts.height} ch {ts.num_channels} grids {row['grids']} layers {row['layers']}")
            print(f"  macs {leg['macs']} flops {sc['flops']} weight_bytes {row['weight_bytes']}")
            print(f"  legacy loads {leg['loads']} (latent {leg['latent_loads']} weight {leg['weight_loads']} "
                  f"scale {leg['scale_loads']}) bytes {leg['load_bytes']} alu~{leg['alu']}")
            print(f"  fast   loads {fast['loads']} bytes {fast['load_bytes']} uniform_words "
                  f"{fast['uniform_words']} alu~{fast['alu']} latent_resident {row['fast_latent_bytes']}")
            print(f"  loads x{leg['loads'] / max(1, fast['loads']):.1f} fewer, "
                  f"non-MAC alu x{leg['alu'] / max(1, fast['alu']):.1f} fewer")
    if args.json:
        print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
