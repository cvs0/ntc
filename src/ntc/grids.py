"""Latent shape selection at a target total bits per material pixel.

``bpp`` counts latent codes, fp16 latent scales, and quantized MLP
weights, bias, and scales. It does not count the 96-byte header or the
descriptor tables. Those are reported separately as file bpp.
"""

from __future__ import annotations

import itertools

from ntc.container import pe_feature_count
from ntc.quant import weight_group_count


def grid_dim(base: int, shift: int) -> int:
    d = max(1, base >> shift)
    if d >= 4:
        aligned = ((d + 3) // 4) * 4
        if aligned > base:
            aligned = max(4, (base // 4) * 4)
        return aligned
    return d


def latent_payload_bits(grids: list[tuple[int, int, int, int]]) -> int:
    bits = 0
    for gw, gh, ch, b in grids:
        stride = (gw * b + 7) // 8
        bits += stride * gh * ch * 8
        bits += ch * 16  # fp16 scale
    return bits


def weight_payload_bits(
    in_features: int,
    hidden: list[int],
    out_features: int,
    weight_bits: int,
    group_size: int,
) -> int:
    dims = [in_features, *hidden, out_features]
    total = 0
    for a, b in zip(dims[:-1], dims[1:]):
        if weight_bits == 8:
            total += a * b * 8
        elif weight_bits == 4:
            total += b * ((a + 1) // 2) * 8
        else:
            raise ValueError(weight_bits)
        groups = weight_group_count(a, group_size)
        total += b * 16  # bias fp16
        total += b * groups * 16  # scales
    return total


def shape_bpp(
    width: int,
    height: int,
    grids: list[tuple[int, int, int, int]],
    hidden: list[int],
    out_features: int,
    pe_mode: int,
    pe_octaves: int,
    weight_bits: int,
    group_size: int,
) -> float:
    in_f = sum(g[2] for g in grids) + pe_feature_count(pe_mode, pe_octaves)
    bits = latent_payload_bits(grids) + weight_payload_bits(
        in_f, hidden, out_features, weight_bits, group_size
    )
    return bits / float(width * height)


def _family(width: int, height: int) -> list[list[tuple[int, int, int, int]]]:
    out = []
    # shift 0 is full resolution. It was missing from the first search, which
    # capped every grid at half resolution and flattened the high-rate curve.
    shifts = [0, 1, 2, 3, 4, 5]
    chs = [2, 4, 8, 12]
    bitss = [2, 4, 8]
    # one grid
    for s, c, b in itertools.product(shifts, chs, bitss):
        gw, gh = grid_dim(width, s), grid_dim(height, s)
        if gw < 4 or gh < 4:
            continue
        out.append([(gw, gh, c, b)])
    # two grids, coarse strictly smaller shift
    for s0, s1, c0, c1, b0, b1 in itertools.product(shifts, shifts, chs, chs, bitss, bitss):
        if s1 <= s0:
            continue
        g0 = (grid_dim(width, s0), grid_dim(height, s0), c0, b0)
        g1 = (grid_dim(width, s1), grid_dim(height, s1), c1, b1)
        if g0[0] < 4 or g1[0] < 4 or g0[1] < 4 or g1[1] < 4:
            continue
        if g0[0] == g1[0] and g0[1] == g1[1] and b0 == b1:
            continue
        out.append([g0, g1])
    # same resolution, mixed bit depth (per-channel allocation split across grids)
    for s, c0, c1 in itertools.product(shifts, chs, chs):
        gw, gh = grid_dim(width, s), grid_dim(height, s)
        if gw < 4 or gh < 4:
            continue
        out.append([(gw, gh, c0, 8), (gw, gh, c1, 4)])
    return out


def select_grids(
    width: int,
    height: int,
    target_bpp: float,
    hidden: list[int],
    out_features: int,
    pe_mode: int,
    pe_octaves: int,
    weight_bits: int,
    group_size: int,
    adaptive_pool: int = 1,
) -> list[list[tuple[int, int, int, int]]]:
    """Return up to ``adaptive_pool`` shapes closest to ``target_bpp``.

    The first entry is the default (closest). ``adaptive_pool > 1`` is the
    short list a rate search trains.
    """
    scored = []
    for grids in _family(width, height):
        in_f = sum(g[2] for g in grids) + pe_feature_count(pe_mode, pe_octaves)
        if in_f > 160 or in_f < 1:
            continue
        if any(g[2] > 16 for g in grids):
            continue
        bpp = shape_bpp(
            width,
            height,
            grids,
            hidden,
            out_features,
            pe_mode,
            pe_octaves,
            weight_bits,
            group_size,
        )
        # prefer shapes that are not far above the target: extra bits are
        # not a free win when the comparison is at matched rate
        err = abs(bpp - target_bpp)
        over = max(0.0, bpp - target_bpp)
        scored.append((err + 0.25 * over, bpp, grids))
    scored.sort(key=lambda t: (t[0], -t[1], len(t[2])))
    # unique by grid tuple
    seen = set()
    picked = []
    for _err, _bpp, grids in scored:
        key = tuple(tuple(g) for g in grids)
        if key in seen:
            continue
        seen.add(key)
        picked.append(grids)
        if len(picked) >= max(1, adaptive_pool):
            break
    if not picked:
        gw, gh = grid_dim(width, 2), grid_dim(height, 2)
        picked = [[(gw, gh, 4, 8)]]
    return picked
