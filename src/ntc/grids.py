"""Latent shape selection.

``rate="payload"`` (the default, and the historical ``target_bpp``) counts
latent codes, fp16 latent scales, and quantized MLP weights, bias, and
scales, per mip 0 texel. It does not count the header or the descriptor
tables.

``rate="latent"`` counts only latent codes and fp16 latent scales, per
mip 0 texel. That is the rate to line up with ``ntc-cli --bitsPerPixel``.
Weights are still stored. They show up in ``file_bpp_chain`` and ``bppc``,
not in the selection target.
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


def grow_grids(
    base: list[tuple[int, int, int, int]],
    width: int,
    height: int,
    target_bpp: float,
) -> list[tuple[int, int, int, int]]:
    """Add channels on ``base`` until the packed latent rate meets ``target_bpp``.

    Spatial sizes and bit depth stay put, and new channels are appended on
    each grid. A checkpoint of ``base`` can be copied into the prefix of
    those channels. Channels stop at 16.
    """
    grids = [list(g) for g in base]
    pixels = float(max(width * height, 1))

    def bpp() -> float:
        return latent_payload_bits([tuple(g) for g in grids]) / pixels

    guard = 0
    while bpp() < float(target_bpp) * 0.95 and guard < 8:
        guard += 1
        order = sorted(range(len(grids)), key=lambda i: grids[i][0] * grids[i][1], reverse=True)
        grew = False
        for index in order:
            if int(grids[index][2]) + 4 <= 16:
                grids[index][2] = int(grids[index][2]) + 4
                grew = True
                break
        if not grew:
            break
    return [tuple(g) for g in grids]


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


def _bit_band(grids: list[tuple[int, int, int, int]]) -> int:
    """0 for an all-4-bit grid, 1 for any other grid that avoids 2-bit, 2 otherwise.

    4-bit codes are the ones that trained on the 2048 materials. 2-bit has
    four levels. 8-bit on a 12-channel 1024 grid fell under its PCA init.
    """
    bits = [g[3] for g in grids]
    if bits and all(b == 4 for b in bits):
        return 0
    if bits and min(bits) >= 4:
        return 1
    return 2


def _grid_quality(grids: list[tuple[int, int, int, int]], width: int) -> float:
    """Higher means more usable features at a matched raw rate.

    The finest grid dominates. A 2-channel full-resolution grid scored
    29.33 dB at 2.69 bpp and lost to a 4-channel half-resolution grid at
    31.80 dB and 1.42 bpp, so a finest grid under 4 channels is discounted
    and a full-resolution channel counts less than a half-resolution one.
    A quarter-resolution grid counts less than its channel count can
    recover: PavingStones at raw 3.89 tied a 12-channel 512 grid with a
    4-channel 1024 grid, and the 512 grid plateaued near 25 dB.
    """
    ordered = sorted(grids, key=lambda g: (g[0] * g[1], g[2]), reverse=True)
    score = 0.0
    for i, (gw, gh, ch, bits) in enumerate(ordered):
        if bits <= 2:
            quant = 0.2
        elif bits >= 8:
            quant = 0.35
        else:
            quant = 1.0
        if i == 0 and ch < 4:
            quant *= 0.25
        side = max(gw, gh)
        if side >= width:
            res = 0.5
        elif side * 2 >= width:
            res = 1.0
        else:
            res = 0.25
        level = 1.0 if i == 0 else 0.15
        score += ch * quant * res * level
    return score


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
    rate: str = "payload",
    min_bits: int = 0,
) -> list[list[tuple[int, int, int, int]]]:
    """Return up to ``adaptive_pool`` shapes closest to ``target_bpp``.

    The first entry is the default (closest). ``adaptive_pool > 1`` is the
    short list a rate search trains. ``rate`` is ``payload`` or ``latent``.
    The bpp here is the packed latent (and, for ``payload``, the MLP), not
    the deflate or range-coded size written into the file.

    ``min_bits`` of 0 keeps the historical order. Above 0, 2-bit grids sort
    last, then every 4-bit grid sorts ahead of 8-bit, then raw-bpp error,
    then ``_grid_quality``. That puts 4-bit half-resolution features ahead of
    a 2-channel full-resolution grid at the same packed size.
    """
    if rate not in ("payload", "latent"):
        raise ValueError(rate)
    scored = []
    for grids in _family(width, height):
        in_f = sum(g[2] for g in grids) + pe_feature_count(pe_mode, pe_octaves)
        if in_f > 160 or in_f < 1:
            continue
        if any(g[2] > 16 for g in grids):
            continue
        if rate == "latent":
            bpp = latent_payload_bits(grids) / float(width * height)
        else:
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
        if min_bits > 0:
            depth = min(g[3] for g in grids)
            # Scale bytes move the raw bpp by about 1e-5. Quantize so grids
            # at the same packed size are ordered by quality, not by that.
            rate_pen = round(err + 0.25 * over, 2)
            below = 0 if depth >= min_bits else 1
            band = _bit_band(grids) if below == 0 else 2
            scored.append((below, band, rate_pen, -_grid_quality(grids, width), grids))
        else:
            scored.append((err + 0.25 * over, -bpp, len(grids), grids))
    if min_bits > 0:
        scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    else:
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
    # unique by grid tuple
    seen = set()
    picked = []
    for item in scored:
        grids = item[-1]
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
