"""Reference decoder.

The function implemented here is the spec. The C++ decoder and the GLSL
library implement the same steps in the same order. Accumulation is
fp32. Per-output (or per-group) scale is applied after the integer-weight
dot, not inside it:

    acc = sum_i float(weight[i]) * x[i]
    y   = bias + scale * acc

Group ``g`` covers inputs ``[g*group_size, min(in, (g+1)*group_size))``.
``group_size <= 0`` is one group covering the row.

Grid mips are not stored. Mip ``k+1`` is a 2x2 box of mip ``k`` codes:

    out = (a + b + c + d + 2) // 4

Odd trailing rows and columns are dropped. A size-1 axis is averaged
along the other axis only, with divisor 2 and rounding ``(sum+1)//2``.
"""

from __future__ import annotations

import math

import numpy as np

from ntc.container import (
    ACT_HGELU,
    ACT_IDENTITY,
    ACT_RELU,
    PE_FOURIER,
    PE_TRIANGLE,
    TextureSet,
    pe_feature_count,
)

_TAU = np.float32(2.0 * math.pi)


def box_down(codes: np.ndarray) -> np.ndarray | None:
    """Integer box filter. ``codes`` is uint8 ``[C,H,W]``."""
    c, h, w = codes.shape
    if h == 1 and w == 1:
        return None
    h2 = max(1, h // 2)
    w2 = max(1, w // 2)
    src = codes.astype(np.int32)
    if h >= 2 and w >= 2:
        src = src[:, : h2 * 2, : w2 * 2]
        src = src.reshape(c, h2, 2, w2, 2)
        acc = src.sum(axis=(2, 4))
        out = (acc + 2) // 4
    elif h >= 2:
        src = src[:, : h2 * 2, :]
        src = src.reshape(c, h2, 2, w)
        acc = src.sum(axis=2)
        out = (acc + 1) // 2
    elif w >= 2:
        src = src[:, :, : w2 * 2]
        src = src.reshape(c, h, w2, 2)
        acc = src.sum(axis=3)
        out = (acc + 1) // 2
    else:
        return None
    return out.astype(np.uint8)


def grid_mip_chain(codes: np.ndarray, limit: int = 8) -> list[np.ndarray]:
    chain = [codes]
    cur = codes
    while len(chain) < limit:
        nxt = box_down(cur)
        if nxt is None:
            break
        if nxt.shape[1] == cur.shape[1] and nxt.shape[2] == cur.shape[2]:
            break
        chain.append(nxt)
        cur = nxt
        if cur.shape[1] == 1 and cur.shape[2] == 1:
            break
    return chain


def latent_mid(bits: int) -> np.float32:
    return np.float32((1 << bits) - 1) / np.float32(2.0)


def _wrap(i: np.ndarray, n: int) -> np.ndarray:
    """Floor-mod. Matches C++ ``r = i % n; if (r < 0) r += n`` for n > 0."""
    i64 = i.astype(np.int64, copy=False)
    r = np.remainder(i64, np.int64(n))
    # np.remainder on integers follows the sign of the divisor (floor mod
    # for positive n). Keep the mask so a C-style remainder still works.
    r = np.where(r < 0, r + n, r)
    return r


def bilinear_codes(codes: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample of integer codes, then the caller dequantizes.

    ``codes`` is ``[C,H,W]``, ``uv`` is ``[N,2]`` in ``[0,1]``.
    Affine dequant commutes with this lerp, so codes are lerped in fp32.
    """
    c, h, w = codes.shape
    x = uv[:, 0].astype(np.float32) * np.float32(w) - np.float32(0.5)
    y = uv[:, 1].astype(np.float32) * np.float32(h) - np.float32(0.5)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    tx = (x - x0.astype(np.float32)).astype(np.float32)[:, None]
    ty = (y - y0.astype(np.float32)).astype(np.float32)[:, None]
    x0w = _wrap(x0, w)
    x1w = _wrap(x0 + 1, w)
    y0w = _wrap(y0, h)
    y1w = _wrap(y0 + 1, h)
    plane = codes.reshape(c, h * w)
    def gather(yy, xx):
        idx = yy * w + xx
        return plane[:, idx].T.astype(np.float32)  # [N,C]

    c00 = gather(y0w, x0w)
    c10 = gather(y1w, x0w)
    c01 = gather(y0w, x1w)
    c11 = gather(y1w, x1w)
    # lerp x then y. One tx, one ty.
    a = c00 + (c01 - c00) * tx
    b = c10 + (c11 - c10) * tx
    return a + (b - a) * ty


def dequant_lerped(lerped: np.ndarray, scale: np.ndarray, bits: int) -> np.ndarray:
    mid = latent_mid(bits)
    return (lerped - mid) * scale.astype(np.float32)[None, :]


def positional(uv: np.ndarray, pe_mode: int, octaves: int, mip: int, max_mip: int) -> np.ndarray:
    n = uv.shape[0]
    if pe_mode == 0 or octaves <= 0:
        return np.zeros((n, 0), dtype=np.float32)
    cols = []
    if pe_mode == PE_FOURIER:
        p = uv.astype(np.float32)
        for o in range(octaves):
            f = np.float32(1 << o)
            a = _TAU * p * f
            cols.extend(
                [
                    np.sin(a[:, 0]),
                    np.cos(a[:, 0]),
                    np.sin(a[:, 1]),
                    np.cos(a[:, 1]),
                ]
            )
    elif pe_mode == PE_TRIANGLE:
        p = uv.astype(np.float32).copy()
        for _o in range(octaves):
            fx = p[:, 0] - np.floor(p[:, 0])
            fy = p[:, 1] - np.floor(p[:, 1])
            fx2 = (p[:, 0] + np.float32(0.25))
            fy2 = (p[:, 1] + np.float32(0.25))
            fx2 = fx2 - np.floor(fx2)
            fy2 = fy2 - np.floor(fy2)
            cols.extend([fx * 2 - 1, fy * 2 - 1, fx2 * 2 - 1, fy2 * 2 - 1])
            p = p * np.float32(2.0)
        denom = float(max(max_mip, 1))
        lod = np.float32(float(mip) / denom)
        cols.extend([np.full(n, lod, dtype=np.float32), np.full(n, lod, dtype=np.float32)])
    else:
        raise ValueError(pe_mode)
    return np.stack(cols, axis=1).astype(np.float32)


def activate(y: np.ndarray, kind: int, requant: bool) -> np.ndarray:
    if kind == ACT_IDENTITY:
        z = y
    elif kind == ACT_RELU:
        z = np.maximum(y, np.float32(0.0))
    elif kind == ACT_HGELU:
        # min(x, 3) * saturate(x/3 + 1/2). Same clamp used by RTXNTC's HGELU.
        clamped = np.minimum(y, np.float32(3.0))
        gate = np.clip(y * np.float32(1.0 / 3.0) + np.float32(0.5), 0.0, 1.0)
        z = clamped * gate
    else:
        raise ValueError(kind)
    z = z.astype(np.float32)
    if requant:
        q = np.rint(z * np.float32(80.0) - np.float32(113.0))
        q = np.clip(q, -128.0, 127.0)
        return q.astype(np.float32)
    return z


def apply_layer(x: np.ndarray, layer) -> np.ndarray:
    """``x`` is ``[N, in]`` fp32. ``layer.weight`` holds signed integer codes."""
    w = layer.weight.astype(np.float32)
    bias = layer.bias.astype(np.float32)
    scale = layer.scale.astype(np.float32)
    _out, inn = w.shape
    gs = layer.group_size
    if gs <= 0 or gs >= inn:
        acc = x @ w.T
        return bias + scale[:, 0] * acc
    y = np.broadcast_to(bias, (x.shape[0], bias.shape[0])).copy()
    groups = scale.shape[1]
    for g in range(groups):
        i0 = g * gs
        i1 = min(inn, i0 + gs)
        acc = x[:, i0:i1] @ w[:, i0:i1].T
        y += scale[:, g] * acc
    return y.astype(np.float32)


def mlp(x: np.ndarray, ts: TextureSet) -> np.ndarray:
    h = x.astype(np.float32)
    last = len(ts.layers) - 1
    for i, layer in enumerate(ts.layers):
        y = apply_layer(h, layer)
        if i != last:
            y = activate(y, ts.activation, ts.activation_requant)
        h = y
    return h


def sample_features(ts: TextureSet, uv: np.ndarray, mip: int, chains=None) -> np.ndarray:
    if chains is None:
        chains = [grid_mip_chain(g.codes) for g in ts.grids]
    cols = []
    for grid, chain in zip(ts.grids, chains):
        k = min(mip, len(chain) - 1)
        codes = chain[k]
        lerped = bilinear_codes(codes, uv)
        cols.append(dequant_lerped(lerped, grid.scale, grid.bits))
    feat = np.concatenate(cols, axis=1) if cols else np.zeros((uv.shape[0], 0), np.float32)
    pe = positional(uv, ts.pe_mode, ts.pe_octaves, mip, max(ts.num_mips - 1, 1))
    if pe.shape[1]:
        feat = np.concatenate([feat, pe], axis=1)
    expect = ts.layers[0].in_features
    if feat.shape[1] != expect:
        raise RuntimeError(f"feature count {feat.shape[1]} != layer in {expect}")
    return feat.astype(np.float32)


def texel_uvs(width: int, height: int, mip: int) -> tuple[np.ndarray, int, int]:
    w = max(1, width >> mip)
    h = max(1, height >> mip)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys, indexing="xy")
    u = (xg + 0.5) / np.float32(w)
    v = (yg + 0.5) / np.float32(h)
    uv = np.stack([u.reshape(-1), v.reshape(-1)], axis=1)
    return uv, w, h


def decode_mip(ts: TextureSet, mip: int = 0, chains=None) -> np.ndarray:
    """Point reconstruction of color mip ``mip``. Shape ``[H,W,C]`` fp32.

    Each color texel is one MLP evaluation at the texel center. Latents
    are bilinear at that center (feature-space sample, not a color filter).
    """
    uv, w, h = texel_uvs(ts.width, ts.height, mip)
    feat = sample_features(ts, uv, mip, chains)
    out = mlp(feat, ts)
    return out.reshape(h, w, ts.num_channels)


def decode_uv(ts: TextureSet, uv: np.ndarray, mip: int = 0, chains=None) -> np.ndarray:
    """Feature-bilinear decode at continuous uv. One MLP per row of ``uv``."""
    feat = sample_features(ts, uv.astype(np.float32), mip, chains)
    return mlp(feat, ts)


def color_bilinear_uv(ts: TextureSet, uv: np.ndarray, mip: int, chains=None) -> np.ndarray:
    """Bilinear of decoded texel centers. Four MLP evaluations per uv."""
    w = max(1, ts.width >> mip)
    h = max(1, ts.height >> mip)
    x = uv[:, 0] * np.float32(w) - np.float32(0.5)
    y = uv[:, 1] * np.float32(h) - np.float32(0.5)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    tx = (x - x0.astype(np.float32)).astype(np.float32)
    ty = (y - y0.astype(np.float32)).astype(np.float32)
    x0w = _wrap(x0, w)
    x1w = _wrap(x0 + 1, w)
    y0w = _wrap(y0, h)
    y1w = _wrap(y0 + 1, h)

    def at(xx, yy):
        u = (xx.astype(np.float32) + 0.5) / np.float32(w)
        v = (yy.astype(np.float32) + 0.5) / np.float32(h)
        return decode_uv(ts, np.stack([u, v], axis=1), mip, chains)

    c00 = at(x0w, y0w)
    c10 = at(x1w, y0w)
    c01 = at(x0w, y1w)
    c11 = at(x1w, y1w)
    tx = tx[:, None]
    ty = ty[:, None]
    a = c00 + (c10 - c00) * tx
    b = c01 + (c11 - c01) * tx
    return (a + (b - a) * ty).astype(np.float32)


def stochastic_choice(tx: np.ndarray, ty: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Return corner id 0..3 with probability equal to the bilinear weight.

    Corner order is (x0,y0), (x1,y0), (x0,y1), (x1,y1). ``u`` is uniform
    in ``[0,1)``.
    """
    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    # cumulative
    c0 = w00
    c1 = w00 + w10
    c2 = c1 + w01
    choice = np.zeros(tx.shape[0], dtype=np.int32)
    choice[u >= c0] = 1
    choice[u >= c1] = 2
    choice[u >= c2] = 3
    return choice


def feature_count_ok(ts: TextureSet) -> None:
    n = sum(g.channels for g in ts.grids) + pe_feature_count(ts.pe_mode, ts.pe_octaves)
    if n != ts.layers[0].in_features:
        raise RuntimeError("input feature mismatch")


def feature_trilinear_uv(ts: TextureSet, uv: np.ndarray, lod: float, chains=None) -> np.ndarray:
    """Two feature-bilinear MLP evaluations, lerped in color by ``fract(lod)``."""
    mip0 = int(math.floor(lod))
    frac = np.float32(lod - mip0)
    a = decode_uv(ts, uv, mip0, chains)
    b = decode_uv(ts, uv, mip0 + 1, chains)
    return ((1.0 - frac) * a + frac * b).astype(np.float32)


def stochastic_u(x: np.ndarray, y: np.ndarray, frame: int) -> np.ndarray:
    """Uniform ``[0,1)`` hash shared with the shader.

    ``h = x*1664525 + y*1013904223 + frame*22695477`` in uint32.
    The low 24 bits are divided by 2^24.
    """
    x = np.asarray(x, dtype=np.uint32)
    y = np.asarray(y, dtype=np.uint32)
    # uint64 product, then the low 32 bits, so the wrap matches GLSL uint.
    h = (
        np.uint64(x) * np.uint64(1664525)
        + np.uint64(y) * np.uint64(1013904223)
        + np.uint64(frame) * np.uint64(22695477)
    ).astype(np.uint64)
    h = (h & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    return (h & np.uint32(0xFFFFFF)).astype(np.float64) / 16777216.0


def stochastic_uv(
    ts: TextureSet, uv: np.ndarray, mip: int, frame: int, chains=None
) -> np.ndarray:
    """One decoded corner per uv. Corner probabilities are the bilinear weights."""
    w = max(1, ts.width >> mip)
    h = max(1, ts.height >> mip)
    x = uv[:, 0] * np.float32(w) - np.float32(0.5)
    y = uv[:, 1] * np.float32(h) - np.float32(0.5)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    tx = (x - x0.astype(np.float32)).astype(np.float64)
    ty = (y - y0.astype(np.float32)).astype(np.float64)
    x0w = _wrap(x0, w)
    x1w = _wrap(x0 + 1, w)
    y0w = _wrap(y0, h)
    y1w = _wrap(y0 + 1, h)
    # Hash the pre-wrap floor so the shader, which only has the continuous
    # uv, can rebuild the same integers.
    u = stochastic_u(x0.astype(np.int64), y0.astype(np.int64), frame)
    choice = stochastic_choice(tx, ty, u)
    xx = np.where((choice == 1) | (choice == 3), x1w, x0w)
    yy = np.where((choice == 2) | (choice == 3), y1w, y0w)
    su = (xx.astype(np.float32) + 0.5) / np.float32(w)
    sv = (yy.astype(np.float32) + 0.5) / np.float32(h)
    return decode_uv(ts, np.stack([su, sv], axis=1), mip, chains)


def anisotropic_offsets(dudx: np.ndarray, dudy: np.ndarray, max_aniso: int) -> np.ndarray:
    """Sample offsets along the major screen-space derivative, in uv units.

    ``N = clamp(ceil(major/minor), 1, max_aniso)``. Returns ``[N, 2]``.
    """
    dudx = np.asarray(dudx, dtype=np.float64)
    dudy = np.asarray(dudy, dtype=np.float64)
    lx = float(np.linalg.norm(dudx))
    ly = float(np.linalg.norm(dudy))
    major = max(lx, ly)
    minor = max(min(lx, ly), 1e-8)
    n = int(math.ceil(major / minor))
    n = max(1, min(int(max_aniso), n))
    axis = dudx if lx >= ly else dudy
    t = (np.arange(n, dtype=np.float64) + 0.5) / float(n) - 0.5
    return (axis.reshape(1, 2) * t.reshape(-1, 1)).astype(np.float32)


def stochastic_aniso_uv(
    ts: TextureSet,
    uv: np.ndarray,
    mip: int,
    dudx: np.ndarray,
    dudy: np.ndarray,
    max_aniso: int,
    frame: int,
    chains=None,
) -> np.ndarray:
    """Mean of N stochastic corners along the major axis. Matches the shader."""
    offs = anisotropic_offsets(dudx, dudy, max_aniso)
    w = max(1, ts.width >> mip)
    h = max(1, ts.height >> mip)
    acc = None
    for i, off in enumerate(offs):
        sample = (uv + off.reshape(1, 2)).astype(np.float32)
        x = sample[:, 0] * np.float32(w) - np.float32(0.5)
        y = sample[:, 1] * np.float32(h) - np.float32(0.5)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        tx = (x - x0.astype(np.float32)).astype(np.float64)
        ty = (y - y0.astype(np.float32)).astype(np.float64)
        # frame is mixed per tap the same way as ntc_hash01 in the shader
        mixed = (np.uint32(frame) + np.uint32(i) * np.uint32(747796405)) & np.uint32(0xFFFFFFFF)
        u = stochastic_u(x0, y0, int(mixed))
        choice = stochastic_choice(tx, ty, u)
        x0w = _wrap(x0, w)
        x1w = _wrap(x0 + 1, w)
        y0w = _wrap(y0, h)
        y1w = _wrap(y0 + 1, h)
        xx = np.where((choice == 1) | (choice == 3), x1w, x0w)
        yy = np.where((choice == 2) | (choice == 3), y1w, y0w)
        su = (xx.astype(np.float32) + 0.5) / np.float32(w)
        sv = (yy.astype(np.float32) + 0.5) / np.float32(h)
        got = decode_uv(ts, np.stack([su, sv], axis=1), mip, chains)
        acc = got if acc is None else acc + got
    return (acc / float(len(offs))).astype(np.float32)


def anisotropic_uv(
    ts: TextureSet,
    uv: np.ndarray,
    mip: int,
    dudx: np.ndarray,
    dudy: np.ndarray,
    max_aniso: int,
    chains=None,
) -> np.ndarray:
    """Mean of feature-bilinear taps along the major axis. Deterministic reference."""
    offs = anisotropic_offsets(dudx, dudy, max_aniso)
    acc = None
    for off in offs:
        sample_uv = uv + off.reshape(1, 2)
        got = decode_uv(ts, sample_uv.astype(np.float32), mip, chains)
        acc = got if acc is None else acc + got
    return (acc / float(len(offs))).astype(np.float32)
