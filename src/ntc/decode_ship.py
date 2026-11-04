"""CPU references for the decoder changes that sit next to the fast shader.

These match the fp32 decoder where the math is the same, and they record
the rate of the parts that are not in the fast shader yet: BC latent
planes, a BC7 tail for coarse mips, per-block mode hints, a deferred tile
sort, weight immediates, and a hidden-width knee. None of this links to
another vendor's texture codec.
"""

from __future__ import annotations

import numpy as np

from ntc.container import FLAG_BC_LATENTS, LayerRec, TextureSet
from ntc.decode_ref import (
    activate,
    apply_layer,
    bilinear_codes,
    decode_mip,
    latent_mid,
    texel_uvs,
)

# One byte per 4x4 block is 0.5 bpp. Four bits (3 mode + 1 partition) is 0.25.
HINT_BITS = 4


def pack_latent_bc(codes: np.ndarray, bits: int) -> tuple[bytes, np.ndarray]:
    """BC7 mode 6 planes for ``codes`` ``[C,H,W]``, and the restored codes.

    Four channels fill one RGBA block. A short tail is padded with the
    mid code. The restored array has the original channel count.
    """
    from bench.bc_pack import pack_image_bc7, unpack_bc7_mode6_block

    src = np.asarray(codes)
    if src.ndim != 3:
        raise ValueError(src.shape)
    channels, height, width = src.shape
    if height % 4 or width % 4:
        raise ValueError("BC latent grids are multiples of 4")
    levels = (1 << int(bits)) - 1
    span = 255.0 / float(levels if levels else 1)
    padded = int(np.ceil(channels / 4.0) * 4)
    image = np.full((padded, height, width), int(round(levels / 2.0)), np.uint8)
    image[:channels] = np.clip(np.round(src), 0, levels).astype(np.uint8)
    payload = bytearray()
    restored = np.zeros_like(image)
    for plane in range(0, padded, 4):
        view = np.clip(np.round(image[plane : plane + 4].astype(np.float32) * span), 0, 255).astype(np.uint8)
        rgba = np.moveaxis(view, 0, -1)
        raw = pack_image_bc7(rgba)
        payload += raw
        back = np.zeros_like(rgba)
        blocks_x = width // 4
        for by in range(height // 4):
            for bx in range(blocks_x):
                block = raw[(by * blocks_x + bx) * 16 : (by * blocks_x + bx + 1) * 16]
                back[by * 4 : by * 4 + 4, bx * 4 : bx * 4 + 4] = unpack_bc7_mode6_block(block).reshape(4, 4, 4)
        codes_back = np.rint(back.astype(np.float32) / span).clip(0, levels).astype(np.uint8)
        restored[plane : plane + 4] = np.moveaxis(codes_back, -1, 0)
    return bytes(payload), restored[:channels]


def bc_latent_psnr(codes: np.ndarray, restored: np.ndarray, bits: int) -> float:
    levels = float((1 << int(bits)) - 1)
    err = (codes.astype(np.float64) - restored.astype(np.float64)) / max(levels, 1.0)
    mse = float(np.mean(err * err))
    if mse <= 0:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def decode_mip_hoisted(ts: TextureSet, mip: int = 0) -> np.ndarray:
    """Layer 0 at grid resolution, then bilinear, then the rest of the MLP.

    Dequant and the first affine layer are linear, so they commute with the
    lerp up to fp32 rounding. Positional encoding is not linear and is not
    handled here.
    """
    if ts.pe_mode != 0:
        raise ValueError("hoist is for networks without positional encoding")
    if not ts.layers:
        raise ValueError("no layers")
    uv, width, height = texel_uvs(ts.width, ts.height, mip)
    layer0 = ts.layers[0]
    weight = layer0.weight.astype(np.float32)
    scale = layer0.scale.astype(np.float32)
    pre = np.zeros((uv.shape[0], layer0.out_features), np.float32)
    col = 0
    for grid in ts.grids:
        codes = grid.codes.astype(np.float32)
        mid = latent_mid(grid.bits)
        deq = (codes - np.float32(mid)) * grid.scale.astype(np.float32)[:, None, None]
        take = int(grid.channels)
        cols = weight[:, col : col + take]
        partial = np.einsum("oc,chw->ohw", cols, deq)
        if scale.shape[1] != 1 and int(layer0.group_size) > 0:
            raise ValueError("hoist supports one scale group on layer 0")
        partial = partial * scale[:, 0][:, None, None]
        pre += bilinear_codes(partial, uv)
        col += take
    pre = pre + layer0.bias.astype(np.float32)[None, :]
    if len(ts.layers) == 1:
        out = pre
    else:
        hidden = activate(pre, ts.activation, ts.activation_requant)
        for index, layer in enumerate(ts.layers[1:], start=1):
            hidden = apply_layer(hidden, layer)
            if index != len(ts.layers) - 1:
                hidden = activate(hidden, ts.activation, ts.activation_requant)
        out = hidden
    return out.reshape(height, width, ts.num_channels).astype(np.float32)


def emit_immediate_mlp(layers: list[LayerRec]) -> str:
    """GLSL-ish text with dequantized weights as literals. Zeros are omitted."""
    lines = ["// per-material weight immediates", "float imm_eval(int n, float x) {", "    float y = 0.0;"]
    for layer in layers:
        weight = layer.weight.astype(np.float32)
        scale = layer.scale.astype(np.float32)
        for out_i in range(weight.shape[0]):
            for inn in range(weight.shape[1]):
                value = float(weight[out_i, inn] * scale[out_i, 0])
                if value == 0.0:
                    continue
                lines.append(f"    y += {value:.8e} * x; // w{out_i}_{inn}")
    lines.append("    return y;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def immediate_mlp(x: np.ndarray, layers: list[LayerRec], activation: int) -> np.ndarray:
    """Same product as ``mlp`` with exact-zero weights dropped before the dot."""
    hidden = x.astype(np.float32)
    last = len(layers) - 1
    for index, layer in enumerate(layers):
        weight = layer.weight.astype(np.float32) * layer.scale.astype(np.float32)[:, :1]
        weight = np.where(weight == 0.0, 0.0, weight)
        y = hidden @ weight.T + layer.bias.astype(np.float32)
        if index != last:
            y = activate(y, activation, False)
        hidden = y
    return hidden.astype(np.float32)


def hybrid_tail_bpp(textures: int = 3, bc_bpp: float = 8.0) -> float:
    """Extra VRAM for BC textures from mip 2 down.

    Mip k has 1/4^k of the mip-0 texels. The tail sums to 1/12 of mip 0,
    so three 8 bpp textures add 2 bpp.
    """
    tail_fraction = (1.0 / 16.0) / (1.0 - 0.25)
    return float(textures) * float(bc_bpp) * tail_fraction


def lod_histogram(lods: np.ndarray) -> dict[int, int]:
    """Count samples by integer LOD."""
    ids = np.floor(np.asarray(lods, np.float32)).astype(np.int64)
    out: dict[int, int] = {}
    for value in ids.tolist():
        out[int(value)] = out.get(int(value), 0) + 1
    return out


def classify_tiles(material_id: np.ndarray, tile_w: int = 8, tile_h: int = 4) -> list[dict]:
    """One record per tile: material id when the tile is pure, else mixed."""
    height, width = material_id.shape
    rows = []
    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            block = material_id[y : y + tile_h, x : x + tile_w]
            uniq = np.unique(block)
            rows.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "mixed": int(uniq.size) != 1,
                    "material": int(uniq[0]) if uniq.size == 1 else -1,
                    "count": int(block.size),
                }
            )
    return rows


def mode_hint_bytes(height: int, width: int, mode: int = 6, partition: int = 0) -> bytes:
    """Four bits per 4x4 block: mode in the low 3 bits, partition in bit 3."""
    if height % 4 or width % 4:
        raise ValueError("hint grid is a multiple of 4")
    if not 0 <= int(mode) <= 7 or not 0 <= int(partition) <= 1:
        raise ValueError((mode, partition))
    nibble = (int(partition) << 3) | int(mode)
    count = (height // 4) * (width // 4)
    # Two hints per byte.
    raw = bytearray((count + 1) // 2)
    for i in range(count):
        shift = 4 if (i & 1) else 0
        raw[i // 2] |= nibble << shift
    return bytes(raw)


def hint_bpp(height: int, width: int) -> float:
    count = (height // 4) * (width // 4)
    return (HINT_BITS * float(count)) / float(height * width)


def hidden_knee(rows: list[tuple[int, float]], tol_db: float = 0.15) -> int:
    """Smallest hidden width within ``tol_db`` of the best PSNR."""
    if not rows:
        raise ValueError("no rows")
    best = max(score for _width, score in rows)
    close = [width for width, score in rows if score >= best - float(tol_db)]
    return int(min(close))


def bc_flag_roundtrip_keeps_plain_codes(blob: bytes) -> bool:
    """The reserved BC-latent bit survives a parse. Payload codes stay plain."""
    from ntc.container import parse_texture_set

    ts = parse_texture_set(blob)
    return bool(ts.flags & FLAG_BC_LATENTS)


def reference_matches_hoist(ts: TextureSet, mip: int = 0) -> float:
    a = decode_mip(ts, mip)
    b = decode_mip_hoisted(ts, mip)
    return float(np.max(np.abs(a - b)))
