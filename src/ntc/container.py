"""NTC1 container, version 1.

Little-endian. Multi-byte integers are explicit. The header is 96 bytes
and is not a C struct on the wire; the C++ reader uses the same offsets.

See the format tables in the README. ``crc32`` is zlib CRC-32 of every
byte after the header.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

MAGIC = 0x3143544E
VERSION = 1
HEADER_BYTES = 96
MAX_GRIDS = 8
MAX_LAYERS = 4
MAX_CHANNELS = 16
MAX_HIDDEN = 64
MAX_INPUT = 160
MAX_MIPS = 8

FLAG_DERIVED_MIPS = 1 << 0
FLAG_ACTIVATION_REQUANT = 1 << 1
# Latent region is a raw deflate stream (window -15, no zlib header).
# Grid data_offset values are then relative to the inflated bytes.
# Header u32 at offset 84 is the inflated length. bpp counts the
# compressed length, not the inflated grids.
FLAG_LATENT_DEFLATE = 1 << 2
# Order-1 nibble range coder of the same plain latent bytes. Mutually
# exclusive with FLAG_LATENT_DEFLATE. Offset 84 is the plain length.
FLAG_LATENT_RANGE = 1 << 3
# Neighbor- and position-conditioned nibble range coder. Not selected by
# ``latent_coder="best"``, so train_max files stay on deflate or order-1 range.
FLAG_LATENT_CTX = 1 << 5
FLAG_LATENT_COMPRESSED = FLAG_LATENT_DEFLATE | FLAG_LATENT_RANGE | FLAG_LATENT_CTX
# Hidden inputs to each MLP layer are int8 with one fp16 scale per layer,
# stored as a tail of float16 values (one per layer, input side). Default
# files do not set this bit. train_max does not set it.
FLAG_ACT_INT8 = 1 << 4
# Integer MLP reference (int8 activations, int32 dots, multiplier-shift).
# The fp32 decoder ignores this bit. Old files leave it clear.
FLAG_INT_DECODE = 1 << 6

ACT_IDENTITY = 0
ACT_RELU = 1
ACT_HGELU = 2

PE_NONE = 0
PE_FOURIER = 1
PE_TRIANGLE = 2

SEM_ALBEDO_R = 0
SEM_ALBEDO_G = 1
SEM_ALBEDO_B = 2
SEM_NORMAL_X = 3
SEM_NORMAL_Y = 4
SEM_NORMAL_Z = 5
SEM_ROUGHNESS = 6
SEM_METALNESS = 7
SEM_AO = 8
SEM_HEIGHT = 9
SEM_OPACITY = 10
SEM_EXTRA = 11

SEM_NAME = {
    0: "albedo_r",
    1: "albedo_g",
    2: "albedo_b",
    3: "normal_x",
    4: "normal_y",
    5: "normal_z",
    6: "roughness",
    7: "metalness",
    8: "ao",
    9: "height",
    10: "opacity",
    11: "extra",
}

_HEADER = struct.Struct("<IHHIIIHHHHHHHHHHIIIIIIIIIIIIQ")


def pe_feature_count(pe_mode: int, pe_octaves: int) -> int:
    if pe_mode == PE_NONE:
        return 0
    if pe_mode == PE_FOURIER:
        return 4 * pe_octaves
    if pe_mode == PE_TRIANGLE:
        return 4 * pe_octaves + 2
    raise ValueError(f"bad pe mode {pe_mode}")


def row_stride_bytes(width: int, bits: int) -> int:
    return (width * bits + 7) // 8


@dataclass
class ChannelRec:
    semantic: int
    color_space: int = 0


@dataclass
class GridRec:
    width: int
    height: int
    channels: int
    bits: int
    row_stride: int
    data_offset: int
    data_bytes: int
    scale_offset: int
    plane_bytes: int
    scale: np.ndarray = field(repr=False, default=None)
    codes: np.ndarray = field(repr=False, default=None)  # uint8 [C,H,W]
    # Present when this grid uses per-tile endpoints. Both are [C, nty, ntx].
    tile_scale: np.ndarray | None = field(repr=False, default=None)
    tile_offset: np.ndarray | None = field(repr=False, default=None)


@dataclass
class LayerRec:
    in_features: int
    out_features: int
    group_size: int
    num_groups: int
    weight_offset: int
    weight_bytes: int
    bias_offset: int
    scale_offset: int
    weight: np.ndarray = field(repr=False, default=None)  # float codes [out, in]
    bias: np.ndarray = field(repr=False, default=None)
    scale: np.ndarray = field(repr=False, default=None)  # [out, groups]
    # 0 means the header weight_bits value.
    weight_bits: int = 0


@dataclass
class TextureSet:
    width: int
    height: int
    num_mips: int
    activation: int
    pe_mode: int
    pe_octaves: int
    flags: int
    weight_bits: int
    group_size: int
    channels: list[ChannelRec]
    grids: list[GridRec]
    layers: list[LayerRec]
    raw: bytes
    tile_size: int = 0
    latent_stored_bytes: int = 0
    latent_plain_bytes: int = 0
    scale_stored_bytes: int = 0
    # One positive scale per MLP layer input. Empty unless FLAG_ACT_INT8.
    act_scales: list[float] = field(default_factory=list)

    @property
    def activation_requant(self) -> bool:
        return bool(self.flags & FLAG_ACTIVATION_REQUANT)

    @property
    def act_int8(self) -> bool:
        return bool(self.flags & FLAG_ACT_INT8)

    @property
    def num_channels(self) -> int:
        return len(self.channels)


def pack_codes(codes: np.ndarray, bits: int) -> tuple[bytes, int, int]:
    """Pack ``codes`` uint ``[C,H,W]`` into planar rows.

    Returns ``(blob, row_stride, plane_bytes)``.
    4-bit: even x in bits 3:0, odd x in bits 7:4.
    2-bit: x%4 == k in bits ``2k+1:2k``.
    """
    if codes.dtype != np.uint8:
        codes = codes.astype(np.uint8)
    c, h, w = codes.shape
    stride = row_stride_bytes(w, bits)
    plane = stride * h
    out = np.zeros((c, h, stride), dtype=np.uint8)
    if bits == 8:
        out[:, :, :w] = codes
    elif bits == 4:
        if w % 2:
            raise ValueError("4-bit width must be even")
        pairs = codes.reshape(c, h, w // 2, 2)
        out[:, :, : w // 2] = (pairs[:, :, :, 0] & 0x0F) | ((pairs[:, :, :, 1] & 0x0F) << 4)
    elif bits == 2:
        if w % 4:
            raise ValueError("2-bit width must be a multiple of 4")
        quads = codes.reshape(c, h, w // 4, 4)
        byte = quads[:, :, :, 0] & 0x03
        byte = byte | ((quads[:, :, :, 1] & 0x03) << 2)
        byte = byte | ((quads[:, :, :, 2] & 0x03) << 4)
        byte = byte | ((quads[:, :, :, 3] & 0x03) << 6)
        out[:, :, : w // 4] = byte.astype(np.uint8)
    else:
        raise ValueError(f"unsupported latent bits {bits}")
    return out.tobytes(), stride, plane


def _need(blob: bytes | memoryview, offset: int, nbytes: int, what: str) -> None:
    """Reject a table or payload that would read past ``blob``."""
    size = len(blob)
    if offset < 0 or nbytes < 0 or offset > size or nbytes > size - offset:
        raise ValueError(f"{what} out of range")


def unpack_codes(blob: bytes | memoryview, offset: int, c: int, h: int, w: int, bits: int, stride: int):
    if c < 0 or h < 0 or w < 0 or stride < 0 or bits not in (2, 4, 8):
        raise ValueError(f"bad latent shape bits={bits}")
    if w > 0 and stride <= 0:
        raise ValueError("latent stride")
    nbytes = c * stride * h
    _need(blob, offset, nbytes, "latent codes")
    raw = np.frombuffer(blob, dtype=np.uint8, count=nbytes, offset=offset)
    raw = raw.reshape(c, h, stride)
    if bits == 8:
        return raw[:, :, :w].copy()
    if bits == 4:
        packed = raw[:, :, : w // 2]
        lo = packed & 0x0F
        hi = packed >> 4
        codes = np.empty((c, h, w), dtype=np.uint8)
        codes[:, :, 0::2] = lo
        codes[:, :, 1::2] = hi
        return codes
    if bits == 2:
        packed = raw[:, :, : w // 4]
        codes = np.empty((c, h, w), dtype=np.uint8)
        for k in range(4):
            codes[:, :, k::4] = (packed >> (2 * k)) & 0x03
        return codes
    raise ValueError(bits)


def tile_counts(width: int, height: int, tile: int) -> tuple[int, int]:
    """Return ``(nty, ntx)``. ``(0, 0)`` means one scale per channel."""
    if tile < 2 or (width <= tile and height <= tile):
        return 0, 0
    return math.ceil(height / tile), math.ceil(width / tile)


def deflate_raw(data: bytes) -> bytes:
    co = zlib.compressobj(level=9, wbits=-15)
    return co.compress(data) + co.flush()


def inflate_raw(data: bytes, plain_len: int) -> bytes:
    out = zlib.decompress(data, wbits=-15)
    if len(out) != plain_len:
        raise ValueError(f"deflate length {len(out)} != {plain_len}")
    return out


def _f16_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype=np.float32).astype(np.float16).tobytes()


def _read_f16(blob: bytes, offset: int, count: int) -> np.ndarray:
    if count < 0:
        raise ValueError("fp16 count")
    if offset % 2:
        raise ValueError("fp16 offset must be 2-aligned")
    _need(blob, offset, count * 2, "fp16")
    raw = np.frombuffer(blob, dtype=np.float16, count=count, offset=offset)
    return np.array(raw, dtype=np.float32)


def pack_weight_matrix(codes: np.ndarray, bits: int) -> bytes:
    """``codes`` signed integers ``[out, in]``."""
    codes = np.asarray(codes)
    out, inn = codes.shape
    if bits == 8:
        return np.ascontiguousarray(codes, dtype=np.int8).tobytes()
    if bits != 4:
        raise ValueError(bits)
    if inn % 2:
        codes = np.pad(codes, ((0, 0), (0, 1)))
    pairs = codes.reshape(out, -1, 2).astype(np.int16)
    lo = pairs[:, :, 0] & 0x0F
    hi = pairs[:, :, 1] & 0x0F
    return np.ascontiguousarray((lo | (hi << 4)).astype(np.uint8)).tobytes()


def unpack_weight_matrix(blob: bytes, offset: int, out: int, inn: int, bits: int) -> np.ndarray:
    if out < 0 or inn < 0:
        raise ValueError("bad weight shape")
    if bits == 8:
        _need(blob, offset, out * inn, "weights")
        raw = np.frombuffer(blob, dtype=np.int8, count=out * inn, offset=offset)
        return np.array(raw, dtype=np.float32).reshape(out, inn)
    if bits != 4:
        raise ValueError(bits)
    row_bytes = (inn + 1) // 2
    _need(blob, offset, out * row_bytes, "weights")
    raw = np.frombuffer(blob, dtype=np.uint8, count=out * row_bytes, offset=offset)
    raw = raw.reshape(out, row_bytes)
    lo = (raw & 0x0F).astype(np.int16)
    hi = (raw >> 4).astype(np.int16)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    stacked = np.stack([lo, hi], axis=-1).reshape(out, -1)
    return stacked[:, :inn].astype(np.float32)


def build_texture_set(
    *,
    width: int,
    height: int,
    num_mips: int,
    activation: int,
    pe_mode: int,
    pe_octaves: int,
    flags: int,
    weight_bits: int,
    group_size: int,
    channels: list[ChannelRec],
    grids: list[dict],
    layers: list[dict],
    latent_deflate: bool = False,
    tile_size: int = 0,
    latent_coder: str | None = None,
    act_scales: list[float] | None = None,
) -> bytes:
    """Build a file.

    ``grids`` items: codes uint8 ``[C,H,W]``, bits, scale float ``[C]``.
    A tiled grid also carries ``tile_scale`` and ``tile_offset``, each
    ``[C, nty, ntx]``. ``layers`` items: codes signed ``[out,in]``, bias
    ``[out]``, scale ``[out,groups]``, group_size, optional ``weight_bits``.
    """
    if not (1 <= len(channels) <= MAX_CHANNELS):
        raise ValueError("channel count")
    if not (1 <= len(grids) <= MAX_GRIDS):
        raise ValueError("grid count")
    if not (1 <= len(layers) <= MAX_LAYERS):
        raise ValueError("layer count")
    flags = flags | FLAG_DERIVED_MIPS

    ch_blob = bytearray()
    for ch in channels:
        ch_blob += struct.pack("<BBHI", ch.semantic & 0xFF, ch.color_space & 0xFF, 0, 0)

    grid_recs = []
    for g in grids:
        codes = np.asarray(g["codes"])
        bits = int(g["bits"])
        scale = np.asarray(g["scale"], dtype=np.float32).reshape(-1)
        packed, stride, plane = pack_codes(codes, bits)
        gw, gh, gc = int(codes.shape[2]), int(codes.shape[1]), int(codes.shape[0])
        nty, ntx = tile_counts(gw, gh, tile_size)
        tscale = g.get("tile_scale")
        toffset = g.get("tile_offset")
        if nty:
            if tscale is None or toffset is None:
                raise ValueError("tiled grid missing endpoints")
            tscale = np.asarray(tscale, dtype=np.float32).reshape(gc, nty, ntx)
            toffset = np.asarray(toffset, dtype=np.float32).reshape(gc, nty, ntx)
            scale_bytes = _f16_bytes(tscale.reshape(-1)) + _f16_bytes(toffset.reshape(-1))
        else:
            if len(scale) != gc:
                raise ValueError("scale count")
            scale_bytes = _f16_bytes(scale)
        grid_recs.append(
            {
                "width": gw,
                "height": gh,
                "channels": gc,
                "bits": bits,
                "row_stride": stride,
                "plane_bytes": plane,
                "packed": packed,
                "scale_bytes": scale_bytes,
            }
        )

    weight = bytearray()
    layer_recs = []
    for layer in layers:
        codes = np.asarray(layer["codes"])
        bias = np.asarray(layer["bias"], dtype=np.float32).reshape(-1)
        scale = np.asarray(layer["scale"], dtype=np.float32)
        if scale.ndim == 1:
            scale = scale.reshape(-1, 1)
        gs = int(layer["group_size"])
        out_f, in_f = codes.shape
        groups = scale.shape[1]
        layer_bits = int(layer.get("weight_bits") or weight_bits)
        wbytes = pack_weight_matrix(codes, layer_bits)
        if len(weight) % 2:
            weight += b"\x00"
        w_off = len(weight)
        weight += wbytes
        if len(weight) % 2:
            weight += b"\x00"
        b_off = len(weight)
        weight += _f16_bytes(bias)
        s_off = len(weight)
        weight += _f16_bytes(scale.reshape(-1))
        layer_recs.append(
            {
                "in": in_f,
                "out": out_f,
                "gs": gs,
                "groups": groups,
                "w_off": w_off,
                "w_len": len(wbytes),
                "b_off": b_off,
                "s_off": s_off,
                "bits": layer_bits,
            }
        )

    # Place tables then payloads. Offsets inside payloads are relative
    # until the base is known.
    offset = HEADER_BYTES
    channel_off = offset
    offset += len(ch_blob)
    grid_off = offset
    grid_table_bytes = 24 * len(grid_recs)
    offset += grid_table_bytes
    layer_off = offset
    layer_table_bytes = 32 * len(layer_recs)
    offset += layer_table_bytes
    if offset % 2:
        offset += 1
    latent_off = offset
    # Assign absolute offsets while concatenating latents and scales.
    latent_blob = bytearray()
    scale_blob = bytearray()
    abs_grids = []
    for rec in grid_recs:
        if len(latent_blob) % 2:
            latent_blob += b"\x00"
        rel = len(latent_blob)
        latent_blob += rec["packed"]
        abs_grids.append((rec, rel))
    plain = bytes(latent_blob)
    plain_len = len(plain)
    coder = latent_coder
    if coder is None:
        coder = "deflate" if latent_deflate else "raw"
    if coder == "deflate":
        stored_latent = deflate_raw(plain)
        flags = flags | FLAG_LATENT_DEFLATE
    elif coder == "range":
        from ntc.range_coder import range_encode

        stored_latent = range_encode(plain)
        flags = flags | FLAG_LATENT_RANGE
    elif coder == "best":
        from ntc.range_coder import range_encode

        deflated = deflate_raw(plain)
        ranged = range_encode(plain)
        if len(ranged) < len(deflated):
            stored_latent = ranged
            flags = flags | FLAG_LATENT_RANGE
        else:
            stored_latent = deflated
            flags = flags | FLAG_LATENT_DEFLATE
    elif coder in ("context", "best_all"):
        from ntc.range_coder import range_encode, range_encode_ctx

        coded = range_encode_ctx(plain)
        flags_here = FLAG_LATENT_CTX
        if coder == "best_all":
            deflated = deflate_raw(plain)
            ranged = range_encode(plain)
            coded, flags_here = min(
                ((coded, FLAG_LATENT_CTX), (ranged, FLAG_LATENT_RANGE), (deflated, FLAG_LATENT_DEFLATE)),
                key=lambda item: len(item[0]),
            )
        stored_latent = coded
        flags = flags | flags_here
    elif coder == "raw":
        stored_latent = plain
    else:
        raise ValueError(f"latent coder {coder}")
    latent_bytes = len(stored_latent)
    offset = latent_off + latent_bytes
    if offset % 2:
        offset += 1
    scale_off = offset
    abs_scales = []
    for rec, rel in abs_grids:
        s_off = scale_off + len(scale_blob)
        scale_blob += rec["scale_bytes"]
        data_off = rel if (flags & FLAG_LATENT_COMPRESSED) else latent_off + rel
        abs_scales.append((rec, data_off, s_off))
    scale_bytes = len(scale_blob)
    offset = scale_off + scale_bytes
    weight_off = offset
    weight_blob = bytes(weight)
    weight_bytes = len(weight_blob)
    tail = b""
    if act_scales:
        flags = flags | FLAG_ACT_INT8
        tail = np.asarray(act_scales, np.float32).astype(np.float16).tobytes()
        if len(tail) != len(act_scales) * 2:
            raise RuntimeError("act scale tail")
    file_bytes = weight_off + weight_bytes + len(tail)

    grid_table = bytearray()
    for rec, data_off, s_off in abs_scales:
        grid_table += struct.pack(
            "<HHBBHIIII",
            rec["width"],
            rec["height"],
            rec["channels"],
            rec["bits"],
            rec["row_stride"],
            data_off,
            len(rec["packed"]),
            s_off,
            rec["plane_bytes"],
        )
    if len(grid_table) != grid_table_bytes:
        raise RuntimeError("grid table size")

    layer_table = bytearray()
    for rec in layer_recs:
        layer_table += struct.pack(
            "<HHIIIIHHII",
            rec["in"],
            rec["out"],
            weight_off + rec["w_off"],
            rec["w_len"],
            weight_off + rec["b_off"],
            weight_off + rec["s_off"],
            rec["gs"] & 0xFFFF,
            rec["groups"] & 0xFFFF,
            rec["bits"] & 0xFFFFFFFF,
            0,
        )

    body = bytearray(file_bytes)
    body[channel_off : channel_off + len(ch_blob)] = ch_blob
    body[grid_off : grid_off + len(grid_table)] = grid_table
    body[layer_off : layer_off + len(layer_table)] = layer_table
    body[latent_off : latent_off + latent_bytes] = stored_latent
    body[scale_off : scale_off + scale_bytes] = scale_blob
    body[weight_off : weight_off + weight_bytes] = weight_blob
    if tail:
        body[weight_off + weight_bytes : weight_off + weight_bytes + len(tail)] = tail

    crc = zlib.crc32(body[HEADER_BYTES:]) & 0xFFFFFFFF
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        HEADER_BYTES,
        flags,
        width,
        height,
        len(channels),
        num_mips,
        len(grids),
        len(layers),
        activation,
        pe_mode,
        pe_octaves,
        weight_bits,
        group_size & 0xFFFF,
        tile_size & 0xFFFF,
        channel_off,
        grid_off,
        layer_off,
        latent_off,
        latent_bytes,
        weight_off,
        weight_bytes,
        scale_off,
        scale_bytes,
        crc,
        file_bytes,
        plain_len if (flags & FLAG_LATENT_COMPRESSED) else 0,
        0,
    )
    if len(header) != HEADER_BYTES:
        raise RuntimeError(f"header {len(header)}")
    body[:HEADER_BYTES] = header
    return bytes(body)


def parse_texture_set(blob: bytes) -> TextureSet:
    if len(blob) < HEADER_BYTES:
        raise ValueError("short file")
    fields = _HEADER.unpack_from(blob, 0)
    (
        magic,
        version,
        header_bytes,
        flags,
        width,
        height,
        num_channels,
        num_mips,
        num_grids,
        num_layers,
        activation,
        pe_mode,
        pe_octaves,
        weight_bits,
        group_size,
        tile_size,
        channel_off,
        grid_off,
        layer_off,
        latent_off,
        latent_bytes,
        weight_off,
        weight_bytes,
        scale_off,
        scale_bytes,
        crc,
        file_bytes,
        latent_plain,
        _r2,
    ) = fields
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic:#x}")
    if version != VERSION:
        raise ValueError(f"bad version {version}")
    if header_bytes != HEADER_BYTES:
        raise ValueError("bad header size")
    if file_bytes != len(blob):
        raise ValueError(f"file_bytes {file_bytes} != {len(blob)}")
    if width <= 0 or height <= 0 or num_mips <= 0:
        raise ValueError("empty image")
    got = zlib.crc32(blob[HEADER_BYTES:]) & 0xFFFFFFFF
    if got != crc:
        raise ValueError(f"crc mismatch {got:#x} != {crc:#x}")
    if _HEADER.size != HEADER_BYTES:
        raise RuntimeError("struct size")
    _need(blob, channel_off, num_channels * 8, "channel table")
    _need(blob, grid_off, num_grids * 24, "grid table")
    _need(blob, layer_off, num_layers * 32, "layer table")

    nflags = bin(flags & (FLAG_LATENT_DEFLATE | FLAG_LATENT_RANGE | FLAG_LATENT_CTX)).count("1")
    if nflags > 1:
        raise ValueError("latent coder flags are mutually exclusive")
    code_src: bytes | memoryview = blob
    if flags & FLAG_LATENT_COMPRESSED:
        if latent_plain <= 0 or latent_plain > 64 * 1024 * 1024:
            raise ValueError("compressed latent without inflated length")
        _need(blob, int(latent_off), int(latent_bytes), "compressed latent")
        packed = blob[latent_off : latent_off + latent_bytes]
        if flags & FLAG_LATENT_DEFLATE:
            code_src = inflate_raw(packed, latent_plain)
        elif flags & FLAG_LATENT_CTX:
            from ntc.range_coder import range_decode_ctx

            code_src = range_decode_ctx(packed, latent_plain)
        else:
            from ntc.range_coder import range_decode

            code_src = range_decode(packed, latent_plain)

    channels = []
    for i in range(num_channels):
        sem, cs, _, _ = struct.unpack_from("<BBHI", blob, channel_off + 8 * i)
        channels.append(ChannelRec(sem, cs))

    grids = []
    for i in range(num_grids):
        w, h, ch, bits, stride, data_off, data_len, s_off, plane = struct.unpack_from(
            "<HHBBHIIII", blob, grid_off + 24 * i
        )
        codes = unpack_codes(code_src, data_off, ch, h, w, bits, stride)
        nty, ntx = tile_counts(w, h, tile_size)
        tile_scale = None
        tile_offset = None
        if nty:
            n = ch * nty * ntx
            tile_scale = _read_f16(blob, s_off, n).reshape(ch, nty, ntx)
            tile_offset = _read_f16(blob, s_off + n * 2, n).reshape(ch, nty, ntx)
            scale = tile_scale.reshape(ch, -1).mean(axis=1)
        else:
            scale = _read_f16(blob, s_off, ch)
        grids.append(
            GridRec(
                w, h, ch, bits, stride, data_off, data_len, s_off, plane, scale, codes,
                tile_scale, tile_offset,
            )
        )

    layers = []
    for i in range(num_layers):
        inn, out, w_off, w_len, b_off, s_off, gs, groups, layer_bits, _ = struct.unpack_from(
            "<HHIIIIHHII", blob, layer_off + 32 * i
        )
        use_bits = int(layer_bits) if layer_bits else int(weight_bits)
        weight = unpack_weight_matrix(blob, w_off, out, inn, use_bits)
        bias = _read_f16(blob, b_off, out)
        scale = _read_f16(blob, s_off, out * groups).reshape(out, groups)
        layers.append(
            LayerRec(
                inn, out, gs, groups, w_off, w_len, b_off, s_off, weight, bias, scale, use_bits
            )
        )

    return TextureSet(
        width=width,
        height=height,
        num_mips=num_mips,
        activation=activation,
        pe_mode=pe_mode,
        pe_octaves=pe_octaves,
        flags=flags,
        weight_bits=weight_bits,
        group_size=group_size,
        channels=channels,
        grids=grids,
        layers=layers,
        raw=blob,
        tile_size=int(tile_size),
        latent_stored_bytes=int(latent_bytes),
        latent_plain_bytes=int(latent_plain) if flags & FLAG_LATENT_COMPRESSED else int(latent_bytes),
        scale_stored_bytes=int(scale_bytes),
        act_scales=_read_act_scales(blob, flags, num_layers),
    )


def _read_act_scales(blob: bytes, flags: int, num_layers: int) -> list[float]:
    if not (flags & FLAG_ACT_INT8) or num_layers <= 0:
        return []
    nbytes = num_layers * 2
    if len(blob) < nbytes:
        raise ValueError("act int8 tail missing")
    raw = np.frombuffer(blob[len(blob) - nbytes :], dtype=np.float16).astype(np.float32)
    return [float(v) for v in raw]


def payload_bytes(ts: TextureSet) -> int:
    """Latent codes + scales + weights + bias. Excludes header and tables.

    When the latent region is deflated, the compressed length is the rate.
    """
    if ts.flags & FLAG_LATENT_COMPRESSED:
        latent = ts.latent_stored_bytes
    else:
        latent = sum(g.data_bytes for g in ts.grids)
    scales = ts.scale_stored_bytes if ts.scale_stored_bytes else sum(g.channels * 2 for g in ts.grids)
    weights = 0
    for layer in ts.layers:
        weights += layer.weight_bytes
        weights += layer.out_features * 2
        weights += layer.out_features * layer.num_groups * 2
    return latent + scales + weights


def total_bpp(ts: TextureSet) -> float:
    """Historical rate. Payload bits over mip 0 texels.

    Payload is compressed latent codes + fp16 latent scales + integer
    weights + fp16 bias + fp16 weight scales. The 96-byte header and the
    descriptor tables are not included. The denominator is ``width * height``,
    not the mip chain, and the count is not divided by the channel count.
    """
    return 8.0 * payload_bytes(ts) / float(ts.width * ts.height)


def file_bpp(blob: bytes, width: int, height: int) -> float:
    """Whole-file bits over mip 0 texels. Not the paper rate and not the SDK rate."""
    return 8.0 * len(blob) / float(width * height)


def mip_chain_texels(width: int, height: int, num_mips: int) -> int:
    """Texels represented by mips ``0 .. num_mips-1``. A full chain uses
    ``floor(log2(max(width, height))) + 1`` levels, down to 1x1.
    """
    total = 0
    for mip in range(max(int(num_mips), 1)):
        total += max(width >> mip, 1) * max(height >> mip, 1)
    return total


def _latent_code_bytes(ts: TextureSet) -> int:
    if ts.flags & FLAG_LATENT_COMPRESSED:
        return int(ts.latent_stored_bytes)
    return int(sum(g.data_bytes for g in ts.grids))


def _scale_bytes(ts: TextureSet) -> int:
    if ts.scale_stored_bytes:
        return int(ts.scale_stored_bytes)
    return int(sum(g.channels * 2 for g in ts.grids))


def rate_report(ts: TextureSet) -> dict:
    """Three rates, plus the historical ``total_bpp``.

    ``latent_bpp_mip0`` is the RTXNTC SDK comparison rate: stored latent
    codes plus the fp16 per-channel latent scales, in bits, divided by mip 0
    texels. Network weights, bias, weight scales, the header, and the
    descriptor tables are excluded. ``--bitsPerPixel`` on ``ntc-cli`` is a
    latent-grid rate of this kind. It is not ``total_bpp``.

    ``file_bpp_chain`` is the whole file (latents + scales + MLP weights +
    header + descriptor tables) in bits, divided by the texels in every mip
    the header declares. Derived color mips are not a second copy of those
    texels in the file. The denominator is the chain the file represents.

    ``bppc`` is the paper rate (Vaidyanathan et al. 2023): the same payload
    as ``total_bpp`` (latents, scales, and weights, no header), divided by
    mip-chain texels and by the channel count.
    """
    chain = mip_chain_texels(ts.width, ts.height, ts.num_mips)
    channels = max(len(ts.channels), 1)
    latent = _latent_code_bytes(ts) + _scale_bytes(ts)
    payload = payload_bytes(ts)
    mip0 = float(ts.width * ts.height)
    return {
        "latent_bpp_mip0": 8.0 * latent / mip0,
        "file_bpp_chain": 8.0 * len(ts.raw) / float(chain),
        "bppc": 8.0 * payload / float(chain * channels),
        "total_bpp": 8.0 * payload / mip0,
        "mip_chain_texels": chain,
        "num_channels": len(ts.channels),
        "latent_code_bytes": _latent_code_bytes(ts),
        "latent_scale_bytes": _scale_bytes(ts),
        "payload_bytes": payload,
        "file_bytes": len(ts.raw),
        "header_bytes": HEADER_BYTES,
        "network_bytes": _network_bytes(ts),
    }


def _network_bytes(ts: TextureSet) -> int:
    """Integer weights plus fp16 bias and fp16 weight scales. Not the header."""
    n = 0
    for layer in ts.layers:
        n += layer.weight_bytes
        n += layer.out_features * 2
        n += layer.out_features * layer.num_groups * 2
    return n
