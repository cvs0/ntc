"""NTC1 container, version 1.

Little-endian. Multi-byte integers are explicit. The header is 96 bytes
and is not a C struct on the wire; the C++ reader uses the same offsets.

See the format tables in the README. ``crc32`` is zlib CRC-32 of every
byte after the header.
"""

from __future__ import annotations

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

    @property
    def activation_requant(self) -> bool:
        return bool(self.flags & FLAG_ACTIVATION_REQUANT)

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


def unpack_codes(blob: bytes | memoryview, offset: int, c: int, h: int, w: int, bits: int, stride: int):
    raw = np.frombuffer(blob, dtype=np.uint8, count=c * stride * h, offset=offset)
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


def _f16_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype=np.float32).astype(np.float16).tobytes()


def _read_f16(blob: bytes, offset: int, count: int) -> np.ndarray:
    if offset % 2:
        raise ValueError("fp16 offset must be 2-aligned")
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
    if bits == 8:
        raw = np.frombuffer(blob, dtype=np.int8, count=out * inn, offset=offset)
        return np.array(raw, dtype=np.float32).reshape(out, inn)
    if bits != 4:
        raise ValueError(bits)
    row_bytes = (inn + 1) // 2
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
) -> bytes:
    """Build a file.

    ``grids`` items: codes uint8 ``[C,H,W]``, bits, scale float ``[C]``.
    ``layers`` items: codes signed ``[out,in]``, bias ``[out]``, scale ``[out,groups]``,
    group_size.
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
        grid_recs.append(
            {
                "width": codes.shape[2],
                "height": codes.shape[1],
                "channels": codes.shape[0],
                "bits": bits,
                "row_stride": stride,
                "plane_bytes": plane,
                "packed": packed,
                "scale_bytes": _f16_bytes(scale),
            }
        )
        if len(scale) != codes.shape[0]:
            raise ValueError("scale count")

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
        wbytes = pack_weight_matrix(codes, weight_bits)
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
        data_off = latent_off + len(latent_blob)
        latent_blob += rec["packed"]
        abs_grids.append((rec, data_off))
    latent_bytes = len(latent_blob)
    offset = latent_off + latent_bytes
    if offset % 2:
        offset += 1
    scale_off = offset
    abs_scales = []
    for rec, data_off in abs_grids:
        s_off = scale_off + len(scale_blob)
        scale_blob += rec["scale_bytes"]
        abs_scales.append((rec, data_off, s_off))
    scale_bytes = len(scale_blob)
    offset = scale_off + scale_bytes
    weight_off = offset
    weight_blob = bytes(weight)
    weight_bytes = len(weight_blob)
    file_bytes = weight_off + weight_bytes

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
            0,
            0,
        )

    body = bytearray(file_bytes)
    body[channel_off : channel_off + len(ch_blob)] = ch_blob
    body[grid_off : grid_off + len(grid_table)] = grid_table
    body[layer_off : layer_off + len(layer_table)] = layer_table
    body[latent_off : latent_off + latent_bytes] = latent_blob
    body[scale_off : scale_off + scale_bytes] = scale_blob
    body[weight_off : weight_off + weight_bytes] = weight_blob

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
        0,
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
        0,
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
        _reserved0,
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
        _r1,
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
    got = zlib.crc32(blob[HEADER_BYTES:]) & 0xFFFFFFFF
    if got != crc:
        raise ValueError(f"crc mismatch {got:#x} != {crc:#x}")
    if _HEADER.size != HEADER_BYTES:
        raise RuntimeError("struct size")

    channels = []
    for i in range(num_channels):
        sem, cs, _, _ = struct.unpack_from("<BBHI", blob, channel_off + 8 * i)
        channels.append(ChannelRec(sem, cs))

    grids = []
    for i in range(num_grids):
        w, h, ch, bits, stride, data_off, data_len, s_off, plane = struct.unpack_from(
            "<HHBBHIIII", blob, grid_off + 24 * i
        )
        codes = unpack_codes(blob, data_off, ch, h, w, bits, stride)
        scale = _read_f16(blob, s_off, ch)
        grids.append(
            GridRec(w, h, ch, bits, stride, data_off, data_len, s_off, plane, scale, codes)
        )

    layers = []
    for i in range(num_layers):
        inn, out, w_off, w_len, b_off, s_off, gs, groups, _, _ = struct.unpack_from(
            "<HHIIIIHHII", blob, layer_off + 32 * i
        )
        weight = unpack_weight_matrix(blob, w_off, out, inn, weight_bits)
        bias = _read_f16(blob, b_off, out)
        scale = _read_f16(blob, s_off, out * groups).reshape(out, groups)
        layers.append(
            LayerRec(inn, out, gs, groups, w_off, w_len, b_off, s_off, weight, bias, scale)
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
    )


def payload_bytes(ts: TextureSet) -> int:
    """Latent codes + scales + weights + bias. Excludes header and tables."""
    latent = sum(g.data_bytes for g in ts.grids)
    scales = sum(g.channels * 2 for g in ts.grids)
    weights = 0
    for layer in ts.layers:
        weights += layer.weight_bytes
        weights += layer.out_features * 2
        weights += layer.out_features * layer.num_groups * 2
    return latent + scales + weights


def total_bpp(ts: TextureSet) -> float:
    return 8.0 * payload_bytes(ts) / float(ts.width * ts.height)


def file_bpp(blob: bytes, width: int, height: int) -> float:
    return 8.0 * len(blob) / float(width * height)
