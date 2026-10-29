"""BC1, BC5, and BC7 mode 6 packer. Writes an NTCB file.

NTCB layout, little-endian: u32 width, u32 height, u32 format, u32 payload_bytes,
u32 magic ``0x4243544E``, then the payload. Format 1 is BC7, 2 is BC1, 3 is BC5.

BC7 is mode 6 only (one subset, 7-bit endpoints, two p-bits, 4-bit indices, anchor
index stored in 3 bits). This is not ``bc7enc``. Screenshot and timing files say
``mode6`` so they are not compared to the ``bc7enc`` rate-distortion table.

BC1 is DXT1. BC5 is two BC4 blocks (X then Y).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

FMT_BC7 = 1
FMT_BC1 = 2
FMT_BC5 = 3
MAGIC = 0x4243544E

_W4 = (0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64)


def write_ntcb(path: Path, width: int, height: int, fmt: int, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = struct.pack("<IIIII", width, height, fmt, len(payload), MAGIC) + payload
    path.write_bytes(blob)


def _rgb565(r: int, g: int, b: int) -> int:
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def _from565(c: int) -> tuple[int, int, int]:
    r5 = (c >> 11) & 31
    g6 = (c >> 5) & 63
    b5 = c & 31
    return (r5 << 3) | (r5 >> 2), (g6 << 2) | (g6 >> 4), (b5 << 3) | (b5 >> 2)


def pack_bc1_block(px: np.ndarray) -> bytes:
    """``px`` is uint8 ``[16, 3]`` in raster order."""
    cols = px.astype(np.int32)
    lum = cols[:, 0] * 2 + cols[:, 1] * 4 + cols[:, 2]
    c0 = cols[int(np.argmax(lum))]
    c1 = cols[int(np.argmin(lum))]
    e0 = _rgb565(int(c0[0]), int(c0[1]), int(c0[2]))
    e1 = _rgb565(int(c1[0]), int(c1[1]), int(c1[2]))
    if e0 == e1:
        return struct.pack("<HH", e0, e1) + b"\x00\x00\x00\x00"
    if e0 < e1:
        e0, e1 = e1, e0
    p0 = np.array(_from565(e0), dtype=np.int32)
    p1 = np.array(_from565(e1), dtype=np.int32)
    pal = (p0, p1, (2 * p0 + p1) // 3, (p0 + 2 * p1) // 3)
    indices = 0
    for i in range(16):
        d = [int(np.sum((cols[i] - p) ** 2)) for p in pal]
        indices |= int(np.argmin(d)) << (2 * i)
    return struct.pack("<HHI", e0, e1, indices)


def unpack_bc1_block(raw: bytes) -> np.ndarray:
    e0, e1, indices = struct.unpack("<HHI", raw[:8])
    p0 = np.array(_from565(e0), dtype=np.int32)
    p1 = np.array(_from565(e1), dtype=np.int32)
    if e0 > e1:
        pal = (p0, p1, (2 * p0 + p1) // 3, (p0 + 2 * p1) // 3)
    else:
        pal = (p0, p1, (p0 + p1) // 2, np.array([0, 0, 0], dtype=np.int32))
    out = np.zeros((16, 4), dtype=np.uint8)
    for i in range(16):
        k = (indices >> (2 * i)) & 3
        out[i, :3] = pal[k]
        out[i, 3] = 0 if (e0 <= e1 and k == 3) else 255
    return out


def _bc4_block(values: np.ndarray) -> bytes:
    """``values`` uint8 length 16. Returns 8 bytes."""
    v = values.astype(np.int32)
    a = int(v.max())
    b = int(v.min())
    if a == b:
        return struct.pack("<BB", a, b) + b"\x00" * 6
    if a > b:
        pal = [a, b] + [((7 - i) * a + (i + 1) * b) // 8 for i in range(6)]
    else:
        pal = [a, b] + [((5 - i) * a + (i + 1) * b) // 6 for i in range(4)] + [0, 255]
    bits = 0
    for i in range(16):
        k = min(range(len(pal)), key=lambda t, i=i: abs(int(pal[t]) - int(v[i])))
        bits |= k << (3 * i)
    raw = struct.pack("<BB", a, b) + bits.to_bytes(6, "little")
    return raw


def unpack_bc4_block(raw: bytes) -> np.ndarray:
    a = raw[0]
    b = raw[1]
    bits = int.from_bytes(raw[2:8], "little")
    if a > b:
        pal = [a, b] + [((7 - i) * a + (i + 1) * b) // 8 for i in range(6)]
    else:
        pal = [a, b] + [((5 - i) * a + (i + 1) * b) // 6 for i in range(4)] + [0, 255]
    out = np.zeros(16, dtype=np.uint8)
    for i in range(16):
        out[i] = pal[(bits >> (3 * i)) & 7]
    return out


def pack_bc5_block(px: np.ndarray) -> bytes:
    """``px`` uint8 ``[16, 2]``."""
    return _bc4_block(px[:, 0]) + _bc4_block(px[:, 1])


def _put_bits(words: list[int], bitpos: int, value: int, nbits: int) -> int:
    value &= (1 << nbits) - 1
    for i in range(nbits):
        if (value >> i) & 1:
            words[(bitpos + i) // 64] |= 1 << ((bitpos + i) % 64)
    return bitpos + nbits


def pack_bc7_mode6_block(px: np.ndarray) -> bytes:
    """``px`` uint8 ``[16, 4]``."""
    cols = px.astype(np.int32)
    # Endpoints are the min and max of each channel. The p-bit is the low bit of
    # red. Indices are the nearest of the 16 mode-6 weights.
    lo = cols.min(axis=0)
    hi = cols.max(axis=0)
    # One p-bit per endpoint, shared across channels. Use the red channel's low bit.
    ends0 = []
    ends1 = []
    p0 = int(lo[0]) & 1
    p1 = int(hi[0]) & 1
    for ch in range(4):
        e0 = int(np.clip(lo[ch], 0, 255)) >> 1
        e1 = int(np.clip(hi[ch], 0, 255)) >> 1
        ends0.append(e0)
        ends1.append(e1)
    r0 = (np.array(ends0) << 1) | p0
    r1 = (np.array(ends1) << 1) | p1
    palette = []
    for i, w in enumerate(_W4):
        iw = 64 - w
        palette.append(((r0 * iw + r1 * w + 32) >> 6).astype(np.int32))
    idx = []
    for i in range(16):
        best = 0
        bestd = 1 << 30
        for k, pal in enumerate(palette):
            d = int(np.sum((cols[i] - pal) ** 2))
            if d < bestd:
                bestd = d
                best = k
        idx.append(best)
    if idx[0] >= 8:
        idx = [15 - k for k in idx]
        ends0, ends1 = ends1, ends0
        p0, p1 = p1, p0
    words = [0, 0]
    pos = 0
    pos = _put_bits(words, pos, 1 << 6, 7)
    for a, b in zip(ends0, ends1, strict=True):
        pos = _put_bits(words, pos, a, 7)
        pos = _put_bits(words, pos, b, 7)
    pos = _put_bits(words, pos, p0, 1)
    pos = _put_bits(words, pos, p1, 1)
    pos = _put_bits(words, pos, idx[0], 3)
    for k in idx[1:]:
        pos = _put_bits(words, pos, k, 4)
    if pos != 128:
        raise RuntimeError(f"mode 6 bit count {pos}")
    return struct.pack("<QQ", words[0], words[1])


def unpack_bc7_mode6_block(raw: bytes) -> np.ndarray:
    lo, hi = struct.unpack("<QQ", raw[:16])

    def take(word_bits: list[int], n: int) -> int:
        v = 0
        for i in range(n):
            bit = word_bits.pop(0)
            v |= bit << i
        return v

    bits = [((lo >> i) & 1) for i in range(64)] + [((hi >> i) & 1) for i in range(64)]
    mode = take(bits, 7)
    if mode != (1 << 6):
        raise ValueError("not mode 6")
    ends = [(take(bits, 7), take(bits, 7)) for _ in range(4)]
    p0 = take(bits, 1)
    p1 = take(bits, 1)
    r0 = np.array([(e[0] << 1) | p0 for e in ends], dtype=np.int32)
    r1 = np.array([(e[1] << 1) | p1 for e in ends], dtype=np.int32)
    palette = []
    for w in _W4:
        iw = 64 - w
        palette.append(((r0 * iw + r1 * w + 32) >> 6).astype(np.uint8))
    index = [take(bits, 3)]
    index.extend(take(bits, 4) for _ in range(15))
    out = np.zeros((16, 4), dtype=np.uint8)
    for i, k in enumerate(index):
        out[i] = palette[k]
    return out


def pack_image_bc1(image: np.ndarray) -> bytes:
    h, w, _c = image.shape
    out = bytearray()
    src = image[:, :, :3]
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            out += pack_bc1_block(src[y : y + 4, x : x + 4].reshape(16, 3))
    return bytes(out)


def pack_image_bc5(image: np.ndarray) -> bytes:
    out = bytearray()
    src = image[:, :, :2]
    h, w, _c = src.shape
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            out += pack_bc5_block(src[y : y + 4, x : x + 4].reshape(16, 2))
    return bytes(out)


def pack_image_bc7(image: np.ndarray) -> bytes:
    h, w, c = image.shape
    src = image
    if c < 4:
        pad = np.full((h, w, 4), 255, dtype=np.uint8)
        pad[:, :, :c] = image
        src = pad
    out = bytearray()
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            out += pack_bc7_mode6_block(src[y : y + 4, x : x + 4].reshape(16, 4))
    return bytes(out)


def image_u8(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image)
    if x.dtype != np.uint8:
        x = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    return x


def write_material_blocks(image_f32: np.ndarray, dest: Path, semantics: list[int]) -> dict:
    """Write albedo BC7, albedo BC1, and normal BC5 when those channels exist.

    ``image_f32`` is HWC in ``[0, 1]``.
    """
    img = image_u8(image_f32)
    h, w, _c = img.shape
    sem = {s: i for i, s in enumerate(semantics)}
    written = {}
    if all(s in sem for s in (0, 1, 2)):
        alb = img[:, :, [sem[0], sem[1], sem[2]]]
        rgba = np.full((h, w, 4), 255, dtype=np.uint8)
        rgba[:, :, :3] = alb
        write_ntcb(dest / "albedo_bc7.ntcb", w, h, FMT_BC7, pack_image_bc7(rgba))
        write_ntcb(dest / "albedo_bc1.ntcb", w, h, FMT_BC1, pack_image_bc1(alb))
        written["bc7"] = "albedo_bc7.ntcb"
        written["bc1"] = "albedo_bc1.ntcb"
    if all(s in sem for s in (3, 4)):
        nrm = img[:, :, [sem[3], sem[4]]]
        write_ntcb(dest / "normal_bc5.ntcb", w, h, FMT_BC5, pack_image_bc5(nrm))
        written["bc5"] = "normal_bc5.ntcb"
    if all(s in sem for s in (6, 7, 8)):
        orm = np.full((h, w, 4), 255, dtype=np.uint8)
        orm[:, :, 0] = img[:, :, sem[8]]
        orm[:, :, 1] = img[:, :, sem[6]]
        orm[:, :, 2] = img[:, :, sem[7]]
        write_ntcb(dest / "orm_bc7.ntcb", w, h, FMT_BC7, pack_image_bc7(orm))
        written["orm_bc7"] = "orm_bc7.ntcb"
    written["encoder"] = "bc_pack.py mode6 / dxt1 / bc4"
    return written
