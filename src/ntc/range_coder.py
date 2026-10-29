"""Adaptive range coders over nibbles.

``range_encode`` is order-1: 16 contexts, one per previous nibble.
``range_encode_ctx`` uses 256 contexts from the previous nibble, two high
bits of the previous byte, and four position bands along the plain latent.
Low nibble first. Counts start at 1 and rescale at 4096. The same arithmetic
is in ``cpp/src/range_coder.hpp``.
"""

from __future__ import annotations

_ALPH = 16
_RESCALE = 4096


def _rescale(freq: list[int]) -> int:
    total = 0
    for i in range(len(freq)):
        freq[i] = (freq[i] + 1) // 2
        total += freq[i]
    return total


def _new_model() -> tuple[list[list[int]], list[int]]:
    freq = [[1] * _ALPH for _ in range(_ALPH)]
    total = [_ALPH] * _ALPH
    return freq, total


def range_encode(data: bytes) -> bytes:
    low = 0
    high = 0xFFFFFFFF
    out = bytearray()
    freq, total = _new_model()
    ctx = 0

    def renorm() -> None:
        nonlocal low, high
        while ((low ^ high) & 0xFF000000) == 0:
            out.append((low >> 24) & 0xFF)
            low = (low << 8) & 0xFFFFFFFF
            high = ((high << 8) | 0xFF) & 0xFFFFFFFF

    def put(cum: int, fr: int, tot: int) -> None:
        nonlocal low, high
        span = (high - low + 1)
        high = (low + (span * (cum + fr)) // tot - 1) & 0xFFFFFFFF
        low = (low + (span * cum) // tot) & 0xFFFFFFFF
        renorm()

    for byte in data:
        for shift in (0, 4):
            sym = (byte >> shift) & 0xF
            row = freq[ctx]
            cum = sum(row[:sym])
            put(cum, row[sym], total[ctx])
            row[sym] += 1
            total[ctx] += 1
            if total[ctx] >= _RESCALE:
                total[ctx] = _rescale(row)
            ctx = sym
    for _ in range(4):
        out.append((low >> 24) & 0xFF)
        low = (low << 8) & 0xFFFFFFFF
    return bytes(out)


def range_decode(data: bytes, plain_len: int) -> bytes:
    if len(data) < 4:
        raise ValueError("short range stream")
    low = 0
    high = 0xFFFFFFFF
    pos = 0

    def read() -> int:
        nonlocal pos
        if pos >= len(data):
            return 0
        b = data[pos]
        pos += 1
        return b

    code = 0
    for _ in range(4):
        code = ((code << 8) | read()) & 0xFFFFFFFF
    freq, total = _new_model()
    ctx = 0
    out = bytearray()

    def renorm() -> None:
        nonlocal low, high, code
        while ((low ^ high) & 0xFF000000) == 0:
            code = ((code << 8) | read()) & 0xFFFFFFFF
            low = (low << 8) & 0xFFFFFFFF
            high = ((high << 8) | 0xFF) & 0xFFFFFFFF

    def take(cum: int, fr: int, tot: int) -> None:
        nonlocal low, high
        span = high - low + 1
        high = (low + (span * (cum + fr)) // tot - 1) & 0xFFFFFFFF
        low = (low + (span * cum) // tot) & 0xFFFFFFFF
        renorm()

    for _ in range(plain_len):
        byte = 0
        for shift in (0, 4):
            span = high - low + 1
            tot = total[ctx]
            value = ((code - low + 1) * tot - 1) // span
            row = freq[ctx]
            cum = 0
            sym = _ALPH - 1
            for s in range(_ALPH):
                nxt = cum + row[s]
                if value < nxt:
                    sym = s
                    break
                cum = nxt
            take(cum, row[sym], tot)
            row[sym] += 1
            total[ctx] += 1
            if total[ctx] >= _RESCALE:
                total[ctx] = _rescale(row)
            byte |= sym << shift
            ctx = sym
        out.append(byte)
    if len(out) != plain_len:
        raise ValueError("range length")
    return bytes(out)


def _ctx_id(prev: int, left: int, index: int, n: int) -> int:
    """256 contexts: previous nibble, 2 bits of the left byte, 4 stream bands.

    The band is ``index`` along the plain latent, which moves from the fine
    grid to the coarse grid as the bytes are concatenated. No side info.
    """
    band = 0 if n <= 1 else min(3, (index * 4) // n)
    return (prev & 15) | (((left >> 6) & 3) << 4) | (band << 6)


def range_encode_ctx(data: bytes) -> bytes:
    """Same arithmetic as ``range_encode``, with neighbor and position context."""
    low = 0
    high = 0xFFFFFFFF
    out = bytearray()
    freq = [[1] * _ALPH for _ in range(256)]
    total = [_ALPH] * 256
    prev = 0
    n = len(data)

    def renorm() -> None:
        nonlocal low, high
        while ((low ^ high) & 0xFF000000) == 0:
            out.append((low >> 24) & 0xFF)
            low = (low << 8) & 0xFFFFFFFF
            high = ((high << 8) | 0xFF) & 0xFFFFFFFF

    def put(cum: int, fr: int, tot: int) -> None:
        nonlocal low, high
        span = high - low + 1
        high = (low + (span * (cum + fr)) // tot - 1) & 0xFFFFFFFF
        low = (low + (span * cum) // tot) & 0xFFFFFFFF
        renorm()

    for i, byte in enumerate(data):
        left = data[i - 1] if i else 0
        for shift in (0, 4):
            sym = (byte >> shift) & 0xF
            ctx = _ctx_id(prev, left, i, n)
            row = freq[ctx]
            cum = sum(row[:sym])
            put(cum, row[sym], total[ctx])
            row[sym] += 1
            total[ctx] += 1
            if total[ctx] >= _RESCALE:
                total[ctx] = _rescale(row)
            prev = sym
    for _ in range(4):
        out.append((low >> 24) & 0xFF)
        low = (low << 8) & 0xFFFFFFFF
    return bytes(out)


def range_decode_ctx(data: bytes, plain_len: int) -> bytes:
    if len(data) < 4:
        raise ValueError("short range stream")
    low = 0
    high = 0xFFFFFFFF
    pos = 0

    def read() -> int:
        nonlocal pos
        if pos >= len(data):
            return 0
        b = data[pos]
        pos += 1
        return b

    code = 0
    for _ in range(4):
        code = ((code << 8) | read()) & 0xFFFFFFFF
    freq = [[1] * _ALPH for _ in range(256)]
    total = [_ALPH] * 256
    prev = 0
    out = bytearray()

    def renorm() -> None:
        nonlocal low, high, code
        while ((low ^ high) & 0xFF000000) == 0:
            code = ((code << 8) | read()) & 0xFFFFFFFF
            low = (low << 8) & 0xFFFFFFFF
            high = ((high << 8) | 0xFF) & 0xFFFFFFFF

    def take(cum: int, fr: int, tot: int) -> None:
        nonlocal low, high
        span = high - low + 1
        high = (low + (span * (cum + fr)) // tot - 1) & 0xFFFFFFFF
        low = (low + (span * cum) // tot) & 0xFFFFFFFF
        renorm()

    for i in range(plain_len):
        left = out[-1] if out else 0
        byte = 0
        for shift in (0, 4):
            ctx = _ctx_id(prev, left, i, plain_len)
            span = high - low + 1
            tot = total[ctx]
            value = ((code - low + 1) * tot - 1) // span
            row = freq[ctx]
            cum = 0
            sym = _ALPH - 1
            for s in range(_ALPH):
                nxt = cum + row[s]
                if value < nxt:
                    sym = s
                    break
                cum = nxt
            take(cum, row[sym], tot)
            row[sym] += 1
            total[ctx] += 1
            if total[ctx] >= _RESCALE:
                total[ctx] = _rescale(row)
            byte |= sym << shift
            prev = sym
        out.append(byte)
    if len(out) != plain_len:
        raise ValueError("range length")
    return bytes(out)
