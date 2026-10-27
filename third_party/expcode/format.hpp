#pragma once

#include <cstddef>
#include <cstdint>

// On-disk and in-memory layout v1 for one matrix.
//
// Little-endian. Multi-byte integers are stored explicitly, not as a C
// struct, so padding and host endianness cannot change the layout.
//
// Header (64 bytes)
//   offset  size  field
//   0       4     magic = 0x31435845 ('E','X','C','1' on a little-endian host)
//   4       2     version = 1
//   6       2     code_bits = 4
//   8       4     rows
//   12      4     cols
//   16      4     group_rows   (rows that share one codebook; >= 1)
//   20      4     code_stride  (bytes of nibble codes per row, (cols+1)/2)
//   24      4     sm_stride    (sign/mantissa bytes per row, equal to cols)
//   28      4     n_groups     (ceil(rows / group_rows))
//   32      4     flags        (bit 0: escape stream is non-empty)
//   36      4     codebook_nbytes
//   40      4     codes_nbytes
//   44      4     sm_nbytes
//   48      4     escape_index_nbytes
//   52      4     escape_nbytes
//   56      8     reserved, zero
//
// Sections follow the header with no alignment padding, in this order:
//
//   codebook: n_groups * 16 bytes.
//     Byte i in 0..14 is the BF16 exponent selected by code i.
//     Byte 15 is unused and written as 0. Code 15 is the escape code.
//     Group g covers rows [g * group_rows, min(rows, (g+1) * group_rows)).
//
//   codes: rows * code_stride bytes, row-major.
//     The byte at column-pair j holds code(2j) in bits 3:0 and code(2j+1)
//     in bits 7:4. If cols is odd, the high nibble of the last byte is 0.
//
//   sign_mant: rows * cols bytes, row-major.
//     Bit 7 is the BF16 sign. Bits 6:0 are the BF16 mantissa.
//     Reconstruction:
//       bf16 = ((sm & 0x80) << 8) | (exponent << 7) | (sm & 0x7F)
//
//   escape_index: (rows + 1) little-endian uint32 byte offsets into the
//     escape section. Row r owns escapes [index[r], index[r+1]).
//     Offsets are byte offsets, not element indexes. The table is a
//     sentinel so a worker can enter any row in O(1). The index may be
//     misaligned; readers must not issue a typed uint32 load.
//
//   escapes: raw BF16 exponent bytes, row-major, and within a row in
//     increasing column order. A weight whose code nibble is 15 consumes
//     exactly one byte. There is no position side-index: the nibble is
//     the position marker.

namespace expcode {

inline constexpr uint32_t kMagic = 0x31435845u;
inline constexpr uint16_t kFormatVersion = 1;
inline constexpr int kCodeBits = 4;
inline constexpr int kDirectCodes = 15;
inline constexpr int kEscapeCode = 15;
inline constexpr int kCodebookBytes = 16;
inline constexpr std::size_t kHeaderBytes = 64;
inline constexpr uint32_t kFlagHasEscapes = 1u;

// Canonical GEMV accumulation width. Every kernel (scalar, AVX2, AVX-512)
// adds into this many independent binary32 partials:
//   partial[c & 63] = fma(w[c], x[c], partial[c & 63])
// then sums partial[0] + partial[1] + ... + partial[63] in that order,
// with round-to-nearest-even and no contraction of the final adds.
// A single left-to-right accumulator is a different function and is not
// the bit-exact contract. 64 partials are 4 AVX-512 registers or 8 AVX2
// registers, enough to cover FMA latency on the target cores.
inline constexpr int kAccStripe = 64;

inline std::size_t code_stride_bytes(uint32_t cols) {
    return (std::size_t(cols) + 1u) / 2u;
}

// Byte size of a v1 blob. n_escape is the number of escaped weights.
inline std::size_t encoded_storage_bytes(uint32_t rows, uint32_t cols, uint32_t n_groups,
                                         uint64_t n_escape) {
    const std::size_t code_bytes = std::size_t(rows) * code_stride_bytes(cols);
    const std::size_t sm_bytes = std::size_t(rows) * std::size_t(cols);
    const std::size_t index_bytes = 4u * (std::size_t(rows) + 1u);
    return kHeaderBytes + std::size_t(n_groups) * std::size_t(kCodebookBytes) + code_bytes +
           sm_bytes + std::size_t(n_escape) + index_bytes;
}

} // namespace expcode
