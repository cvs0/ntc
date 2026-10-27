#pragma once

#include "expcode/bf16.hpp"
#include "expcode/format.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace expcode {

inline void store_u16(uint8_t* p, uint16_t v) {
    p[0] = uint8_t(v);
    p[1] = uint8_t(v >> 8);
}

inline void store_u32(uint8_t* p, uint32_t v) {
    p[0] = uint8_t(v);
    p[1] = uint8_t(v >> 8);
    p[2] = uint8_t(v >> 16);
    p[3] = uint8_t(v >> 24);
}

inline void store_u64(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i)
        p[i] = uint8_t(v >> (8 * i));
}

inline uint16_t load_u16(const uint8_t* p) {
    return uint16_t(p[0] | (uint16_t(p[1]) << 8));
}

inline uint32_t load_u32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

inline uint64_t load_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v |= uint64_t(p[i]) << (8 * i);
    return v;
}

struct Codebook {
    // exp[i] for i in 0..14. exp[15] is unused.
    uint8_t exp[kCodebookBytes]{};
    // 0..14, or 255 if the exponent is not in the direct set.
    uint8_t inv[256]{};
};

// Descending frequency. Equal counts break toward the smaller exponent,
// so the assignment is a pure function of the histogram.
inline Codebook codebook_from_histogram(const uint64_t hist[256]) {
    struct Item {
        uint64_t count;
        uint16_t exp;
    };
    Item items[256];
    for (int e = 0; e < 256; ++e) {
        items[e].count = hist[e];
        items[e].exp = uint16_t(e);
    }
    std::sort(items, items + 256, [](Item a, Item b) {
        if (a.count != b.count)
            return a.count > b.count;
        return a.exp < b.exp;
    });
    Codebook cb;
    std::fill(std::begin(cb.inv), std::end(cb.inv), uint8_t(255));
    int n = 0;
    for (int i = 0; i < 256 && n < kDirectCodes; ++i) {
        if (items[i].count == 0)
            break;
        cb.exp[n] = uint8_t(items[i].exp);
        cb.inv[items[i].exp] = uint8_t(n);
        ++n;
    }
    return cb;
}

// Non-owning view of a v1 blob. Pointers alias the blob bytes.
struct MatrixView {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t group_rows = 1;
    uint32_t code_stride = 0;
    uint32_t sm_stride = 0;
    uint32_t n_groups = 0;
    uint32_t flags = 0;
    const uint8_t* codebook = nullptr;
    const uint8_t* codes = nullptr;
    const uint8_t* sign_mant = nullptr;
    const uint8_t* escape_index = nullptr;
    const uint8_t* escapes = nullptr;
    uint32_t escape_nbytes = 0;

    const uint8_t* codebook_for_row(uint32_t row) const {
        const uint32_t g = row / group_rows;
        return codebook + std::size_t(g) * std::size_t(kCodebookBytes);
    }

    uint32_t escape_offset_at(uint32_t row) const {
        return load_u32(escape_index + std::size_t(row) * 4u);
    }
};

inline uint8_t code_at(const uint8_t* codes, uint32_t col) {
    const uint8_t byte = codes[col >> 1];
    return (col & 1u) ? uint8_t(byte >> 4) : uint8_t(byte & 0x0F);
}

class EncodedMatrix {
public:
    // group_rows == 0 selects one codebook for the whole matrix.
    static EncodedMatrix encode(std::span<const uint16_t> row_major, uint32_t rows, uint32_t cols,
                                uint32_t group_rows = 0) {
        if (rows == 0 || cols == 0) {
            throw std::runtime_error("expcode: rows and cols must be non-zero");
        }
        if (row_major.size() != std::size_t(rows) * std::size_t(cols)) {
            throw std::runtime_error("expcode: weight span size does not match rows*cols");
        }
        if (group_rows == 0 || group_rows > rows)
            group_rows = rows;
        const uint32_t n_groups = (rows + group_rows - 1u) / group_rows;
        const uint32_t code_stride = uint32_t(code_stride_bytes(cols));

        std::vector<uint8_t> codebooks(std::size_t(n_groups) * std::size_t(kCodebookBytes), 0);
        std::vector<uint8_t> codes(std::size_t(rows) * code_stride, 0);
        std::vector<uint8_t> sign_mant(std::size_t(rows) * cols);
        std::vector<uint32_t> escape_ofs(std::size_t(rows) + 1u, 0);
        std::vector<uint8_t> escapes;

        std::array<uint64_t, 256> hist{};
        for (uint32_t g = 0; g < n_groups; ++g) {
            hist.fill(0);
            const uint32_t r0 = g * group_rows;
            const uint32_t r1 = std::min(rows, r0 + group_rows);
            for (uint32_t r = r0; r < r1; ++r) {
                const uint16_t* row = row_major.data() + std::size_t(r) * cols;
                for (uint32_t c = 0; c < cols; ++c)
                    hist[bf16_exponent(row[c])]++;
            }
            const Codebook cb = codebook_from_histogram(hist.data());
            std::memcpy(codebooks.data() + std::size_t(g) * kCodebookBytes, cb.exp, kCodebookBytes);
            for (uint32_t r = r0; r < r1; ++r) {
                if (escapes.size() > 0xFFFFFFFFu) {
                    throw std::runtime_error("expcode: escape stream exceeds 2^32 bytes");
                }
                escape_ofs[r] = uint32_t(escapes.size());
                const uint16_t* row = row_major.data() + std::size_t(r) * cols;
                uint8_t* code_row = codes.data() + std::size_t(r) * code_stride;
                uint8_t* sm_row = sign_mant.data() + std::size_t(r) * cols;
                for (uint32_t c = 0; c < cols; ++c) {
                    const uint8_t exponent = bf16_exponent(row[c]);
                    sm_row[c] = bf16_sign_mant(row[c]);
                    uint8_t code = cb.inv[exponent];
                    if (code == 255) {
                        code = uint8_t(kEscapeCode);
                        escapes.push_back(exponent);
                    }
                    if ((c & 1u) == 0) {
                        code_row[c >> 1] = code;
                    } else {
                        code_row[c >> 1] = uint8_t(code_row[c >> 1] | uint8_t(code << 4));
                    }
                }
            }
        }
        escape_ofs[rows] = uint32_t(escapes.size());

        const uint32_t flags = escapes.empty() ? 0u : kFlagHasEscapes;
        const std::size_t total = encoded_storage_bytes(rows, cols, n_groups, escapes.size());
        std::vector<uint8_t> blob(total, 0);
        uint8_t* h = blob.data();
        store_u32(h + 0, kMagic);
        store_u16(h + 4, kFormatVersion);
        store_u16(h + 6, uint16_t(kCodeBits));
        store_u32(h + 8, rows);
        store_u32(h + 12, cols);
        store_u32(h + 16, group_rows);
        store_u32(h + 20, code_stride);
        store_u32(h + 24, cols);
        store_u32(h + 28, n_groups);
        store_u32(h + 32, flags);
        store_u32(h + 36, uint32_t(codebooks.size()));
        store_u32(h + 40, uint32_t(codes.size()));
        store_u32(h + 44, uint32_t(sign_mant.size()));
        store_u32(h + 48, uint32_t(escape_ofs.size() * 4u));
        store_u32(h + 52, uint32_t(escapes.size()));
        store_u64(h + 56, 0);

        uint8_t* p = blob.data() + kHeaderBytes;
        auto copy_bytes = [&](const uint8_t* src, std::size_t n) {
            if (n != 0) {
                std::memcpy(p, src, n);
                p += n;
            }
        };
        copy_bytes(codebooks.data(), codebooks.size());
        copy_bytes(codes.data(), codes.size());
        copy_bytes(sign_mant.data(), sign_mant.size());
        for (uint32_t off : escape_ofs) {
            store_u32(p, off);
            p += 4;
        }
        copy_bytes(escapes.data(), escapes.size());

        EncodedMatrix out;
        out.bytes_ = std::move(blob);
        out.bind_fields();
        return out;
    }

    static EncodedMatrix parse_copy(std::span<const uint8_t> blob) {
        // Validate before copying so a short buffer cannot be stored.
        (void)view_blob(blob);
        EncodedMatrix out;
        out.bytes_.assign(blob.begin(), blob.end());
        out.bind_fields();
        return out;
    }

    static MatrixView view_blob(std::span<const uint8_t> blob) {
        if (blob.size() < kHeaderBytes) {
            throw std::runtime_error("expcode: blob shorter than header");
        }
        const uint8_t* h = blob.data();
        if (load_u32(h + 0) != kMagic)
            throw std::runtime_error("expcode: bad magic");
        if (load_u16(h + 4) != kFormatVersion)
            throw std::runtime_error("expcode: bad version");
        if (load_u16(h + 6) != uint16_t(kCodeBits)) {
            throw std::runtime_error("expcode: only 4-bit codes are supported");
        }
        MatrixView v;
        v.rows = load_u32(h + 8);
        v.cols = load_u32(h + 12);
        v.group_rows = load_u32(h + 16);
        v.code_stride = load_u32(h + 20);
        v.sm_stride = load_u32(h + 24);
        v.n_groups = load_u32(h + 28);
        v.flags = load_u32(h + 32);
        const uint32_t codebook_nbytes = load_u32(h + 36);
        const uint32_t codes_nbytes = load_u32(h + 40);
        const uint32_t sm_nbytes = load_u32(h + 44);
        const uint32_t index_nbytes = load_u32(h + 48);
        v.escape_nbytes = load_u32(h + 52);
        if (v.rows == 0 || v.cols == 0 || v.group_rows == 0) {
            throw std::runtime_error("expcode: degenerate shape");
        }
        const uint32_t expect_groups = (v.rows + v.group_rows - 1u) / v.group_rows;
        if (v.n_groups != expect_groups)
            throw std::runtime_error("expcode: n_groups mismatch");
        if (v.code_stride != code_stride_bytes(v.cols) || v.sm_stride != v.cols) {
            throw std::runtime_error("expcode: stride mismatch");
        }
        if (codebook_nbytes != v.n_groups * uint32_t(kCodebookBytes)) {
            throw std::runtime_error("expcode: codebook size mismatch");
        }
        if (codes_nbytes != uint64_t(v.rows) * v.code_stride ||
            sm_nbytes != uint64_t(v.rows) * v.sm_stride) {
            throw std::runtime_error("expcode: payload size mismatch");
        }
        if (index_nbytes != 4u * (v.rows + 1u)) {
            throw std::runtime_error("expcode: escape index size mismatch");
        }
        const uint64_t body =
            uint64_t(codebook_nbytes) + codes_nbytes + sm_nbytes + index_nbytes + v.escape_nbytes;
        if (body != blob.size() - kHeaderBytes) {
            throw std::runtime_error("expcode: section sizes do not fill the blob");
        }
        if ((v.flags & kFlagHasEscapes) == 0 && v.escape_nbytes != 0) {
            throw std::runtime_error("expcode: escape flag mismatch");
        }
        const uint8_t* p = blob.data() + kHeaderBytes;
        v.codebook = p;
        p += codebook_nbytes;
        v.codes = p;
        p += codes_nbytes;
        v.sign_mant = p;
        p += sm_nbytes;
        v.escape_index = p;
        p += index_nbytes;
        v.escapes = p;
        // Index must be monotonic and in range. This is O(rows), paid at parse.
        uint32_t prev = 0;
        for (uint32_t r = 0; r <= v.rows; ++r) {
            const uint32_t off = v.escape_offset_at(r);
            if (off < prev || off > v.escape_nbytes) {
                throw std::runtime_error("expcode: escape index is not monotonic");
            }
            prev = off;
        }
        if (prev != v.escape_nbytes) {
            throw std::runtime_error("expcode: escape index does not end at escape_nbytes");
        }
        return v;
    }

    MatrixView view() const { return view_blob(bytes_); }

    std::span<const uint8_t> bytes() const { return bytes_; }
    uint32_t rows() const { return rows_; }
    uint32_t cols() const { return cols_; }
    uint32_t group_rows() const { return group_rows_; }
    uint32_t n_groups() const { return n_groups_; }
    uint64_t n_escape() const { return n_escape_; }

    double bits_per_weight() const {
        const double n = double(rows_) * double(cols_);
        return 8.0 * double(bytes_.size()) / n;
    }

    void decode(std::span<uint16_t> out) const {
        const MatrixView m = view();
        if (out.size() != std::size_t(m.rows) * m.cols) {
            throw std::runtime_error("expcode: decode output size mismatch");
        }
        for (uint32_t r = 0; r < m.rows; ++r) {
            const uint8_t* cb = m.codebook_for_row(r);
            const uint8_t* codes = m.codes + std::size_t(r) * m.code_stride;
            const uint8_t* sm = m.sign_mant + std::size_t(r) * m.sm_stride;
            const uint8_t* esc = m.escapes + m.escape_offset_at(r);
            const uint8_t* esc_end = m.escapes + m.escape_offset_at(r + 1);
            uint16_t* dst = out.data() + std::size_t(r) * m.cols;
            for (uint32_t c = 0; c < m.cols; ++c) {
                const uint8_t code = code_at(codes, c);
                uint8_t exponent;
                if (code == kEscapeCode) {
                    if (esc >= esc_end)
                        throw std::runtime_error("expcode: escape underrun");
                    exponent = *esc++;
                } else {
                    exponent = cb[code];
                }
                dst[c] = bf16_pack(sm[c], exponent);
            }
            if (esc != esc_end)
                throw std::runtime_error("expcode: escape overrun");
        }
    }

private:
    void bind_fields() {
        const MatrixView v = view();
        rows_ = v.rows;
        cols_ = v.cols;
        group_rows_ = v.group_rows;
        n_groups_ = v.n_groups;
        n_escape_ = v.escape_nbytes;
    }

    std::vector<uint8_t> bytes_;
    uint32_t rows_ = 0;
    uint32_t cols_ = 0;
    uint32_t group_rows_ = 1;
    uint32_t n_groups_ = 0;
    uint64_t n_escape_ = 0;
};

} // namespace expcode
