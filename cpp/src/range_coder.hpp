#pragma once

// Order-1 adaptive range coder over nibbles. Matches src/ntc/range_coder.py.

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ntc {

inline void range_rescale(int* freq, int& total) {
    total = 0;
    for (int i = 0; i < 16; ++i) {
        freq[i] = (freq[i] + 1) / 2;
        total += freq[i];
    }
}

inline std::vector<uint8_t> range_encode(const uint8_t* data, size_t n) {
    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFu;
    std::vector<uint8_t> out;
    int freq[16][16];
    int total[16];
    for (int c = 0; c < 16; ++c) {
        total[c] = 16;
        for (int s = 0; s < 16; ++s) freq[c][s] = 1;
    }
    int ctx = 0;
    auto renorm = [&]() {
        while (((low ^ high) & 0xFF000000u) == 0u) {
            out.push_back(static_cast<uint8_t>(low >> 24));
            low <<= 8;
            high = (high << 8) | 0xFFu;
        }
    };
    auto put = [&](int cum, int fr, int tot) {
        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1ull;
        high = static_cast<uint32_t>(static_cast<uint64_t>(low) + (span * static_cast<uint64_t>(cum + fr)) / static_cast<uint64_t>(tot) - 1ull);
        low = static_cast<uint32_t>(static_cast<uint64_t>(low) + (span * static_cast<uint64_t>(cum)) / static_cast<uint64_t>(tot));
        renorm();
    };
    for (size_t i = 0; i < n; ++i) {
        const int byte = data[i];
        for (int shift = 0; shift <= 4; shift += 4) {
            const int sym = (byte >> shift) & 0xF;
            int cum = 0;
            for (int s = 0; s < sym; ++s) cum += freq[ctx][s];
            put(cum, freq[ctx][sym], total[ctx]);
            freq[ctx][sym] += 1;
            total[ctx] += 1;
            if (total[ctx] >= 4096) range_rescale(freq[ctx], total[ctx]);
            ctx = sym;
        }
    }
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<uint8_t>(low >> 24));
        low <<= 8;
    }
    return out;
}

inline std::vector<uint8_t> range_decode(const uint8_t* data, size_t n, size_t plain_len) {
    if (n < 4) throw std::runtime_error("short range stream");
    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFu;
    size_t pos = 0;
    auto read = [&]() -> uint32_t {
        if (pos >= n) return 0;
        return data[pos++];
    };
    uint32_t code = 0;
    for (int i = 0; i < 4; ++i) code = (code << 8) | read();
    int freq[16][16];
    int total[16];
    for (int c = 0; c < 16; ++c) {
        total[c] = 16;
        for (int s = 0; s < 16; ++s) freq[c][s] = 1;
    }
    int ctx = 0;
    std::vector<uint8_t> out;
    out.reserve(plain_len);
    auto renorm = [&]() {
        while (((low ^ high) & 0xFF000000u) == 0u) {
            code = (code << 8) | read();
            low <<= 8;
            high = (high << 8) | 0xFFu;
        }
    };
    auto take = [&](int cum, int fr, int tot) {
        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1ull;
        high = static_cast<uint32_t>(static_cast<uint64_t>(low) + (span * static_cast<uint64_t>(cum + fr)) / static_cast<uint64_t>(tot) - 1ull);
        low = static_cast<uint32_t>(static_cast<uint64_t>(low) + (span * static_cast<uint64_t>(cum)) / static_cast<uint64_t>(tot));
        renorm();
    };
    for (size_t i = 0; i < plain_len; ++i) {
        int byte = 0;
        for (int shift = 0; shift <= 4; shift += 4) {
            const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1ull;
            const int tot = total[ctx];
            const uint32_t value = static_cast<uint32_t>(((static_cast<uint64_t>(code - low) + 1ull) * static_cast<uint64_t>(tot) - 1ull) / span);
            int cum = 0;
            int sym = 15;
            for (int s = 0; s < 16; ++s) {
                const int nxt = cum + freq[ctx][s];
                if (value < static_cast<uint32_t>(nxt)) {
                    sym = s;
                    break;
                }
                cum = nxt;
            }
            take(cum, freq[ctx][sym], tot);
            freq[ctx][sym] += 1;
            total[ctx] += 1;
            if (total[ctx] >= 4096) range_rescale(freq[ctx], total[ctx]);
            byte |= sym << shift;
            ctx = sym;
        }
        out.push_back(static_cast<uint8_t>(byte));
    }
    return out;
}

}  // namespace ntc
