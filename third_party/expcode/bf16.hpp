#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include <immintrin.h>

namespace expcode {

// BF16 is the high 16 bits of an IEEE-754 binary32 value.
// Promotion is a left shift by 16. It is exact: every BF16 number is a
// binary32 number, including subnormals, infinities, and NaN payloads
// (the BF16 mantissa occupies binary32 mantissa bits [22:16]).

inline float bf16_to_fp32(uint16_t bits) {
    uint32_t u = uint32_t(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Round a binary32 value to BF16, round-to-nearest-even.
// The addition is done in 64 bits so a value that rounds up to infinity
// does not wrap the 32-bit encoding.
inline uint16_t fp32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    uint64_t lsb = (uint64_t(u) >> 16) & 1u;
    uint64_t rounded = uint64_t(u) + 0x7FFFu + lsb;
    return uint16_t(rounded >> 16);
}

inline uint8_t bf16_exponent(uint16_t bits) {
    return uint8_t((bits >> 7) & 0xFF);
}

// Bit 7 is the BF16 sign. Bits 6:0 are the BF16 mantissa.
inline uint8_t bf16_sign_mant(uint16_t bits) {
    return uint8_t(((bits >> 8) & 0x80) | (bits & 0x7F));
}

// One binary32 FMA. With FMA in the ISA this is the hardware instruction.
// libm fma and vfmaddps can return different NaN payloads, so every GEMV
// path, including scalar tails, has to use the same operation.
inline float exact_fma(float a, float b, float c) {
#if defined(__FMA__)
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
#else
    return std::fma(a, b, c);
#endif
}

// Left fold a + b as vaddss with the accumulator in src1.
// Addition of two NaNs keeps the src1 payload on Intel. A C `+` lets the
// compiler swap operands, which changes that payload. The GEMV horizontal
// sum has to be the same instruction on every path.
inline float exact_add(float a, float b) {
#if defined(__SSE2__)
    asm("vaddss %1, %0, %0" : "+x"(a) : "x"(b));
    return a;
#else
    return a + b;
#endif
}

inline uint16_t bf16_pack(uint8_t sign_mant, uint8_t exponent) {
    return uint16_t((uint16_t(sign_mant & 0x80) << 8) | (uint16_t(exponent) << 7) |
                    uint16_t(sign_mant & 0x7F));
}

inline uint32_t float_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline bool float_bits_equal(float a, float b) {
    return float_bits(a) == float_bits(b);
}

// GEMV bit-exactness assumes IEEE-754 binary32 with denormals preserved
// and round-to-nearest-even. FTZ or DAZ would flush BF16 subnormals.
inline void set_ieee_fp_env() {
    unsigned csr = _mm_getcsr();
    csr &= ~0x8040u;    // DAZ (bit 6), FTZ (bit 15)
    csr &= ~(3u << 13); // RC = round to nearest even
    _mm_setcsr(csr);
}

} // namespace expcode
