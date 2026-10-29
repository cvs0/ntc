// Specialized NTC decode. Host side: cpp/include/ntc/fast.hpp.
//
// Every shape, width, and parameter offset is a specialization constant, so
// the driver sees constant loop bounds, unrolls every loop, and keeps the
// feature and hidden vectors in registers. MLP parameters live in a uniform
// buffer at offsets known at pipeline creation: constant-bank operands on
// NVIDIA, scalar (SGPR) loads on AMD. Latents are channel planes whose
// record is a power of two of at most 32 bits, one aligned load per tap.
//
// Variants (pick one define before the include):
//   NTC_FAST_EXACT  NoContraction on every lerp and product. Bit-identical
//                   to ntc::decode_mip under default fp32 rounding.
//   (none)          fma for lerps and dots. Within 1e-5 of the reference.
//   NTC_FAST_FP16   packed fp16 weights and activations, f16vec2 fma.
//                   Needs shaderFloat16. Not exact.
//   NTC_FAST_I8     int8 activations times int8 weights with
//                   dotPacked4x8EXT. For act_int8 files the integer dot is
//                   exact and only the final fp32 rounding differs; other
//                   files get a per-vector max-abs / 127 activation scale,
//                   a quality change. Grouped layers with group % 4 != 0 fall
//                   back to fp32.
//
// Bindings (set 0): 0 uniform params, 1 storage latents. The including
// shader owns every other binding and the push constants.

#extension GL_EXT_control_flow_attributes : require
#ifdef NTC_FAST_FP16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif
#ifdef NTC_FAST_I8
#extension GL_EXT_integer_dot_product : require
#endif

layout(constant_id = 0) const int NTCF_GRIDS = 1;
layout(constant_id = 1) const int NTCF_LAYERS = 1;
layout(constant_id = 2) const int NTCF_OUTCH = 1;
layout(constant_id = 3) const int NTCF_PE = 0;
layout(constant_id = 4) const int NTCF_PEOCT = 0;
layout(constant_id = 5) const int NTCF_ACT = 0;
layout(constant_id = 6) const int NTCF_REQUANT = 0;
layout(constant_id = 7) const int NTCF_ACTI8 = 0;
layout(constant_id = 8) const int NTCF_NUMMIPS = 1;
layout(constant_id = 9) const int NTCF_LSOFF = 0;
layout(constant_id = 10) const int NTCF_ASOFF = 0;
layout(constant_id = 11) const int NTCF_W0 = 1;
layout(constant_id = 12) const int NTCF_W1 = 1;
layout(constant_id = 13) const int NTCF_W2 = 1;
layout(constant_id = 14) const int NTCF_W3 = 1;
layout(constant_id = 15) const int NTCF_W4 = 1;
layout(constant_id = 16) const int NTCF_GC0 = 0;
layout(constant_id = 17) const int NTCF_GC1 = 0;
layout(constant_id = 18) const int NTCF_GC2 = 0;
layout(constant_id = 19) const int NTCF_GC3 = 0;
layout(constant_id = 20) const int NTCF_GC4 = 0;
layout(constant_id = 21) const int NTCF_GC5 = 0;
layout(constant_id = 22) const int NTCF_GC6 = 0;
layout(constant_id = 23) const int NTCF_GC7 = 0;
layout(constant_id = 24) const int NTCF_GB0 = 4;
layout(constant_id = 25) const int NTCF_GB1 = 4;
layout(constant_id = 26) const int NTCF_GB2 = 4;
layout(constant_id = 27) const int NTCF_GB3 = 4;
layout(constant_id = 28) const int NTCF_GB4 = 4;
layout(constant_id = 29) const int NTCF_GB5 = 4;
layout(constant_id = 30) const int NTCF_GB6 = 4;
layout(constant_id = 31) const int NTCF_GB7 = 4;
layout(constant_id = 32) const int NTCF_GS0 = 0;
layout(constant_id = 33) const int NTCF_GS1 = 0;
layout(constant_id = 34) const int NTCF_GS2 = 0;
layout(constant_id = 35) const int NTCF_GS3 = 0;
layout(constant_id = 36) const int NTCF_NG0 = 1;
layout(constant_id = 37) const int NTCF_NG1 = 1;
layout(constant_id = 38) const int NTCF_NG2 = 1;
layout(constant_id = 39) const int NTCF_NG3 = 1;
layout(constant_id = 40) const int NTCF_WO0 = 0;
layout(constant_id = 41) const int NTCF_WO1 = 0;
layout(constant_id = 42) const int NTCF_WO2 = 0;
layout(constant_id = 43) const int NTCF_WO3 = 0;
layout(constant_id = 44) const int NTCF_BO0 = 0;
layout(constant_id = 45) const int NTCF_BO1 = 0;
layout(constant_id = 46) const int NTCF_BO2 = 0;
layout(constant_id = 47) const int NTCF_BO3 = 0;
layout(constant_id = 48) const int NTCF_SO0 = 0;
layout(constant_id = 49) const int NTCF_SO1 = 0;
layout(constant_id = 50) const int NTCF_SO2 = 0;
layout(constant_id = 51) const int NTCF_SO3 = 0;
layout(constant_id = 52) const int NTCF_QO0 = 0;
layout(constant_id = 53) const int NTCF_QO1 = 0;
layout(constant_id = 54) const int NTCF_QO2 = 0;
layout(constant_id = 55) const int NTCF_QO3 = 0;
layout(constant_id = 56) const int NTCF_HO0 = 0;
layout(constant_id = 57) const int NTCF_HO1 = 0;
layout(constant_id = 58) const int NTCF_HO2 = 0;
layout(constant_id = 59) const int NTCF_HO3 = 0;

#define NTCF_MAX_IN 160
#define NTCF_MAX_HID 64
#define NTCF_MAX_OUT 16

layout(std140, set = 0, binding = 0) uniform NtcfParams { uvec4 ntcf_p[4096]; };
layout(std430, set = 0, binding = 1) readonly buffer NtcfLatents { uint ntcf_l[]; };

uint ntcf_pu(int i) { return ntcf_p[i >> 2][i & 3]; }
float ntcf_pf(int i) { return uintBitsToFloat(ntcf_pu(i)); }

int ntcf_gc(int g) {
    return g == 0 ? NTCF_GC0 : g == 1 ? NTCF_GC1 : g == 2 ? NTCF_GC2 : g == 3 ? NTCF_GC3
         : g == 4 ? NTCF_GC4 : g == 5 ? NTCF_GC5 : g == 6 ? NTCF_GC6 : NTCF_GC7;
}
int ntcf_gb(int g) {
    return g == 0 ? NTCF_GB0 : g == 1 ? NTCF_GB1 : g == 2 ? NTCF_GB2 : g == 3 ? NTCF_GB3
         : g == 4 ? NTCF_GB4 : g == 5 ? NTCF_GB5 : g == 6 ? NTCF_GB6 : NTCF_GB7;
}
int ntcf_width(int l) {
    return l == 0 ? NTCF_W0 : l == 1 ? NTCF_W1 : l == 2 ? NTCF_W2 : l == 3 ? NTCF_W3 : NTCF_W4;
}
int ntcf_gs(int l) { return l == 0 ? NTCF_GS0 : l == 1 ? NTCF_GS1 : l == 2 ? NTCF_GS2 : NTCF_GS3; }
int ntcf_ng(int l) { return l == 0 ? NTCF_NG0 : l == 1 ? NTCF_NG1 : l == 2 ? NTCF_NG2 : NTCF_NG3; }
int ntcf_wo(int l) { return l == 0 ? NTCF_WO0 : l == 1 ? NTCF_WO1 : l == 2 ? NTCF_WO2 : NTCF_WO3; }
int ntcf_bo(int l) { return l == 0 ? NTCF_BO0 : l == 1 ? NTCF_BO1 : l == 2 ? NTCF_BO2 : NTCF_BO3; }
int ntcf_so(int l) { return l == 0 ? NTCF_SO0 : l == 1 ? NTCF_SO1 : l == 2 ? NTCF_SO2 : NTCF_SO3; }
int ntcf_qo(int l) { return l == 0 ? NTCF_QO0 : l == 1 ? NTCF_QO1 : l == 2 ? NTCF_QO2 : NTCF_QO3; }
int ntcf_ho(int l) { return l == 0 ? NTCF_HO0 : l == 1 ? NTCF_HO1 : l == 2 ? NTCF_HO2 : NTCF_HO3; }

// Largest power of two <= n, capped at cap. Matches ntc::fast_planes.
int ntcf_plane_n(int rem, int bits) {
    int n = rem >= 16 ? 16 : rem >= 8 ? 8 : rem >= 4 ? 4 : rem >= 2 ? 2 : 1;
    return min(n, 32 / bits);
}

#ifdef NTC_FAST_EXACT
#define NTCF_P precise
precise float ntcf_lerp(float a, float b, float t) { precise float d = (b - a) * t; precise float r = a + d; return r; }
precise float ntcf_mad(float acc, float w, float x) { precise float p = w * x; precise float r = acc + p; return r; }
#else
#define NTCF_P
float ntcf_lerp(float a, float b, float t) { return fma(b - a, t, a); }
float ntcf_mad(float acc, float w, float x) { return fma(w, x, acc); }
#endif

// One latent record at texel (x, y) of a plane. R is a constant.
uint ntcf_rec(uint row, int x, int R) {
    uint bit = uint(x) * uint(R);
    uint w = ntcf_l[row + (bit >> 5u)];
    return R == 32 ? w : bitfieldExtract(w, int(bit & 31u), R);
}

float ntcf_activate(float y) {
    float z;
    if (NTCF_ACT == 1) {
        z = max(y, 0.0);
    } else if (NTCF_ACT == 2) {
        z = min(y, 3.0) * clamp(y * (1.0 / 3.0) + 0.5, 0.0, 1.0);
    } else {
        z = y;
    }
    if (NTCF_REQUANT != 0) z = clamp(round(z * 80.0 - 113.0), -128.0, 127.0);
    return z;
}

// Features: bilinear latents of every grid at mip, then positional terms.
void ntcf_features(vec2 uv, int mip, inout float x[NTCF_MAX_IN]) {
    vec2 u = uv - floor(uv);  // exact for uv in [0, 1); same taps as the integer wrap
    int n = 0;
    int ls = NTCF_LSOFF;
    int km = min(mip, 7);
    [[unroll]] for (int g = 0; g < 8; ++g) {
        if (g >= NTCF_GRIDS) break;
        const int C = ntcf_gc(g);
        const int bits = ntcf_gb(g);
        uvec4 md = ntcf_p[g * 8 + km];
        int GW = int(md.y), GH = int(md.z);
        NTCF_P float fx = u.x * float(GW) - 0.5;
        NTCF_P float fy = u.y * float(GH) - 0.5;
        int x0 = int(floor(fx));
        int y0 = int(floor(fy));
        float tx = fx - float(x0);
        float ty = fy - float(y0);
        int xa = x0 < 0 ? x0 + GW : x0;
        int xb = x0 + 1 >= GW ? x0 + 1 - GW : x0 + 1;
        int ya = y0 < 0 ? y0 + GH : y0;
        int yb = y0 + 1 >= GH ? y0 + 1 - GH : y0 + 1;
        float mid = float((1 << bits) - 1) * 0.5;
        uint pbase = md.x;
        int c = 0;
        [[unroll]] for (int p = 0; p < 8; ++p) {
            if (c >= C) break;
            const int pn = ntcf_plane_n(C - c, bits);
            const int R = pn * bits;
            int rw = (GW * R + 31) >> 5;
            uint ra = pbase + uint(ya * rw), rb = pbase + uint(yb * rw);
            uint r00 = ntcf_rec(ra, xa, R), r01 = ntcf_rec(ra, xb, R);
            uint r10 = ntcf_rec(rb, xa, R), r11 = ntcf_rec(rb, xb, R);
            [[unroll]] for (int k = 0; k < 16; ++k) {
                if (k >= pn) break;
                float c00 = float(bitfieldExtract(r00, k * bits, bits));
                float c01 = float(bitfieldExtract(r01, k * bits, bits));
                float c10 = float(bitfieldExtract(r10, k * bits, bits));
                float c11 = float(bitfieldExtract(r11, k * bits, bits));
                float a = ntcf_lerp(c00, c01, tx);
                float b = ntcf_lerp(c10, c11, tx);
                float l = ntcf_lerp(a, b, ty);
                x[n + c + k] = (l - mid) * ntcf_pf(ls + c + k);
            }
            c += pn;
            pbase += uint(rw * GH);
        }
        n += C;
        ls += C;
    }
    if (NTCF_PE == 1 && NTCF_PEOCT > 0) {
        const float tau = 6.283185307179586;
        [[unroll]] for (int o = 0; o < 16; ++o) {
            if (o >= NTCF_PEOCT) break;
            float f = float(1 << o);
            NTCF_P float au = tau * uv.x * f;
            NTCF_P float av = tau * uv.y * f;
            x[n++] = sin(au);
            x[n++] = cos(au);
            x[n++] = sin(av);
            x[n++] = cos(av);
        }
    } else if (NTCF_PE == 2 && NTCF_PEOCT > 0) {
        vec2 p = uv;
        [[unroll]] for (int o = 0; o < 16; ++o) {
            if (o >= NTCF_PEOCT) break;
            NTCF_P float t0 = (p.x - floor(p.x)) * 2.0 - 1.0;
            NTCF_P float t1 = (p.y - floor(p.y)) * 2.0 - 1.0;
            NTCF_P float t2 = ((p.x + 0.25) - floor(p.x + 0.25)) * 2.0 - 1.0;
            NTCF_P float t3 = ((p.y + 0.25) - floor(p.y + 0.25)) * 2.0 - 1.0;
            x[n++] = t0;
            x[n++] = t1;
            x[n++] = t2;
            x[n++] = t3;
            p *= 2.0;
        }
        float lod = float(mip) / float(max(NTCF_NUMMIPS - 1, 1));
        x[n++] = lod;
        x[n++] = lod;
    }
}

#ifdef NTC_FAST_I8
// True when layer l can run as packed int8 dots.
bool ntcf_i8_layer(int l) { int gs = ntcf_gs(l); return gs == 0 || (gs & 3) == 0; }
#endif

// x: features in, outputs (NTCF_OUTCH) out.
void ntcf_mlp(inout float x[NTCF_MAX_IN]) {
    float t[NTCF_MAX_HID];
    [[unroll]] for (int l = 0; l < 4; ++l) {
        if (l >= NTCF_LAYERS) break;
        const int inn = ntcf_width(l);
        const int outn = ntcf_width(l + 1);
        const int gs = ntcf_gs(l);
        const int ng = ntcf_ng(l);
        const int wo = ntcf_wo(l), bo = ntcf_bo(l), so = ntcf_so(l);
        const bool last = l + 1 == NTCF_LAYERS;
#ifdef NTC_FAST_I8
        if (ntcf_i8_layer(l)) {
            // Activation scale: the file's when act_int8, else max-abs / 127.
            float s;
            if (NTCF_ACTI8 != 0) {
                s = max(ntcf_pf(NTCF_ASOFF + l), 1e-6);
            } else {
                float m = 0.0;
                [[unroll]] for (int i = 0; i < NTCF_MAX_IN; ++i) { if (i >= inn) break; m = max(m, abs(x[i])); }
                s = max(m * (1.0 / 127.0), 1e-6);
            }
            float rs = 1.0 / s;
            const int in4 = (inn + 3) >> 2;
            // Signed operands: the uint overload of dotPacked4x8EXT is the
            // unsigned dot.
            uint q[NTCF_MAX_IN / 4];
            [[unroll]] for (int j = 0; j < NTCF_MAX_IN / 4; ++j) {
                if (j >= in4) break;
                ivec4 v = ivec4(0);
                [[unroll]] for (int b = 0; b < 4; ++b) {
                    int i = j * 4 + b;
                    if (i < inn) v[b] = clamp(int(round(NTCF_ACTI8 != 0 ? x[i] / s : x[i] * rs)), -127, 127);
                }
                q[j] = uint(v.x & 255) | (uint(v.y & 255) << 8) | (uint(v.z & 255) << 16) | (uint(v.w & 255) << 24);
            }
            const int qo = ntcf_qo(l);
            [[unroll]] for (int o = 0; o < NTCF_MAX_HID; ++o) {
                if (o >= outn) break;
                float r = ntcf_pf(bo + o);
                if (gs == 0) {
                    int acc = 0;
                    [[unroll]] for (int j = 0; j < NTCF_MAX_IN / 4; ++j) {
                        if (j >= in4) break;
                        acc += dotPacked4x8EXT(int(ntcf_pu(qo + o * in4 + j)), int(q[j]));
                    }
                    r += ntcf_pf(so + o) * (float(acc) * s);
                } else {
                    [[unroll]] for (int g = 0; g < 40; ++g) {
                        if (g >= ng) break;
                        int acc = 0;
                        [[unroll]] for (int j = 0; j < NTCF_MAX_IN / 4; ++j) {
                            if (j >= (gs >> 2)) break;
                            int jj = g * (gs >> 2) + j;
                            if (jj < in4) acc += dotPacked4x8EXT(int(ntcf_pu(qo + o * in4 + jj)), int(q[jj]));
                        }
                        r += ntcf_pf(so + o * ng + g) * (float(acc) * s);
                    }
                }
                t[o] = last ? r : ntcf_activate(r);
            }
            [[unroll]] for (int o = 0; o < NTCF_MAX_HID; ++o) { if (o >= outn) break; x[o] = t[o]; }
            continue;
        }
#endif
        if (NTCF_ACTI8 != 0) {
            float s = max(ntcf_pf(NTCF_ASOFF + l), 1e-6);
            [[unroll]] for (int i = 0; i < NTCF_MAX_IN; ++i) {
                if (i >= inn) break;
                x[i] = clamp(round(x[i] / s), -127.0, 127.0) * s;
            }
        }
#ifdef NTC_FAST_FP16
        const int ho = ntcf_ho(l);
        const int in2 = (inn + 1) >> 1;
        f16vec2 h[NTCF_MAX_IN / 2];
        [[unroll]] for (int j = 0; j < NTCF_MAX_IN / 2; ++j) {
            if (j >= in2) break;
            h[j] = f16vec2(x[2 * j], 2 * j + 1 < inn ? x[2 * j + 1] : 0.0);
        }
#endif
        [[unroll]] for (int o = 0; o < NTCF_MAX_HID; ++o) {
            if (o >= outn) break;
            float r;
#ifdef NTC_FAST_FP16
            if (gs == 0) {
                f16vec2 acc = f16vec2(0.0);
                [[unroll]] for (int j = 0; j < NTCF_MAX_IN / 2; ++j) {
                    if (j >= in2) break;
                    acc = fma(unpackFloat2x16(ntcf_pu(ho + o * in2 + j)), h[j], acc);
                }
                r = ntcf_pf(bo + o) + ntcf_pf(so + o) * (float(acc.x) + float(acc.y));
            } else
#endif
            if (gs == 0) {
                float acc = 0.0;
                [[unroll]] for (int i = 0; i < NTCF_MAX_IN; ++i) {
                    if (i >= inn) break;
                    acc = ntcf_mad(acc, ntcf_pf(wo + o * inn + i), x[i]);
                }
#ifdef NTC_FAST_EXACT
                precise float sc = ntcf_pf(so + o) * acc;
                precise float rr = ntcf_pf(bo + o) + sc;
                r = rr;
#else
                r = fma(ntcf_pf(so + o), acc, ntcf_pf(bo + o));
#endif
            } else {
                r = ntcf_pf(bo + o);
                [[unroll]] for (int g = 0; g < 40; ++g) {
                    if (g >= ng) break;
                    float acc = 0.0;
                    [[unroll]] for (int i = 0; i < NTCF_MAX_IN; ++i) {
                        if (i >= gs) break;
                        int ii = g * gs + i;
                        if (ii < inn) acc = ntcf_mad(acc, ntcf_pf(wo + o * inn + ii), x[ii]);
                    }
#ifdef NTC_FAST_EXACT
                    precise float sc = ntcf_pf(so + o * ng + g) * acc;
                    precise float rr = r + sc;
                    r = rr;
#else
                    r = fma(ntcf_pf(so + o * ng + g), acc, r);
#endif
                }
            }
            t[o] = last ? r : ntcf_activate(r);
        }
        [[unroll]] for (int o = 0; o < NTCF_MAX_HID; ++o) { if (o >= outn) break; x[o] = t[o]; }
    }
}

// Full decode at one mip. Channels 0..NTCF_OUTCH-1 of x hold the result.
void ntcf_eval(vec2 uv, int mip, out float x[NTCF_MAX_IN]) {
    ntcf_features(uv, mip, x);
    ntcf_mlp(x);
}
