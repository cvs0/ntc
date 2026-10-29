#include "ntc/fast.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ntc {

namespace {

uint32_t f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

float bits_f32(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Round to nearest even. Weight codes are integers in [-127, 127], exact in fp16.
uint16_t f32_to_f16(float f) {
    const uint32_t x = f32_bits(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return static_cast<uint16_t>(sign | h);
    }
    uint32_t h = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return static_cast<uint16_t>(sign | h);
}

int wrap_i(int i, int n) {
    int r = i % n;
    if (r < 0) r += n;
    return r;
}

float activate(float y, uint32_t kind, bool requant) {
    float z;
    if (kind == kActRelu) {
        z = y > 0.f ? y : 0.f;
    } else if (kind == kActHgelu) {
        const float clamped = y < 3.f ? y : 3.f;
        float gate = y * (1.f / 3.f) + 0.5f;
        if (gate < 0.f) gate = 0.f;
        if (gate > 1.f) gate = 1.f;
        z = clamped * gate;
    } else {
        z = y;
    }
    if (requant) {
        float q = std::round(z * 80.f - 113.f);
        if (q < -128.f) q = -128.f;
        if (q > 127.f) q = 127.f;
        return q;
    }
    return z;
}

}  // namespace

std::vector<FastPlane> fast_planes(int channels, int bits) {
    std::vector<FastPlane> out;
    const int cap = 32 / bits;
    int c = 0;
    while (c < channels) {
        const int rem = channels - c;
        int n = 1;
        while (n * 2 <= rem) n *= 2;
        n = std::min(n, cap);
        out.push_back({c, n, n * bits});
        c += n;
    }
    return out;
}

int fast_row_words(int width, int record_bits) { return (width * record_bits + 31) / 32; }

FastUpload make_fast_upload(const TextureSet& ts) {
    FastUpload fu;
    fu.spec.assign(kSpecCount, 0);
    const int ngrids = static_cast<int>(ts.grids.size());
    const int nlayers = static_cast<int>(ts.layers.size());
    if (ngrids < 1 || ngrids > kMaxGrids || nlayers < 1 || nlayers > kMaxLayers) {
        fu.why = "grid or layer count out of range";
        return fu;
    }
    for (const Grid& g : ts.grids) {
        if (g.bits != 2 && g.bits != 4 && g.bits != 8) {
            fu.why = "grid bits not 2, 4, or 8";
            return fu;
        }
    }
    std::vector<uint32_t>& p = fu.params;
    p.assign(kFastMipTableWords, 0);

    // Latents: planes of every stored and derived mip.
    for (int g = 0; g < ngrids; ++g) {
        const Grid& grid = ts.grids[g];
        const auto planes = fast_planes(grid.channels, grid.bits);
        const int nm = std::min(static_cast<int>(grid.mips.size()), kMaxMips);
        for (int k = 0; k < nm; ++k) {
            const int w = grid.mip_w[k];
            const int h = grid.mip_h[k];
            const auto& codes = grid.mips[k];
            const uint32_t base = static_cast<uint32_t>(fu.latents.size());
            for (const FastPlane& pl : planes) {
                const int rw = fast_row_words(w, pl.record_bits);
                const size_t off = fu.latents.size();
                fu.latents.resize(off + static_cast<size_t>(rw) * h, 0u);
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        uint32_t rec = 0;
                        for (int c = 0; c < pl.channels; ++c) {
                            const int ci = pl.ch0 + c;
                            const uint32_t code = codes[(static_cast<size_t>(ci) * h + y) * w + x];
                            rec |= (code & ((1u << grid.bits) - 1u)) << (c * grid.bits);
                        }
                        const uint32_t bit = static_cast<uint32_t>(x) * static_cast<uint32_t>(pl.record_bits);
                        fu.latents[off + static_cast<size_t>(y) * rw + (bit >> 5)] |= rec << (bit & 31u);
                    }
                }
            }
            const int mi = (g * 8 + k) * 4;
            p[mi + 0] = base;
            p[mi + 1] = static_cast<uint32_t>(w);
            p[mi + 2] = static_cast<uint32_t>(h);
            p[mi + 3] = 0;
            int stride = 0;
            stride = (w * grid.bits + 7) / 8;
            fu.legacy_latent_bytes += static_cast<uint32_t>(stride * h * grid.channels);
        }
        for (int k = nm; k < kMaxMips; ++k) {
            const int src = (g * 8 + nm - 1) * 4;
            const int mi = (g * 8 + k) * 4;
            for (int j = 0; j < 4; ++j) p[mi + j] = p[src + j];
        }
    }
    if (fu.latents.empty()) fu.latents.push_back(0);
    fu.latent_bytes = static_cast<uint32_t>(fu.latents.size() * 4);

    const uint32_t ls_off = static_cast<uint32_t>(p.size());
    for (const Grid& g : ts.grids) {
        for (int c = 0; c < g.channels; ++c) p.push_back(f32_bits(g.scale[c]));
    }
    const uint32_t as_off = static_cast<uint32_t>(p.size());
    for (int l = 0; l < kMaxLayers; ++l) {
        float s = 1.f;
        if (l < static_cast<int>(ts.act_scales.size())) s = ts.act_scales[l];
        p.push_back(f32_bits(s));
    }

    fu.spec[kSpecGrids] = static_cast<uint32_t>(ngrids);
    fu.spec[kSpecLayers] = static_cast<uint32_t>(nlayers);
    fu.spec[kSpecOutCh] = static_cast<uint32_t>(ts.num_channels());
    fu.spec[kSpecPeMode] = ts.pe_mode;
    fu.spec[kSpecPeOct] = ts.pe_octaves;
    fu.spec[kSpecAct] = ts.activation;
    fu.spec[kSpecRequant] = ts.activation_requant() ? 1u : 0u;
    fu.spec[kSpecActI8] = ts.act_scales.empty() ? 0u : 1u;
    fu.spec[kSpecNumMips] = ts.num_mips;
    fu.spec[kSpecLatentScaleOff] = ls_off;
    fu.spec[kSpecActScaleOff] = as_off;
    for (int g = 0; g < ngrids; ++g) {
        fu.spec[kSpecGridCh0 + g] = static_cast<uint32_t>(ts.grids[g].channels);
        fu.spec[kSpecGridBits0 + g] = static_cast<uint32_t>(ts.grids[g].bits);
    }
    fu.spec[kSpecWidth0] = static_cast<uint32_t>(ts.layers[0].in_features);
    for (int l = 0; l < nlayers; ++l) {
        const Layer& layer = ts.layers[l];
        const int inn = layer.in_features;
        const int out = layer.out_features;
        if (inn > kMaxInput || out > (l + 1 == nlayers ? kMaxChannels : kMaxHidden)) {
            fu.why = "layer width over the shader limit";
            return fu;
        }
        fu.spec[kSpecWidth0 + l + 1] = static_cast<uint32_t>(out);
        const bool one = layer.group_size <= 0 || layer.group_size >= inn;
        const int ng = one ? 1 : std::max(1, layer.num_groups);
        fu.spec[kSpecGroupSize0 + l] = one ? 0u : static_cast<uint32_t>(layer.group_size);
        fu.spec[kSpecNumGroups0 + l] = static_cast<uint32_t>(ng);
        const int stride_groups = std::max(1, layer.num_groups);

        fu.spec[kSpecWOff0 + l] = static_cast<uint32_t>(p.size());
        for (int o = 0; o < out; ++o)
            for (int i = 0; i < inn; ++i) p.push_back(f32_bits(layer.weight[static_cast<size_t>(o) * inn + i]));
        fu.spec[kSpecBOff0 + l] = static_cast<uint32_t>(p.size());
        for (int o = 0; o < out; ++o) p.push_back(f32_bits(layer.bias[o]));
        fu.spec[kSpecSOff0 + l] = static_cast<uint32_t>(p.size());
        for (int o = 0; o < out; ++o)
            for (int g = 0; g < ng; ++g) p.push_back(f32_bits(layer.scale[static_cast<size_t>(o) * stride_groups + g]));
        fu.spec[kSpecQOff0 + l] = static_cast<uint32_t>(p.size());
        const int in4 = (inn + 3) / 4;
        for (int o = 0; o < out; ++o) {
            for (int j = 0; j < in4; ++j) {
                uint32_t wd = 0;
                for (int b = 0; b < 4; ++b) {
                    const int i = j * 4 + b;
                    const int v = i < inn ? static_cast<int>(layer.weight[static_cast<size_t>(o) * inn + i]) : 0;
                    wd |= (static_cast<uint32_t>(v) & 0xFFu) << (8 * b);
                }
                p.push_back(wd);
            }
        }
        fu.spec[kSpecHOff0 + l] = static_cast<uint32_t>(p.size());
        const int in2 = (inn + 1) / 2;
        for (int o = 0; o < out; ++o) {
            for (int j = 0; j < in2; ++j) {
                const int i0 = j * 2;
                const float a = layer.weight[static_cast<size_t>(o) * inn + i0];
                const float b = i0 + 1 < inn ? layer.weight[static_cast<size_t>(o) * inn + i0 + 1] : 0.f;
                p.push_back(static_cast<uint32_t>(f32_to_f16(a)) | (static_cast<uint32_t>(f32_to_f16(b)) << 16));
            }
        }
    }
    if (p.size() > static_cast<size_t>(kFastParamWords)) {
        fu.why = "parameters exceed 64 KiB";
        return fu;
    }
    while (p.size() % 4) p.push_back(0);
    fu.ok = true;
    return fu;
}

std::vector<float> decode_fast_mip(const FastUpload& fu, int mip, int width, int height) {
    const auto& S = fu.spec;
    const auto& P = fu.params;
    const auto& L = fu.latents;
    auto pf = [&](uint32_t i) { return bits_f32(P[i]); };
    const int ngrids = static_cast<int>(S[kSpecGrids]);
    const int nlayers = static_cast<int>(S[kSpecLayers]);
    const int nch = static_cast<int>(S[kSpecOutCh]);
    const int w = std::max(1, width >> mip);
    const int h = std::max(1, height >> mip);
    std::vector<float> img(static_cast<size_t>(w) * h * nch);
    for (int py = 0; py < h; ++py) {
        for (int px = 0; px < w; ++px) {
            const float u = (static_cast<float>(px) + 0.5f) / static_cast<float>(w);
            const float v = (static_cast<float>(py) + 0.5f) / static_cast<float>(h);
            float x[kMaxInput];
            int n = 0;
            uint32_t ls = S[kSpecLatentScaleOff];
            for (int g = 0; g < ngrids; ++g) {
                const int C = static_cast<int>(S[kSpecGridCh0 + g]);
                const int bits = static_cast<int>(S[kSpecGridBits0 + g]);
                const int mi = (g * 8 + std::min(mip, kMaxMips - 1)) * 4;
                const uint32_t base = P[mi];
                const int GW = static_cast<int>(P[mi + 1]);
                const int GH = static_cast<int>(P[mi + 2]);
                const float fx = u * static_cast<float>(GW) - 0.5f;
                const float fy = v * static_cast<float>(GH) - 0.5f;
                const int x0 = static_cast<int>(std::floor(fx));
                const int y0 = static_cast<int>(std::floor(fy));
                const float tx = fx - static_cast<float>(x0);
                const float ty = fy - static_cast<float>(y0);
                const int xa = wrap_i(x0, GW), xb = wrap_i(x0 + 1, GW);
                const int ya = wrap_i(y0, GH), yb = wrap_i(y0 + 1, GH);
                const float mid = static_cast<float>((1 << bits) - 1) * 0.5f;
                uint32_t pbase = base;
                for (const FastPlane& pl : fast_planes(C, bits)) {
                    const int rw = fast_row_words(GW, pl.record_bits);
                    auto rec = [&](int xx, int yy) {
                        const uint32_t bit = static_cast<uint32_t>(xx) * static_cast<uint32_t>(pl.record_bits);
                        const uint32_t wd = L[pbase + static_cast<uint32_t>(yy * rw) + (bit >> 5)];
                        return pl.record_bits == 32 ? wd : (wd >> (bit & 31u)) & ((1u << pl.record_bits) - 1u);
                    };
                    const uint32_t r00 = rec(xa, ya), r01 = rec(xb, ya), r10 = rec(xa, yb), r11 = rec(xb, yb);
                    const uint32_t m = (1u << bits) - 1u;
                    for (int c = 0; c < pl.channels; ++c) {
                        const uint32_t sh = static_cast<uint32_t>(c * bits);
                        const float c00 = static_cast<float>((r00 >> sh) & m);
                        const float c01 = static_cast<float>((r01 >> sh) & m);
                        const float c10 = static_cast<float>((r10 >> sh) & m);
                        const float c11 = static_cast<float>((r11 >> sh) & m);
                        const float a = c00 + (c01 - c00) * tx;
                        const float b = c10 + (c11 - c10) * tx;
                        const float lerped = a + (b - a) * ty;
                        x[n + pl.ch0 + c] = (lerped - mid) * pf(ls + static_cast<uint32_t>(pl.ch0 + c));
                    }
                    pbase += static_cast<uint32_t>(rw * GH);
                }
                n += C;
                ls += static_cast<uint32_t>(C);
            }
            const uint32_t pe = S[kSpecPeMode], oct = S[kSpecPeOct];
            if (pe == kPeFourier && oct) {
                for (uint32_t o = 0; o < oct; ++o) {
                    const float f = static_cast<float>(1u << o);
                    const float tau = static_cast<float>(6.28318530717958647692);
                    x[n++] = std::sin(tau * u * f);
                    x[n++] = std::cos(tau * u * f);
                    x[n++] = std::sin(tau * v * f);
                    x[n++] = std::cos(tau * v * f);
                }
            } else if (pe == kPeTriangle && oct) {
                float pu = u, pv = v;
                for (uint32_t o = 0; o < oct; ++o) {
                    x[n++] = (pu - std::floor(pu)) * 2.f - 1.f;
                    x[n++] = (pv - std::floor(pv)) * 2.f - 1.f;
                    x[n++] = ((pu + 0.25f) - std::floor(pu + 0.25f)) * 2.f - 1.f;
                    x[n++] = ((pv + 0.25f) - std::floor(pv + 0.25f)) * 2.f - 1.f;
                    pu *= 2.f;
                    pv *= 2.f;
                }
                const float lod = static_cast<float>(mip) / static_cast<float>(std::max(1u, S[kSpecNumMips] - 1));
                x[n++] = lod;
                x[n++] = lod;
            }
            float y[kMaxInput];
            int cur = static_cast<int>(S[kSpecWidth0]);
            for (int l = 0; l < nlayers; ++l) {
                const int inn = cur;
                const int out = static_cast<int>(S[kSpecWidth0 + l + 1]);
                if (S[kSpecActI8]) {
                    float s = pf(S[kSpecActScaleOff] + l);
                    s = s > 1e-6f ? s : 1e-6f;
                    for (int i = 0; i < inn; ++i) {
                        float q = std::round(x[i] / s);
                        q = std::min(127.f, std::max(-127.f, q));
                        x[i] = q * s;
                    }
                }
                const int gs = static_cast<int>(S[kSpecGroupSize0 + l]);
                const int ng = static_cast<int>(S[kSpecNumGroups0 + l]);
                const uint32_t wo = S[kSpecWOff0 + l], bo = S[kSpecBOff0 + l], so = S[kSpecSOff0 + l];
                for (int o = 0; o < out; ++o) {
                    float r;
                    if (gs == 0) {
                        float acc = 0.f;
                        for (int i = 0; i < inn; ++i) acc += pf(wo + static_cast<uint32_t>(o * inn + i)) * x[i];
                        r = pf(bo + o) + pf(so + static_cast<uint32_t>(o * ng)) * acc;
                    } else {
                        r = pf(bo + o);
                        for (int g = 0; g < ng; ++g) {
                            const int i0 = g * gs;
                            const int i1 = std::min(inn, i0 + gs);
                            float acc = 0.f;
                            for (int i = i0; i < i1; ++i) acc += pf(wo + static_cast<uint32_t>(o * inn + i)) * x[i];
                            r += pf(so + static_cast<uint32_t>(o * ng + g)) * acc;
                        }
                    }
                    y[o] = (l + 1 < nlayers) ? activate(r, S[kSpecAct], S[kSpecRequant] != 0) : r;
                }
                std::memcpy(x, y, sizeof(float) * static_cast<size_t>(out));
                cur = out;
            }
            std::memcpy(img.data() + (static_cast<size_t>(py) * w + px) * nch, x, sizeof(float) * static_cast<size_t>(nch));
        }
    }
    return img;
}

namespace {

int pe_terms(const TextureSet& ts) { return pe_feature_count(ts.pe_mode, ts.pe_octaves); }

void add_mlp(const TextureSet& ts, SampleCost& c) {
    for (const Layer& l : ts.layers) {
        c.macs += l.in_features * l.out_features;
        const int ng = (l.group_size <= 0 || l.group_size >= l.in_features) ? 1 : std::max(1, l.num_groups);
        // scale fma per group, bias add, activation (2 ops: max, or the hgelu chain)
        c.alu += l.out_features * (ng + 2);
        if (!ts.act_scales.empty()) c.alu += l.in_features * 4;  // div, round, clamp, mul
    }
}

}  // namespace

SampleCost legacy_sample_cost(const TextureSet& ts) {
    SampleCost c;
    int feats = 0;
    for (const Grid& g : ts.grids) {
        feats += g.channels;
        c.latent_loads += 4 * g.channels;  // ntc_u8: one 32-bit load per code byte, 4 taps
        c.scale_loads += 2 * g.channels;   // fp16 scale as two byte loads
        c.alu += g.channels * (4 * 3 + 6 + 3) + 20;  // byte shift and mask per tap, 3 lerps, dequant; wrap
    }
    c.alu += pe_terms(ts) * 4;
    add_mlp(ts, c);
    for (const Layer& l : ts.layers) {
        c.weight_loads += l.in_features * l.out_features;  // one byte load per weight
        const int ng = (l.group_size <= 0 || l.group_size >= l.in_features) ? 1 : std::max(1, l.num_groups);
        c.weight_loads += l.out_features * (2 + 2 * ng);   // fp16 bias and scales, two byte loads each
        c.alu += l.in_features * l.out_features * 4;        // per weight: address, shift, sign extend, convert
    }
    (void)feats;
    c.load_bytes = 4 * (c.latent_loads + c.weight_loads + c.scale_loads);
    return c;
}

SampleCost fast_sample_cost(const TextureSet& ts) {
    SampleCost c;
    for (const Grid& g : ts.grids) {
        const auto planes = fast_planes(g.channels, g.bits);
        c.latent_loads += 4 * static_cast<int>(planes.size());
        c.uniform_words += g.channels + 4;  // latent scales, mip table entry
        c.alu += g.channels * (4 + 6 + 2) + static_cast<int>(planes.size()) * 8 + 16;  // bfe per tap, 3 fma lerps, dequant; addressing
    }
    c.alu += pe_terms(ts) * 4;
    add_mlp(ts, c);
    for (const Layer& l : ts.layers) {
        const int ng = (l.group_size <= 0 || l.group_size >= l.in_features) ? 1 : std::max(1, l.num_groups);
        c.uniform_words += l.in_features * l.out_features + l.out_features * (1 + ng);
    }
    c.load_bytes = 4 * c.latent_loads;
    return c;
}

}  // namespace ntc
