#include "ntc/ntc.hpp"
#include "range_coder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

extern "C" {
#include "puff.h"
}

namespace ntc {
namespace {

uint16_t rd_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t rd_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            uint32_t e = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x3FFu;
            out = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

float read_f16(const uint8_t* data, uint32_t offset) {
    return f16_to_f32(rd_u16(data + offset));
}

uint32_t crc32(const uint8_t* data, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= data[i];
        for (int b = 0; b < 8; ++b) {
            const uint32_t mask = -(c & 1u);
            c = (c >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~c;
}

void unpack_codes(const uint8_t* data, uint32_t offset, int c, int h, int w, int bits, int stride,
                  std::vector<uint8_t>& dst) {
    dst.assign(static_cast<size_t>(c) * h * w, 0);
    for (int ci = 0; ci < c; ++ci) {
        for (int y = 0; y < h; ++y) {
            const uint8_t* row = data + offset + static_cast<uint32_t>(ci) * stride * h +
                                 static_cast<uint32_t>(y) * stride;
            uint8_t* out = dst.data() + (static_cast<size_t>(ci) * h + y) * w;
            if (bits == 8) {
                std::memcpy(out, row, static_cast<size_t>(w));
            } else if (bits == 4) {
                for (int x = 0; x < w; x += 2) {
                    const uint8_t b = row[x >> 1];
                    out[x] = b & 0x0F;
                    if (x + 1 < w) out[x + 1] = b >> 4;
                }
            } else if (bits == 2) {
                for (int x = 0; x < w; ++x) {
                    const uint8_t b = row[x >> 2];
                    out[x] = (b >> ((x & 3) * 2)) & 0x03;
                }
            } else {
                throw std::runtime_error("bad latent bits");
            }
        }
    }
}

std::vector<uint8_t> box_down(const std::vector<uint8_t>& codes, int c, int h, int w, int& oh, int& ow) {
    if (h == 1 && w == 1) return {};
    oh = std::max(1, h / 2);
    ow = std::max(1, w / 2);
    std::vector<uint8_t> out(static_cast<size_t>(c) * oh * ow, 0);
    auto at = [&](int ci, int y, int x) -> int {
        return codes[(static_cast<size_t>(ci) * h + y) * w + x];
    };
    for (int ci = 0; ci < c; ++ci) {
        for (int y = 0; y < oh; ++y) {
            for (int x = 0; x < ow; ++x) {
                int acc = 0;
                int n = 0;
                const int y0 = (h >= 2) ? y * 2 : 0;
                const int x0 = (w >= 2) ? x * 2 : 0;
                const int y1 = (h >= 2) ? y0 + 1 : 0;
                const int x1 = (w >= 2) ? x0 + 1 : 0;
                acc += at(ci, y0, x0);
                ++n;
                if (h >= 2) {
                    acc += at(ci, y1, x0);
                    ++n;
                }
                if (w >= 2) {
                    acc += at(ci, y0, x1);
                    ++n;
                }
                if (h >= 2 && w >= 2) {
                    acc += at(ci, y1, x1);
                    ++n;
                }
                const int rounded = (n == 4) ? (acc + 2) / 4 : (acc + n / 2) / n;
                out[(static_cast<size_t>(ci) * oh + y) * ow + x] = static_cast<uint8_t>(rounded);
            }
        }
    }
    return out;
}

void build_mips(Grid& g) {
    g.mips.clear();
    g.mip_w.clear();
    g.mip_h.clear();
    g.mips.push_back(g.codes);
    g.mip_w.push_back(g.width);
    g.mip_h.push_back(g.height);
    int h = g.height;
    int w = g.width;
    std::vector<uint8_t> cur = g.codes;
    while (static_cast<int>(g.mips.size()) < kMaxMips && !(h == 1 && w == 1)) {
        int oh = 0, ow = 0;
        auto nxt = box_down(cur, g.channels, h, w, oh, ow);
        if (nxt.empty()) break;
        if (oh == h && ow == w) break;
        g.mips.push_back(nxt);
        g.mip_w.push_back(ow);
        g.mip_h.push_back(oh);
        cur.swap(nxt);
        h = oh;
        w = ow;
    }
}

int wrap_index(int i, int n) {
    int r = i % n;
    if (r < 0) r += n;
    return r;
}

float code_at(const std::vector<uint8_t>& codes, int c, int h, int w, int ci, int y, int x) {
    return static_cast<float>(codes[(static_cast<size_t>(ci) * h + y) * w + x]);
}

void bilinear(const std::vector<uint8_t>& codes, int channels, int h, int w, float u, float v,
              float* lerped) {
    const float x = u * static_cast<float>(w) - 0.5f;
    const float y = v * static_cast<float>(h) - 0.5f;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const int x0w = wrap_index(x0, w);
    const int x1w = wrap_index(x0 + 1, w);
    const int y0w = wrap_index(y0, h);
    const int y1w = wrap_index(y0 + 1, h);
    for (int ci = 0; ci < channels; ++ci) {
        const float c00 = code_at(codes, channels, h, w, ci, y0w, x0w);
        const float c01 = code_at(codes, channels, h, w, ci, y0w, x1w);
        const float c10 = code_at(codes, channels, h, w, ci, y1w, x0w);
        const float c11 = code_at(codes, channels, h, w, ci, y1w, x1w);
        const float a = c00 + (c01 - c00) * tx;
        const float b = c10 + (c11 - c10) * tx;
        lerped[ci] = a + (b - a) * ty;
    }
}

void positional(const TextureSet& ts, float u, float v, int mip, float* dst, int& n) {
    n = 0;
    if (ts.pe_mode == kPeNone || ts.pe_octaves == 0) return;
    if (ts.pe_mode == kPeFourier) {
        float pu = u, pv = v;
        for (uint32_t o = 0; o < ts.pe_octaves; ++o) {
            const float f = static_cast<float>(1u << o);
            const float tau = static_cast<float>(6.28318530717958647692);
            const float au = tau * pu * f;
            const float av = tau * pv * f;
            dst[n++] = std::sin(au);
            dst[n++] = std::cos(au);
            dst[n++] = std::sin(av);
            dst[n++] = std::cos(av);
            (void)pu;
            (void)pv;
        }
        // frequencies applied via f, uv itself is not doubled. Matches Python.
        return;
    }
    if (ts.pe_mode == kPeTriangle) {
        float pu = u, pv = v;
        for (uint32_t o = 0; o < ts.pe_octaves; ++o) {
            const float fx = pu - std::floor(pu);
            const float fy = pv - std::floor(pv);
            const float fx2 = (pu + 0.25f) - std::floor(pu + 0.25f);
            const float fy2 = (pv + 0.25f) - std::floor(pv + 0.25f);
            dst[n++] = fx * 2.f - 1.f;
            dst[n++] = fy * 2.f - 1.f;
            dst[n++] = fx2 * 2.f - 1.f;
            dst[n++] = fy2 * 2.f - 1.f;
            pu *= 2.f;
            pv *= 2.f;
            (void)o;
        }
        const float denom = static_cast<float>(std::max(1u, ts.num_mips - 1));
        const float lod = static_cast<float>(mip) / denom;
        dst[n++] = lod;
        dst[n++] = lod;
    }
}

float activate_one(float y, uint32_t kind, bool requant) {
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

void apply_layer(const Layer& layer, const float* x, float* y) {
    const int inn = layer.in_features;
    const int out = layer.out_features;
    const int gs = layer.group_size;
    const bool one = gs <= 0 || gs >= inn;
    for (int o = 0; o < out; ++o) {
        const float* wrow = layer.weight.data() + static_cast<size_t>(o) * inn;
        if (one) {
            float acc = 0.f;
            for (int i = 0; i < inn; ++i) acc += wrow[i] * x[i];
            y[o] = layer.bias[o] + layer.scale[static_cast<size_t>(o) * layer.num_groups] * acc;
        } else {
            float yv = layer.bias[o];
            for (int g = 0; g < layer.num_groups; ++g) {
                const int i0 = g * gs;
                const int i1 = std::min(inn, i0 + gs);
                float acc = 0.f;
                for (int i = i0; i < i1; ++i) acc += wrow[i] * x[i];
                yv += layer.scale[static_cast<size_t>(o) * layer.num_groups + g] * acc;
            }
            y[o] = yv;
        }
    }
}

void features_at(const TextureSet& ts, float u, float v, int mip, float* feat) {
    int n = 0;
    for (const Grid& g : ts.grids) {
        const int k = std::min(mip, static_cast<int>(g.mips.size()) - 1);
        const int h = g.mip_h[k];
        const int w = g.mip_w[k];
        float lerped[16];
        bilinear(g.mips[k], g.channels, h, w, u, v, lerped);
        const float mid = (static_cast<float>((1 << g.bits) - 1)) * 0.5f;
        for (int c = 0; c < g.channels; ++c) {
            feat[n++] = (lerped[c] - mid) * g.scale[c];
        }
    }
    float pe[64];
    int pn = 0;
    positional(ts, u, v, mip, pe, pn);
    for (int i = 0; i < pn; ++i) feat[n++] = pe[i];
    if (n != ts.layers[0].in_features) {
        throw std::runtime_error("feature count does not match layer 0");
    }
}

void mlp(const TextureSet& ts, const float* feat, float* out) {
    float a[kMaxHidden];
    float b[kMaxHidden];
    const float* cur = feat;
    int cur_n = ts.layers[0].in_features;
    const int last = static_cast<int>(ts.layers.size()) - 1;
    for (int li = 0; li <= last; ++li) {
        const Layer& layer = ts.layers[li];
        if (layer.in_features != cur_n) throw std::runtime_error("layer input mismatch");
        float* dst = (li == last) ? out : ((li % 2 == 0) ? a : b);
        // first hidden writes a, but if last, out. For li==0 and not last, dst=a.
        if (li != last && li % 2 == 1) dst = b;
        if (li != last && li % 2 == 0) dst = a;
        apply_layer(layer, cur, dst);
        if (li != last) {
            for (int o = 0; o < layer.out_features; ++o) {
                dst[o] = activate_one(dst[o], ts.activation, ts.activation_requant());
            }
            cur = dst;
            cur_n = layer.out_features;
        }
    }
}

std::vector<uint8_t> pack_codes(const std::vector<uint8_t>& codes, int c, int h, int w, int bits,
                                int& stride) {
    stride = (w * bits + 7) / 8;
    std::vector<uint8_t> out(static_cast<size_t>(c) * stride * h, 0);
    for (int ci = 0; ci < c; ++ci) {
        for (int y = 0; y < h; ++y) {
            uint8_t* row = out.data() + (static_cast<size_t>(ci) * h + y) * stride;
            const uint8_t* src = codes.data() + (static_cast<size_t>(ci) * h + y) * w;
            if (bits == 8) {
                std::memcpy(row, src, static_cast<size_t>(w));
            } else if (bits == 4) {
                for (int x = 0; x < w; x += 2) {
                    const uint8_t lo = src[x] & 0x0F;
                    const uint8_t hi = (x + 1 < w) ? (src[x + 1] & 0x0F) : 0;
                    row[x >> 1] = static_cast<uint8_t>(lo | (hi << 4));
                }
            } else if (bits == 2) {
                for (int x = 0; x < w; ++x) {
                    row[x >> 2] |= static_cast<uint8_t>((src[x] & 0x03) << ((x & 3) * 2));
                }
            }
        }
    }
    return out;
}

}  // namespace

int pe_feature_count(int pe_mode, int pe_octaves) {
    if (pe_mode == kPeNone || pe_octaves <= 0) return 0;
    if (pe_mode == kPeFourier) return 4 * pe_octaves;
    if (pe_mode == kPeTriangle) return 4 * pe_octaves + 2;
    return 0;
}

TextureSet parse(const uint8_t* data, size_t size) {
    if (size < kHeaderBytes) throw std::runtime_error("short ntc file");
    if (rd_u32(data) != kMagic) throw std::runtime_error("bad ntc magic");
    if (rd_u16(data + 4) != kVersion) throw std::runtime_error("bad ntc version");
    if (rd_u16(data + 6) != kHeaderBytes) throw std::runtime_error("bad header size");
    const uint32_t file_bytes = rd_u32(data + 80);
    if (file_bytes != size) throw std::runtime_error("file_bytes mismatch");
    const uint32_t expect = rd_u32(data + 76);
    if (crc32(data + kHeaderBytes, size - kHeaderBytes) != expect) {
        throw std::runtime_error("crc mismatch");
    }
    TextureSet ts;
    ts.flags = rd_u32(data + 8);
    ts.width = rd_u32(data + 12);
    ts.height = rd_u32(data + 16);
    const uint32_t num_channels = rd_u16(data + 20);
    ts.num_mips = rd_u16(data + 22);
    const uint32_t num_grids = rd_u16(data + 24);
    const uint32_t num_layers = rd_u16(data + 26);
    ts.activation = rd_u16(data + 28);
    ts.pe_mode = rd_u16(data + 30);
    ts.pe_octaves = rd_u16(data + 32);
    ts.weight_bits = rd_u16(data + 34);
    ts.group_size = rd_u16(data + 36);
    ts.tile_size = rd_u16(data + 38);
    if (ts.tile_size != 0) {
        throw std::runtime_error("per-tile latent endpoints are not in this reader");
    }
    const uint32_t channel_off = rd_u32(data + 40);
    const uint32_t latent_off = rd_u32(data + 52);
    const uint32_t latent_bytes = rd_u32(data + 56);
    ts.latent_stored_bytes = latent_bytes;
    ts.scale_stored_bytes = rd_u32(data + 72);
    const uint32_t plain_len = rd_u32(data + 84);
    if ((ts.flags & kFlagLatentDeflate) && (ts.flags & kFlagLatentRange)) {
        throw std::runtime_error("deflate and range flags are mutually exclusive");
    }
    ts.latent_plain_bytes = (ts.flags & kFlagLatentCompressed) ? plain_len : latent_bytes;
    std::vector<uint8_t> inflated;
    const uint8_t* code_base = data;
    if (ts.flags & kFlagLatentCompressed) {
        if (plain_len == 0 || static_cast<size_t>(latent_off) + latent_bytes > size) {
            throw std::runtime_error("bad compressed latent span");
        }
        if (ts.flags & kFlagLatentDeflate) {
            inflated.assign(plain_len, 0);
            unsigned long dlen = plain_len;
            unsigned long slen = latent_bytes;
            const int rc = puff(inflated.data(), &dlen, data + latent_off, &slen);
            if (rc != 0 || dlen != plain_len) throw std::runtime_error("latent deflate");
        } else {
            inflated = range_decode(data + latent_off, latent_bytes, plain_len);
            if (inflated.size() != plain_len) throw std::runtime_error("latent range");
        }
        code_base = inflated.data();
    }
    const uint32_t grid_off = rd_u32(data + 44);
    const uint32_t layer_off = rd_u32(data + 48);
    ts.raw.assign(data, data + size);

    ts.channels.resize(num_channels);
    for (uint32_t i = 0; i < num_channels; ++i) {
        const uint8_t* p = data + channel_off + 8 * i;
        ts.channels[i].semantic = p[0];
        ts.channels[i].color_space = p[1];
    }
    ts.grids.resize(num_grids);
    for (uint32_t i = 0; i < num_grids; ++i) {
        const uint8_t* p = data + grid_off + 24 * i;
        Grid g;
        g.width = rd_u16(p + 0);
        g.height = rd_u16(p + 2);
        g.channels = p[4];
        g.bits = p[5];
        g.row_stride = rd_u16(p + 6);
        g.data_offset = rd_u32(p + 8);
        g.data_bytes = rd_u32(p + 12);
        g.scale_offset = rd_u32(p + 16);
        g.plane_bytes = rd_u32(p + 20);
        unpack_codes(code_base, g.data_offset, g.channels, g.height, g.width, g.bits, g.row_stride,
                     g.codes);
        g.scale.resize(g.channels);
        for (int c = 0; c < g.channels; ++c) {
            g.scale[c] = read_f16(data, g.scale_offset + static_cast<uint32_t>(c) * 2);
        }
        build_mips(g);
        ts.grids[i] = std::move(g);
    }
    ts.layers.resize(num_layers);
    for (uint32_t i = 0; i < num_layers; ++i) {
        const uint8_t* p = data + layer_off + 32 * i;
        Layer layer;
        layer.in_features = rd_u16(p + 0);
        layer.out_features = rd_u16(p + 2);
        layer.weight_offset = rd_u32(p + 4);
        layer.weight_bytes = rd_u32(p + 8);
        layer.bias_offset = rd_u32(p + 12);
        layer.scale_offset = rd_u32(p + 16);
        layer.group_size = rd_u16(p + 20);
        layer.num_groups = rd_u16(p + 22);
        const uint32_t lbits = rd_u32(p + 24);
        layer.weight_bits = lbits ? lbits : ts.weight_bits;
        const int out = layer.out_features;
        const int inn = layer.in_features;
        layer.weight.assign(static_cast<size_t>(out) * inn, 0.f);
        if (layer.weight_bits == 8) {
            const int8_t* w = reinterpret_cast<const int8_t*>(data + layer.weight_offset);
            for (int k = 0; k < out * inn; ++k) layer.weight[k] = static_cast<float>(w[k]);
        } else if (layer.weight_bits == 4) {
            const int row_bytes = (inn + 1) / 2;
            const uint8_t* raw = data + layer.weight_offset;
            for (int o = 0; o < out; ++o) {
                for (int col = 0; col < inn; ++col) {
                    const uint8_t byte = raw[o * row_bytes + (col >> 1)];
                    int nib = (col & 1) ? (byte >> 4) : (byte & 0x0F);
                    if (nib >= 8) nib -= 16;
                    layer.weight[static_cast<size_t>(o) * inn + col] = static_cast<float>(nib);
                }
            }
        } else {
            throw std::runtime_error("bad weight bits");
        }
        layer.bias.resize(out);
        for (int o = 0; o < out; ++o) {
            layer.bias[o] = read_f16(data, layer.bias_offset + static_cast<uint32_t>(o) * 2);
        }
        layer.scale.resize(static_cast<size_t>(out) * layer.num_groups);
        for (size_t k = 0; k < layer.scale.size(); ++k) {
            layer.scale[k] = read_f16(data, layer.scale_offset + static_cast<uint32_t>(k) * 2);
        }
        ts.layers[i] = std::move(layer);
    }
    return ts;
}

TextureSet load_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return parse(buf.data(), buf.size());
}

void decode_uv(const TextureSet& ts, float u, float v, int mip, float* out) {
    float feat[kMaxInput];
    features_at(ts, u, v, mip, feat);
    mlp(ts, feat, out);
}

std::vector<float> decode_mip(const TextureSet& ts, int mip) {
    const int w = std::max(1, static_cast<int>(ts.width >> mip));
    const int h = std::max(1, static_cast<int>(ts.height >> mip));
    const int c = ts.num_channels();
    std::vector<float> img(static_cast<size_t>(h) * w * c);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(w);
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(h);
            decode_uv(ts, u, v, mip, img.data() + (static_cast<size_t>(y) * w + x) * c);
        }
    }
    return img;
}

uint32_t payload_bytes(const TextureSet& ts) {
    uint32_t n = 0;
    if (ts.flags & kFlagLatentCompressed) n += ts.latent_stored_bytes;
    else {
        for (const auto& g : ts.grids) n += g.data_bytes;
    }
    n += ts.scale_stored_bytes;
    for (const auto& layer : ts.layers) {
        n += layer.weight_bytes;
        n += static_cast<uint32_t>(layer.out_features) * 2u;
        n += static_cast<uint32_t>(layer.out_features) * layer.num_groups * 2u;
    }
    return n;
}

float total_bpp(const TextureSet& ts) {
    return 8.f * static_cast<float>(payload_bytes(ts)) /
           static_cast<float>(ts.width * ts.height);
}

GpuUpload make_gpu_upload(const TextureSet& ts, uint32_t frame, uint32_t filter_mode,
                          uint32_t max_aniso) {
    GpuUpload up;
    up.blob = ts.raw;
    up.desc.assign(kDescWords, 0);
    auto set4 = [&](int uvec, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
        up.desc[uvec * 4 + 0] = x;
        up.desc[uvec * 4 + 1] = y;
        up.desc[uvec * 4 + 2] = z;
        up.desc[uvec * 4 + 3] = w;
    };
    set4(0, ts.width, ts.height, static_cast<uint32_t>(ts.num_channels()),
         static_cast<uint32_t>(ts.grids.size()));
    set4(1, static_cast<uint32_t>(ts.layers.size()), ts.activation, ts.pe_mode, ts.pe_octaves);
    set4(2, ts.flags, ts.weight_bits, ts.num_mips, frame);
    set4(3, filter_mode, max_aniso, kMaxMips, 0);
    for (size_t g = 0; g < ts.grids.size() && g < 8; ++g) {
        const Grid& grid = ts.grids[g];
        set4(4 + static_cast<int>(g), static_cast<uint32_t>(grid.channels),
             static_cast<uint32_t>(grid.bits), grid.scale_offset,
             static_cast<uint32_t>(grid.mips.size()));
        for (size_t k = 0; k < grid.mips.size() && k < 8; ++k) {
            uint32_t data_off = grid.data_offset;
            int stride = grid.row_stride;
            if (k > 0 || (ts.flags & kFlagLatentCompressed)) {
                int ps = 0;
                auto packed = pack_codes(grid.mips[k], grid.channels, grid.mip_h[k], grid.mip_w[k],
                                         grid.bits, ps);
                if (up.blob.size() % 2) up.blob.push_back(0);
                data_off = static_cast<uint32_t>(up.blob.size());
                up.blob.insert(up.blob.end(), packed.begin(), packed.end());
                stride = ps;
            }
            const int mi = 20 + static_cast<int>(g) * 8 + static_cast<int>(k);
            set4(mi, data_off, static_cast<uint32_t>(stride), static_cast<uint32_t>(grid.mip_w[k]),
                 static_cast<uint32_t>(grid.mip_h[k]));
        }
    }
    for (size_t l = 0; l < ts.layers.size() && l < 4; ++l) {
        const Layer& layer = ts.layers[l];
        set4(12 + static_cast<int>(l), static_cast<uint32_t>(layer.in_features),
             static_cast<uint32_t>(layer.out_features), layer.weight_offset, layer.bias_offset);
        const uint32_t lbits = layer.weight_bits ? layer.weight_bits : ts.weight_bits;
        set4(16 + static_cast<int>(l), layer.scale_offset, static_cast<uint32_t>(layer.group_size),
             static_cast<uint32_t>(layer.num_groups), lbits);
    }
    while (up.blob.size() % 4) up.blob.push_back(0);
    return up;
}

}  // namespace ntc
