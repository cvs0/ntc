// NTC1 decode library. The arithmetic matches src/ntc/decode_ref.py.
//
// Bindings:
//   0  uvec4 descriptor, 84 vectors, written by ntc::make_gpu_upload
//   1  uint  blob (file bytes, then packed derived mips, 4-byte padded)
//   2  float output, compute shaders only (NTC_NO_OUT skips it)
//
// Descriptor, uvec4 index:
//   0  width, height, channels, grids
//   1  layers, activation, pe_mode, pe_octaves
//   2  flags, weight_bits, num_mips, frame
//   3  filter_mode, max_aniso, 8, 0
//   4..11   grid (channels, bits, scale_offset, nmips)
//   12..15  layer (in, out, weight_offset, bias_offset)
//   16..19  layer (scale_offset, group_size, num_groups, weight_bits)
//   20..83  mip g*8+k (data_offset, row_stride, width, height)
//
// Required path is fp32 dots of integer weights. Define NTC_FP16 before
// including this file to accumulate the dot in float16. That path is
// optional and is not bit-exact.

#ifdef NTC_FP16
#ifndef NTC_FP16_EXT_DONE
#ifdef NTC_FP16_AMD
#extension GL_AMD_gpu_shader_half_float : require
#elif defined(NTC_FP16_NV)
#extension GL_NV_gpu_shader5 : require
#else
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif
#endif
#endif

#ifdef NTC_GL
layout(std430, binding = 0) readonly buffer NtcDesc { uvec4 g_desc[]; };
layout(std430, binding = 1) readonly buffer NtcBlob { uint g_blob[]; };
#ifndef NTC_NO_OUT
layout(std430, binding = 2) writeonly buffer NtcOut { float g_outimg[]; };
#endif
layout(std140, binding = 3) uniform NtcPc { uvec4 g_pc; };
#else
layout(std430, set = 0, binding = 0) readonly buffer NtcDesc { uvec4 g_desc[]; };
layout(std430, set = 0, binding = 1) readonly buffer NtcBlob { uint g_blob[]; };
#ifndef NTC_NO_OUT
layout(std430, set = 0, binding = 2) writeonly buffer NtcOut { float g_outimg[]; };
#endif
layout(push_constant) uniform NtcPc { uvec4 g_pc; };
#endif

float g_feat[160];
float g_ha[64];
float g_hb[64];
float g_y[16];
int g_nfeat;

#ifdef NTC_STAGE
// Workgroup staging. Weights are one contiguous span. Each grid stages the
// full rows the 8x8 tile bilinear-samples, including the halo. A grid that
// does not fit is left on the global load. s_ready stays 0 until the copies
// finish, so the fill itself reads global memory.
shared uint s_words[4096];
shared uint s_ready;
shared uint s_wsrc;
shared uint s_wlen;
shared int s_iy0[8];
shared int s_ny[8];
shared int s_gh_s[8];
shared int s_ch_s[8];
shared uint s_off[8];
shared uint s_stride_s[8];
shared uint s_base_s[8];
shared uint s_grid_ok[8];
#endif

uint ntc_u8(uint off) {
#ifdef NTC_STAGE
    if (s_ready != 0u && s_wlen > 0u && off >= s_wsrc && off < s_wsrc + s_wlen) {
        uint local = off - s_wsrc;
        uint w = s_words[local >> 2u];
        return (w >> ((local & 3u) * 8u)) & 0xFFu;
    }
#endif
    uint w = g_blob[off >> 2u];
    return (w >> ((off & 3u) * 8u)) & 0xFFu;
}

float ntc_f16(uint off) {
    uint bits = ntc_u8(off) | (ntc_u8(off + 1u) << 8u);
    return unpackHalf2x16(bits).x;
}

int ntc_wrap(int i, int n) {
    int r = i % n;
    if (r < 0) r += n;
    return r;
}

#ifdef NTC_STAGE
uint ntc_stage_u8(uint local) {
    uint w = s_words[local >> 2u];
    return (w >> ((local & 3u) * 8u)) & 0xFFu;
}
#endif

float ntc_code(uint base, uint stride, int bits, int W, int H, int ch, int x, int y) {
#ifdef NTC_STAGE
    if (s_ready != 0u) {
        uint ngrids = g_desc[0].w;
        for (uint gi = 0u; gi < ngrids; ++gi) {
            if (s_grid_ok[gi] == 0u || base != s_base_s[gi]) continue;
            int origin = ntc_wrap(s_iy0[gi], s_gh_s[gi]);
            int ly = y - origin;
            if (ly < 0) ly += s_gh_s[gi];
            if (ly < 0 || ly >= s_ny[gi]) continue;
            uint pitch = (s_stride_s[gi] + 3u) & ~3u;
            uint row = s_off[gi] + uint(ly * s_ch_s[gi] + ch) * pitch;
            if (bits == 8) return float(ntc_stage_u8(row + uint(x)));
            if (bits == 4) {
                uint b = ntc_stage_u8(row + uint(x >> 1));
                return float((x & 1) == 0 ? (b & 0xFu) : (b >> 4u));
            }
            uint b2 = ntc_stage_u8(row + uint(x >> 2));
            return float((b2 >> (uint(x & 3) * 2u)) & 3u);
        }
    }
#endif
    uint row = base + uint(ch) * stride * uint(H) + uint(y) * stride;
    if (bits == 8) {
        return float(ntc_u8(row + uint(x)));
    }
    if (bits == 4) {
        uint b = ntc_u8(row + uint(x >> 1));
        return float((x & 1) == 0 ? (b & 0xFu) : (b >> 4u));
    }
    uint b = ntc_u8(row + uint(x >> 2));
    return float((b >> (uint(x & 3) * 2u)) & 3u);
}

float ntc_acc(float acc, float w, float x) {
#ifdef NTC_FP16
    float16_t a = float16_t(acc) + float16_t(w) * float16_t(x);
    return float(a);
#else
    return acc + w * x;
#endif
}

float ntc_weight(uint base, int row, int col, int inn, uint bits) {
    if (bits == 8u) {
        uint b = ntc_u8(base + uint(row * inn + col));
        int s = int(b);
        if (s >= 128) s -= 256;
        return float(s);
    }
    int row_bytes = (inn + 1) >> 1;
    uint b = ntc_u8(base + uint(row * row_bytes + (col >> 1)));
    uint nib = ((col & 1) == 0) ? (b & 0xFu) : (b >> 4u);
    int s = int(nib);
    if (s >= 8) s -= 16;
    return float(s);
}

void ntc_bilinear(uint base, uint stride, int bits, int W, int H, int channels, uint scale_off, vec2 uv) {
    float x = uv.x * float(W) - 0.5;
    float y = uv.y * float(H) - 0.5;
    int x0 = int(floor(x));
    int y0 = int(floor(y));
    float tx = x - float(x0);
    float ty = y - float(y0);
    int x0w = ntc_wrap(x0, W);
    int x1w = ntc_wrap(x0 + 1, W);
    int y0w = ntc_wrap(y0, H);
    int y1w = ntc_wrap(y0 + 1, H);
    float mid = float((1 << bits) - 1) * 0.5;
    for (int c = 0; c < channels; ++c) {
        float c00 = ntc_code(base, stride, bits, W, H, c, x0w, y0w);
        float c01 = ntc_code(base, stride, bits, W, H, c, x1w, y0w);
        float c10 = ntc_code(base, stride, bits, W, H, c, x0w, y1w);
        float c11 = ntc_code(base, stride, bits, W, H, c, x1w, y1w);
        float a = c00 + (c01 - c00) * tx;
        float b = c10 + (c11 - c10) * tx;
        float lerped = a + (b - a) * ty;
        float scale = ntc_f16(scale_off + uint(c) * 2u);
        g_feat[g_nfeat++] = (lerped - mid) * scale;
    }
}

void ntc_positional(vec2 uv, int mip) {
    uint pe = g_desc[1].z;
    uint octaves = g_desc[1].w;
    if (pe == 0u || octaves == 0u) return;
    if (pe == 1u) {
        float tau = 6.283185307179586;
        for (uint o = 0u; o < octaves; ++o) {
            float f = float(1u << o);
            float au = tau * uv.x * f;
            float av = tau * uv.y * f;
            g_feat[g_nfeat++] = sin(au);
            g_feat[g_nfeat++] = cos(au);
            g_feat[g_nfeat++] = sin(av);
            g_feat[g_nfeat++] = cos(av);
        }
        return;
    }
    vec2 p = uv;
    for (uint o = 0u; o < octaves; ++o) {
        float fx = p.x - floor(p.x);
        float fy = p.y - floor(p.y);
        float fx2 = (p.x + 0.25) - floor(p.x + 0.25);
        float fy2 = (p.y + 0.25) - floor(p.y + 0.25);
        g_feat[g_nfeat++] = fx * 2.0 - 1.0;
        g_feat[g_nfeat++] = fy * 2.0 - 1.0;
        g_feat[g_nfeat++] = fx2 * 2.0 - 1.0;
        g_feat[g_nfeat++] = fy2 * 2.0 - 1.0;
        p *= 2.0;
    }
    uint num_mips = g_desc[2].z;
    float denom = float(max(num_mips, 1u) - 1u);
    if (denom < 1.0) denom = 1.0;
    float lod = float(mip) / denom;
    g_feat[g_nfeat++] = lod;
    g_feat[g_nfeat++] = lod;
}

void ntc_features(vec2 uv, int mip) {
    g_nfeat = 0;
    uint ngrids = g_desc[0].w;
    for (uint gi = 0u; gi < ngrids; ++gi) {
        uvec4 grid = g_desc[4u + gi];
        uint nm = grid.w;
        uint k = uint(mip);
        if (k >= nm) k = nm - 1u;
        uvec4 info = g_desc[20u + gi * 8u + k];
        ntc_bilinear(info.x, info.y, int(grid.y), int(info.z), int(info.w), int(grid.x), grid.z, uv);
    }
    ntc_positional(uv, mip);
}

float ntc_activate(float y, uint kind, bool requant) {
    float z = y;
    if (kind == 1u) {
        z = max(y, 0.0);
    } else if (kind == 2u) {
        float clamped = min(y, 3.0);
        float gate = clamp(y * (1.0 / 3.0) + 0.5, 0.0, 1.0);
        z = clamped * gate;
    }
    if (requant) {
        float q = round(z * 80.0 - 113.0);
        z = clamp(q, -128.0, 127.0);
    }
    return z;
}

void ntc_quantize_inputs(uint li, int n) {
    if (g_desc[3].w == 0u || n <= 0) return;
    float s = ntc_f16(g_desc[3].w + li * 2u);
    if (s < 1e-6) s = 1e-6;
    for (int i = 0; i < n; ++i) {
        float v = (li == 0u) ? g_feat[i] : (((li & 1u) == 1u) ? g_ha[i] : g_hb[i]);
        int q = int(round(v / s));
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        v = float(q) * s;
        if (li == 0u) g_feat[i] = v;
        else if ((li & 1u) == 1u) g_ha[i] = v;
        else g_hb[i] = v;
    }
}

void ntc_layer(uint li, int cur_n) {
    ntc_quantize_inputs(li, cur_n);
    uvec4 a = g_desc[12u + li];
    uvec4 b = g_desc[16u + li];
    int inn = int(a.x);
    int outn = int(a.y);
    uint woff = a.z;
    uint boff = a.w;
    uint soff = b.x;
    int gs = int(b.y);
    int groups = int(b.z);
    uint bits = b.w;
    bool one = gs <= 0 || gs >= inn;
    bool last = li + 1u == g_desc[1].x;
    uint kind = g_desc[1].y;
    bool requant = (g_desc[2].x & 2u) != 0u;
    for (int o = 0; o < outn; ++o) {
        float acc_out;
        if (one) {
            float acc = 0.0;
            for (int i = 0; i < inn; ++i) {
                float wv = ntc_weight(woff, o, i, inn, bits);
                float xv = (li == 0u) ? g_feat[i] : ((li & 1u) == 1u ? g_ha[i] : g_hb[i]);
                acc = ntc_acc(acc, wv, xv);
            }
            float scale = ntc_f16(soff + uint(o) * 2u);
            float bias = ntc_f16(boff + uint(o) * 2u);
            acc_out = bias + scale * acc;
        } else {
            float yv = ntc_f16(boff + uint(o) * 2u);
            for (int g = 0; g < groups; ++g) {
                int i0 = g * gs;
                int i1 = min(inn, i0 + gs);
                float acc = 0.0;
                for (int i = i0; i < i1; ++i) {
                    float wv = ntc_weight(woff, o, i, inn, bits);
                    float xv = (li == 0u) ? g_feat[i] : ((li & 1u) == 1u ? g_ha[i] : g_hb[i]);
                    acc = ntc_acc(acc, wv, xv);
                }
                float scale = ntc_f16(soff + uint(o * groups + g) * 2u);
                yv += scale * acc;
            }
            acc_out = yv;
        }
        if (!last) acc_out = ntc_activate(acc_out, kind, requant);
        if (last) g_y[o] = acc_out;
        else if ((li & 1u) == 0u) g_ha[o] = acc_out;
        else g_hb[o] = acc_out;
    }
}

void ntc_mlp() {
    uint nlayers = g_desc[1].x;
    int cur = g_nfeat;
    for (uint li = 0u; li < nlayers; ++li) {
        ntc_layer(li, cur);
        if (li + 1u != nlayers) cur = int(g_desc[12u + li].y);
    }
}

#ifdef NTC_STAGE
void ntc_stage_prepare() {
    uint tid = gl_LocalInvocationIndex;
    if (tid == 0u) {
        s_ready = 0u;
        s_wsrc = 0u;
        s_wlen = 0u;
        for (uint i = 0u; i < 8u; ++i) s_grid_ok[i] = 0u;
        uint nl = g_desc[1].x;
        uint wmin = 0xFFFFFFFFu;
        uint wmax = 0u;
        for (uint li = 0u; li < nl; ++li) {
            uvec4 a = g_desc[12u + li];
            uvec4 b = g_desc[16u + li];
            int inn = int(a.x);
            int outn = int(a.y);
            uint bits = b.w;
            uint wbytes = (bits == 8u) ? uint(outn * inn) : uint(outn * ((inn + 1) / 2));
            wmin = min(wmin, a.z);
            wmax = max(wmax, a.z + wbytes);
            wmin = min(wmin, a.w);
            wmax = max(wmax, a.w + uint(outn) * 2u);
            int groups = max(int(b.z), 1);
            uint sbytes = uint(outn * groups) * 2u;
            wmin = min(wmin, b.x);
            wmax = max(wmax, b.x + sbytes);
        }
        if (wmax > wmin && (wmax - wmin) <= 8192u) {
            s_wsrc = wmin;
            s_wlen = wmax - wmin;
        }
    }
    barrier();
    if (s_wlen > 0u) {
        uint words = (s_wlen + 3u) / 4u;
        for (uint i = tid; i < words; i += 64u) {
            uint b0 = i * 4u;
            uint v = 0u;
            uint src = s_wsrc + b0;
            if (b0 < s_wlen) v |= ntc_u8(src);
            if (b0 + 1u < s_wlen) v |= ntc_u8(src + 1u) << 8u;
            if (b0 + 2u < s_wlen) v |= ntc_u8(src + 2u) << 16u;
            if (b0 + 3u < s_wlen) v |= ntc_u8(src + 3u) << 24u;
            s_words[i] = v;
        }
    }
    barrier();
    uint used = (s_wlen + 3u) & ~3u;
    uint mip = g_pc.x;
    uvec4 h0 = g_desc[0];
    uint W = max(h0.x >> mip, 1u);
    uint H = max(h0.y >> mip, 1u);
    uint y0p = gl_WorkGroupID.y * 8u;
    uint y1p = min(H, y0p + 8u);
    uint ngrids = h0.w;
    for (uint gi = 0u; gi < ngrids && gi < 8u; ++gi) {
        if (tid == 0u && y0p < H) {
            uvec4 grid = g_desc[4u + gi];
            uint nm = grid.w;
            uint k = mip;
            if (k >= nm) k = nm - 1u;
            uvec4 info = g_desc[20u + gi * 8u + k];
            int GH = int(info.w);
            float y_a = (float(y0p) + 0.5) / float(H) * float(GH) - 0.5;
            float y_b = (float(max(y1p, y0p + 1u) - 1u) + 0.5) / float(H) * float(GH) - 0.5;
            int iy0 = int(floor(y_a));
            int iy1 = int(floor(y_b)) + 1;
            int ny = iy1 - iy0 + 1;
            if (ny >= GH) {
                iy0 = 0;
                ny = GH;
            }
            uint stride = info.y;
            uint chs = grid.x;
            uint pitch = (stride + 3u) & ~3u;
            uint bytes = uint(ny) * chs * pitch;
            uint aligned = (used + 3u) & ~3u;
            if (bytes > 0u && aligned + bytes <= 4096u * 4u) {
                s_iy0[gi] = iy0;
                s_ny[gi] = ny;
                s_gh_s[gi] = GH;
                s_ch_s[gi] = int(chs);
                s_stride_s[gi] = stride;
                s_base_s[gi] = info.x;
                s_off[gi] = aligned;
                s_grid_ok[gi] = 1u;
                used = aligned + bytes;
            }
        }
        barrier();
        if (s_grid_ok[gi] != 0u) {
            int ny = s_ny[gi];
            int chs = s_ch_s[gi];
            uint stride = s_stride_s[gi];
            uint pitch = (stride + 3u) & ~3u;
            uint nrows = uint(ny * chs);
            for (uint row = tid; row < nrows; row += 64u) {
                int ly = int(row) / chs;
                int ch = int(row) % chs;
                int gy = ntc_wrap(s_iy0[gi] + ly, s_gh_s[gi]);
                uint src = s_base_s[gi] + uint(ch) * stride * uint(s_gh_s[gi]) + uint(gy) * stride;
                uint dst = s_off[gi] + row * pitch;
                uint words = (stride + 3u) / 4u;
                for (uint i = 0u; i < words; ++i) {
                    uint b0 = i * 4u;
                    uint v = 0u;
                    if (b0 < stride) v |= ntc_u8(src + b0);
                    if (b0 + 1u < stride) v |= ntc_u8(src + b0 + 1u) << 8u;
                    if (b0 + 2u < stride) v |= ntc_u8(src + b0 + 2u) << 16u;
                    if (b0 + 3u < stride) v |= ntc_u8(src + b0 + 3u) << 24u;
                    s_words[(dst >> 2u) + i] = v;
                }
            }
        }
        barrier();
    }
    if (tid == 0u) s_ready = 1u;
    barrier();
}
#endif

void ntc_eval(vec2 uv, int mip) {
    ntc_features(uv, mip);
    ntc_mlp();
}

float ntc_hash01(int x, int y, uint frame) {
    uint h = uint(x) * 1664525u + uint(y) * 1013904223u + frame * 22695477u;
    return float(h & 0x00FFFFFFu) / 16777216.0;
}

void ntc_copy_y(inout float dst[16]) {
    for (int i = 0; i < 16; ++i) dst[i] = g_y[i];
}

void ntc_eval_at_texel(int x, int y, int mip, int W, int H) {
    int xx = ntc_wrap(x, W);
    int yy = ntc_wrap(y, H);
    vec2 uv = (vec2(float(xx), float(yy)) + 0.5) / vec2(float(W), float(H));
    ntc_eval(uv, mip);
}

void ntc_eval_filtered(vec2 uv, vec2 dudx, vec2 dudy, float lod, uint frame, uint filt, uint max_aniso) {
    int mip0 = int(floor(lod));
    if (mip0 < 0) mip0 = 0;
    if (filt == 0u || filt == 1u) {
        ntc_eval(uv, mip0);
        return;
    }
    if (filt == 4u) {
        float t = lod - float(mip0);
        ntc_eval(uv, mip0);
        float a0[16];
        ntc_copy_y(a0);
        ntc_eval(uv, mip0 + 1);
        uint ch = g_desc[0].z;
        for (uint i = 0u; i < ch; ++i) g_y[i] = a0[i] + (g_y[i] - a0[i]) * t;
        return;
    }
    uvec4 h0 = g_desc[0];
    int W = int(max(h0.x >> uint(mip0), 1u));
    int H = int(max(h0.y >> uint(mip0), 1u));
    float x = uv.x * float(W) - 0.5;
    float y = uv.y * float(H) - 0.5;
    int x0 = int(floor(x));
    int y0 = int(floor(y));
    float tx = x - float(x0);
    float ty = y - float(y0);
    if (filt == 3u) {
        float u = ntc_hash01(x0, y0, frame);
        float w00 = (1.0 - tx) * (1.0 - ty);
        float w10 = tx * (1.0 - ty);
        float w01 = (1.0 - tx) * ty;
        int choice = 0;
        if (u >= w00) choice = 1;
        if (u >= w00 + w10) choice = 2;
        if (u >= w00 + w10 + w01) choice = 3;
        int xx = x0 + (((choice == 1) || (choice == 3)) ? 1 : 0);
        int yy = y0 + (((choice == 2) || (choice == 3)) ? 1 : 0);
        ntc_eval_at_texel(xx, yy, mip0, W, H);
        return;
    }
    if (filt == 5u) {
        float lx = length(dudx);
        float ly = length(dudy);
        float major = max(lx, ly);
        float minor = max(min(lx, ly), 1e-6);
        int N = int(ceil(major / minor));
        if (N < 1) N = 1;
        int cap = int(max_aniso);
        if (cap < 1) cap = 1;
        if (N > cap) N = cap;
        vec2 axis = lx >= ly ? dudx : dudy;
        float acc[16];
        for (int i = 0; i < 16; ++i) acc[i] = 0.0;
        for (int i = 0; i < N; ++i) {
            float t = (float(i) + 0.5) / float(N) - 0.5;
            vec2 p = uv + axis * t;
            float px = p.x * float(W) - 0.5;
            float py = p.y * float(H) - 0.5;
            int px0 = int(floor(px));
            int py0 = int(floor(py));
            float ptx = px - float(px0);
            float pty = py - float(py0);
            float u = ntc_hash01(px0, py0, frame + uint(i) * 747796405u);
            float w00 = (1.0 - ptx) * (1.0 - pty);
            float w10 = ptx * (1.0 - pty);
            float w01 = (1.0 - ptx) * pty;
            int choice = 0;
            if (u >= w00) choice = 1;
            if (u >= w00 + w10) choice = 2;
            if (u >= w00 + w10 + w01) choice = 3;
            int xx = px0 + (((choice == 1) || (choice == 3)) ? 1 : 0);
            int yy = py0 + (((choice == 2) || (choice == 3)) ? 1 : 0);
            ntc_eval_at_texel(xx, yy, mip0, W, H);
            for (int c = 0; c < 16; ++c) acc[c] += g_y[c];
        }
        uint ch = g_desc[0].z;
        for (uint c = 0u; c < ch; ++c) g_y[c] = acc[c] / float(N);
        return;
    }
    // filt == 2: color bilinear, four texel-center evaluations
    float c00[16];
    float c10[16];
    float c01[16];
    float c11[16];
    ntc_eval_at_texel(x0, y0, mip0, W, H);
    ntc_copy_y(c00);
    ntc_eval_at_texel(x0 + 1, y0, mip0, W, H);
    ntc_copy_y(c10);
    ntc_eval_at_texel(x0, y0 + 1, mip0, W, H);
    ntc_copy_y(c01);
    ntc_eval_at_texel(x0 + 1, y0 + 1, mip0, W, H);
    ntc_copy_y(c11);
    uint ch = g_desc[0].z;
    for (uint i = 0u; i < ch; ++i) {
        float a = c00[i] + (c10[i] - c00[i]) * tx;
        float b = c01[i] + (c11[i] - c01[i]) * tx;
        g_y[i] = a + (b - a) * ty;
    }
}
