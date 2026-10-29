// Real-time material viewer.
// Inference-on-sample: the fragment shader calls ntc_eval_filtered every pixel,
// with lod from dFdx/dFdy of the mesh uv (feature trilinear, filt = 4).
// Decode-on-load: the same mesh samples BC7/BC5/BC1 packed at load from the
// CPU decode. That packer is mode 6 / DXT1 / BC4, the same family as
// bench/bc_pack.py, not bc7enc. ASTC is sampled only when the device
// supports VK_FORMAT_ASTC_4x4_UNORM_BLOCK and --astc supplies a file.
// There is no ASTC encoder in this binary.
//
// --benchmark renders offscreen. A CPU device skips 1920x1080 and 2560x1440
// and records the skip. Those rows are not a performance claim.

#include "ntc/ntc.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <windowsx.h>
#define NTC_WIN32 1
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(NTC_X11)
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#define VK_USE_PLATFORM_XLIB_KHR
#endif

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(expr)                                                                 \
    do {                                                                            \
        VkResult _r = (expr);                                                       \
        if (_r != VK_SUCCESS) throw std::runtime_error(std::string(#expr) + " " + std::to_string(_r)); \
    } while (0)

static constexpr int kW4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

struct FrameUBO {
    float mvp[16];
    float model[16];
    float eye[4];
    float light[4];
    uint32_t sw[4];
    uint32_t sem0[4];
    uint32_t sem1[4];
    uint32_t sem2[4];
};
static_assert(sizeof(FrameUBO) == 224, "frame ubo");

static std::vector<uint8_t> read_bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static uint32_t find_mem(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags) return i;
    }
    throw std::runtime_error("no memory type");
}

struct Buf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize alloc = 0;
    void* mapped = nullptr;
};

static Buf make_buf(VkDevice dev, VkPhysicalDevice phys, VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags mem, const void* src) {
    Buf b;
    b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = std::max<VkDeviceSize>(size, 4);
    bi.usage = usage;
    CHECK(vkCreateBuffer(dev, &bi, nullptr, &b.buffer));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, b.buffer, &req);
    b.alloc = req.size;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    try {
        ai.memoryTypeIndex = find_mem(phys, req.memoryTypeBits, mem);
    } catch (const std::runtime_error&) {
        ai.memoryTypeIndex = find_mem(phys, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.memory));
    CHECK(vkBindBufferMemory(dev, b.buffer, b.memory, 0));
    if (mem & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        CHECK(vkMapMemory(dev, b.memory, 0, req.size, 0, &b.mapped));
        if (src && size) std::memcpy(b.mapped, src, static_cast<size_t>(size));
    }
    return b;
}

static void destroy_buf(VkDevice dev, Buf& b) {
    if (b.mapped) vkUnmapMemory(dev, b.memory);
    if (b.buffer) vkDestroyBuffer(dev, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(dev, b.memory, nullptr);
    b = {};
}

static void write_png(const std::string& path, const uint8_t* bgra, int w, int h) {
    auto crc_table = []() {
        static uint32_t t[256];
        static bool init = false;
        if (!init) {
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = i;
                for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
                t[i] = c;
            }
            init = true;
        }
        return t;
    };
    auto crc32 = [&](const uint8_t* data, size_t n) {
        uint32_t c = 0xFFFFFFFFu;
        const uint32_t* t = crc_table();
        for (size_t i = 0; i < n; ++i) c = t[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
        return c ^ 0xFFFFFFFFu;
    };
    std::vector<uint8_t> raw;
    raw.reserve(static_cast<size_t>(h) * (1 + w * 4));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        for (int x = 0; x < w; ++x) {
            const uint8_t* p = bgra + (static_cast<size_t>(y) * w + x) * 4;
            raw.push_back(p[2]);
            raw.push_back(p[1]);
            raw.push_back(p[0]);
            raw.push_back(255);
        }
    }
    uint32_t adler = 1;
    uint32_t s1 = 1, s2 = 0;
    for (uint8_t b : raw) {
        s1 = (s1 + b) % 65521u;
        s2 = (s2 + s1) % 65521u;
    }
    adler = (s2 << 16) | s1;
    std::vector<uint8_t> z;
    z.push_back(0x78);
    z.push_back(0x01);
    size_t off = 0;
    while (off < raw.size()) {
        size_t n = std::min<size_t>(65535, raw.size() - off);
        const bool last = off + n == raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(static_cast<uint8_t>(n & 0xFF));
        z.push_back(static_cast<uint8_t>((n >> 8) & 0xFF));
        uint16_t nlen = static_cast<uint16_t>(~static_cast<uint16_t>(n));
        z.push_back(static_cast<uint8_t>(nlen & 0xFF));
        z.push_back(static_cast<uint8_t>((nlen >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(off),
                 raw.begin() + static_cast<std::ptrdiff_t>(off + n));
        off += n;
    }
    z.push_back(static_cast<uint8_t>((adler >> 24) & 0xFF));
    z.push_back(static_cast<uint8_t>((adler >> 16) & 0xFF));
    z.push_back(static_cast<uint8_t>((adler >> 8) & 0xFF));
    z.push_back(static_cast<uint8_t>(adler & 0xFF));
    auto chunk = [&](std::vector<uint8_t>& out, const char* type, const uint8_t* data, size_t n) {
        uint32_t len = static_cast<uint32_t>(n);
        uint8_t lh[4] = {uint8_t(len >> 24), uint8_t(len >> 16), uint8_t(len >> 8), uint8_t(len)};
        out.insert(out.end(), lh, lh + 4);
        out.insert(out.end(), type, type + 4);
        if (n) out.insert(out.end(), data, data + n);
        uint32_t c = crc32(out.data() + out.size() - n - 4, n + 4);
        uint8_t ch[4] = {uint8_t(c >> 24), uint8_t(c >> 16), uint8_t(c >> 8), uint8_t(c)};
        out.insert(out.end(), ch, ch + 4);
    };
    std::vector<uint8_t> png = {137, 80, 78, 71, 13, 10, 26, 10};
    uint8_t ihdr[13] = {uint8_t(w >> 24), uint8_t(w >> 16), uint8_t(w >> 8), uint8_t(w),
                        uint8_t(h >> 24), uint8_t(h >> 16), uint8_t(h >> 8), uint8_t(h), 8, 2, 0, 0, 0};
    chunk(png, "IHDR", ihdr, 13);
    chunk(png, "IDAT", z.data(), z.size());
    chunk(png, "IEND", nullptr, 0);
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
}

static int rgb565(int r, int g, int b) { return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3); }
static void from565(int c, int& r, int& g, int& b) {
    int r5 = (c >> 11) & 31, g6 = (c >> 5) & 63, b5 = c & 31;
    r = (r5 << 3) | (r5 >> 2);
    g = (g6 << 2) | (g6 >> 4);
    b = (b5 << 3) | (b5 >> 2);
}

static void pack_bc1_block(const uint8_t px[16][4], uint8_t out[8]) {
    int best_i = 0, worst_i = 0, best_l = -1, worst_l = 1 << 30;
    for (int i = 0; i < 16; ++i) {
        int l = px[i][0] * 2 + px[i][1] * 4 + px[i][2];
        if (l > best_l) {
            best_l = l;
            best_i = i;
        }
        if (l < worst_l) {
            worst_l = l;
            worst_i = i;
        }
    }
    int e0 = rgb565(px[best_i][0], px[best_i][1], px[best_i][2]);
    int e1 = rgb565(px[worst_i][0], px[worst_i][1], px[worst_i][2]);
    if (e0 < e1) std::swap(e0, e1);
    int p0[3], p1[3];
    from565(e0, p0[0], p0[1], p0[2]);
    from565(e1, p1[0], p1[1], p1[2]);
    int pal[4][3];
    for (int c = 0; c < 3; ++c) {
        pal[0][c] = p0[c];
        pal[1][c] = p1[c];
        pal[2][c] = (2 * p0[c] + p1[c]) / 3;
        pal[3][c] = (p0[c] + 2 * p1[c]) / 3;
    }
    uint32_t idx = 0;
    if (e0 != e1) {
        for (int i = 0; i < 16; ++i) {
            int best = 0, bestd = 1 << 30;
            for (int k = 0; k < 4; ++k) {
                int d = 0;
                for (int c = 0; c < 3; ++c) {
                    int diff = int(px[i][c]) - pal[k][c];
                    d += diff * diff;
                }
                if (d < bestd) {
                    bestd = d;
                    best = k;
                }
            }
            idx |= uint32_t(best) << (2 * i);
        }
    }
    out[0] = uint8_t(e0);
    out[1] = uint8_t(e0 >> 8);
    out[2] = uint8_t(e1);
    out[3] = uint8_t(e1 >> 8);
    std::memcpy(out + 4, &idx, 4);
}

static void pack_bc4_block(const uint8_t v[16], uint8_t out[8]) {
    int a = v[0], b = v[0];
    for (int i = 1; i < 16; ++i) {
        a = std::max(a, int(v[i]));
        b = std::min(b, int(v[i]));
    }
    out[0] = uint8_t(a);
    out[1] = uint8_t(b);
    std::memset(out + 2, 0, 6);
    if (a == b) return;
    int pal[8];
    pal[0] = a;
    pal[1] = b;
    for (int i = 0; i < 6; ++i) pal[2 + i] = ((7 - i) * a + (i + 1) * b) / 8;
    uint64_t bits = 0;
    for (int i = 0; i < 16; ++i) {
        int best = 0, bestd = 1 << 30;
        for (int k = 0; k < 8; ++k) {
            int d = std::abs(pal[k] - int(v[i]));
            if (d < bestd) {
                bestd = d;
                best = k;
            }
        }
        bits |= uint64_t(best) << (3 * i);
    }
    for (int i = 0; i < 6; ++i) out[2 + i] = uint8_t((bits >> (8 * i)) & 0xFF);
}

static void put_bits(uint64_t words[2], int& pos, uint32_t value, int nbits) {
    value &= (1u << nbits) - 1u;
    for (int i = 0; i < nbits; ++i) {
        if ((value >> i) & 1u) words[(pos + i) / 64] |= 1ull << ((pos + i) % 64);
    }
    pos += nbits;
}

static void pack_bc7_mode6(const uint8_t px[16][4], uint8_t out[16]) {
    int lo[4], hi[4];
    for (int c = 0; c < 4; ++c) {
        lo[c] = 255;
        hi[c] = 0;
        for (int i = 0; i < 16; ++i) {
            lo[c] = std::min(lo[c], int(px[i][c]));
            hi[c] = std::max(hi[c], int(px[i][c]));
        }
    }
    int p0 = lo[0] & 1, p1 = hi[0] & 1;
    int e0[4], e1[4];
    for (int c = 0; c < 4; ++c) {
        e0[c] = lo[c] >> 1;
        e1[c] = hi[c] >> 1;
    }
    auto endpoint = [&](int which, int c) {
        return ((which ? e1[c] : e0[c]) << 1) | (which ? p1 : p0);
    };
    int idx[16];
    for (int i = 0; i < 16; ++i) {
        int best = 0, bestd = 1 << 30;
        for (int k = 0; k < 16; ++k) {
            int d = 0;
            int iw = 64 - kW4[k];
            for (int c = 0; c < 4; ++c) {
                int r0 = endpoint(0, c);
                int r1 = endpoint(1, c);
                int pal = (r0 * iw + r1 * kW4[k] + 32) >> 6;
                int diff = int(px[i][c]) - pal;
                d += diff * diff;
            }
            if (d < bestd) {
                bestd = d;
                best = k;
            }
        }
        idx[i] = best;
    }
    if (idx[0] >= 8) {
        for (int i = 0; i < 16; ++i) idx[i] = 15 - idx[i];
        std::swap(e0[0], e1[0]);
        std::swap(e0[1], e1[1]);
        std::swap(e0[2], e1[2]);
        std::swap(e0[3], e1[3]);
        std::swap(p0, p1);
    }
    uint64_t words[2] = {};
    int pos = 0;
    put_bits(words, pos, 1u << 6, 7);
    for (int c = 0; c < 4; ++c) {
        put_bits(words, pos, uint32_t(e0[c]), 7);
        put_bits(words, pos, uint32_t(e1[c]), 7);
    }
    put_bits(words, pos, uint32_t(p0), 1);
    put_bits(words, pos, uint32_t(p1), 1);
    put_bits(words, pos, uint32_t(idx[0]), 3);
    for (int i = 1; i < 16; ++i) put_bits(words, pos, uint32_t(idx[i]), 4);
    std::memcpy(out, words, 16);
}

struct Mesh {
    std::vector<float> v;
    std::vector<uint16_t> idx;
    void vert(float x, float y, float z, float nx, float ny, float nz, float u, float vv) {
        v.insert(v.end(), {x, y, z, nx, ny, nz, u, vv});
    }
};

static Mesh make_sphere() {
    Mesh m;
    const int stacks = 24, slices = 32;
    for (int iy = 0; iy <= stacks; ++iy) {
        float v = float(iy) / stacks;
        float phi = v * 3.14159265f;
        for (int ix = 0; ix <= slices; ++ix) {
            float u = float(ix) / slices;
            float th = u * 6.2831853f;
            float x = std::sin(phi) * std::cos(th);
            float y = std::cos(phi);
            float z = std::sin(phi) * std::sin(th);
            m.vert(x, y, z, x, y, z, u, v);
        }
    }
    int stride = slices + 1;
    for (int iy = 0; iy < stacks; ++iy) {
        for (int ix = 0; ix < slices; ++ix) {
            uint16_t a = uint16_t(iy * stride + ix);
            uint16_t b = uint16_t(a + stride);
            m.idx.insert(m.idx.end(), {a, b, uint16_t(a + 1), uint16_t(a + 1), b, uint16_t(b + 1)});
        }
    }
    return m;
}

static Mesh make_plane() {
    Mesh m;
    m.vert(-1, 0, -1, 0, 1, 0, 0, 0);
    m.vert(1, 0, -1, 0, 1, 0, 2, 0);
    m.vert(1, 0, 1, 0, 1, 0, 2, 2);
    m.vert(-1, 0, 1, 0, 1, 0, 0, 2);
    m.idx = {0, 1, 2, 0, 2, 3};
    return m;
}

static Mesh make_cube() {
    Mesh m;
    const float p[6][4][3] = {
        {{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}},   {{1, -1, -1}, {-1, -1, -1}, {-1, 1, -1}, {1, 1, -1}},
        {{-1, 1, 1}, {1, 1, 1}, {1, 1, -1}, {-1, 1, -1}},    {{-1, -1, -1}, {1, -1, -1}, {1, -1, 1}, {-1, -1, 1}},
        {{1, -1, 1}, {1, -1, -1}, {1, 1, -1}, {1, 1, 1}},    {{-1, -1, -1}, {-1, -1, 1}, {-1, 1, 1}, {-1, 1, -1}},
    };
    const float n[6][3] = {{0, 0, 1}, {0, 0, -1}, {0, 1, 0}, {0, -1, 0}, {1, 0, 0}, {-1, 0, 0}};
    const float uv[4][2] = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    for (int f = 0; f < 6; ++f) {
        uint16_t base = uint16_t(m.v.size() / 8);
        for (int i = 0; i < 4; ++i) m.vert(p[f][i][0], p[f][i][1], p[f][i][2], n[f][0], n[f][1], n[f][2], uv[i][0], uv[i][1]);
        m.idx.insert(m.idx.end(), {base, uint16_t(base + 1), uint16_t(base + 2), base, uint16_t(base + 2), uint16_t(base + 3)});
    }
    return m;
}

static void mul(float* o, const float* a, const float* b) {
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            o[c * 4 + r] = a[0 * 4 + r] * b[c * 4 + 0] + a[1 * 4 + r] * b[c * 4 + 1] + a[2 * 4 + r] * b[c * 4 + 2] +
                           a[3 * 4 + r] * b[c * 4 + 3];
        }
    }
}

static void perspective(float* m, float fovy, float aspect, float zn, float zf) {
    std::memset(m, 0, 64);
    float f = 1.f / std::tan(fovy * 0.5f);
    m[0] = f / aspect;
    m[5] = -f;
    m[10] = zf / (zn - zf);
    m[11] = -1.f;
    m[14] = (zf * zn) / (zn - zf);
}

static void look_at(float* m, const float* eye, const float* center) {
    float f[3] = {center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]};
    float fl = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    f[0] /= fl;
    f[1] /= fl;
    f[2] /= fl;
    float up[3] = {0, 1, 0};
    float s[3] = {f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]};
    float sl = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    s[0] /= sl;
    s[1] /= sl;
    s[2] /= sl;
    float u[3] = {s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0]};
    std::memset(m, 0, 64);
    m[0] = s[0];
    m[4] = s[1];
    m[8] = s[2];
    m[1] = u[0];
    m[5] = u[1];
    m[9] = u[2];
    m[2] = -f[0];
    m[6] = -f[1];
    m[10] = -f[2];
    m[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
    m[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
    m[14] = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];
    m[15] = 1.f;
}

static void translate_scale(float* m, float x, float y, float z, float s) {
    std::memset(m, 0, 64);
    m[0] = s;
    m[5] = s;
    m[10] = s;
    m[12] = x;
    m[13] = y;
    m[14] = z;
    m[15] = 1.f;
}

// 8x8 glyphs live in font_rows.
static const uint8_t* font_rows(char ch) {
    struct Table {
        uint8_t g[128][8]{};
        Table() {
            auto set = [&](char c, std::initializer_list<int> rows) {
                int i = 0;
                for (int r : rows) g[int(c)][i++] = uint8_t(r);
            };
            set('0', {0x3C, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x3C, 0});
            set('1', {0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0});
            set('2', {0x3C, 0x66, 0x06, 0x0C, 0x18, 0x30, 0x7E, 0});
            set('3', {0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0});
            set('4', {0x0C, 0x1C, 0x3C, 0x6C, 0x7E, 0x0C, 0x0C, 0});
            set('5', {0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0});
            set('6', {0x1C, 0x30, 0x60, 0x7C, 0x66, 0x66, 0x3C, 0});
            set('7', {0x7E, 0x06, 0x0C, 0x18, 0x30, 0x30, 0x30, 0});
            set('8', {0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0});
            set('9', {0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0});
            set('A', {0x18, 0x3C, 0x66, 0x66, 0x7E, 0x66, 0x66, 0});
            set('B', {0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0});
            set('C', {0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0});
            set('D', {0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0});
            set('E', {0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0});
            set('F', {0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0});
            set('G', {0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3C, 0});
            set('H', {0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0});
            set('I', {0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0});
            set('K', {0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0});
            set('L', {0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0});
            set('M', {0xC3, 0xE7, 0xFF, 0xDB, 0xC3, 0xC3, 0xC3, 0});
            set('N', {0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0});
            set('O', {0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0});
            set('P', {0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0});
            set('R', {0x7C, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0});
            set('S', {0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0});
            set('T', {0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0});
            set('U', {0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0});
            set('V', {0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0});
            set('W', {0xC3, 0xC3, 0xC3, 0xDB, 0xFF, 0xE7, 0xC3, 0});
            set('X', {0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0});
            set('.', {0, 0, 0, 0, 0, 0x18, 0x18, 0});
            set(':', {0, 0x18, 0x18, 0, 0, 0x18, 0x18, 0});
            set('-', {0, 0, 0, 0x7E, 0, 0, 0, 0});
            set('/', {0x02, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0});
            set('%', {0x62, 0x66, 0x0C, 0x18, 0x30, 0x66, 0x46, 0});
            set(' ', {0, 0, 0, 0, 0, 0, 0, 0});
        }
    };
    static Table table;
    if (static_cast<unsigned char>(ch) > 127) return table.g[0];
    return table.g[int(static_cast<unsigned char>(ch))];
}

static void draw_text(uint8_t* rgba, int w, int h, int x, int y, const std::string& s) {
    int cx = x;
    for (char ch : s) {
        char up = ch;
        if (up >= 'a' && up <= 'z') up = char(up - 'a' + 'A');
        const uint8_t* rows = font_rows(up);
        for (int row = 0; row < 8; ++row) {
            int py = y + row;
            if (py < 0 || py >= h) continue;
            for (int col = 0; col < 8; ++col) {
                if ((rows[row] & (0x80 >> col)) == 0) continue;
                int px = cx + col;
                if (px < 0 || px >= w) continue;
                uint8_t* p = rgba + (static_cast<size_t>(py) * w + px) * 4;
                p[0] = 240;
                p[1] = 240;
                p[2] = 240;
                p[3] = 255;
            }
        }
        cx += 8;
    }
}

static std::string slug_of(std::string name) {
    std::string o;
    for (char ch : name) {
        char c = ch;
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) o.push_back(c);
        else if (!o.empty() && o.back() != '-') o.push_back('-');
    }
    while (!o.empty() && o.back() == '-') o.pop_back();
    if (o.size() > 80) o.resize(80);
    return o.empty() ? "gpu" : o;
}

static double median_of(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static double p95_of(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t i = std::min(v.size() - 1, static_cast<size_t>(std::ceil(0.95 * v.size())) - 1);
    return v[i];
}

int main(int argc, char** argv) {
    std::string shader_dir = "build/shaders";
    std::string device_spec;
    std::vector<std::string> ntcs;
    std::vector<std::string> dirs = {"results/train_sweep", "results/train_max"};
    bool benchmark = false;
    bool force_res = false;
    int frames = 32;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value");
            return argv[++i];
        };
        try {
            if (a == "--benchmark") benchmark = true;
            else if (a == "--force-res") force_res = true;
            else if (a == "--shader-dir") shader_dir = need();
            else if (a == "--device") device_spec = need();
            else if (a == "--ntc") ntcs.push_back(need());
            else if (a == "--ntc-dir") dirs.push_back(need());
            else if (a == "--frames") frames = std::stoi(need());
            else {
                std::cerr << "unknown " << a << "\n";
                return 2;
            }
        } catch (const std::exception& ex) {
            std::cerr << ex.what() << "\n";
            return 2;
        }
    }
    if (ntcs.empty()) {
        for (const auto& d : dirs) {
            std::error_code ec;
            if (!std::filesystem::exists(d, ec)) continue;
            for (const auto& ent : std::filesystem::recursive_directory_iterator(d, ec)) {
                if (ent.path().extension() == ".ntc") ntcs.push_back(ent.path().string());
            }
        }
    }
    std::sort(ntcs.begin(), ntcs.end());
    ntcs.erase(std::unique(ntcs.begin(), ntcs.end()), ntcs.end());
    if (ntcs.empty()) {
        std::cout << "ntc_view: no .ntc files in results/train_sweep or results/train_max\n";
        return benchmark ? 0 : 2;
    }
    try {
        VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        ai.pApplicationName = "ntc_view";
        ai.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &ai;
        VkInstance inst = VK_NULL_HANDLE;
        CHECK(vkCreateInstance(&ici, nullptr, &inst));
        uint32_t ndev = 0;
        CHECK(vkEnumeratePhysicalDevices(inst, &ndev, nullptr));
        std::vector<VkPhysicalDevice> phys_list(ndev);
        CHECK(vkEnumeratePhysicalDevices(inst, &ndev, phys_list.data()));
        auto score = [](VkPhysicalDevice p) {
            VkPhysicalDeviceProperties pr{};
            vkGetPhysicalDeviceProperties(p, &pr);
            if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return 3;
            if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) return 2;
            if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) return 1;
            return 0;
        };
        VkPhysicalDevice phys = phys_list[0];
        if (!device_spec.empty()) {
            bool found = false;
            for (uint32_t i = 0; i < ndev; ++i) {
                VkPhysicalDeviceProperties pr{};
                vkGetPhysicalDeviceProperties(phys_list[i], &pr);
                std::string nm = pr.deviceName;
                if (device_spec == std::to_string(i) ||
                    nm.find(device_spec) != std::string::npos) {
                    phys = phys_list[i];
                    found = true;
                    break;
                }
            }
            if (!found) throw std::runtime_error("device not found");
        } else {
            for (auto p : phys_list)
                if (score(p) > score(phys)) phys = p;
        }
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys, &props);
        const bool cpu_dev = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
        std::cout << "device_name " << props.deviceName << "\n";
        std::cout << "vendor_id " << std::hex << props.vendorID << std::dec << "\n";
        std::cout << "device_id " << std::hex << props.deviceID << std::dec << "\n";
        std::cout << "driver_raw " << props.driverVersion << "\n";
        std::cout << "api_version " << VK_VERSION_MAJOR(props.apiVersion) << "." << VK_VERSION_MINOR(props.apiVersion)
                  << "." << VK_VERSION_PATCH(props.apiVersion) << "\n";
        VkPhysicalDeviceSubgroupProperties sub{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sub;
        vkGetPhysicalDeviceProperties2(phys, &p2);
        std::cout << "subgroup_size " << sub.subgroupSize << "\n";
        uint32_t nfam = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nfam, nullptr);
        std::vector<VkQueueFamilyProperties> fams(nfam);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nfam, fams.data());
        uint32_t qfam = 0;
        for (uint32_t i = 0; i < nfam; ++i) {
            if (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                qfam = i;
                break;
            }
        }
        float prio = 1.f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = qfam;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkPhysicalDeviceDynamicRenderingFeatures dyn{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
        dyn.dynamicRendering = VK_TRUE;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &dyn;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        VkDevice dev = VK_NULL_HANDLE;
        CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(dev, qfam, 0, &queue);

        auto fmt_ok = [&](VkFormat fmt) {
            VkFormatProperties fp{};
            vkGetPhysicalDeviceFormatProperties(phys, fmt, &fp);
            const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                               VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                               VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
            return (fp.optimalTilingFeatures & need) == need;
        };
        const bool bc7_ok = fmt_ok(VK_FORMAT_BC7_UNORM_BLOCK);
        const bool bc5_ok = fmt_ok(VK_FORMAT_BC5_UNORM_BLOCK);
        const bool bc1_ok = fmt_ok(VK_FORMAT_BC1_RGBA_UNORM_BLOCK);
        const bool astc_ok = fmt_ok(VK_FORMAT_ASTC_4x4_UNORM_BLOCK);
        std::cout << "bc7 " << bc7_ok << " bc5 " << bc5_ok << " bc1 " << bc1_ok << " astc " << astc_ok << "\n";
        if (!astc_ok) std::cout << "astc skipped format\n";
        else std::cout << "astc skipped no_encoder\n";

        // One file is enough to prove the path here. Benchmark walks all of them.
        const std::string gpu_slug = slug_of(props.deviceName);
        const std::string hw_dir = "results/hw/" + gpu_slug + "/realtime";
        const std::string shot_dir = "screenshots/" + gpu_slug + "/realtime";
        std::filesystem::create_directories(hw_dir);
        std::filesystem::create_directories(shot_dir);

        std::ostringstream json;
        json << "{\n  \"device_name\": \"" << props.deviceName << "\",\n";
        json << "  \"vendor_id\": " << props.vendorID << ",\n";
        json << "  \"device_id\": " << props.deviceID << ",\n";
        json << "  \"driver_raw\": " << props.driverVersion << ",\n";
        json << "  \"subgroup_size\": " << sub.subgroupSize << ",\n";
        json << "  \"cpu_device\": " << (cpu_dev ? "true" : "false") << ",\n";
        json << "  \"bc7\": " << (bc7_ok ? "true" : "false") << ",\n";
        json << "  \"bc1\": " << (bc1_ok ? "true" : "false") << ",\n";
        json << "  \"bc5\": " << (bc5_ok ? "true" : "false") << ",\n";
        json << "  \"astc\": false,\n";
        json << "  \"astc_reason\": \"" << (astc_ok ? "no encoder in this viewer" : "format unsupported") << "\",\n";
        json << "  \"note\": \"BC views are the CPU decode packed at load (mode 6, BC1, BC4). "
                "Inference-on-sample is the fragment ntc_eval_filtered path.\",\n";
        json << "  \"files\": [\n";

        auto load_spirv = [&](const std::string& name) {
            auto code = read_bin(shader_dir + "/" + name);
            VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            ci.codeSize = code.size();
            ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
            VkShaderModule m = VK_NULL_HANDLE;
            CHECK(vkCreateShaderModule(dev, &ci, nullptr, &m));
            return m;
        };
        VkShaderModule vs = load_spirv("ntc_pbr.vert.spv");
        VkShaderModule fs = load_spirv("ntc_pbr.frag.spv");

        VkDescriptorSetLayoutBinding binds[11]{};
        binds[0].binding = 0;
        binds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[0].descriptorCount = 1;
        binds[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        binds[1] = binds[0];
        binds[1].binding = 1;
        for (uint32_t i = 2; i <= 7; ++i) {
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        binds[8].binding = 8;
        binds[8].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        binds[8].descriptorCount = 1;
        binds[8].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        binds[9].binding = 9;
        binds[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binds[9].descriptorCount = 1;
        binds[9].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        binds[10] = binds[9];
        binds[10].binding = 10;
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = 11;
        lci.pBindings = binds;
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        CHECK(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &set_layout));
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pcr.size = 16;
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &set_layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
        CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pipe_layout));

        VkVertexInputBindingDescription vb{};
        vb.binding = 0;
        vb.stride = 32;
        vb.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        VkVertexInputAttributeDescription at[3]{};
        at[0].location = 0;
        at[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        at[1].location = 1;
        at[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        at[1].offset = 12;
        at[2].location = 2;
        at[2].format = VK_FORMAT_R32G32_SFLOAT;
        at[2].offset = 24;
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &vb;
        vi.vertexAttributeDescriptionCount = 3;
        vi.pVertexAttributeDescriptions = at;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;
        VkDynamicState dyn_states[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dys{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dys.dynamicStateCount = 2;
        dys.pDynamicStates = dyn_states;
        VkFormat color_fmt = VK_FORMAT_B8G8R8A8_UNORM;
        VkPipelineRenderingCreateInfo rend{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rend.colorAttachmentCount = 1;
        rend.pColorAttachmentFormats = &color_fmt;
        rend.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;
        stages[1].pName = "main";
        VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gp.pNext = &rend;
        gp.stageCount = 2;
        gp.pStages = stages;
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vp;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &ds;
        gp.pColorBlendState = &cb;
        gp.pDynamicState = &dys;
        gp.layout = pipe_layout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipeline));

        Mesh sphere = make_sphere();
        Mesh plane = make_plane();
        Mesh cube = make_cube();
        std::vector<float> verts = sphere.v;
        std::vector<uint16_t> indices = sphere.idx;
        uint32_t plane_first = uint32_t(indices.size());
        uint32_t plane_base = uint32_t(verts.size() / 8);
        for (auto i : plane.idx) indices.push_back(uint16_t(i + plane_base));
        verts.insert(verts.end(), plane.v.begin(), plane.v.end());
        uint32_t plane_count = uint32_t(plane.idx.size());
        uint32_t cube_first = uint32_t(indices.size());
        uint32_t cube_base = uint32_t(verts.size() / 8);
        for (auto i : cube.idx) indices.push_back(uint16_t(i + cube_base));
        verts.insert(verts.end(), cube.v.begin(), cube.v.end());
        uint32_t cube_count = uint32_t(cube.idx.size());
        uint32_t sphere_count = uint32_t(sphere.idx.size());
        Buf vbuf = make_buf(dev, phys, verts.size() * 4,
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, verts.data());
        Buf ibuf = make_buf(dev, phys, indices.size() * 2, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, indices.data());
        const VkDeviceSize align = 256;
        Buf ubo = make_buf(dev, phys, align * 3, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, nullptr);

        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.queueFamilyIndex = qfam;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VkCommandPool pool = VK_NULL_HANDLE;
        CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &pool));
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        VkQueryPool qpool = VK_NULL_HANDLE;
        CHECK(vkCreateQueryPool(dev, &qpci, nullptr, &qpool));

        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        VkSampler samp = VK_NULL_HANDLE;
        CHECK(vkCreateSampler(dev, &sci, nullptr, &samp));

        auto make_image = [&](uint32_t w, uint32_t h, uint32_t layers, VkFormat fmt, VkImageUsageFlags usage,
                              VkImage& image, VkDeviceMemory& mem, VkImageView& view, uint64_t& bytes) {
            VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ii.imageType = VK_IMAGE_TYPE_2D;
            ii.format = fmt;
            ii.extent = {w, h, 1};
            ii.mipLevels = 1;
            ii.arrayLayers = layers;
            ii.samples = VK_SAMPLE_COUNT_1_BIT;
            ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.usage = usage;
            ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            CHECK(vkCreateImage(dev, &ii, nullptr, &image));
            VkMemoryRequirements req{};
            vkGetImageMemoryRequirements(dev, image, &req);
            bytes = req.size;
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = req.size;
            try {
                mai.memoryTypeIndex = find_mem(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            } catch (const std::runtime_error&) {
                mai.memoryTypeIndex = find_mem(phys, req.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            }
            CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
            CHECK(vkBindImageMemory(dev, image, mem, 0));
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = image;
            vi.viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
            vi.format = fmt;
            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vi.subresourceRange.levelCount = 1;
            vi.subresourceRange.layerCount = layers;
            CHECK(vkCreateImageView(dev, &vi, nullptr, &view));
        };

        auto upload_image = [&](VkCommandBuffer c, VkImage image, uint32_t w, uint32_t h, uint32_t layers,
                                const void* pixels, VkDeviceSize nbytes) {
            Buf staging = make_buf(dev, phys, nbytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, pixels);
            VkImageMemoryBarrier to{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to.image = image;
            to.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to.subresourceRange.levelCount = 1;
            to.subresourceRange.layerCount = layers;
            to.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                                 nullptr, 1, &to);
            VkDeviceSize layer_bytes = nbytes / layers;
            for (uint32_t layer = 0; layer < layers; ++layer) {
                VkBufferImageCopy cp{};
                cp.bufferOffset = layer_bytes * layer;
                cp.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                cp.imageSubresource.baseArrayLayer = layer;
                cp.imageSubresource.layerCount = 1;
                cp.imageExtent = {w, h, 1};
                vkCmdCopyBufferToImage(c, staging.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
            }
            VkImageMemoryBarrier to2 = to;
            to2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            to2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &to2);
            // staging lives until submit; caller must keep it. We wait idle then free by leaking into a list.
            return staging;
        };

        std::vector<Buf> keep_staging;

        struct Target {
            VkImage color = VK_NULL_HANDLE;
            VkDeviceMemory color_mem = VK_NULL_HANDLE;
            VkImageView color_view = VK_NULL_HANDLE;
            VkImage depth = VK_NULL_HANDLE;
            VkDeviceMemory depth_mem = VK_NULL_HANDLE;
            VkImageView depth_view = VK_NULL_HANDLE;
            Buf readback;
            uint32_t w = 0, h = 0;
        };
        auto make_target = [&](uint32_t w, uint32_t h) {
            Target t;
            t.w = w;
            t.h = h;
            uint64_t dummy = 0;
            make_image(w, h, 1, color_fmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, t.color,
                       t.color_mem, t.color_view, dummy);
            VkImageCreateInfo di{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            di.imageType = VK_IMAGE_TYPE_2D;
            di.format = VK_FORMAT_D32_SFLOAT;
            di.extent = {w, h, 1};
            di.mipLevels = 1;
            di.arrayLayers = 1;
            di.samples = VK_SAMPLE_COUNT_1_BIT;
            di.tiling = VK_IMAGE_TILING_OPTIMAL;
            di.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            CHECK(vkCreateImage(dev, &di, nullptr, &t.depth));
            VkMemoryRequirements req{};
            vkGetImageMemoryRequirements(dev, t.depth, &req);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = req.size;
            try {
                mai.memoryTypeIndex = find_mem(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            } catch (const std::runtime_error&) {
                mai.memoryTypeIndex =
                    find_mem(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            }
            CHECK(vkAllocateMemory(dev, &mai, nullptr, &t.depth_mem));
            CHECK(vkBindImageMemory(dev, t.depth, t.depth_mem, 0));
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = t.depth;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = VK_FORMAT_D32_SFLOAT;
            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            vi.subresourceRange.levelCount = 1;
            vi.subresourceRange.layerCount = 1;
            CHECK(vkCreateImageView(dev, &vi, nullptr, &t.depth_view));
            t.readback = make_buf(dev, phys, VkDeviceSize(w) * h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, nullptr);
            return t;
        };

        const char* mode_name[] = {"ntc", "bc7", "bc1", "astc", "ref"};
        bool first_file = true;
        for (size_t fi = 0; fi < ntcs.size(); ++fi) {
            if (fi) json << ",\n";
            ntc::TextureSet ts = ntc::load_file(ntcs[fi]);
            auto up = ntc::make_gpu_upload(ts, 0, ntc::kFilterFeatureTrilinear, 8);
            auto decoded = ntc::decode_mip(ts, 0);
            const int W = int(ts.width), H = int(ts.height), C = ts.num_channels();
            const float bpp_total = ntc::total_bpp(ts);
            const float bpp_latent = ntc::latent_bpp_mip0(ts);
            const float bpp_file = ntc::file_bpp_chain(ts);
            const float bpp_pc = ntc::bppc(ts);
            uint32_t sem[9];
            for (int i = 0; i < 9; ++i) sem[i] = 255;
            for (int i = 0; i < C && i < 16; ++i) {
                int s = ts.channels[size_t(i)].semantic;
                if (s >= 0 && s < 9) sem[s] = uint32_t(i);
            }
            std::vector<uint8_t> ref(size_t(W) * H * 4 * 3, 0);
            auto put = [&](int layer, int x, int y, uint8_t r, uint8_t g, uint8_t b) {
                size_t i = (size_t(layer) * H * W + size_t(y) * W + x) * 4;
                ref[i] = b;
                ref[i + 1] = g;
                ref[i + 2] = r;
                ref[i + 3] = 255;
            };
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    auto pix = [&](int sem_i) -> uint8_t {
                        if (sem_i > 8 || sem[sem_i] == 255) return sem_i == 8 ? 255 : (sem_i == 6 ? 128 : 0);
                        float v = decoded[(size_t(y) * W + x) * C + sem[sem_i]];
                        int q = int(std::lround(std::min(1.f, std::max(0.f, v)) * 255.f));
                        return uint8_t(q);
                    };
                    put(0, x, y, pix(0), pix(1), pix(2));
                    put(1, x, y, pix(3), pix(4), pix(5));
                    put(2, x, y, pix(8), pix(6), pix(7));
                }
            }
            auto blocks = [&](auto&& pack_one, int bpp_block, int comps, int layer) {
                int bw = (W + 3) / 4, bh = (H + 3) / 4;
                std::vector<uint8_t> out(size_t(bw) * bh * bpp_block);
                for (int by = 0; by < bh; ++by) {
                    for (int bx = 0; bx < bw; ++bx) {
                        uint8_t px[16][4] = {};
                        for (int iy = 0; iy < 4; ++iy) {
                            for (int ix = 0; ix < 4; ++ix) {
                                int x = std::min(W - 1, bx * 4 + ix);
                                int y = std::min(H - 1, by * 4 + iy);
                                const uint8_t* s = ref.data() + ((size_t(layer) * H * W + size_t(y) * W + x) * 4);
                                px[iy * 4 + ix][0] = s[2];
                                px[iy * 4 + ix][1] = s[1];
                                px[iy * 4 + ix][2] = s[0];
                                px[iy * 4 + ix][3] = 255;
                                (void)comps;
                            }
                        }
                        pack_one(px, out.data() + (size_t(by * bw + bx) * bpp_block));
                    }
                }
                return out;
            };
            // Reference was stored BGRA. packers above swizzle back to RGBA in px[].
            std::vector<uint8_t> bc7, bc5, bc7orm, bc1, bc1orm;
            if (bc7_ok) {
                bc7 = blocks([&](uint8_t px[16][4], uint8_t* dst) { pack_bc7_mode6(px, dst); }, 16, 4, 0);
                bc7orm = blocks([&](uint8_t px[16][4], uint8_t* dst) { pack_bc7_mode6(px, dst); }, 16, 4, 2);
            }
            if (bc1_ok) {
                bc1 = blocks([&](uint8_t px[16][4], uint8_t* dst) { pack_bc1_block(px, dst); }, 8, 3, 0);
                bc1orm = blocks([&](uint8_t px[16][4], uint8_t* dst) { pack_bc1_block(px, dst); }, 8, 3, 2);
            }
            if (bc5_ok) {
                int bw = (W + 3) / 4, bh = (H + 3) / 4;
                bc5.resize(size_t(bw) * bh * 16);
                for (int by = 0; by < bh; ++by) {
                    for (int bx = 0; bx < bw; ++bx) {
                        uint8_t x16[16], y16[16];
                        for (int iy = 0; iy < 4; ++iy) {
                            for (int ix = 0; ix < 4; ++ix) {
                                int x = std::min(W - 1, bx * 4 + ix);
                                int y = std::min(H - 1, by * 4 + iy);
                                const uint8_t* s = ref.data() + ((size_t(1) * H * W + size_t(y) * W + x) * 4);
                                x16[iy * 4 + ix] = s[2];
                                y16[iy * 4 + ix] = s[1];
                            }
                        }
                        uint8_t* dst = bc5.data() + (size_t(by * bw + bx) * 16);
                        pack_bc4_block(x16, dst);
                        pack_bc4_block(y16, dst + 8);
                    }
                }
            }
            uint8_t white[4] = {255, 255, 255, 255};
            VkImage ref_i{}, bc7_i{}, bc5_i{}, bc7o_i{}, bc1_i{}, bc1o_i{}, astc_i{}, ov_i{};
            VkDeviceMemory ref_m{}, bc7_m{}, bc5_m{}, bc7o_m{}, bc1_m{}, bc1o_m{}, astc_m{}, ov_m{};
            VkImageView ref_v{}, bc7_v{}, bc5_v{}, bc7o_v{}, bc1_v{}, bc1o_v{}, astc_v{}, ov_v{};
            uint64_t b_ref = 0, b_bc7 = 0, b_bc5 = 0, b_bc7o = 0, b_bc1 = 0, b_bc1o = 0, b_astc = 0, b_ov = 0;
            make_image(W, H, 3, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, ref_i,
                       ref_m, ref_v, b_ref);
            auto solid = [&](VkImage& im, VkDeviceMemory& mem, VkImageView& view, uint64_t& bytes) {
                make_image(1, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, im,
                           mem, view, bytes);
            };
            if (bc7_ok) {
                make_image((W + 3) & ~3, (H + 3) & ~3, 1, VK_FORMAT_BC7_UNORM_BLOCK,
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, bc7_i, bc7_m, bc7_v, b_bc7);
                make_image((W + 3) & ~3, (H + 3) & ~3, 1, VK_FORMAT_BC7_UNORM_BLOCK,
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, bc7o_i, bc7o_m, bc7o_v, b_bc7o);
            } else solid(bc7_i, bc7_m, bc7_v, b_bc7), solid(bc7o_i, bc7o_m, bc7o_v, b_bc7o);
            if (bc5_ok)
                make_image((W + 3) & ~3, (H + 3) & ~3, 1, VK_FORMAT_BC5_UNORM_BLOCK,
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, bc5_i, bc5_m, bc5_v, b_bc5);
            else solid(bc5_i, bc5_m, bc5_v, b_bc5);
            if (bc1_ok) {
                make_image((W + 3) & ~3, (H + 3) & ~3, 1, VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, bc1_i, bc1_m, bc1_v, b_bc1);
                make_image((W + 3) & ~3, (H + 3) & ~3, 1, VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, bc1o_i, bc1o_m, bc1o_v, b_bc1o);
            } else solid(bc1_i, bc1_m, bc1_v, b_bc1), solid(bc1o_i, bc1o_m, bc1o_v, b_bc1o);
            solid(astc_i, astc_m, astc_v, b_astc);
            make_image(64, 16, 1, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, ov_i,
                       ov_m, ov_v, b_ov);

            Buf desc = make_buf(dev, phys, up.desc.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, up.desc.data());
            Buf blob = make_buf(dev, phys, up.blob.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, up.blob.data());
            uint64_t vram = desc.alloc + blob.alloc + b_ref + b_bc7 + b_bc5 + b_bc7o + b_bc1 + b_bc1o;

            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            CHECK(vkBeginCommandBuffer(cmd, &begin));
            keep_staging.push_back(upload_image(cmd, ref_i, W, H, 3, ref.data(), ref.size()));
            if (bc7_ok) {
                keep_staging.push_back(upload_image(cmd, bc7_i, (W + 3) & ~3, (H + 3) & ~3, 1, bc7.data(), bc7.size()));
                keep_staging.push_back(upload_image(cmd, bc7o_i, (W + 3) & ~3, (H + 3) & ~3, 1, bc7orm.data(), bc7orm.size()));
            } else {
                keep_staging.push_back(upload_image(cmd, bc7_i, 1, 1, 1, white, 4));
                keep_staging.push_back(upload_image(cmd, bc7o_i, 1, 1, 1, white, 4));
            }
            if (bc5_ok) keep_staging.push_back(upload_image(cmd, bc5_i, (W + 3) & ~3, (H + 3) & ~3, 1, bc5.data(), bc5.size()));
            else keep_staging.push_back(upload_image(cmd, bc5_i, 1, 1, 1, white, 4));
            if (bc1_ok) {
                keep_staging.push_back(upload_image(cmd, bc1_i, (W + 3) & ~3, (H + 3) & ~3, 1, bc1.data(), bc1.size()));
                keep_staging.push_back(upload_image(cmd, bc1o_i, (W + 3) & ~3, (H + 3) & ~3, 1, bc1orm.data(), bc1orm.size()));
            } else {
                keep_staging.push_back(upload_image(cmd, bc1_i, 1, 1, 1, white, 4));
                keep_staging.push_back(upload_image(cmd, bc1o_i, 1, 1, 1, white, 4));
            }
            keep_staging.push_back(upload_image(cmd, astc_i, 1, 1, 1, white, 4));
            std::vector<uint8_t> blank(64 * 16 * 4, 0);
            keep_staging.push_back(upload_image(cmd, ov_i, 64, 16, 1, blank.data(), blank.size()));
            CHECK(vkEndCommandBuffer(cmd));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
            CHECK(vkQueueWaitIdle(queue));
            for (auto& s : keep_staging) destroy_buf(dev, s);
            keep_staging.clear();

            VkDescriptorPoolSize ps[3]{};
            ps[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ps[0].descriptorCount = 2;
            ps[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ps[1].descriptorCount = 8;
            ps[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            ps[2].descriptorCount = 1;
            VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            dpci.maxSets = 1;
            dpci.poolSizeCount = 3;
            dpci.pPoolSizes = ps;
            VkDescriptorPool dpool = VK_NULL_HANDLE;
            CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
            VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dsai.descriptorPool = dpool;
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts = &set_layout;
            VkDescriptorSet set = VK_NULL_HANDLE;
            CHECK(vkAllocateDescriptorSets(dev, &dsai, &set));
            VkDescriptorBufferInfo dbi{desc.buffer, 0, desc.size};
            VkDescriptorBufferInfo bbi{blob.buffer, 0, blob.size};
            VkDescriptorBufferInfo ubi{ubo.buffer, 0, sizeof(FrameUBO)};
            VkDescriptorImageInfo infos[8]{};
            VkImageView views[8] = {ref_v, bc7_v, bc5_v, bc7o_v, bc1_v, bc1o_v, astc_v, ov_v};
            for (int i = 0; i < 8; ++i) {
                infos[i].sampler = samp;
                infos[i].imageView = views[i];
                infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            VkWriteDescriptorSet wr[11]{};
            auto write_buf = [&](int i, VkDescriptorBufferInfo* info, VkDescriptorType ty) {
                wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wr[i].dstSet = set;
                wr[i].dstBinding = uint32_t(i == 2 ? 8 : i);
                wr[i].descriptorCount = 1;
                wr[i].descriptorType = ty;
                wr[i].pBufferInfo = info;
            };
            wr[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[0].dstSet = set;
            wr[0].dstBinding = 0;
            wr[0].descriptorCount = 1;
            wr[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[0].pBufferInfo = &dbi;
            wr[1] = wr[0];
            wr[1].dstBinding = 1;
            wr[1].pBufferInfo = &bbi;
            for (int i = 0; i < 6; ++i) {
                wr[2 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wr[2 + i].dstSet = set;
                wr[2 + i].dstBinding = uint32_t(2 + i);
                wr[2 + i].descriptorCount = 1;
                wr[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                wr[2 + i].pImageInfo = &infos[i];
            }
            wr[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[8].dstSet = set;
            wr[8].dstBinding = 9;
            wr[8].descriptorCount = 1;
            wr[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            wr[8].pImageInfo = &infos[6];
            wr[9] = wr[8];
            wr[9].dstBinding = 10;
            wr[9].pImageInfo = &infos[7];
            wr[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[10].dstSet = set;
            wr[10].dstBinding = 8;
            wr[10].descriptorCount = 1;
            wr[10].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            wr[10].pBufferInfo = &ubi;
            vkUpdateDescriptorSets(dev, 11, wr, 0, nullptr);
            (void)write_buf;

            auto render_frame = [&](Target& t, int mode, int wipe, bool heatmap, float yaw, float pitch, float dist,
                                    bool shot, bool read, const std::string& label, double& ms_out) {
                float eye[3] = {std::cos(pitch) * std::sin(yaw) * dist, std::sin(pitch) * dist,
                                std::cos(pitch) * std::cos(yaw) * dist};
                float center[3] = {0, 0.2f, 0};
                float view[16], proj[16], vp[16];
                look_at(view, eye, center);
                perspective(proj, 0.9f, float(t.w) / float(t.h), 0.05f, 50.f);
                mul(vp, proj, view);
                float models[3][16];
                translate_scale(models[0], -1.15f, 0.55f, 0, 0.55f);
                translate_scale(models[1], 0, -0.55f, 0, 1.6f);
                translate_scale(models[2], 1.15f, 0.45f, 0, 0.45f);
                for (int m = 0; m < 3; ++m) {
                    FrameUBO u{};
                    float mvp[16];
                    mul(mvp, vp, models[m]);
                    std::memcpy(u.mvp, mvp, 64);
                    std::memcpy(u.model, models[m], 64);
                    u.eye[0] = eye[0];
                    u.eye[1] = eye[1];
                    u.eye[2] = eye[2];
                    u.light[0] = 0.35f;
                    u.light[1] = 0.55f;
                    u.light[2] = 0.76f;
                    u.sw[0] = uint32_t(mode);
                    u.sw[1] = uint32_t(wipe);
                    u.sw[2] = 4;
                    u.sw[3] = heatmap ? 1u : 0u;
                    u.sem0[0] = sem[0];
                    u.sem0[1] = sem[1];
                    u.sem0[2] = sem[2];
                    u.sem0[3] = sem[3];
                    u.sem1[0] = sem[4];
                    u.sem1[1] = sem[5];
                    u.sem1[2] = sem[6];
                    u.sem1[3] = sem[7];
                    u.sem2[0] = sem[8];
                    u.sem2[1] = t.w;
                    u.sem2[2] = t.h;
                    std::memcpy(static_cast<uint8_t*>(ubo.mapped) + m * align, &u, sizeof(u));
                }
                CHECK(vkResetCommandBuffer(cmd, 0));
                CHECK(vkBeginCommandBuffer(cmd, &begin));
                vkCmdResetQueryPool(cmd, qpool, 0, 2);
                VkImageMemoryBarrier to_color{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                to_color.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                to_color.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                to_color.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                to_color.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                to_color.image = t.color;
                to_color.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                to_color.subresourceRange.levelCount = 1;
                to_color.subresourceRange.layerCount = 1;
                to_color.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                VkImageMemoryBarrier to_depth{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                to_depth.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                to_depth.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                to_depth.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                to_depth.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                to_depth.image = t.depth;
                to_depth.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                to_depth.subresourceRange.levelCount = 1;
                to_depth.subresourceRange.layerCount = 1;
                to_depth.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                VkImageMemoryBarrier bars[2] = {to_color, to_depth};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                                     0, nullptr, 0, nullptr, 2, bars);
                VkRenderingAttachmentInfo color_att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                color_att.imageView = t.color_view;
                color_att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                color_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                color_att.clearValue.color.float32[0] = 0.05f;
                color_att.clearValue.color.float32[1] = 0.05f;
                color_att.clearValue.color.float32[2] = 0.06f;
                color_att.clearValue.color.float32[3] = 1.f;
                VkRenderingAttachmentInfo depth_att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                depth_att.imageView = t.depth_view;
                depth_att.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                depth_att.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                depth_att.clearValue.depthStencil.depth = 1.f;
                VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
                ri.renderArea = {{0, 0}, {t.w, t.h}};
                ri.layerCount = 1;
                ri.colorAttachmentCount = 1;
                ri.pColorAttachments = &color_att;
                ri.pDepthAttachment = &depth_att;
                vkCmdBeginRendering(cmd, &ri);
                VkViewport viewport{};
                viewport.width = float(t.w);
                viewport.height = float(t.h);
                viewport.minDepth = 0.f;
                viewport.maxDepth = 1.f;
                VkRect2D scissor{{0, 0}, {t.w, t.h}};
                vkCmdSetViewport(cmd, 0, 1, &viewport);
                vkCmdSetScissor(cmd, 0, 1, &scissor);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                VkDeviceSize zero = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf.buffer, &zero);
                vkCmdBindIndexBuffer(cmd, ibuf.buffer, 0, VK_INDEX_TYPE_UINT16);
                uint32_t pc[4] = {0, 4, 0, 0};
                vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, pc);
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, qpool, 0);
                uint32_t counts[3] = {sphere_count, plane_count, cube_count};
                uint32_t firsts[3] = {0, plane_first, cube_first};
                for (int m = 0; m < 3; ++m) {
                    uint32_t dyn = uint32_t(m * align);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_layout, 0, 1, &set, 1, &dyn);
                    vkCmdDrawIndexed(cmd, counts[m], 1, firsts[m], 0, 0);
                }
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, qpool, 1);
                vkCmdEndRendering(cmd);
                if (shot || read) {
                    VkImageMemoryBarrier to_src{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                    to_src.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    to_src.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    to_src.image = t.color;
                    to_src.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    to_src.subresourceRange.levelCount = 1;
                    to_src.subresourceRange.layerCount = 1;
                    to_src.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                    to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                         0, nullptr, 0, nullptr, 1, &to_src);
                    VkBufferImageCopy cp{};
                    cp.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    cp.imageSubresource.layerCount = 1;
                    cp.imageExtent = {t.w, t.h, 1};
                    vkCmdCopyImageToBuffer(cmd, t.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.readback.buffer, 1, &cp);
                    VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                    bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    bb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    bb.buffer = t.readback.buffer;
                    bb.size = VK_WHOLE_SIZE;
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &bb,
                                         0, nullptr);
                }
                CHECK(vkEndCommandBuffer(cmd));
                CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
                CHECK(vkQueueWaitIdle(queue));
                uint64_t stamps[2] = {};
                vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(stamps), stamps, 8,
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                ms_out = 0;
                if (stamps[1] > stamps[0] && props.limits.timestampPeriod > 0.f)
                    ms_out = double(stamps[1] - stamps[0]) * double(props.limits.timestampPeriod) / 1e6;
                if (shot || read) {
                    char line[180];
                    std::snprintf(line, sizeof(line), "%s %dx%d %.2fMS FPS %.0f VRAM %llu", label.c_str(), t.w, t.h,
                                  ms_out, ms_out > 0.0 ? 1000.0 / ms_out : 0.0, static_cast<unsigned long long>(vram));
                    draw_text(static_cast<uint8_t*>(t.readback.mapped), int(t.w), int(t.h), 8, 8, line);
                    std::snprintf(line, sizeof(line), "LAT %.2f FILE %.2f BPPC %.3f TOT %.2f", double(bpp_latent),
                                  double(bpp_file), double(bpp_pc), double(bpp_total));
                    draw_text(static_cast<uint8_t*>(t.readback.mapped), int(t.w), int(t.h), 8, 18, line);
                    if (shot) {
                        std::string path = shot_dir + "/" + label + ".png";
                        write_png(path, static_cast<uint8_t*>(t.readback.mapped), int(t.w), int(t.h));
                        std::cout << "wrote " << path << "\n";
                    }
                }
            };

            std::string path_json = ntcs[fi];
            for (char& ch : path_json)
                if (ch == '\\') ch = '/';
            json << "    {\"path\": \"" << path_json << "\", \"total_bpp\": " << bpp_total
                 << ", \"latent_bpp_mip0\": " << bpp_latent << ", \"file_bpp_chain\": " << bpp_file
                 << ", \"bppc\": " << bpp_pc << ", \"vram_bytes\": " << vram
                 << ", \"width\": " << W << ", \"height\": " << H << ", \"channels\": " << C;
            if (!benchmark) {
                json << ", \"interactive\": true}";
                std::cout << ntcs[fi] << " total_bpp " << bpp_total << " latent_bpp_mip0 " << bpp_latent
                          << " file_bpp_chain " << bpp_file << " bppc " << bpp_pc << " vram " << vram << "\n";
                std::cout << "keys: 1 ntc  2 bc7  3 bc1  4 astc  5 ref  h heatmap  j wipe  esc quit\n";
                const int vw = 1280, vh = 720;
                Target live = make_target(vw, vh);
                float yaw = 0.5f, pitch = 0.4f, dist = 3.6f;
                int mode = 0;
                bool heat = false, wipe = true, quit = false;
                int mx = 0, my = 0;
                bool dragging = false;
#if defined(NTC_WIN32)
                WNDCLASSW wc{};
                wc.lpfnWndProc = DefWindowProcW;
                wc.hInstance = GetModuleHandleW(nullptr);
                wc.lpszClassName = L"ntc_view";
                RegisterClassW(&wc);
                HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"ntc_view", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 80, 80,
                                            vw, vh, nullptr, nullptr, wc.hInstance, nullptr);
                if (!hwnd) throw std::runtime_error("CreateWindow");
#elif defined(NTC_X11)
                Display* dpy = XOpenDisplay(nullptr);
                if (!dpy) throw std::runtime_error("XOpenDisplay");
                int screen = DefaultScreen(dpy);
                Window win = XCreateSimpleWindow(dpy, RootWindow(dpy, screen), 80, 80, vw, vh, 0, 0, 0);
                XSelectInput(dpy, win, KeyPressMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask);
                XMapWindow(dpy, win);
                XImage* ximg = XCreateImage(dpy, DefaultVisual(dpy, screen), 24, ZPixmap, 0, nullptr, vw, vh, 32, 0);
#else
                std::cout << "interactive needs Win32 or X11. Use --benchmark for the offscreen path.\n";
                quit = true;
#endif
                while (!quit) {
#if defined(NTC_WIN32)
                    MSG msg;
                    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                        if (msg.message == WM_QUIT) quit = true;
                        if (msg.message == WM_KEYDOWN) {
                            if (msg.wParam == VK_ESCAPE) quit = true;
                            if (msg.wParam == '1') mode = 0;
                            if (msg.wParam == '2') mode = 1;
                            if (msg.wParam == '3') mode = 2;
                            if (msg.wParam == '4') mode = 3;
                            if (msg.wParam == '5') mode = 4;
                            if (msg.wParam == 'H') heat = !heat;
                            if (msg.wParam == 'J') wipe = !wipe;
                        }
                        if (msg.message == WM_LBUTTONDOWN) {
                            dragging = true;
                            mx = GET_X_LPARAM(msg.lParam);
                            my = GET_Y_LPARAM(msg.lParam);
                        }
                        if (msg.message == WM_LBUTTONUP) dragging = false;
                        if (msg.message == WM_MOUSEMOVE && dragging) {
                            int x = GET_X_LPARAM(msg.lParam);
                            int y = GET_Y_LPARAM(msg.lParam);
                            yaw += float(x - mx) * 0.01f;
                            pitch += float(y - my) * 0.01f;
                            mx = x;
                            my = y;
                        }
                        if (msg.message == WM_MOUSEWHEEL) {
                            int delta = GET_WHEEL_DELTA_WPARAM(msg.wParam);
                            dist *= delta > 0 ? 0.9f : 1.1f;
                        }
                        TranslateMessage(&msg);
                        DispatchMessageW(&msg);
                    }
#elif defined(NTC_X11)
                    while (XPending(dpy)) {
                        XEvent ev;
                        XNextEvent(dpy, &ev);
                        if (ev.type == KeyPress) {
                            KeySym sym = XLookupKeysym(&ev.xkey, 0);
                            if (sym == XK_Escape) quit = true;
                            if (sym == XK_1) mode = 0;
                            if (sym == XK_2) mode = 1;
                            if (sym == XK_3) mode = 2;
                            if (sym == XK_4) mode = 3;
                            if (sym == XK_5) mode = 4;
                            if (sym == XK_h || sym == XK_H) heat = !heat;
                            if (sym == XK_j || sym == XK_J) wipe = !wipe;
                        }
                        if (ev.type == ButtonPress && ev.xbutton.button == 1) {
                            dragging = true;
                            mx = ev.xbutton.x;
                            my = ev.xbutton.y;
                        }
                        if (ev.type == ButtonRelease && ev.xbutton.button == 1) dragging = false;
                        if (ev.type == ButtonPress && ev.xbutton.button == 4) dist *= 0.9f;
                        if (ev.type == ButtonPress && ev.xbutton.button == 5) dist *= 1.1f;
                        if (ev.type == MotionNotify && dragging) {
                            yaw += float(ev.xmotion.x - mx) * 0.01f;
                            pitch += float(ev.xmotion.y - my) * 0.01f;
                            mx = ev.xmotion.x;
                            my = ev.xmotion.y;
                        }
                    }
#endif
                    if (pitch < -1.2f) pitch = -1.2f;
                    if (pitch > 1.2f) pitch = 1.2f;
                    if (dist < 0.6f) dist = 0.6f;
                    double ms = 0;
                    const char* names[] = {"NTC", "BC7", "BC1", "ASTC", "REF"};
                    int wipe_x = wipe ? vw / 2 : 0;
                    render_frame(live, mode, wipe_x, heat, yaw, pitch, dist, false, true, names[mode], ms);
#if defined(NTC_WIN32)
                    HDC hdc = GetDC(hwnd);
                    BITMAPINFO bmi{};
                    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                    bmi.bmiHeader.biWidth = vw;
                    bmi.bmiHeader.biHeight = -vh;
                    bmi.bmiHeader.biPlanes = 1;
                    bmi.bmiHeader.biBitCount = 32;
                    bmi.bmiHeader.biCompression = BI_RGB;
                    StretchDIBits(hdc, 0, 0, vw, vh, 0, 0, vw, vh, live.readback.mapped, &bmi, DIB_RGB_COLORS, SRCCOPY);
                    ReleaseDC(hwnd, hdc);
#elif defined(NTC_X11)
                    ximg->data = static_cast<char*>(live.readback.mapped);
                    XPutImage(dpy, win, DefaultGC(dpy, screen), ximg, 0, 0, 0, 0, vw, vh);
                    ximg->data = nullptr;
                    XFlush(dpy);
#endif
                }
#if defined(NTC_WIN32)
                DestroyWindow(hwnd);
#elif defined(NTC_X11)
                ximg->data = nullptr;
                XDestroyImage(ximg);
                XDestroyWindow(dpy, win);
                XCloseDisplay(dpy);
#endif
                vkDestroyImageView(dev, live.color_view, nullptr);
                vkDestroyImage(dev, live.color, nullptr);
                vkFreeMemory(dev, live.color_mem, nullptr);
                vkDestroyImageView(dev, live.depth_view, nullptr);
                vkDestroyImage(dev, live.depth, nullptr);
                vkFreeMemory(dev, live.depth_mem, nullptr);
                destroy_buf(dev, live.readback);
            } else {
                json << ", \"runs\": [\n";
                struct Job {
                    int w, h, mode;
                    bool heat;
                    const char* tag;
                };
                std::vector<Job> jobs;
                if (cpu_dev && !force_res) {
                    jobs.push_back({256, 256, 0, false, "ntc_256"});
                    std::cout << "bench 1920x1080 skipped cpu_device\n";
                    std::cout << "bench 2560x1440 skipped cpu_device\n";
                } else {
                    jobs.push_back({1920, 1080, 0, false, "ntc_1920x1080"});
                    jobs.push_back({1920, 1080, 1, false, "bc7_1920x1080"});
                    jobs.push_back({1920, 1080, 4, false, "ref_1920x1080"});
                    jobs.push_back({1920, 1080, 0, true, "heat_1920x1080"});
                    jobs.push_back({2560, 1440, 0, false, "ntc_2560x1440"});
                    jobs.push_back({2560, 1440, 1, false, "bc7_2560x1440"});
                }
                bool first_job = true;
                for (const Job& job : jobs) {
                    if (!(bc7_ok) && job.mode == 1) {
                        std::cout << "bench " << job.tag << " skipped format\n";
                        continue;
                    }
                    Target target = make_target(job.w, job.h);
                    std::vector<double> samples;
                    for (int f = 0; f < frames; ++f) {
                        double ms = 0;
                        bool shot = (f == frames / 2) && first_file;
                        float yaw = 6.2831853f * float(f) / float(std::max(frames, 1));
                        int wipe = (job.mode == 0 && !job.heat) ? job.w / 2 : 0;
                        render_frame(target, job.mode, wipe, job.heat, yaw, 0.35f, 3.4f, shot, shot, std::string(job.tag), ms);
                        samples.push_back(ms);
                    }
                    double med = median_of(samples);
                    double p95 = p95_of(samples);
                    std::cout << "bench " << job.tag << " median_ms " << med << " p95_ms " << p95;
                    if (cpu_dev) std::cout << " not_a_performance_claim";
                    std::cout << "\n";
                    if (!first_job) json << ",\n";
                    first_job = false;
                    json << "      {\"tag\": \"" << job.tag << "\", \"mode\": \"" << mode_name[job.mode]
                         << "\", \"heatmap\": " << (job.heat ? "true" : "false") << ", \"width\": " << job.w
                         << ", \"height\": " << job.h << ", \"median_ms\": " << med << ", \"p95_ms\": " << p95
                         << ", \"frames\": " << frames << "}";
                    vkDestroyImageView(dev, target.color_view, nullptr);
                    vkDestroyImage(dev, target.color, nullptr);
                    vkFreeMemory(dev, target.color_mem, nullptr);
                    vkDestroyImageView(dev, target.depth_view, nullptr);
                    vkDestroyImage(dev, target.depth, nullptr);
                    vkFreeMemory(dev, target.depth_mem, nullptr);
                    destroy_buf(dev, target.readback);
                }
                if (cpu_dev && !force_res) {
                    json << ",\n      {\"tag\": \"1920x1080\", \"skipped\": \"cpu_device\"}";
                    json << ",\n      {\"tag\": \"2560x1440\", \"skipped\": \"cpu_device\"}";
                }
                json << "\n    ]}";
            }
            first_file = false;
            vkDestroyDescriptorPool(dev, dpool, nullptr);
            destroy_buf(dev, desc);
            destroy_buf(dev, blob);
            auto kill_img = [&](VkImage i, VkDeviceMemory m, VkImageView v) {
                vkDestroyImageView(dev, v, nullptr);
                vkDestroyImage(dev, i, nullptr);
                vkFreeMemory(dev, m, nullptr);
            };
            kill_img(ref_i, ref_m, ref_v);
            kill_img(bc7_i, bc7_m, bc7_v);
            kill_img(bc5_i, bc5_m, bc5_v);
            kill_img(bc7o_i, bc7o_m, bc7o_v);
            kill_img(bc1_i, bc1_m, bc1_v);
            kill_img(bc1o_i, bc1o_m, bc1o_v);
            kill_img(astc_i, astc_m, astc_v);
            kill_img(ov_i, ov_m, ov_v);
            if (!benchmark) break;
        }
        json << "\n  ]\n}\n";
        if (benchmark) {
            std::ofstream js(hw_dir + "/report.json");
            js << json.str();
            std::cout << "wrote " << hw_dir << "/report.json\n";
        } else {
            std::cout << "interactive window is the next step when a surface is available\n";
            std::cout << "run with --benchmark for the offscreen path\n";
        }
        if (!benchmark) {
#if defined(NTC_WIN32) || defined(NTC_X11)
            std::cout << "keys: 1 ntc, 2 bc7 transcode, 3 bc1+bc5, 4 astc, 5 cpu decode, h heatmap, j wipe\n";
#else
            std::cout << "interactive needs a Win32 or X11 build\n";
#endif
        }
        destroy_buf(dev, vbuf);
        destroy_buf(dev, ibuf);
        destroy_buf(dev, ubo);
        vkDestroyPipeline(dev, pipeline, nullptr);
        vkDestroyShaderModule(dev, vs, nullptr);
        vkDestroyShaderModule(dev, fs, nullptr);
        vkDestroyPipelineLayout(dev, pipe_layout, nullptr);
        vkDestroyDescriptorSetLayout(dev, set_layout, nullptr);
        vkDestroySampler(dev, samp, nullptr);
        vkDestroyQueryPool(dev, qpool, nullptr);
        vkDestroyCommandPool(dev, pool, nullptr);
        vkDestroyDevice(dev, nullptr);
        vkDestroyInstance(inst, nullptr);
        (void)mode_name;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
    return 0;
}
