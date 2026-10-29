#include "ntc/fast.hpp"
#include "ntc/group_quant.hpp"
#include "ntc/ntc.hpp"
#include "../src/range_coder.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static int g_fails = 0;

static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL " << msg << "\n";
        ++g_fails;
    }
}

static std::vector<uint8_t> read_bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static void test_quant() {
    float w[8] = {1.f, -1.f, 0.5f, -0.25f, 0.1f, 0.2f, -0.3f, 0.4f};
    auto q = ntc::quantize_weight(w, 2, 4, 8, 0);
    expect(q.groups == 1, "one group");
    expect(q.codes.size() == 8, "code count");
    for (int c : q.codes) expect(c >= -127 && c <= 127, "int8 range");
    auto q4 = ntc::quantize_weight(w, 2, 4, 4, 2);
    expect(q4.groups == 2, "two groups");
    for (int c : q4.codes) expect(c >= -7 && c <= 7, "int4 range");
}

static void test_range() {
    const uint8_t src[] = {0, 1, 2, 15, 15, 15, 0, 4, 8, 255, 7, 7, 7, 1};
    auto enc = ntc::range_encode(src, sizeof(src));
    auto dec = ntc::range_decode(enc.data(), enc.size(), sizeof(src));
    expect(dec.size() == sizeof(src), "range length");
    if (dec.size() == sizeof(src)) {
        for (size_t i = 0; i < sizeof(src); ++i) expect(dec[i] == src[i], "range byte");
    }
}

static void test_range_ctx() {
    // Smooth ramp. The coded bytes are the Python range_encode_ctx stream.
    uint8_t src[64];
    for (int i = 0; i < 64; ++i) src[i] = static_cast<uint8_t>((i * 3) / 8);
    const uint8_t py[] = {
        0, 0, 98, 165, 12, 107, 37, 243, 31, 134, 244, 189, 88, 128, 78, 202,
        22, 9, 233, 162, 9, 21, 130, 56, 143, 180, 20, 82, 152, 233, 64, 42,
        131, 210, 62, 49, 193, 43, 251, 204, 251, 31, 112, 90, 162, 144, 68, 81,
        62, 240, 253, 147, 205, 121, 172, 197, 33, 131, 163, 0,
    };
    auto enc = ntc::range_encode_ctx(src, sizeof(src));
    expect(enc.size() == sizeof(py), "ctx length matches python");
    if (enc.size() == sizeof(py)) {
        for (size_t i = 0; i < sizeof(py); ++i) expect(enc[i] == py[i], "ctx byte matches python");
    }
    auto dec = ntc::range_decode_ctx(py, sizeof(py), sizeof(src));
    expect(dec.size() == sizeof(src), "ctx decode length");
    if (dec.size() == sizeof(src)) {
        for (size_t i = 0; i < sizeof(src); ++i) expect(dec[i] == src[i], "ctx decode byte");
    }
}

static void test_ctx_file(const std::string& dir) {
    const auto bytes = read_bin(dir + "/ctx.ntc");
    expect(!bytes.empty(), "ctx fixture present");
    if (bytes.empty()) return;
    ntc::TextureSet ts = ntc::parse(bytes.data(), bytes.size());
    expect((ts.flags & ntc::kFlagLatentCtx) != 0, "ctx flag");
    expect((ts.flags & ntc::kFlagLatentRange) == 0, "ctx not order-1");
    expect((ts.flags & ntc::kFlagLatentDeflate) == 0, "ctx not deflate");
    expect(ntc::network_bytes(ts) > 0, "network bytes");
    auto img = ntc::decode_mip(ts, 0);
    std::ifstream ref(dir + "/ctx_mip0.f32", std::ios::binary);
    uint32_t hdr[4];
    ref.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    expect(hdr[0] == ts.width && hdr[1] == ts.height, "ctx ref size");
    std::vector<float> want(img.size());
    ref.read(reinterpret_cast<char*>(want.data()), static_cast<std::streamsize>(want.size() * 4));
    float max_abs = 0.f;
    for (size_t i = 0; i < img.size() && i < want.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(img[i] - want[i]));
    }
    std::cout << "ctx file max abs " << max_abs << "\n";
    expect(max_abs < 1e-4f, "ctx file matches python");
}

static void test_fixture(const std::string& dir) {
    const auto bytes = read_bin(dir + "/tiny.ntc");
    expect(!bytes.empty(), "fixture present");
    if (bytes.empty()) return;
    ntc::TextureSet ts = ntc::parse(bytes.data(), bytes.size());
    expect(ts.width == 8 && ts.height == 8, "fixture size");
    auto img = ntc::decode_mip(ts, 0);
    std::ifstream ref(dir + "/tiny_mip0.f32", std::ios::binary);
    uint32_t hdr[4];
    ref.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    expect(hdr[0] == 8 && hdr[1] == 8, "ref header");
    std::vector<float> want(img.size());
    ref.read(reinterpret_cast<char*>(want.data()), static_cast<std::streamsize>(want.size() * 4));
    float max_abs = 0.f;
    for (size_t i = 0; i < img.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(img[i] - want[i]));
    }
    std::cout << "cpp vs python max abs " << max_abs << "\n";
    expect(max_abs < 1e-4f, "decode matches python");

    auto img1 = ntc::decode_mip(ts, 1);
    std::ifstream ref1(dir + "/tiny_mip1.f32", std::ios::binary);
    uint32_t h1[4];
    ref1.read(reinterpret_cast<char*>(h1), sizeof(h1));
    std::vector<float> want1(static_cast<size_t>(h1[0]) * h1[1] * h1[2]);
    ref1.read(reinterpret_cast<char*>(want1.data()), static_cast<std::streamsize>(want1.size() * 4));
    expect(img1.size() == want1.size(), "mip1 size");
    float max1 = 0.f;
    for (size_t i = 0; i < img1.size(); ++i) max1 = std::max(max1, std::fabs(img1[i] - want1[i]));
    std::cout << "mip1 max abs " << max1 << "\n";
    expect(max1 < 1e-4f, "mip1 matches python");

    auto gpu = ntc::make_gpu_upload(ts, 0, ntc::kFilterPoint, 1);
    expect(gpu.desc.size() == static_cast<size_t>(ntc::kDescWords), "desc words");
    expect(gpu.blob.size() >= ts.raw.size(), "blob holds the file");
}

static uint32_t crc32_ref(const uint8_t* data, size_t n) {
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

static void test_wild_offset(const std::string& dir) {
    auto bytes = read_bin(dir + "/tiny.ntc");
    expect(bytes.size() > 96, "tiny present for offset test");
    if (bytes.size() <= 96) return;
    bytes[40] = 0xF0;
    bytes[41] = 0xFF;
    bytes[42] = 0xFF;
    bytes[43] = 0xFF;
    const uint32_t crc = crc32_ref(bytes.data() + 96, bytes.size() - 96);
    bytes[76] = static_cast<uint8_t>(crc);
    bytes[77] = static_cast<uint8_t>(crc >> 8);
    bytes[78] = static_cast<uint8_t>(crc >> 16);
    bytes[79] = static_cast<uint8_t>(crc >> 24);
    bool threw = false;
    try {
        ntc::parse(bytes.data(), bytes.size());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    expect(threw, "wild channel offset throws");
}

// The fast layout must decode bit-identically to the reference, every mip.
static void test_fast_layout(const std::string& path) {
    const auto bytes = read_bin(path);
    if (bytes.empty()) return;
    ntc::TextureSet ts = ntc::parse(bytes.data(), bytes.size());
    ntc::FastUpload fu = ntc::make_fast_upload(ts);
    expect(fu.ok, "fast upload ok");
    if (!fu.ok) return;
    for (uint32_t m = 0; m < ts.num_mips && m < 4; ++m) {
        auto want = ntc::decode_mip(ts, static_cast<int>(m));
        auto got = ntc::decode_fast_mip(fu, static_cast<int>(m), static_cast<int>(ts.width), static_cast<int>(ts.height));
        expect(want.size() == got.size(), "fast size");
        size_t diff = 0;
        for (size_t i = 0; i < want.size() && i < got.size(); ++i) diff += want[i] != got[i];
        if (diff) std::cerr << path << " mip " << m << " fast differs at " << diff << " values\n";
        expect(diff == 0, "fast layout bit-identical");
    }
}

static void test_fast_planes() {
    auto p = ntc::fast_planes(12, 4);
    expect(p.size() == 2 && p[0].channels == 8 && p[1].channels == 4, "12x4 planes 8+4");
    p = ntc::fast_planes(6, 8);
    expect(p.size() == 2 && p[0].record_bits == 32 && p[1].record_bits == 16, "6x8 planes 32+16");
    p = ntc::fast_planes(2, 2);
    expect(p.size() == 1 && p[0].record_bits == 4, "2x2 one 4-bit record");
    p = ntc::fast_planes(16, 4);
    expect(p.size() == 2 && p[1].ch0 == 8, "16x4 planes 8+8");
    expect(ntc::fast_row_words(5, 16) == 3, "row words");
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "tests/fixtures";
    test_quant();
    test_range();
    test_range_ctx();
    test_ctx_file(dir);
    test_fixture(dir);
    test_wild_offset(dir);
    test_fast_planes();
    test_fast_layout(dir + "/tiny.ntc");
    test_fast_layout(dir + "/ctx.ntc");
    for (int i = 2; i < argc; ++i) test_fast_layout(argv[i]);
    if (g_fails) {
        std::cerr << g_fails << " failures\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
