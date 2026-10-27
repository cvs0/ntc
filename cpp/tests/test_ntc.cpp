#include "ntc/group_quant.hpp"
#include "ntc/ntc.hpp"

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

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "tests/fixtures";
    test_quant();
    test_fixture(dir);
    if (g_fails) {
        std::cerr << g_fails << " failures\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
