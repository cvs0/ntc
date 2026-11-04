#include "ntc/ntc.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ntc_decode file.ntc out.f32 [--mip N]
// out.f32: uint32 width, height, channels, mip, then float32 HWC.

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: ntc_decode file.ntc out.f32 [--mip N]\n";
        return 2;
    }
    int mip = 0;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mip" && i + 1 < argc) mip = std::stoi(argv[++i]);
    }
    try {
        ntc::TextureSet ts = ntc::load_file(argv[1]);
        std::vector<float> img = ntc::decode_mip(ts, mip);
        const uint32_t w = std::max(1u, ts.width >> mip);
        const uint32_t h = std::max(1u, ts.height >> mip);
        const uint32_t c = static_cast<uint32_t>(ts.num_channels());
        std::ofstream out(argv[2], std::ios::binary);
        const uint32_t hdr[4] = {w, h, c, static_cast<uint32_t>(mip)};
        out.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
        out.write(reinterpret_cast<const char*>(img.data()),
                  static_cast<std::streamsize>(img.size() * sizeof(float)));
        std::cout << "decoded " << w << "x" << h << "x" << c << " mip " << mip
                  << " total_bpp " << ntc::total_bpp(ts)
                  << " latent_bpp_mip0 " << ntc::latent_bpp_mip0(ts)
                  << " file_bpp_chain " << ntc::file_bpp_chain(ts)
                  << " bppc " << ntc::bppc(ts)
                  << " header_bytes " << ntc::kHeaderBytes
                  << " network_bytes " << ntc::network_bytes(ts) << "\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
    return 0;
}
