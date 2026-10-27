// Compare int8 payload bytes with a lossless BF16 exponent coding of the
// same matrix. The codec is third_party/expcode (cvs0/bf16-exponent-gemv).
// It is not on the decode path: the shader multiplies integer codes.

#include "expcode/codec.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: expcode_bytes rows cols f32_rowmajor.bin\n";
        return 2;
    }
    const uint32_t rows = static_cast<uint32_t>(std::stoul(argv[1]));
    const uint32_t cols = static_cast<uint32_t>(std::stoul(argv[2]));
    std::ifstream in(argv[3], std::ios::binary);
    std::vector<float> f(static_cast<size_t>(rows) * cols);
    in.read(reinterpret_cast<char*>(f.data()), static_cast<std::streamsize>(f.size() * 4));
    if (!in) {
        std::cerr << "short float file\n";
        return 1;
    }
    std::vector<uint16_t> bf(f.size());
    for (size_t i = 0; i < f.size(); ++i) bf[i] = expcode::fp32_to_bf16(f[i]);
    auto enc = expcode::EncodedMatrix::encode(bf, rows, cols, 0);
    const uint32_t int8_bytes = rows * cols;
    const uint32_t bf16_bytes = rows * cols * 2u;
    std::cout << "rows " << rows << " cols " << cols << " int8 " << int8_bytes << " bf16 " << bf16_bytes
              << " expcode " << enc.bytes().size() << "\n";
    return 0;
}
