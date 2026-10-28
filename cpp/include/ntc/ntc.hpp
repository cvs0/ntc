#pragma once

// NTC1 reader and reference decoder. The math matches src/ntc/decode_ref.py.
// Latent parameters on disk are integer codes. A code q decodes as
// (q - mid) * scale, mid = (2^bits - 1) / 2. Grid mips are a 2x2 integer
// box, (sum + 2) / 4, odd trailing texels dropped.

#include <cstdint>
#include <string>
#include <vector>

namespace ntc {

constexpr uint32_t kMagic = 0x3143544Eu;
constexpr uint16_t kVersion = 1;
constexpr uint32_t kHeaderBytes = 96;
constexpr int kMaxChannels = 16;
constexpr int kMaxGrids = 8;
constexpr int kMaxLayers = 4;
constexpr int kMaxHidden = 64;
constexpr int kMaxInput = 160;
constexpr int kMaxMips = 8;

constexpr uint32_t kFlagDerivedMips = 1u << 0;
constexpr uint32_t kFlagActivationRequant = 1u << 1;
// Raw deflate (window -15) of the latent region. data_offset is then
// relative to the inflated bytes. Header u32 at offset 84 is that length.
constexpr uint32_t kFlagLatentDeflate = 1u << 2;
// Order-1 nibble range coder. Same plain bytes as deflate. Offset 84 is the length.
constexpr uint32_t kFlagLatentRange = 1u << 3;
constexpr uint32_t kFlagLatentCompressed = kFlagLatentDeflate | kFlagLatentRange;

constexpr int kActIdentity = 0;
constexpr int kActRelu = 1;
constexpr int kActHgelu = 2;
constexpr int kPeNone = 0;
constexpr int kPeFourier = 1;
constexpr int kPeTriangle = 2;

struct Channel {
    uint8_t semantic = 0;
    uint8_t color_space = 0;
};

struct Grid {
    int width = 0;
    int height = 0;
    int channels = 0;
    int bits = 0;
    int row_stride = 0;
    uint32_t data_offset = 0;
    uint32_t data_bytes = 0;
    uint32_t scale_offset = 0;
    uint32_t plane_bytes = 0;
    std::vector<float> scale;           // [C]
    std::vector<uint8_t> codes;         // [C,H,W] row-major, unpacked
    std::vector<std::vector<uint8_t>> mips;  // mips[k] is [C,Hk,Wk]
    std::vector<int> mip_w;
    std::vector<int> mip_h;
};

struct Layer {
    int in_features = 0;
    int out_features = 0;
    int group_size = 0;
    int num_groups = 0;
    uint32_t weight_offset = 0;
    uint32_t weight_bytes = 0;
    uint32_t bias_offset = 0;
    uint32_t scale_offset = 0;
    std::vector<float> weight;  // [out, in] integer codes as float
    std::vector<float> bias;    // [out]
    std::vector<float> scale;   // [out, groups] row-major
    uint32_t weight_bits = 0;   // 0 means the header value
};

struct TextureSet {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t num_mips = 0;
    uint32_t activation = 0;
    uint32_t pe_mode = 0;
    uint32_t pe_octaves = 0;
    uint32_t flags = 0;
    uint32_t weight_bits = 8;
    uint32_t group_size = 0;
    std::vector<Channel> channels;
    std::vector<Grid> grids;
    std::vector<Layer> layers;
    std::vector<uint8_t> raw;
    uint32_t tile_size = 0;
    uint32_t latent_stored_bytes = 0;
    uint32_t latent_plain_bytes = 0;
    uint32_t scale_stored_bytes = 0;

    int num_channels() const { return static_cast<int>(channels.size()); }
    bool activation_requant() const { return (flags & kFlagActivationRequant) != 0; }
};

TextureSet load_file(const std::string& path);
TextureSet parse(const uint8_t* data, size_t size);

// HWC float32, color mip `mip`, one MLP at each texel center.
std::vector<float> decode_mip(const TextureSet& ts, int mip);

// One feature-bilinear sample. `uv` is in [0,1]. Output length is num_channels.
void decode_uv(const TextureSet& ts, float u, float v, int mip, float* out);

int pe_feature_count(int pe_mode, int pe_octaves);
uint32_t payload_bytes(const TextureSet& ts);
float total_bpp(const TextureSet& ts);
// SDK latent rate: compressed latent codes + fp16 latent scales, per mip 0
// texel. Excludes MLP weights, the header, and the descriptor tables.
float latent_bpp_mip0(const TextureSet& ts);
// Whole file, per texel of every mip declared in the header.
float file_bpp_chain(const TextureSet& ts);
// Paper rate: payload bits / (mip-chain texels * channel count).
float bppc(const TextureSet& ts);
int mip_chain_texels(const TextureSet& ts);

// GPU descriptor. 84 uvec4s, matching the layout in shaders/ntc_decode.glsl.
// `blob` is the file bytes followed by packed derived-mip payloads.
// Descriptor mip offsets point into `blob`.
constexpr int kDescUvec4 = 84;
constexpr int kDescWords = kDescUvec4 * 4;

struct GpuUpload {
    std::vector<uint32_t> desc;  // kDescWords
    std::vector<uint8_t> blob;
};

GpuUpload make_gpu_upload(const TextureSet& ts, uint32_t frame, uint32_t filter_mode, uint32_t max_aniso);

// Filter modes stored in the GPU descriptor. The CPU reference decode_mip
// is filter mode "point" (texel-center feature sample).
constexpr uint32_t kFilterPoint = 0;
constexpr uint32_t kFilterFeatureBilinear = 1;
constexpr uint32_t kFilterColorBilinear = 2;
constexpr uint32_t kFilterStochastic = 3;
constexpr uint32_t kFilterFeatureTrilinear = 4;
constexpr uint32_t kFilterStochasticAniso = 5;

}  // namespace ntc
