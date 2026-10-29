#pragma once

// Fast inference layout for shaders/ntc_fast.glsl.
//
// The file format is unchanged. This is a GPU-only layout derived at upload:
//
//   spec     Vulkan specialization constants, index = constant_id. Every
//            loop bound, width, grid shape, and parameter offset is a
//            constant, so the driver unrolls the MLP and keeps the feature
//            and hidden vectors in registers.
//   params   Uniform buffer words (at most kFastParamWords). Mip table,
//            fp32 latent scales, fp32 activation scales, then per layer
//            fp32 weights [out][in], fp32 bias [out], fp32 scales
//            [out][groups], packed int8 weights [out][ceil(in/4)] and
//            packed fp16 weights [out][ceil(in/2)]. Weight codes are small
//            integers, so the fp32 and fp16 copies are exact.
//   latents  Storage buffer words. Each grid mip is split into channel
//            planes whose record (channels * bits) is a power of two of at
//            most 32 bits, so one texel of one plane is one aligned load.
//            Codes are LSB first inside the record. Rows are word padded.
//
// Mip table entry (grid g, mip k) at uvec4 index g * 8 + k:
//   x  word offset of plane 0
//   y  width
//   z  height
//   w  0
// Plane p starts after the rows of planes 0..p-1 of the same mip.

#include "ntc/ntc.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ntc {

constexpr int kFastParamWords = 16384;  // 64 KiB, the minimum maxUniformBufferRange
constexpr int kFastMipTableWords = 8 * 8 * 4;

enum FastSpec : int {
    kSpecGrids = 0,
    kSpecLayers = 1,
    kSpecOutCh = 2,
    kSpecPeMode = 3,
    kSpecPeOct = 4,
    kSpecAct = 5,
    kSpecRequant = 6,
    kSpecActI8 = 7,
    kSpecNumMips = 8,
    kSpecLatentScaleOff = 9,
    kSpecActScaleOff = 10,
    kSpecWidth0 = 11,   // 11..15: layer widths, width0 = input features
    kSpecGridCh0 = 16,  // 16..23
    kSpecGridBits0 = 24,  // 24..31
    kSpecGroupSize0 = 32,  // 32..35, 0 means one group per row
    kSpecNumGroups0 = 36,  // 36..39
    kSpecWOff0 = 40,  // 40..43 fp32 weights
    kSpecBOff0 = 44,  // 44..47 fp32 bias
    kSpecSOff0 = 48,  // 48..51 fp32 scales
    kSpecQOff0 = 52,  // 52..55 packed int8 weights
    kSpecHOff0 = 56,  // 56..59 packed fp16 weights
    kSpecCount = 60,
};

struct FastPlane {
    int ch0 = 0;       // first channel of the grid in this plane
    int channels = 0;  // channels in the plane
    int record_bits = 0;  // channels * bits, a power of two <= 32
};

// Channel planes for one grid. Matches ntcf_plane_* in ntc_fast.glsl.
std::vector<FastPlane> fast_planes(int channels, int bits);
int fast_row_words(int width, int record_bits);

struct FastUpload {
    std::vector<uint32_t> spec;     // kSpecCount
    std::vector<uint32_t> params;   // uniform buffer words
    std::vector<uint32_t> latents;  // storage buffer words
    uint32_t latent_bytes = 0;      // latents.size() * 4
    uint32_t legacy_latent_bytes = 0;  // stored + derived packed codes in the legacy blob
    bool ok = false;
    std::string why;                // reason when !ok
};

FastUpload make_fast_upload(const TextureSet& ts);

// CPU decode that reads only the FastUpload buffers, in the exact operation
// order of ntc_fast.glsl with NTC_FAST_EXACT. It is bit-identical to
// decode_mip when the layout transform is correct.
std::vector<float> decode_fast_mip(const FastUpload& fu, int mip, int width, int height);

// Per-sample work of one decode (one feature-bilinear sample at one mip),
// counted from the shader source. Mirrors ntc.cost.decode_ops in Python.
struct SampleCost {
    int macs = 0;             // weight * activation products
    int latent_loads = 0;     // 32-bit storage loads for latent codes
    int weight_loads = 0;     // 32-bit loads for weights, bias, and scales
    int scale_loads = 0;      // 32-bit loads for latent scales
    int uniform_words = 0;    // words read from the uniform buffer (fast path)
    int load_bytes = 0;       // storage bytes requested, 4 per load
    int alu = 0;              // non-MAC ALU estimate: unpack, lerp, dequant, act
};

SampleCost legacy_sample_cost(const TextureSet& ts);
SampleCost fast_sample_cost(const TextureSet& ts);

}  // namespace ntc
