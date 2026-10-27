#pragma once

// Symmetric per-group integer quantization for MLP weights.
// Same scheme as cvs0/weight-quantizer: one scale per group of the input
// axis, round to nearest, clamp to the symmetric range (the extreme
// negative code is unused). group_size <= 0 means one scale per row.
//
// cvs0/weight-quantizer documents the method. This header is the
// implementation used by the tests.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ntc {

inline int quant_qmax(int bits) {
    if (bits == 8) return 127;
    if (bits == 4) return 7;
    if (bits == 2) return 1;
    return (1 << (bits - 1)) - 1;
}

struct QuantizedWeight {
    int out = 0;
    int in = 0;
    int bits = 8;
    int group_size = 0;
    int groups = 1;
    std::vector<int> codes;    // [out * in]
    std::vector<float> scale;  // [out * groups]
};

inline QuantizedWeight quantize_weight(const float* w, int out, int in, int bits, int group_size) {
    const int qmax = quant_qmax(bits);
    int gs = (group_size <= 0 || group_size >= in) ? in : group_size;
    const int groups = (in + gs - 1) / gs;
    QuantizedWeight q;
    q.out = out;
    q.in = in;
    q.bits = bits;
    q.group_size = (gs == in) ? 0 : gs;
    q.groups = groups;
    q.codes.assign(static_cast<size_t>(out) * in, 0);
    q.scale.assign(static_cast<size_t>(out) * groups, 0.f);
    for (int o = 0; o < out; ++o) {
        for (int g = 0; g < groups; ++g) {
            const int i0 = g * gs;
            const int i1 = std::min(in, i0 + gs);
            float amax = 0.f;
            for (int i = i0; i < i1; ++i) {
                amax = std::max(amax, std::fabs(w[o * in + i]));
            }
            float scale = amax / static_cast<float>(qmax);
            if (scale < 1e-12f) scale = 1e-12f;
            q.scale[static_cast<size_t>(o) * groups + g] = scale;
            for (int i = i0; i < i1; ++i) {
                float r = std::round(w[o * in + i] / scale);
                if (r > qmax) r = static_cast<float>(qmax);
                if (r < -qmax) r = static_cast<float>(-qmax);
                q.codes[static_cast<size_t>(o) * in + i] = static_cast<int>(r);
            }
        }
    }
    return q;
}

}  // namespace ntc
