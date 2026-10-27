# ntc

NTC1 stores one PBR material as quantized latent grids plus an MLP of integer weights, and decodes each texel with portable fp32 dots in Vulkan 1.3 and OpenGL 4.5.

The container is not NVIDIA NTEX. RTXNTC is not vendored. Official `ntc-cli` numbers are pending a machine that can run it.

## Format

Little-endian. Magic `0x3143544E` (`NTC1`). Version 1. Header 96 bytes. `crc32` is zlib CRC-32 of every byte after the header.

| offset | type | field |
| --- | --- | --- |
| 0 | u32 | magic |
| 4 | u16 | version |
| 6 | u16 | header_bytes = 96 |
| 8 | u32 | flags |
| 12 | u32 | width |
| 16 | u32 | height |
| 20 | u16 | channels |
| 22 | u16 | num_mips |
| 24 | u16 | grids |
| 26 | u16 | layers |
| 28 | u16 | activation |
| 30 | u16 | pe_mode |
| 32 | u16 | pe_octaves |
| 34 | u16 | weight_bits |
| 36 | u16 | group_size |
| 38 | u16 | reserved |
| 40 | u32 | channel_off |
| 44 | u32 | grid_off |
| 48 | u32 | layer_off |
| 52 | u32 | latent_off |
| 56 | u32 | latent_bytes |
| 60 | u32 | weight_off |
| 64 | u32 | weight_bytes |
| 68 | u32 | scale_off |
| 72 | u32 | scale_bytes |
| 76 | u32 | crc32 |
| 80 | u32 | file_bytes |
| 84 | u32 | reserved |
| 88 | u64 | reserved |

Flag bit 0: derived mips (set on every file this encoder writes). Flag bit 1: activation requant. It is off in shipped files.

Channel record, 8 bytes: `u8 semantic, u8 color_space, u16 reserved, u32 reserved`.

| semantic | name |
| --- | --- |
| 0..2 | albedo r, g, b |
| 3..5 | normal x, y, z |
| 6 | roughness |
| 7 | metalness |
| 8 | ao |
| 9 | height |
| 10 | opacity |
| 11 | extra |

Grid record, 24 bytes: `u16 width, u16 height, u8 channels, u8 bits, u16 row_stride, u32 data_off, u32 data_bytes, u32 scale_off, u32 plane_bytes`.

Layer record, 32 bytes: `u16 in, u16 out, u32 weight_off, u32 weight_bytes, u32 bias_off, u32 scale_off, u16 group_size, u16 num_groups, u32 reserved, u32 reserved`.

Limits: 8 grids, 4 layers, 16 channels, hidden width 64, 160 input features, 8 mips.

`total_bpp = 8 * (packed latent codes + fp16 latent scales + integer weights + fp16 bias + fp16 weight scales) / (width * height)`. The 96-byte header and the descriptor tables are not in `total_bpp`. They are in `file_bytes`.

## Network

Input features are the concatenated bilinear latent samples, then optional positional features. `pe_mode` 0 produces none. Mode 1 (Fourier) produces `4 * octaves` values, `sin/cos(2π * uv * 2^o)` on each axis, with `uv` not doubled. Mode 2 (triangle) produces `4 * octaves + 2` values, the second pair being `lod = mip / (num_mips - 1)` duplicated. Shipped files use mode 0.

The MLP is a stack of affine layers. Hidden activation 0 is identity, 1 is ReLU, 2 is the hard GELU `min(x, 3) * saturate(x / 3 + 0.5)`. Shipped files use ReLU. The last layer has no activation. Output channels follow the channel table order.

A shipped high-rate network for a 9-channel material is:

| item | value |
| --- | --- |
| grids | 256x256x4 at 4 bits, then 8x8x2 at 2 bits |
| MLP | 6 to 32 ReLU, 32 to 9 |
| weights | int8, one fp16 scale per output row |
| MACs, one MLP | 480 |
| position features | 0 |

The low-rate search may pick other grids. The decoder does not assume this shape.

## Latent grids and addressing

Codes are planar, channel-major, row-major. Row stride is `(width * bits + 7) / 8` bytes.

| bits | packing |
| --- | --- |
| 8 | one code per byte |
| 4 | even x in bits 3:0, odd x in bits 7:4. Width even. |
| 2 | `x % 4 == k` in bits `2k+1:2k`. Width multiple of 4. |

Only mip 0 is stored. Mip `k+1` is a 2x2 integer box of mip `k`: `(a + b + c + d + 2) / 4`. A size-1 axis uses `(sum + 1) / 2`. Odd trailing rows and columns are dropped. `num_mips` is `min(8, floor(log2(max(width, height))) + 1)`. The GPU upload appends the packed derived mips after the file bytes and 4-byte-pads the blob. Descriptor mip offsets point into that blob.

For the 16.084 bpp metal_plate network the stored latent payload is 131104 bytes and the derived chain is 43702 bytes. Those 43702 bytes are computed at upload and are not in the file. A stored chain would add that many bytes, about one third of the stored latent payload (`43702 / 131104`).

Bilinear sample of one grid at `uv` in `[0, 1]`:

```
x = u * W - 0.5
y = v * H - 0.5
x0 = floor(x), y0 = floor(y)
tx = x - x0, ty = y - y0
```

Corners wrap with floor-mod. The four codes are lerped in x, then in y. The affine dequant runs once on the lerped code, which is exact because dequant is affine:

```
value = (lerped - mid) * scale
mid = (2^bits - 1) / 2
```

`scale` is one fp16 value per latent channel, for the whole grid.

## Quantization

Latent parameters are trained in code units. With straight-through estimation:

```
q = round(z) clamped to [0, 2^bits - 1]
q_ste = z + (q - z).detach()
deq = (q_ste - mid) * scale
```

Weight codes are signed and symmetric. int8 uses `[-127, 127]`. int4 uses `[-7, 7]`. The extreme negative bin is unused. A group is a contiguous span of the input axis of one output row. `group_size == 0` is one scale for the whole row. Scales and biases are stored as fp16 and the training quantizer rounds them to fp16 before the forward pass.

```
acc = sum_i float(w[i]) * x[i]
y = bias + scale * acc
```

This is the per-group scale, round-to-nearest, clamp scheme described by cvs0/weight-quantizer. That repository documents the method. The implementation here is `src/ntc/quant.py` and `cpp/include/ntc/group_quant.hpp`.

`third_party/expcode` is the bf16 exponent codec from cvs0/bf16-exponent-gemv (MIT). It is not on the decode path. On three random normal matrices it was larger than int8: 32x8 was 596 bytes vs 256, 9x32 was 552 vs 288, 64x48 was 4948 vs 3072. It beat raw bf16 only on the 64x48 matrix (4948 vs 6144).

## Shader decode and cost

`shaders/ntc_decode.glsl` is the library. Vulkan bindings use `set = 0`. OpenGL bindings omit the set. `NTC_FP16` accumulates the dot in float16 and is not bit-exact. `NTC_NO_OUT` skips the output buffer.

Descriptor, 84 `uvec4`s, written by `ntc::make_gpu_upload`:

| index | contents |
| --- | --- |
| 0 | width, height, channels, grids |
| 1 | layers, activation, pe_mode, pe_octaves |
| 2 | flags, weight_bits, num_mips, frame |
| 3 | filter_mode, max_aniso, 8, 0 |
| 4..11 | grid channels, bits, scale_offset, nmips |
| 12..15 | layer in, out, weight_offset, bias_offset |
| 16..19 | layer scale_offset, group_size, num_groups, weight_bits |
| 20..83 | mip `g * 8 + k`: data_offset, row_stride, width, height |

Required arithmetic is fp32 dots of the integer weights. `GL_EXT_shader_integer_dot_product` is not used: this tree's glslang does not compile it. `VK_KHR_cooperative_matrix` is queried and not used, because one texel's GEMV does not fill a tile. Subgroup size is read from the device. The shader does not assume 32 or 64.

Filter modes:

| mode | name | MLP evals |
| --- | --- | --- |
| 0 | point | 1 |
| 1 | feature bilinear | 1 |
| 2 | color bilinear | 4 |
| 3 | stochastic | 1 |
| 4 | feature trilinear | 2 |
| 5 | stochastic anisotropic | N |

Mode 1 is the latent lerp above, then one MLP. Mode 2 evaluates the MLP at the four texel centers and lerps the colors. Mode 4 lerps two feature-bilinear colors by `fract(lod)`. Mode 3 picks one of the four corners. The hash is `uint(x) * 1664525 + uint(y) * 1013904223 + frame * 22695477`, then the low 24 bits over `2^24`. Corner probabilities are the bilinear weights, in order `(x0,y0), (x1,y0), (x0,y1), (x1,y1)`. Mode 5 takes `N = clamp(ceil(major/minor), 1, max_aniso)` samples along the major screen-space derivative and averages stochastic corners. Tap `i` mixes `frame + i * 747796405`.

Flops for one feature-bilinear sample, counted as scalar adds and multiplies:

| term | count |
| --- | --- |
| code lerp | 6 per latent channel |
| dequant | 2 per latent channel |
| weight dot | 2 per MAC |
| bias and scale | 2 per layer output |
| ReLU | 1 per hidden unit |
| hard GELU | 4 per hidden unit |

Latent bytes are an upper bound of 4 taps times channels, one packed byte per tap per channel. A 4-bit or 2-bit pair that shares a byte is also reported as `latent_bytes_shared_x`. Weight bytes are the cold MLP working set. A workgroup that holds the weights once divides that by the number of invocations.

Shipped 16.084 bpp network, one feature-bilinear texel:

| quantity | value |
| --- | --- |
| MACs | 480 |
| flops | 1122 |
| latent bytes, upper | 24 |
| latent bytes, shared-x bound | 12 |
| weight bytes, cold | 644 |
| weight bytes if shared by 64 invocations | 10.0625 |
| unique latent bytes in an 8x8 tile | 2.9375 per texel |
| same tile, no reuse | 24 per texel |

Color bilinear multiplies the MLP term by 4 (1920 MACs, 4488 flops). Feature trilinear multiplies it by 2. The shipped compute shader is one thread per texel and does not stage the 8x8 tile. The 2.9375 byte figure is the fetch count if a workgroup did. No GPU timing of that optimization exists.

Baking the per-channel latent scale into the first int8 matrix is not exact: the scale is fp16 and the stored weights are integer. The dequant is 12 flops and 12 scale bytes on this network, next to 960 dot flops. It was not implemented.

## Filtering checks

Reference implementation: `src/ntc/decode_ref.py`. Same hash and same corner order as the shader. Measured on one 8x8 random network, seed 2 (`results/filtering.json`):

| check | result |
| --- | --- |
| mean of 400 stochastic frames vs color bilinear | max abs 9.774e-6 |
| feature trilinear vs a manual lerp | max abs 0 |
| deterministic aniso vs the mean of the same taps | max abs 0, N = 5 |
| mean of 300 stochastic-aniso frames vs color-bilinear at those offsets | max abs 2.466e-6 |
| corner rate minus expected weight, 4000 samples | -0.00454, -0.00140, +0.00566, +0.00028 |

## Vulkan and OpenGL

```cpp
ntc::TextureSet ts = ntc::load_file(path);
ntc::GpuUpload up = ntc::make_gpu_upload(ts, frame, ntc::kFilterFeatureBilinear, 8);
// up.desc: 84 uvec4. up.blob: file bytes, then packed derived mips.
```

Compute, one thread per texel:

```glsl
vec2 uv = (vec2(gl_GlobalInvocationID.xy) + vec2(0.5)) / vec2(float(W), float(H));
ntc_eval(uv, mip);
```

Filtered sampling:

```glsl
ntc_eval_filtered(uv, dudx, dudy, lod, frame, /*filt*/ 1u, /*maxAniso*/ 8u);
```

`examples/vulkan/main.cpp`: `ntc_vk file.ntc spv_dir`. Prefers a discrete GPU, then integrated, then CPU. Runs the fp32 compute shader and compares to `ntc::decode_mip` with tolerance `1e-3`. If `shaderFloat16` is set, it also runs `ntc_decompress_fp16.comp`. Cooperative matrix is queried and ignored.

`examples/opengl/main.cpp`: `ntc_gl file.ntc shader_src_dir`. Surfaceless EGL. The host splices `ntc_decode.glsl` in, because the driver is not asked for `GL_GOOGLE_include_directive`. Context 4.6, then 4.5. The GL vertex shader uses `gl_VertexID`. Fragment `uv` is `gl_FragCoord.xy / size` with no Y flip: GL and `glReadPixels` are both lower-left, and mip 0 row 0 is `v = 0.5 / H`.

CPU reference: `ntc_decode file.ntc out.f32 [--mip N]`.

Python: `ntc-encode MATERIAL OUT.ntc --size 256 --bpp 16`. `ntc-decode FILE OUT.npz --reference MATERIAL --size 256`.

## Compatibility

Required path: Vulkan 1.3 or OpenGL 4.5, storage buffers, fp32. No `GL_NV_*`, no `VK_NV_*`, no cooperative-matrix variant.

| vendor | API | mode | required | optional, queried |
| --- | --- | --- | --- | --- |
| AMD, NVIDIA, lavapipe, others | Vulkan 1.3 | inference-on-load compute | storage buffer, int8/fp32 | `shaderFloat16` selects the fp16 dot |
| same | Vulkan 1.3 | inference-on-sample | same, plus a fragment or compute dispatch | subgroup size is reported, not required |
| same | OpenGL 4.5 or 4.6 | compute and fragment | SSBO, GLSL 450 | `GL_KHR_shader_subgroup` is not required |

Subgroup size is whatever the device reports. This tree does not branch on 32 vs 64. A wave64 device runs the same fp32 dot.

Encoder device: `torch.cuda.is_available()` covers CUDA and ROCm. A ROCm build sets `torch.version.hip` and still uses the `cuda` device API. `device_report` labels that backend `rocm`. There is no CUDA-only op. CPU is the fallback. ROCm is a PyTorch ROCm wheel from pytorch.org. This VM has the CPU wheel `2.14.1+cpu`.

## Encoder

`src/ntc/encode.py`. Latent grids are chosen from a family of one- and two-grid shapes, including full resolution (shift 0). The eight shapes closest to the target rate are scored at step 0 (PCA plus a least-squares last layer). The best is trained. For a hidden layer, four seeds are trained on that grid and the higher 8-bit PSNR is kept. Ranking seeds by the step-0 score, without training them, dropped the 8 bpp metal_plate point from 31.574 dB to 31.400 dB, so the pool trains each seed.

Initialization is coarse-to-fine PCA of the residual into each grid, then a least-squares fit of the last layer. Adam then runs for 40 steps at learning rate `5e-3` (latent parameters at `1e-2`) with a cosine decay to 5% of the base rate. After training, the export compares the initialization against the trained weights and keeps the higher 8-bit PSNR.

Loss is mean squared error. Channel weights and a normal-cosine term exist and are off by default.

Defaults: hidden `(32,)`, ReLU, no positional encoding, int8 weights, one scale per row, straight-through latent and weight quantization, `adaptive_pool = 8`, `seed_pool = 4`, 40 steps, `channel_weighting = false`.

## Measurements

Machine: Linux 6.12.94+, x86_64, glibc 2.39, torch 2.14.1+cpu, 4 threads. No GPU. Every quality number below is from a run on that machine. JSON in `results/` holds the unrounded values.

PSNR is `10 log10(1 / mse)` after `round(clip(x, 0, 1) * 255) / 255`, capped at 100. SSIM is Wang 2004, 11x11 Gaussian, sigma 1.5, `k1 = 0.01`, `k2 = 0.03`, on those 8-bit values. FLIP is not installed. `try_flip` returns nothing here.

Block codecs on metal_plate resized to 256 with a box filter. BC7 is `bc7enc -l -u4`. The 16 bpp bundle is BC1 albedo, BC5 normal, BC1 of roughness/metalness/ao, `bc7enc -1/-5 -u5`, normal Z reconstructed from XY. ASTC is `astcenc 5.7.0 -cl -medium`.

| codec | bpp | PSNR | SSIM |
| --- | --- | --- | --- |
| NTC1, target 4 | 4.076 | 28.479 | 0.7737 |
| NTC1, target 8 | 8.080 | 31.574 | 0.8523 |
| NTC1, target 10 | 10.080 | 32.174 | 0.8532 |
| NTC1, target 12 | 12.072 | 32.260 | 0.8645 |
| NTC1, target 16 | 16.084 | 35.749 | 0.9474 |
| NTC1 linear, target 16 | 16.016 | 34.391 | 0.9181 |
| BC1+BC5+BC1 | 16.000 | 33.199 | 0.9258 |
| ASTC 8x8, five textures | 10.000 | 34.072 | 0.9282 |
| ASTC 6x6, five textures | 17.778 | 38.275 | 0.9647 |
| ASTC 4x4, five textures | 40.000 | 45.794 | 0.9906 |
| BC7, five textures | 40.000 | 44.867 | 0.9900 |

Chart: `results/rd_metal_plate_256.png`.

At 16.084 bpp the shipped network is grids `256x256x4@4` and `8x8x2@2`, hidden 32, seed 3, 480 MACs, 1122 flops. Per-channel 8-bit PSNR: albedo 37.018, normal 39.459, roughness 37.525, metalness 31.595, ao 32.293. Normal angle 1.769 degrees. Against the 16 bpp block bundle that is +2.550 dB PSNR and +0.0216 SSIM. It does not beat ASTC 6x6 (38.275 dB at 17.778 bpp) or per-texture BC7 (44.867 dB at 40 bpp). At 10 bpp it does not beat ASTC 8x8 (32.174 vs 34.072).

The linear network on the same grids keeps the PCA initialization (40 Adam steps scored 34.348 dB and were discarded). 54 MACs, 174 flops, 90 weight bytes. It is +1.192 dB PSNR over the 16 bpp bundle and -0.0077 SSIM.

Half-resolution family, hidden 32, channel weighting on, 40 steps, same material (`results/rd_metal_plate_256.json`): 30.734 dB at 16.105 bpp, SSIM 0.8960, grids no larger than 128. The shipped point is +5.015 dB and +0.0514 SSIM at the same rate. That gap is not one factor. The isolated factors are below.

metal_plate 16 bpp, one encode, this CPU (`results/encode_time_metal16.json`): wall 8.833 s, of which the four trained seeds sum to 2.309 s and the winning seed is 0.506 s. The rest is the eight step-0 grid probes, export, and SSIM. CUDA and ROCm encode times are pending hardware.

PavingStones070 resized to 256, same encoder. The material has albedo, normal, roughness, ao, and height, and no metalness.

| codec | bpp | PSNR | SSIM |
| --- | --- | --- | --- |
| NTC1, target 8 | 8.076 | 28.109 | 0.8242 |
| NTC1, target 16 | 16.084 | 31.688 | 0.9117 |
| ASTC 8x8, five textures | 10.000 | 30.981 | 0.8840 |
| ASTC 6x6, five textures | 17.778 | 33.727 | 0.9368 |
| ASTC 4x4, five textures | 40.000 | 38.766 | 0.9802 |
| BC7, five textures | 40.000 | 37.604 | 0.9727 |
| BC1+BC5+BC1, height left at 0 | 16.000 | 12.840 | 0.8079 |

The bundle aggregate is not a matched-channel score: height is 3.382 dB because the bundle does not store it. On the eight channels the bundle writes, mean-square PSNR is 29.481 dB for the bundle and 32.139 dB for NTC1. NTC1 also stores height at 29.159 dB inside the same 16.084 bpp. Per group, NTC1 albedo is 31.449 vs the bundle's 32.659, NTC1 normal is 32.298 vs 26.513 (the bundle's reconstructed Z is 21.834 dB), and NTC1 roughness is 33.382 vs 34.677.

### Isolated deltas

Locked grids `64x64x4@2` and `16x16x12@8`, metal_plate 128, 40 steps, hidden 32, weighting on, seed 0, one seed (`results/ablation_locked_128.json`). Baseline 26.584 dB, SSIM 0.7536, 3.986 bpp, 800 MACs.

| change | PSNR | delta vs baseline |
| --- | --- | --- |
| PCA off | 21.122 | -5.462 |
| latent QAT off | 24.431 | -2.153 |
| uniform-noise proxy instead of STE | 26.143 | -0.442 |
| weighting off | 27.062 | +0.478 overall, normal 31.055 vs 31.790, angle 4.504 vs 4.006 degrees |
| Fourier, 1 octave | 26.299 | -0.285, 928 MACs, 4.049 bpp |
| triangle, 3 octaves | 24.793 | -1.792, 1248 MACs, 4.205 bpp |
| linear readout | 26.600 | +0.016, 3.604 bpp, 144 MACs, 434 flops |
| hidden 16 | 26.462 | -0.123, 400 MACs |
| hidden 48 | 27.158 | +0.573 at 4.213 bpp, 1200 MACs |
| hidden 64-32 | 23.509 | -3.075, 5.361 bpp, 3360 MACs |
| hard GELU | 26.583 | -0.002 |
| hard GELU plus activation requant | 11.363 | -15.221 |
| int4, group 16 | 26.008 | -0.576, 3.800 bpp |
| int8, group 16 | 26.677 | +0.092, 3.995 bpp |
| joint loss on 4 mips | 25.975 | -0.609 at mip 0, mip 2 26.057 vs baseline derived mip 2 of 23.045 |
| 0 Adam steps | 26.139 | -0.445 |

Grid search, not a same-grid ablation (`results/ablation_metal_plate_128.json`): the single closest shape at target 4 bpp scored 24.083 dB (1056 MACs). Scoring a pool of 4 and keeping the best scored 27.155 dB at 3.986 bpp (800 MACs), grids `64x64x4@2` and `16x16x12@8`. Delta +3.072 dB.

Full-resolution grids and the wider pool, metal_plate 256, target 16, hidden 32, weighting off:

| configuration | bpp | PSNR | SSIM | MACs |
| --- | --- | --- | --- | --- |
| half-resolution family, weighting on | 16.105 | 30.734 | 0.8960 | see `rd_metal_plate_256.json` |
| winning grids, linear, init kept | 16.016 | 34.391 | 0.9181 | 54 |
| winning grids, hidden 32, seed 0, 40 steps | 16.084 | 35.240 | 0.9331 | 480 |
| same, 80 steps, seed 0 | 16.084 | 35.354 | 0.9370 | 480 |
| same, seed 0, channel weighting on | 16.084 | 34.614 | 0.9254 | 480 |
| shipped: pool 8, four trained seeds, seed 3 | 16.084 | 35.749 | 0.9474 | 480 |

Seed 0 to the four-seed pool on those grids is +0.509 dB and +0.0143 SSIM. The other seeds scored 35.306, 35.287, and 35.240. Eighty steps on seed 0 gained +0.114 dB over forty steps (0.933 s vs 0.468 s of that seed) and still lost to seed 3 at forty steps. Weighting on lost 0.626 dB and 0.0076 SSIM against seed 0 unweighted.

Same-grid hidden 32 vs linear at seed 0 is +0.849 dB and +0.0150 SSIM for 480 MACs instead of 54. The shipped seed is a further +0.509 dB on the hidden network. The linear result does not depend on the hidden-layer seed.

## Negative results

Kept off, with the delta above:

- Uniform noise as a quantizer proxy. STE is higher.
- Fourier and triangle positional encoding. Both lost PSNR and added MACs.
- Hard GELU. No gain. Activation requant of the RTXNTC form (`round(z * 80 - 113)` to int8) fell to 11.363 dB.
- int4 weights. -0.576 dB on the locked 128 grid.
- int8 group size 16. +0.092 dB on that grid. The shipped linear and 32-wide nets use one scale per row. Group 16 was not adopted.
- Joint training of four mips. Mip 0 fell 0.609 dB. Mip 2 rose about 3.0 dB. Derived mips stay the format.
- Hidden 64-32 and the triangle-PE, hard-GELU, requantized 64-48-32 stack. The latter, trained here, scored 16.614 dB at 7.143 bpp and 6816 MACs on the locked 128 grids, and 18.030 dB at 4.006 bpp when the search gave it small grids. That stack is this encoder running a decoder shape from the RTXNTC paper. It is not `ntc-cli` output.
- Channel weighting and the normal-cosine term. On the 128 locked grid, weighting off raised overall PSNR 0.478 dB and SSIM, and lowered normal PSNR 0.734 dB. On the 256 winning grids at seed 0, weighting on scored 34.614 vs 35.240 unweighted. Normals are not separately protected in the default loss.
- Dedicating the finest grid's PCA to normals, or splitting normals and the other channels across grids (`results/shape_scan_256.json`). On the 16 bpp linear grids, normals-only fine PCA reached normal PSNR 40.47 and overall PSNR 30.53, against joint PCA at 30.92 normal and 33.28 overall. Roughness collapsed. Not adopted.
- A third grid. The best three-grid shape at step 0 was 33.310 dB at 16.021 bpp, against 33.276 dB for the two-grid linear init. SSIM was lower. The family stays at one or two grids.
- Training a linear map for 40 steps. On the winning grids the init is 34.391 dB and the trained weights are 34.348 dB. The export keeps the init.
- bf16 exponent coding of the weights, as above. Larger than int8 on the matrices that were measured.
- Fusing latent scales into the int8 matrix. Not exact. Not implemented.
- A subgroup or quad GEMV split of one texel. The MLP is 480 MACs. The shader stays one thread per texel. The 8x8 unique-byte count is analytical only.

## Correctness on this VM

CPU reference vs the Python decoder on `tests/fixtures/tiny.ntc`: max abs 1.490e-8 at mip 0 and 3.725e-9 at mip 1.

Lavapipe (Mesa 25.2.8, vendor `0x10005`, device type CPU, subgroup size 8, `shaderFloat16` and `shaderInt8` present, cooperative matrix absent) compute vs the CPU reference: max abs 0. The fp16 dot path: max abs 1.345e-4. `bench/gpu_bench.py` on this machine recorded a lavapipe timestamp of 1.277 ms (`results/gpu.json`). That is not a performance number.

OpenGL 4.5 llvmpipe, compute and fragment vs the CPU reference: max abs 0. `GL_KHR_shader_subgroup` is not present, so the reported subgroup size is 0. The 4.6 context request returns `EGL_BAD_MATCH`. The shaders are GLSL 450 and do not use a 4.6-only feature.

`pytest` covers the container, the quantizer, the reference decoder, filtering, and the cost model. `ctest` runs the C++ decoder against the fixture. glslang compiles the SPIR-V. CI is `.github/workflows/ci.yml`.

## What is pending hardware

- GPU milliseconds on AMD and NVIDIA, for inference-on-sample and inference-on-load. Lavapipe numbers stay correctness-only.
- Official RTXNTC `ntc-cli` PSNR and encode time at matched bpp. Compression needs an NVIDIA Turing-or-newer GPU and CUDA. Decompression of an existing file can pass `--no-coopVec` on other vendors. `bench/gpu_bench.py` records the skip reason and does not fail the run. `bench/compare_runs.py` prints two result files side by side.
- CUDA and ROCm encode times. The encoder will use those devices when `torch.cuda.is_available()` is true. This VM is CPU only.

## Limitations

The 10 bpp point loses to ASTC 8x8 on metal_plate. The 16 bpp point loses to ASTC 6x6 and to per-texture BC7. A single fp16 scale per latent channel has no spatial endpoints. Derived mips are a box filter, not a second trained pyramid. Inference-on-load writes fp32. Transcode to BC7 or BC5 is host-side through bc7enc. There is no portable GPU BC7 encoder in this tree. The seed pool trains four copies of the chosen grid.

## Prior work

Vaidyanathan, Salvi, Wronski, Akenine-Möller, Ebeida, Lefohn. Random-Access Neural Compression of Material Textures. 2023. https://research.nvidia.com/labs/rtr/neural_texture_compression/

RTXNTC: https://github.com/NVIDIA-RTX/RTXNTC . License is the NVIDIA RTX SDKs license, not MIT. No shader or binary from that tree is in this repository.

## Layout

| path | role |
| --- | --- |
| `src/ntc` | encoder, container, reference decoder, metrics |
| `cpp` | C++20 reader and reference decoder |
| `shaders` | GLSL library and entry points |
| `examples` | Vulkan 1.3 and OpenGL hosts |
| `bench` | block codecs, ablations, GPU script |
| `third_party/expcode` | vendored bf16 exponent codec, not on the decode path |

Pieces that can be split later without changing the format: the NTC1 container and loader, the GLSL decode library, the group quantizer (it could move into weight-quantizer), and the CPU reference decoder.

Build: CMake, C++20, `glslangValidator`. Python package `ntc` in `pyproject.toml`, dependencies numpy, torch, pillow. `ruff` and `pytest` for the Python tree.
