# Decoder variants

The bit-exact reference is `ntc::decode_mip` and `ntc.decode_ref.decode_mip`. Every GPU variant is checked against it before it is timed. `ntc_decbench` exits 3 when there is no Vulkan 1.3 device, and ctest treats that as a skip.

## How to run

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ntc_decbench
./build/ntc_decbench --check-only --shader-dir build/shaders tests/fixtures/tiny.ntc
./build/ntc_decbench --sizes 1920x1080,3840x2160 --variants fma,fp16,bc7 \
  --shader-dir build/shaders path/to/file.ntc --json results/decbench/out.json
```

macOS, after `brew install cmake ninja glslang molten-vk vulkan-loader vulkan-headers`. MoltenVK 1.4 puts the ICD at `$(brew --prefix)/etc/vulkan/icd.d/MoltenVK_icd.json`. If `vulkaninfo` finds no driver, set `VK_ICD_FILENAMES` to that file.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(brew --prefix)"
cmake --build build
./build/ntc_decbench --check-only --shader-dir build/shaders tests/fixtures/tiny.ntc tests/fixtures/ctx.ntc
```

The hosts set `MVK_CONFIG_FAST_MATH_ENABLED=0` before the ICD loads. On MoltenVK 1.4 that value means never use Metal fast-math. The exact variant can still differ by one rounding. `decbench` passes the exact check on a portability device when max abs is `<= 1e-6`, and requires 0 diff on native Vulkan. `ntc_vk` checks fp16 at max abs 0.02 and at least 35 dB. A pipeline cache is written under `~/.cache/ntc`. `ntc_gl` is not built on macOS.

`--variants` is a comma list. An empty list runs every name below. A missing extension skips that variant and does not fail the process, except `exact` and `legacy`, which must match the CPU.

## Shader variants

| Name | Flag or extension | What it checks | When it is skipped |
| --- | --- | --- | --- |
| `legacy` | `shaders/ntc_decode.glsl` | Interpreter path. Must be within 1e-4 of the CPU. | Never, if the pipeline builds. |
| `exact` | `NTC_FAST_EXACT` | fp32 with `precise` mul and add. 0 diff on native Vulkan. On MoltenVK, max abs `<= 1e-6`. | Never, if the pipeline builds. A miss fails ctest. |
| `fma` | default fast shader | fp32 with contraction. Must be within 1e-4. | Never, if the pipeline builds. |
| `fp16` | `NTC_FAST_FP16` | Packed fp16 fma. PSNR is reported, not a fail. | `shaderFloat16` absent. |
| `i8` | `NTC_FAST_I8` | Packed int8 dot, int32 sums. PSNR is reported. | `shaderIntegerDotProduct` absent, or the SPIR-V is missing because glslang has no `GL_EXT_integer_dot_product`. |
| `i16` | `NTC_FAST_I8` and `NTC_FAST_I16` | 16-bit codes, two dots per four weights. | Same as `i8`. |
| `bc7` | hardware BC | One mode-6 BC7 texture, one bilinear fetch. | `textureCompressionBC` absent. |
| `bc_mat` | hardware BC | BC7 albedo, BC5 normal, BC7 ORM. | `textureCompressionBC` absent. |
| `coopmat_i8` | `VK_KHR_cooperative_matrix` | Integer 16x16 tile product. The CPU reference is `src/ntc/coopmat_i8.py`, which matches a plain int32 gemm. | Extension absent (MoltenVK, lavapipe). When the extension is present the harness still does not bind the scratch buffer, so it does not dispatch. |

## Host variants

These run in the process and print `PASS`. They do not time a dispatch. The same checks live in `tests/test_plan_rest.py` and `src/ntc/plan_rest.py`.

| Name | Idea | What the check shows | What is not measured |
| --- | --- | --- | --- |
| `sparse` | 11, prune and 2:4 | Every group of four keeps the two largest magnitudes. `prune_keep` on a train job keeps that fraction at export. | No b5 PSNR sweep. No sparse matrix-unit timing. |
| `i0int` | 22, integer layer 0 | Lerp is `(a*(denom-numer)+b*numer+denom/2)//denom`. The int32 dot matches that product. | Not the GPU MLP. |
| `matryoshka` | 10 | Prefix widths 8, 16, 32. A zero tail matches the full layer on the kept rows. MACs scale with width. Training with `matryoshka` zeros units past 16 for the whole run. | 0.2 dB full-width loss on the four materials. |
| `dedupe` | 14 | At 4x zoom, 8 of 32 lanes share a texel. At 0.5x, all 32 are unique. | No subgroup ballot. |
| `chanmask` | 20 | A 10 by 32 last layer is 320 MACs. Four rows are 128. | No frame time. |
| `shaderout` | 21 | Octahedral xy round-trips a unit normal above cosine 0.98. Roughness is stored squared. | No retrained material. |
| `simd8` | 13 | A blocked int8 dot equals the scalar dot. The code is portable C++. | No NEON or AVX2 timing on an M5 Pro or a Windows PC. |
| `tilebc7` | 23 | A 16x16 RGBA8 tile is 1024 bytes. BC7 of that tile is 256 bytes, 4 times smaller. | `tile_cache_steady` milliseconds on an RX 9070. The live atlas in `ntc_tile_fill.comp` is still RGBA8. |
| `stream` | 24 | Coarse-versus-full PSNR on a ramp is finite. | The container is not yet ordered coarse grid first, and a decode with grid 0 missing is not a file mode. |

## Other CPU paths

These are in `src/ntc/decode_ship.py` and `tests/test_decode_ship.py`. They do not have their own `ntc_decbench` shader.

| Path | Flag or entry | Measured | Not measured |
| --- | --- | --- | --- |
| High-rate lattice | `grow_grids(..., prefer_resolution=True)` from b10 | A 32 lattice at raw 20 bpp on a 64 image becomes 64x4 plus 32x4. | New 2K PSNR. |
| BC latent planes | `FLAG_BC_LATENTS` (bit 7) | The bit round-trips. Plain codes stay. Mode 6 on an 8x8 ramp is within 2 bins, about 28 dB. | Texture-unit fetch. BC in the training loop. |
| Deferred tiles | `classify_tiles` | An 8x4 tile is pure or mixed. | 4K frame time. |
| BC7 mode hints | `mode_hint_bytes` | 4 bits per 4x4 block is 0.25 bpp. The encoder is mode 6. | GPU transcode time. |
| Hybrid tail | `hybrid_tail_bpp` | Three BC7 textures from mip 2 down add 2.00 bpp. | `ntc_view` frame time and a seam image. |
| Hoisted layer 0 | `decode_mip_hoisted` | Matches `decode_mip` within 1e-4 on an 8x8 net with no positional encoding. | Load-time GPU timing. |
| Weight immediates | `emit_immediate_mlp` | Exact-zero weights are omitted and the product matches `mlp`. | Pipeline compile time on LLPC, Mesa, or MoltenVK. |
| Hidden width | `hidden_knee` | Smallest width within 0.15 dB of the best row. | A 2K sweep. Four widths, four materials, 4000 steps at 20 steps/s is about 0.9 GPU-hours. |
| Renderer metrics | `render_loss`, `group_metrics` | SSIM and normal angle. `flip_approx` when the FLIP package is absent. | FLIP against BC7 x3 on the four 2K materials. |
| BC residual | `bc_residual` | Quarter-res base plus the residual restores the image within 1e-5. | Equal-bpp 2K PSNR against plain ntc. |
| VQ latents | `vq_latents` | A repeated tile's index rate is under 4 bpp at 2 codes. | Tiles135D and PavingStones070 at b5 and b10. |
| Shared decoder | `shared_decoder` | One weight matrix applies to two feature batches. | PSNR loss on the four materials. |
| Per-tile linear head | `tile_head` | A flat field scores a linear fraction of 1 within 0.5 dB. | Fractions on real materials. |

## NTC versus BC7

These are the B300 models already on disk and the BC7 baseline from `bc7enc_rdo` on the same 2K materials, all channels, mip 0. They are not a retrain of this branch. BC7 x3 is 24 bits per texel. The lean bundle is BC1 plus BC5 plus BC1, 16 to 20 bits.

| Material | NTC near 1 bpp | NTC near 5 bpp | NTC best in that run | BC7 x3 at 24 bpp | Lean BC |
| --- | --- | --- | --- | --- | --- |
| MetalPlates013 | 32.92 at 0.98 | 37.48 at 4.99 | 39.42 at 9.21 | 43.99 | 39.82 at 20 |
| Bark001 | 22.22 at 0.99 | 27.90 at 4.91 | 27.90 at 4.91 | 33.06 | 33.07 at 16 |
| PavingStones070 | 24.45 at 1.0 | 27.00 at 4.36 | 27.00 at 4.36 | 35.30 | 31.54 at 16 |
| Tiles135D | 29.56 at 1.0 | 34.62 at 4.6 | 34.62 at 4.6 | 41.73 | 32.17 at 16 |

At about 1 bpp, NTC is 1.7 to 2.9 dB above quarter-resolution BC7 x3 (1.5 bpp: Metal 30.51, Bark 20.63, Paving 23.36, Tiles 28.38). At about 5 bpp it is 2.0 to 5.7 dB above half-resolution lean BC. It does not match native BC7 x3.

GPU rows that do exist, from c33648b on an RX 9070, are a compute dispatch into a storage buffer, not a color attachment: NTC 11.245 ms at 4K, BC7 4.425 ms, gap 6.82 ms. The paper's RTX 4090 fragment figures (1.15 to 1.92 ms NTC, 0.49 ms BC7) are a different decoder and a different pass.
