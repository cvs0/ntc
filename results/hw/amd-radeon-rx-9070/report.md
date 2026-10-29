| field | value |
| --- | --- |
| os | Windows-11-10.0.26300-SP0 |
| cpu | AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD |
| gpu | AMD Radeon RX 9070 |
| vendor_id | 0x1002 |
| device_id | 0x7550 |
| driver | 8389024 |
| vulkan_api | 1.4.359 |
| subgroup_size | 64 |
| opengl | AMD Radeon RX 9070 |
| opengl_version | 4.6.0 Core Profile Context 26.10.1.261008 |
| rtxntc | ntc-cli is not on PATH. Compression in NVIDIA-RTX/RTXNTC requires an NVIDIA Turing-or-newer GPU and CUDA. Decompression of an existing file can pass --no-coopVec. This run did not invoke it. |
| astc | astcenc is not on PATH. ASTC screenshot panels are skipped. The rate table still uses the earlier astcenc 5.7.0 numbers. |
| bc_encoder | bc_pack.py BC7 mode 6, BC1, BC5. Not bc7enc. |

| material | api | pass | median_ms | p95_ms | texels/s | GB/s | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metal_plate | vulkan | load_fp32 | 0.16036 | 0.165 | 544899000.0 | 19.6164 |  |
| metal_plate | vulkan | load_fp16 | 0.16512 | 0.16808 | 529191000.0 | 19.0509 |  |
| metal_plate | vulkan | sample_native_fp32 | 0.11988 | 0.12204 | 546680000.0 | 19.6805 |  |
| metal_plate | vulkan | sample_1920x1080_fp32 | 2.86832 | 2.87564 | 722932000.0 | 26.0255 |  |
| metal_plate | vulkan | sample_3840x2160_fp32 | 11.2448 | 11.2531 | 737624000.0 | 26.5544 |  |
| metal_plate | vulkan | bc7_1920x1080 | 0.942 | 0.94552 | 2201270000.0 | 35.2204 |  |
| metal_plate | vulkan | bc7_3840x2160 | 4.42496 | 4.4302 | 1874460000.0 | 29.9913 |  |
| metal_plate | vulkan | bc1_1920x1080 | 0.94224 | 0.94416 | 2200710000.0 | 35.2114 |  |
| metal_plate | vulkan | bc1_3840x2160 | 4.42496 | 4.42988 | 1874460000.0 | 29.9913 |  |
| metal_plate | vulkan | bc5_1920x1080 | 0.94264 | 0.94656 | 2199780000.0 | 35.1965 |  |
| metal_plate | vulkan | bc5_3840x2160 | 4.42484 | 4.42916 | 1874510000.0 | 29.9921 |  |
| metal_plate | opengl | load_fp32 | 0.14068 | 0.14336 | 621126000.0 | 22.3605 |  |
| metal_plate | opengl | load_fp16 |  |  |  |  | no_gl_fp16 |
| metal_plate | opengl | sample_native_fp32 | 0.1086 | 0.11048 | 603462000.0 | 21.7246 |  |
| metal_plate | opengl | sample_1920x1080_fp32 | 2.86812 | 2.87412 | 722982000.0 | 26.0274 |  |
| metal_plate | opengl | sample_3840x2160_fp32 | 11.3399 | 11.3606 | 731436000.0 | 26.3317 |  |
| metal_plate | opengl | lit_native_fp32 | 0.09148 | 0.09324 | 716397000.0 | 8.59676 |  |
| metal_plate | opengl | bc7_1920x1080 | 0.94904 | 0.9528 | 2184940000.0 | 34.9591 |  |
| metal_plate | opengl | bc7_3840x2160 | 4.43472 | 4.4374 | 1870330000.0 | 29.9253 |  |
| metal_plate | opengl | bc1_1920x1080 | 0.94692 | 0.95224 | 2189840000.0 | 35.0374 |  |
| metal_plate | opengl | bc1_3840x2160 | 4.4348 | 4.44084 | 1870300000.0 | 29.9248 |  |
| metal_plate | opengl | bc5_1920x1080 | 0.94708 | 0.95128 | 2189470000.0 | 35.0315 |  |
| metal_plate | opengl | bc5_3840x2160 | 4.43424 | 4.4372 | 1870530000.0 | 29.9286 |  |
| PavingStones070 | vulkan | load_fp32 | 0.15912 | 0.16284 | 549145000.0 | 19.7692 |  |
| PavingStones070 | vulkan | load_fp16 | 0.16344 | 0.16724 | 534630000.0 | 19.2467 |  |
| PavingStones070 | vulkan | sample_native_fp32 | 0.11876 | 0.12088 | 551836000.0 | 19.8661 |  |
| PavingStones070 | vulkan | sample_1920x1080_fp32 | 2.87604 | 2.88848 | 720991000.0 | 25.9557 |  |
| PavingStones070 | vulkan | sample_3840x2160_fp32 | 11.2672 | 11.277 | 736152000.0 | 26.5015 |  |
| PavingStones070 | vulkan | bc7_1920x1080 | 0.94236 | 0.94404 | 2200430000.0 | 35.2069 |  |
| PavingStones070 | vulkan | bc7_3840x2160 | 4.42512 | 4.42796 | 1874390000.0 | 29.9902 |  |
| PavingStones070 | vulkan | bc1_1920x1080 | 0.94208 | 0.9454 | 2201090000.0 | 35.2174 |  |
| PavingStones070 | vulkan | bc1_3840x2160 | 4.42532 | 4.42736 | 1874310000.0 | 29.9889 |  |
| PavingStones070 | vulkan | bc5_1920x1080 | 0.94268 | 0.94464 | 2199690000.0 | 35.195 |  |
| PavingStones070 | vulkan | bc5_3840x2160 | 4.42532 | 4.42784 | 1874310000.0 | 29.9889 |  |
| PavingStones070 | opengl | load_fp32 | 0.13836 | 0.1424 | 631541000.0 | 22.7355 |  |
| PavingStones070 | opengl | load_fp16 |  |  |  |  | no_gl_fp16 |
| PavingStones070 | opengl | sample_native_fp32 | 0.10848 | 0.11648 | 604130000.0 | 21.7487 |  |
| PavingStones070 | opengl | sample_1920x1080_fp32 | 2.86964 | 2.87748 | 722599000.0 | 26.0136 |  |
| PavingStones070 | opengl | sample_3840x2160_fp32 | 11.3469 | 11.3653 | 730983000.0 | 26.3154 |  |
| PavingStones070 | opengl | lit_native_fp32 | 0.08248 | 0.08376 | 794568000.0 | 9.53482 |  |
| PavingStones070 | opengl | bc7_1920x1080 | 0.95284 | 0.95788 | 2176230000.0 | 34.8197 |  |
| PavingStones070 | opengl | bc7_3840x2160 | 4.43172 | 4.436 | 1871600000.0 | 29.9456 |  |
| PavingStones070 | opengl | bc1_1920x1080 | 0.95308 | 0.95904 | 2175680000.0 | 34.8109 |  |
| PavingStones070 | opengl | bc1_3840x2160 | 4.43168 | 4.43736 | 1871620000.0 | 29.9458 |  |
| PavingStones070 | opengl | bc5_1920x1080 | 0.9542 | 0.95832 | 2173130000.0 | 34.7701 |  |
| PavingStones070 | opengl | bc5_3840x2160 | 4.4316 | 4.43584 | 1871650000.0 | 29.9464 |  |
