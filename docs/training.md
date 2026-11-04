# Training pipeline

`bench/train_max_v2.py` is the resumable queue. `TrainConfig.fast` selects `encode_arrays_fast` in `src/ntc/train_fast.py`. `bench/train_max.py` leaves `fast` off and is a different output directory.

## What one step does

1. Latent grids and mip targets live on the training device. `device_pipeline` draws crop and scatter indices on that device. With the flag off, those indices are drawn on the CPU and copied.
2. The gather is bilinear in feature space. `fused_gather=auto` tries a Triton kernel on CUDA, including the backward. CPU and MPS stay on the eager gather.
3. The MLP is one `bmm` per layer for the whole seed pool. `fused_mlp=auto` uses the compiled MLP when compile is on, cuBLAS for an eager CUDA MLP, and the eager `bmm` on CPU and MPS.
4. Adam updates latents and weights. Codes are clamped into the bin range after the step.
5. The step itself does not call `.item()`. A progress line does, every `log_every` steps.

`torch.compile` runs when `compile_step` is `auto` and the device is CUDA. CPU and MPS stay eager. The CUDA compile mode is `reduce-overhead`, which is a CUDA graph of the step. A manual graph is used when `cuda_graph` is `auto` and compile is off. The learning rate is written into a device tensor before each replay, so `warmup_cosine` does not freeze the first step's rate. `cuda_graph=off` disables both.

bf16 and fp16 autocast wrap the GEMM. Latents, master weights, and the loss stay fp32. TF32 is on for those reduced precisions and off for an fp32 run.

## Schedule, batch, and stop

`schedule=warmup_cosine` ramps the learning rate over `warmup_steps`, then cosine down to 5 percent of the base rate. `schedule=cosine` with `warmup_steps=0` is the older curve that starts at the base rate. `schedule=onecycle` rises for 30 percent of the run and cosines down. `schedule=step` halves at 50 percent and again at 75 percent.

`lr_scale_rule` multiplies the learning rate by `batch / lr_batch_ref` (`linear`), by the square root of that (`sqrt`), or by 1 (`off`). `auto_batch=vram` may grow the batch on a CUDA card of at least 10 GB, at most 4 times the request and at most one image, and may only shrink a smaller card. CPU and MPS keep the request.

`plateau_patience` and `plateau_min_delta` stop a rate when eval PSNR stops clearing the best by that many dB. Patience 0 keeps `early_stop` and a 0.01 dB delta. A score still within 4 dB of the init does not increment patience.

## Rate and lattice

`rate=latent` sizes grids from codes and latent scales, not from the MLP. A label whose `latent_rate_scale` is 0 measures stored/plain on a half-resolution 4-bit probe and searches at label divided by that ratio. From 10 bpp up, a warm start doubles spatial size before it adds channels, then allows 8-bit. The label's own lattice is kept when it already holds the rate, and the lower checkpoint is resampled into it.

`rd_alloc` moves 4 channels at a time from a low-variance image group onto a high-variance one. Grids keep at least 2 channels, and the move stays within 1.15 times the rate.

A result json with a non-finite score, a collapse of more than 8 dB under an init above 20 dB, or a file that does not decode is an error. The queue does not train that label again. `INCOMPLETE` is replaced by `ALL_DONE` only when every planned json is terminal.

## Quality flags

| Flag | Default | Cloud | Effect |
| --- | --- | --- | --- |
| `render_loss` | off | on | Adds a Lambert shade MSE when albedo and a normal are in the file. The export records `flip_approx` (luma and chroma). NVIDIA FLIP is used only when that package imports. |
| `bc_residual` | off | on | The MLP trains on the residual above a quarter-resolution box base. The logged PSNR and the saved PSNR both add that base back before the compare. |
| `vq_latents` | off | off | A 16-entry snap at export dropped the file by more than 0.5 dB on a short CPU stand-in, and run 3's saved scores did not rise with bitrate. The flag still exists. The export keeps the trained codes when the snap would miss by more than 0.5 dB. |
| `tile_head` | off | on | Records the fraction of 8x8 tiles whose quarter-res base is within 0.5 dB of the full reconstruction. It does not change the file. |
| `shared_decoder` | off | on | Packed jobs already share one MLP across materials. A single-material chain records the flag and still trains its own MLP. |
| `prune_keep` | 1 | 1 | Run 3 used 0.5 and the file fell about 13 dB under the loop on a stand-in that had beaten its init. The export now refuses a prune that misses the trained file by more than 0.5 dB. |
| `matryoshka` | off | on | Zeros hidden units past 16 for the whole run. The export writes those rows and the next layer's columns as zeros, so the file matches the mask. |
| `ema_decay` | 0 | 0.999 | The exported weights are the EMA. |
| `mlp_int8_tail` | 0 | 0.90 | The last 10 percent of steps straight-through quantize layer inputs to int8, and the file stores the scales. |

## Profiles

| Profile | Jobs | Batch | Notes |
| --- | --- | --- | --- |
| `laptop` | the 256 and SDK list | 0 on 256 jobs, 262144 area on the SDK 2K list | Unchanged by run 3. |
| `cloud5090` | 12 materials times 10 rates | 2097152 flat | 12.5 GPU-hours at the 20 steps/s budget. Historical queue. |
| `cloudfast` | MetalPlates013, PavingStones070, Tiles135D, Bark001, labels 1, 2.5, 5, 10, 20 | 2097152 flat | Run 3. 20 jobs. |

Labels at 10 bpp and above train 8000 steps. Lower labels train 4000, warm-started from the cheaper neighbor. The seed pool is 3 for the first 500 steps, then halved. `quant_freeze_at` is 0.90.

## Run 3

```bash
sh bench/launch_run3.sh
```

That prints the budget and starts `results/run4`:

```bash
NTC_PROFILE=cloudfast NTC_MAX_GPU=1 NTC_EXIT_WHEN_DONE=1 \
  python bench/train_max_v2.py --profile cloudfast \
  --materials data/materials --out results/run4 --max-gpu --exit-when-done
```

`--max-gpu` starts one chain worker per material when the GPU reports at least 80 GB. Four workers is the `cloudfast` width. They share one card, so they do not multiply the step rate.

Run 3 on a B300 (main `a7221a7`) logged about 36 dB during training and saved 19 to 29 dB. The live line was the residual forward, and the file then applied a 50 percent prune, a 16-code snap, and the unmasked Matryoshka tail. A finished checkpoint was also deleted when the next attempt used a smaller grid. Those exports are rejected or baked now, and a job whose saved `psnr_u8` is more than 0.5 dB under `psnr_u8_loop` aborts the queue.

The launch estimate uses the measured B300 rate, **4.5 steps/s** (the run sat between 4 and 5). Workers share one card, so the wall clock is the full queue at that rate: **9.88 h** for 20 jobs before early stop. At 4 steps/s that is about 11 h, and at 5 steps/s about 9 h. The old 20 steps/s figure (2.22 GPU-hours, 0.56 h wall) was the 5070 laptop budget and is not the B300. `python bench/repro_export_gap.py` is the short CPU toggle of each flag.

Held out of the cloud lists: Leather011, LeafSet017, Gravel022.

## Run 4 quick

MetalPlates013 and Tiles135D at b5 and b10, four jobs, same cloud flags (`prune_keep` 1, `vq_latents` off). b5 is 4000 steps from scratch. b10 is 8000 steps and warm-starts from that b5 file. Two workers share one card.

```bash
sh bench/launch_run4_quick.sh
```

At 4.5 steps/s that is **1.85 h** wall. At 4 steps/s about 2.1 h, and at 5 steps/s about 1.7 h. The guard passed when each `finish` line has `psnr_u8` within 0.5 dB of `loop` and the log has no `ABORT` line. A miss writes `results/run4_quick/STOP_EXPORT_GAP` and stops the other jobs.

## Profile a step

```bash
python bench/profile_train_step.py --device cpu --steps 4 --warmup 2
python bench/profile_train_step.py --device mps
python bench/profile_train_step.py --device cuda
```

The script prints JSON with `before` (compile off, graph off, eager gather, cosine, host index draws) and `after` (the run 3 step switches that the device can actually run). Fields:

| Field | Meaning |
| --- | --- |
| `steady_steps_per_s` | Timed loop after warmup. Compile, when it runs, is outside this number. |
| `steady_step_ms` | 1000 divided by that rate. |
| `profiler_step_ms` | Self time split into `sampling`, `forward`, `backward`, `optimizer`, `other`, plus the inclusive `ntc_train_step_ms`. |
| `plan` | Which backend the device selected. |
| `compiled`, `cuda_graph` | What the step actually used. |

On this CPU, two runs of `--steps 4` were 3.27 ms and 3.46 ms eager, then 3.75 ms and 3.22 ms with the run 3 switches. The ratio is inside the noise of a 3 ms step. Self time on the second run was about 0.02 ms sampling, 0.21 ms forward, 0.35 ms backward, and 0.18 ms Adam. Compile, the CUDA graph, and Triton stayed off because the device was CPU.

Filling 2,097,152 sample indices, which is one cloud batch, took 10.2 ms with an allocate-and-copy and 6.4 ms in place (1.59x). That is the index refill, not a full 2K step.
