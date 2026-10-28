#pragma once

#include "ntc/ntc.hpp"

#include <vulkan/vulkan.h>

#include <string>

// Timings and GPU readbacks. Lavapipe rows are tagged and are not a performance result.
// `bc*_path` is empty or an NTCB block file (see bench/bc_pack.py).
void run_vk_bench(VkDevice dev, VkPhysicalDevice phys, VkQueue queue, uint32_t qfam,
                  const ntc::TextureSet& ts, const ntc::GpuUpload& up, bool float16, bool cpu_dev,
                  const std::string& shader_dir, const std::string& out_dir, int warmup, int iters,
                  const std::string& bc7_path, const std::string& bc1_path, const std::string& bc5_path);
