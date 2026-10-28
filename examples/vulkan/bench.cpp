#include "bench.hpp"
#include "stage.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(expr)                                                                 \
    do {                                                                            \
        VkResult _r = (expr);                                                       \
        if (_r != VK_SUCCESS) {                                                     \
            throw std::runtime_error(std::string(#expr) + " -> " + std::to_string(_r)); \
        }                                                                           \
    } while (0)

namespace {

struct Buf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

uint32_t find_mem(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags) return i;
    }
    throw std::runtime_error("no memory type");
}

Buf make_buf(VkDevice dev, VkPhysicalDevice phys, VkDeviceSize size, VkBufferUsageFlags usage, const void* src) {
    Buf b;
    b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK(vkCreateBuffer(dev, &bi, nullptr, &b.buffer));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, b.buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_mem(phys, req.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.memory));
    CHECK(vkBindBufferMemory(dev, b.buffer, b.memory, 0));
    CHECK(vkMapMemory(dev, b.memory, 0, req.size, 0, &b.mapped));
    if (src && size) std::memcpy(b.mapped, src, static_cast<size_t>(size));
    else if (size) std::memset(b.mapped, 0, static_cast<size_t>(size));
    return b;
}

void destroy_buf(VkDevice dev, Buf& b) {
    if (b.mapped) vkUnmapMemory(dev, b.memory);
    if (b.buffer) vkDestroyBuffer(dev, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(dev, b.memory, nullptr);
    b = {};
}

std::vector<uint8_t> read_bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

VkShaderModule shader_mod(VkDevice dev, const std::string& path) {
    auto code = read_bin(path);
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule m = VK_NULL_HANDLE;
    CHECK(vkCreateShaderModule(dev, &ci, nullptr, &m));
    return m;
}

void write_f32(const std::string& path, const float* data, size_t n) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(float)));
}

struct PassStats {
    double median_ms = 0;
    double p95_ms = 0;
};

PassStats summarize(std::vector<double> ms) {
    PassStats s;
    if (ms.empty()) return s;
    std::sort(ms.begin(), ms.end());
    s.median_ms = ms[ms.size() / 2];
    const size_t i95 = std::min(ms.size() - 1, static_cast<size_t>(std::ceil(0.95 * ms.size())) - 1);
    s.p95_ms = ms[i95];
    return s;
}

void report(const char* name, const PassStats& s, double texels, double out_bytes, bool cpu_dev) {
    const double sec = s.median_ms / 1e3;
    const double tps = sec > 0 ? texels / sec : 0;
    const double gbs = sec > 0 ? (out_bytes / sec) / 1e9 : 0;
    std::cout << "bench " << name << " median_ms " << s.median_ms << " p95_ms " << s.p95_ms << " texels_s "
              << tps << " gb_s " << gbs;
    if (cpu_dev) std::cout << " not_a_performance_claim";
    std::cout << "\n";
}

struct Pipe {
    VkShaderModule mod = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

void destroy_pipe(VkDevice dev, Pipe& p) {
    if (p.pipe) vkDestroyPipeline(dev, p.pipe, nullptr);
    if (p.mod) vkDestroyShaderModule(dev, p.mod, nullptr);
    if (p.pool) vkDestroyDescriptorPool(dev, p.pool, nullptr);
    if (p.layout) vkDestroyPipelineLayout(dev, p.layout, nullptr);
    if (p.set_layout) vkDestroyDescriptorSetLayout(dev, p.set_layout, nullptr);
    p = {};
}

Pipe make_compute(VkDevice dev, VkPhysicalDevice phys, const std::string& spv, VkBuffer desc, VkDeviceSize desc_size,
                  VkBuffer blob, VkDeviceSize blob_size, VkBuffer outb, VkDeviceSize out_size) {
    Pipe p;
    VkDescriptorSetLayoutBinding binds[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 3;
    lci.pBindings = binds;
    CHECK(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &p.set_layout));
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size = 16;
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &p.set_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &p.layout));
    p.mod = shader_mod(dev, spv);
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = p.mod;
    stage.pName = "main";
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_req{};
    ntc_attach_subgroup_32(stage, subgroup_req, phys);
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = stage;
    cpi.layout = p.layout;
    CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &p.pipe));
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 3;
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &p.pool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = p.pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &p.set_layout;
    CHECK(vkAllocateDescriptorSets(dev, &dsai, &p.set));
    VkDescriptorBufferInfo infos[3] = {{desc, 0, desc_size}, {blob, 0, blob_size}, {outb, 0, out_size}};
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = p.set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(dev, 3, writes, 0, nullptr);
    return p;
}

struct BlockFile {
    uint32_t w = 0, h = 0, format = 0;
    std::vector<uint8_t> payload;
};

BlockFile load_blocks(const std::string& path) {
    auto raw = read_bin(path);
    if (raw.size() < 20) throw std::runtime_error("short block file");
    BlockFile f;
    std::memcpy(&f.w, raw.data(), 4);
    std::memcpy(&f.h, raw.data() + 4, 4);
    std::memcpy(&f.format, raw.data() + 8, 4);
    uint32_t n = 0;
    std::memcpy(&n, raw.data() + 12, 4);
    uint32_t magic = 0;
    std::memcpy(&magic, raw.data() + 16, 4);
    if (magic != 0x4243544Eu) throw std::runtime_error("bad block magic");
    if (20u + n > raw.size()) throw std::runtime_error("short block payload");
    f.payload.assign(raw.begin() + 20, raw.begin() + 20 + n);
    return f;
}

VkFormat vk_format_of(uint32_t fmt) {
    if (fmt == 1) return VK_FORMAT_BC7_UNORM_BLOCK;
    if (fmt == 2) return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    if (fmt == 3) return VK_FORMAT_BC5_UNORM_BLOCK;
    return VK_FORMAT_UNDEFINED;
}

bool format_ok(VkPhysicalDevice phys, VkFormat fmt) {
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(phys, fmt, &props);
    const VkFormatFeatureFlags need =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    return (props.optimalTilingFeatures & need) == need;
}

}  // namespace

void run_vk_bench(VkDevice dev, VkPhysicalDevice phys, VkQueue queue, uint32_t qfam,
                  const ntc::TextureSet& ts, const ntc::GpuUpload& up, bool float16, bool cpu_dev,
                  const std::string& shader_dir, const std::string& out_dir, int warmup, int iters,
                  const std::string& bc7_path, const std::string& bc1_path, const std::string& bc5_path) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);
    std::filesystem::create_directories(out_dir);
    const float period = props.limits.timestampPeriod;
    const uint32_t channels = static_cast<uint32_t>(ts.num_channels());
    const uint32_t base_w = ts.width;
    const uint32_t base_h = ts.height;
    uint64_t mip_texels = 0;
    uint32_t nmips = 0;
    for (uint32_t m = 0; m < ts.num_mips; ++m) {
        const uint32_t mw = std::max(1u, base_w >> m);
        const uint32_t mh = std::max(1u, base_h >> m);
        if (mw == 0 || mh == 0) break;
        mip_texels += static_cast<uint64_t>(mw) * mh;
        nmips = m + 1;
        if (mw == 1 && mh == 1) break;
    }

    const uint32_t resolutions[][2] = {{base_w, base_h}, {1920, 1080}, {2560, 1440}, {3840, 2160}};
    uint64_t max_floats = static_cast<uint64_t>(base_w) * base_h * channels;
    for (auto wh : resolutions) {
        max_floats = std::max(max_floats, static_cast<uint64_t>(wh[0]) * wh[1] * std::max(channels, 4u));
    }
    Buf desc = make_buf(dev, phys, up.desc.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, up.desc.data());
    Buf blob = make_buf(dev, phys, up.blob.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, up.blob.data());
    Buf outb = make_buf(dev, phys, max_floats * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr);

    Pipe fp32 = make_compute(dev, phys, shader_dir + "/ntc_decompress.comp.spv", desc.buffer, desc.size, blob.buffer,
                             blob.size, outb.buffer, outb.size);
    Pipe sample = make_compute(dev, phys, shader_dir + "/ntc_sample_res.comp.spv", desc.buffer, desc.size, blob.buffer,
                               blob.size, outb.buffer, outb.size);
    Pipe lit;
    const bool can_lit = channels >= 6;
    if (can_lit) {
        lit = make_compute(dev, phys, shader_dir + "/ntc_lit.comp.spv", desc.buffer, desc.size, blob.buffer, blob.size,
                           outb.buffer, outb.size);
    }
    auto maybe_pipe = [&](const std::string& file) {
        Pipe p;
        std::ifstream probe(file);
        if (probe) {
            p = make_compute(dev, phys, file, desc.buffer, desc.size, blob.buffer, blob.size, outb.buffer, outb.size);
        }
        return p;
    };
    Pipe fp16p;
    if (float16) fp16p = maybe_pipe(shader_dir + "/ntc_decompress_fp16.comp.spv");
    Pipe stage = maybe_pipe(shader_dir + "/ntc_decompress_stage.comp.spv");
    Pipe stage16;
    if (float16) stage16 = maybe_pipe(shader_dir + "/ntc_decompress_fp16_stage.comp.spv");

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = qfam;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cpool = VK_NULL_HANDLE;
    CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &cpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

    const uint32_t qcount = static_cast<uint32_t>(std::max(4, iters * 2));
    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = qcount;
    VkQueryPool qpool = VK_NULL_HANDLE;
    CHECK(vkCreateQueryPool(dev, &qpci, nullptr, &qpool));

    auto submit = [&](VkCommandBuffer c) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c;
        CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
        CHECK(vkQueueWaitIdle(queue));
    };

    auto time_records = [&](auto&& record_one) -> PassStats {
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        for (int i = 0; i < warmup; ++i) {
            CHECK(vkResetCommandBuffer(cmd, 0));
            CHECK(vkBeginCommandBuffer(cmd, &begin));
            record_one(cmd);
            CHECK(vkEndCommandBuffer(cmd));
            submit(cmd);
        }
        CHECK(vkResetCommandBuffer(cmd, 0));
        CHECK(vkBeginCommandBuffer(cmd, &begin));
        vkCmdResetQueryPool(cmd, qpool, 0, qcount);
        for (int i = 0; i < iters; ++i) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, static_cast<uint32_t>(i * 2));
            record_one(cmd);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool,
                                static_cast<uint32_t>(i * 2 + 1));
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &mb, 0, nullptr, 0, nullptr);
        }
        CHECK(vkEndCommandBuffer(cmd));
        submit(cmd);
        std::vector<uint64_t> stamps(static_cast<size_t>(iters * 2));
        CHECK(vkGetQueryPoolResults(dev, qpool, 0, static_cast<uint32_t>(iters * 2), stamps.size() * 8, stamps.data(),
                                    8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        std::vector<double> ms;
        ms.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            const uint64_t a = stamps[static_cast<size_t>(i * 2)];
            const uint64_t b = stamps[static_cast<size_t>(i * 2 + 1)];
            if (b > a && period > 0.f) ms.push_back(double(b - a) * double(period) / 1e6);
        }
        return summarize(ms);
    };

    auto bind_dispatch = [&](VkCommandBuffer c, Pipe& p, uint32_t pc0, uint32_t pc1, uint32_t pc2, uint32_t pc3,
                             uint32_t gx, uint32_t gy) {
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.set, 0, nullptr);
        uint32_t pc[4] = {pc0, pc1, pc2, pc3};
        vkCmdPushConstants(c, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
        vkCmdDispatch(c, gx, gy, 1);
    };

    const double load_bytes = double(mip_texels) * channels * 4.0;
    auto time_load = [&](Pipe& p, const char* name, const char* skip) {
        if (!p.pipe) {
            std::cout << "bench " << name << " skipped " << skip << "\n";
            return;
        }
        const PassStats st = time_records([&](VkCommandBuffer c) {
            for (uint32_t m = 0; m < nmips; ++m) {
                const uint32_t mw = std::max(1u, base_w >> m);
                const uint32_t mh = std::max(1u, base_h >> m);
                bind_dispatch(c, p, m, 0, 0, 0, (mw + 7) / 8, (mh + 7) / 8);
            }
        });
        report(name, st, double(mip_texels), load_bytes, cpu_dev);
    };
    // One dispatch per mip, 8x8, one ntc_eval per texel, point sample.
    // Timestamps bracket those dispatches. The barrier between iterations
    // is outside the pair. Output is an SSBO of fp32 channels.
    std::cout << "note load_pass one_dispatch_per_mip 8x8 point_ntc_eval ssbo_f32\n";
    time_load(fp32, "load_fp32", "missing_shader");
    time_load(fp16p, "load_fp16", "shaderFloat16");
    time_load(stage, "load_fp32_stage", "missing_shader");
    time_load(stage16, "load_fp16_stage", float16 ? "missing_shader" : "shaderFloat16");

    auto sample_at = [&](uint32_t w, uint32_t h, const char* tag) {
        if (cpu_dev && (w > 512 || h > 512)) {
            std::cout << "bench " << tag << " skipped cpu_device\n";
            return;
        }
        const PassStats st = time_records([&](VkCommandBuffer c) {
            bind_dispatch(c, sample, 0, 0, w, h, (w + 7) / 8, (h + 7) / 8);
        });
        const double texels = double(w) * double(h);
        report(tag, st, texels, texels * channels * 4.0, cpu_dev);
    };
    // One 8x8 dispatch. Each thread calls ntc_eval once at mip 0 (point,
    // no derivatives, no footprint). Writes `channels` fp32 values. The
    // 1920 and 3840 passes are that many MLP evaluations, not a blit of
    // the 256 image. No overdraw.
    std::cout << "note sample_pass one_dispatch 8x8 point_ntc_eval mip0 ssbo_f32 no_filter\n";
    sample_at(base_w, base_h, "sample_native_fp32");
    sample_at(1920, 1080, "sample_1920x1080_fp32");
    sample_at(2560, 1440, "sample_2560x1440_fp32");
    sample_at(3840, 2160, "sample_3840x2160_fp32");

    // One readback of mip 0 and of the lit pass at the material resolution.
    {
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vkResetCommandBuffer(cmd, 0));
        CHECK(vkBeginCommandBuffer(cmd, &begin));
        bind_dispatch(cmd, fp32, 0, 0, 0, 0, (base_w + 7) / 8, (base_h + 7) / 8);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
        CHECK(vkEndCommandBuffer(cmd));
        submit(cmd);
        const size_t n = static_cast<size_t>(base_w) * base_h * channels;
        write_f32(out_dir + "/vk_ntc_fp32.f32", static_cast<float*>(outb.mapped), n);
        std::cout << "wrote " << out_dir << "/vk_ntc_fp32.f32 " << base_w << " " << base_h << " " << channels
                  << "\n";
    }
    if (can_lit) {
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vkResetCommandBuffer(cmd, 0));
        CHECK(vkBeginCommandBuffer(cmd, &begin));
        bind_dispatch(cmd, lit, 0, 0, base_w, base_h, (base_w + 7) / 8, (base_h + 7) / 8);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
        CHECK(vkEndCommandBuffer(cmd));
        submit(cmd);
        write_f32(out_dir + "/vk_lit_fp32.f32", static_cast<float*>(outb.mapped),
                  static_cast<size_t>(base_w) * base_h * 3);
        std::cout << "wrote " << out_dir << "/vk_lit_fp32.f32\n";
    } else {
        std::cout << "bench lit skipped channels\n";
    }

    auto sample_blocks = [&](const std::string& path, const char* tag) {
        if (path.empty()) {
            std::cout << "bench " << tag << " skipped no_file\n";
            return;
        }
        BlockFile bf = load_blocks(path);
        const VkFormat fmt = vk_format_of(bf.format);
        if (fmt == VK_FORMAT_UNDEFINED || !format_ok(phys, fmt)) {
            std::cout << "bench " << tag << " skipped format\n";
            return;
        }
        Buf staging = make_buf(dev, phys, bf.payload.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, bf.payload.data());
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = fmt;
        ii.extent = {bf.w, bf.h, 1};
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImage image = VK_NULL_HANDLE;
        CHECK(vkCreateImage(dev, &ii, nullptr, &image));
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(dev, image, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        try {
            ai.memoryTypeIndex = find_mem(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        } catch (const std::runtime_error&) {
            ai.memoryTypeIndex = find_mem(phys, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        VkDeviceMemory mem = VK_NULL_HANDLE;
        CHECK(vkAllocateMemory(dev, &ai, nullptr, &mem));
        CHECK(vkBindImageMemory(dev, image, mem, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = fmt;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        VkImageView view = VK_NULL_HANDLE;
        CHECK(vkCreateImageView(dev, &vi, nullptr, &view));
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VkSampler samp = VK_NULL_HANDLE;
        CHECK(vkCreateSampler(dev, &sci, nullptr, &samp));

        VkDescriptorSetLayoutBinding tb[2]{};
        tb[0].binding = 0;
        tb[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        tb[0].descriptorCount = 1;
        tb[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        tb[1].binding = 1;
        tb[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        tb[1].descriptorCount = 1;
        tb[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = 2;
        lci.pBindings = tb;
        VkDescriptorSetLayout sl = VK_NULL_HANDLE;
        CHECK(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &sl));
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = 16;
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &sl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VkPipelineLayout pl = VK_NULL_HANDLE;
        CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
        VkShaderModule mod = shader_mod(dev, shader_dir + "/tex_sample.comp.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = stage;
        cpi.layout = pl;
        VkPipeline pipe = VK_NULL_HANDLE;
        CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));
        VkDescriptorPoolSize ps[2]{};
        ps[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[0].descriptorCount = 1;
        ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps[1].descriptorCount = 1;
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes = ps;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &pool));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &sl;
        VkDescriptorSet set = VK_NULL_HANDLE;
        CHECK(vkAllocateDescriptorSets(dev, &dsai, &set));
        VkDescriptorImageInfo di{};
        di.sampler = samp;
        di.imageView = view;
        di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorBufferInfo bi{outb.buffer, 0, outb.size};
        VkWriteDescriptorSet wr[2]{};
        wr[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[0].dstSet = set;
        wr[0].dstBinding = 0;
        wr[0].descriptorCount = 1;
        wr[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wr[0].pImageInfo = &di;
        wr[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[1].dstSet = set;
        wr[1].dstBinding = 1;
        wr[1].descriptorCount = 1;
        wr[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[1].pBufferInfo = &bi;
        vkUpdateDescriptorSets(dev, 2, wr, 0, nullptr);

        auto upload = [&](VkCommandBuffer c) {
            VkImageMemoryBarrier to{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to.image = image;
            to.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to.subresourceRange.levelCount = 1;
            to.subresourceRange.layerCount = 1;
            to.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &to);
            VkBufferImageCopy cp{};
            cp.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            cp.imageSubresource.layerCount = 1;
            cp.imageExtent = {bf.w, bf.h, 1};
            vkCmdCopyBufferToImage(c, staging.buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
            VkImageMemoryBarrier to2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            to2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to2.image = image;
            to2.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to2.subresourceRange.levelCount = 1;
            to2.subresourceRange.layerCount = 1;
            to2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &to2);
        };
        // Upload once, then time the sample at 1080p and 4K. Also read back the native size.
        {
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            CHECK(vkResetCommandBuffer(cmd, 0));
            CHECK(vkBeginCommandBuffer(cmd, &begin));
            upload(cmd);
            CHECK(vkEndCommandBuffer(cmd));
            submit(cmd);
        }
        auto draw = [&](VkCommandBuffer c, uint32_t w, uint32_t h) {
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, nullptr);
            uint32_t pc[4] = {w, h, 0, 0};
            vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
            vkCmdDispatch(c, (w + 7) / 8, (h + 7) / 8, 1);
        };
        auto time_res = [&](uint32_t w, uint32_t h, const char* suffix) {
            if (cpu_dev && (w > 512 || h > 512)) {
                std::cout << "bench " << tag << suffix << " skipped cpu_device\n";
                return;
            }
            const PassStats st = time_records([&](VkCommandBuffer c) { draw(c, w, h); });
            report((std::string(tag) + suffix).c_str(), st, double(w) * double(h), double(w) * double(h) * 16.0,
                   cpu_dev);
        };
        // One 8x8 dispatch. One linear texture() of the native-resolution
        // block image, magnified to the output size, written as 4 fp32s.
        // The upload is not inside the timestamp pair.
        std::cout << "note bc_pass one_dispatch 8x8 linear_sample ssbo_rgba32f magnify\n";
        std::cout << "note bc_compute_pass storage_buffer_rgba32f not_a_color_attachment\n";
        time_res(1920, 1080, "_1920x1080");
        time_res(2560, 1440, "_2560x1440");
        time_res(3840, 2160, "_3840x2160");
        {
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            CHECK(vkResetCommandBuffer(cmd, 0));
            CHECK(vkBeginCommandBuffer(cmd, &begin));
            draw(cmd, bf.w, bf.h);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                                 nullptr, 0, nullptr);
            CHECK(vkEndCommandBuffer(cmd));
            submit(cmd);
            write_f32(out_dir + "/" + tag + ".f32", static_cast<float*>(outb.mapped),
                      static_cast<size_t>(bf.w) * bf.h * 4);
            std::cout << "wrote " << out_dir << "/" << tag << ".f32 " << bf.w << " " << bf.h << "\n";
        }
        vkDestroyPipeline(dev, pipe, nullptr);
        vkDestroyShaderModule(dev, mod, nullptr);
        vkDestroyDescriptorPool(dev, pool, nullptr);
        vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyDescriptorSetLayout(dev, sl, nullptr);
        vkDestroySampler(dev, samp, nullptr);
        vkDestroyImageView(dev, view, nullptr);
        vkDestroyImage(dev, image, nullptr);
        vkFreeMemory(dev, mem, nullptr);
        destroy_buf(dev, staging);
    };

    sample_blocks(bc7_path, "bc7");
    sample_blocks(bc1_path, "bc1");
    sample_blocks(bc5_path, "bc5");

    vkDestroyQueryPool(dev, qpool, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    destroy_pipe(dev, fp32);
    destroy_pipe(dev, sample);
    destroy_pipe(dev, lit);
    destroy_pipe(dev, fp16p);
    destroy_pipe(dev, stage);
    destroy_pipe(dev, stage16);
    destroy_buf(dev, desc);
    destroy_buf(dev, blob);
    destroy_buf(dev, outb);
}
