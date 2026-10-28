#include "bench.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(expr)                                                                 \
    do {                                                                            \
        VkResult _r = (expr);                                                       \
        if (_r != VK_SUCCESS) throw std::runtime_error(std::string(#expr) + " -> " + std::to_string(_r)); \
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

Buf make_buf(VkDevice dev, VkPhysicalDevice phys, VkDeviceSize size, VkBufferUsageFlags usage, const void* src,
             bool device_local) {
    Buf b;
    b.size = std::max<VkDeviceSize>(size, 4);
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = b.size;
    bi.usage = usage;
    CHECK(vkCreateBuffer(dev, &bi, nullptr, &b.buffer));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, b.buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    VkMemoryPropertyFlags want = device_local ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                              : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    try {
        ai.memoryTypeIndex = find_mem(phys, req.memoryTypeBits, want);
    } catch (const std::runtime_error&) {
        ai.memoryTypeIndex = find_mem(phys, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.memory));
    CHECK(vkBindBufferMemory(dev, b.buffer, b.memory, 0));
    if (!device_local) {
        CHECK(vkMapMemory(dev, b.memory, 0, req.size, 0, &b.mapped));
        if (src) std::memcpy(b.mapped, src, static_cast<size_t>(size));
        else std::memset(b.mapped, 0, static_cast<size_t>(std::min(size, req.size)));
    }
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
    if (!in) return {};
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

VkShaderModule shader_mod(VkDevice dev, const std::string& path) {
    auto code = read_bin(path);
    if (code.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &ci, nullptr, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
    return m;
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

void report(const char* name, const PassStats& s, double texels, bool cpu_dev) {
    const double sec = s.median_ms / 1e3;
    const double tps = sec > 0 ? texels / sec : 0;
    std::cout << "bench " << name << " median_ms " << s.median_ms << " p95_ms " << s.p95_ms << " texels_s " << tps
              << " gb_s 0";
    if (cpu_dev) std::cout << " not_a_performance_claim";
    std::cout << "\n";
}

double psnr_of(const std::vector<float>& gpu, const std::vector<float>& cpu) {
    const size_t n = std::min(gpu.size(), cpu.size());
    if (!n) return 0;
    double se = 0;
    for (size_t i = 0; i < n; ++i) {
        double a = std::round(std::min(1.0, std::max(0.0, double(gpu[i]))) * 255.0) / 255.0;
        double b = std::round(std::min(1.0, std::max(0.0, double(cpu[i]))) * 255.0) / 255.0;
        double d = a - b;
        se += d * d;
    }
    const double mse = se / double(n);
    if (mse <= 1e-20) return 100.0;
    return std::min(100.0, 10.0 * std::log10(1.0 / mse));
}

struct BlockFile {
    uint32_t w = 0, h = 0, format = 0;
    std::vector<uint8_t> payload;
};

bool load_blocks(const std::string& path, BlockFile& f) {
    auto raw = read_bin(path);
    if (raw.size() < 20) return false;
    std::memcpy(&f.w, raw.data(), 4);
    std::memcpy(&f.h, raw.data() + 4, 4);
    std::memcpy(&f.format, raw.data() + 8, 4);
    uint32_t n = 0, magic = 0;
    std::memcpy(&n, raw.data() + 12, 4);
    std::memcpy(&magic, raw.data() + 16, 4);
    if (magic != 0x4243544Eu || 20u + n > raw.size()) return false;
    f.payload.assign(raw.begin() + 20, raw.begin() + 20 + n);
    return true;
}

}  // namespace

void run_vk_speed(VkDevice dev, VkPhysicalDevice phys, VkQueue queue, uint32_t qfam, bool graphics, bool int8,
                  bool coopmat, bool coopvec, const ntc::TextureSet& ts, const ntc::GpuUpload& up, bool cpu_dev,
                  const std::string& shader_dir, int warmup, int iters, const std::string& bc7_path) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);
    const float period = props.limits.timestampPeriod;
    const uint32_t channels = static_cast<uint32_t>(ts.num_channels());
    const uint32_t base_w = ts.width;
    const uint32_t base_h = ts.height;
    const uint32_t res[][2] = {{1920, 1080}, {2560, 1440}, {3840, 2160}};
    uint64_t max_floats = static_cast<uint64_t>(base_w) * base_h * std::max(channels, 4u);
    for (auto wh : res) max_floats = std::max(max_floats, uint64_t(wh[0]) * wh[1] * 4u);

    Buf desc = make_buf(dev, phys, up.desc.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, up.desc.data(), false);
    Buf blob = make_buf(dev, phys, up.blob.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, up.blob.data(), false);
    Buf outb = make_buf(dev, phys, max_floats * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr, false);
    Buf scratch = make_buf(dev, phys, 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr, false);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = qfam;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool = VK_NULL_HANDLE;
    CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &pool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));
    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = static_cast<uint32_t>(std::max(8, iters * 2));
    VkQueryPool qpool = VK_NULL_HANDLE;
    CHECK(vkCreateQueryPool(dev, &qpci, nullptr, &qpool));

    auto submit = [&](VkCommandBuffer c) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &c;
        CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
        CHECK(vkQueueWaitIdle(queue));
    };
    auto time_records = [&](auto&& record, VkPipelineStageFlagBits stage) -> PassStats {
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        for (int i = 0; i < warmup; ++i) {
            CHECK(vkResetCommandBuffer(cmd, 0));
            CHECK(vkBeginCommandBuffer(cmd, &begin));
            record(cmd);
            CHECK(vkEndCommandBuffer(cmd));
            submit(cmd);
        }
        CHECK(vkResetCommandBuffer(cmd, 0));
        CHECK(vkBeginCommandBuffer(cmd, &begin));
        vkCmdResetQueryPool(cmd, qpool, 0, static_cast<uint32_t>(iters * 2));
        for (int i = 0; i < iters; ++i) {
            vkCmdWriteTimestamp(cmd, stage, qpool, static_cast<uint32_t>(i * 2));
            record(cmd);
            vkCmdWriteTimestamp(cmd, stage, qpool, static_cast<uint32_t>(i * 2 + 1));
        }
        CHECK(vkEndCommandBuffer(cmd));
        submit(cmd);
        std::vector<uint64_t> stamps(static_cast<size_t>(iters * 2));
        CHECK(vkGetQueryPoolResults(dev, qpool, 0, static_cast<uint32_t>(iters * 2), stamps.size() * 8, stamps.data(), 8,
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        std::vector<double> ms;
        for (int i = 0; i < iters; ++i) {
            uint64_t a = stamps[size_t(i * 2)], b = stamps[size_t(i * 2 + 1)];
            if (b > a && period > 0.f) ms.push_back(double(b - a) * double(period) / 1e6);
        }
        return summarize(ms);
    };

    auto make_compute = [&](const std::string& spv, VkDescriptorSetLayoutBinding* binds, uint32_t nb,
                            VkBuffer* bufs, VkDeviceSize* sizes, VkImageView image, VkSampler samp, uint32_t image_binding,
                            VkPipeline& pipe, VkPipelineLayout& layout, VkDescriptorSet& set) {
        pipe = VK_NULL_HANDLE;
        VkShaderModule mod = shader_mod(dev, spv);
        if (!mod) return;
        VkDescriptorSetLayout sl = VK_NULL_HANDLE;
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = nb;
        lci.pBindings = binds;
        CHECK(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &sl));
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = 16;
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &sl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &layout));
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = stage;
        cpi.layout = layout;
        if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe) != VK_SUCCESS) {
            pipe = VK_NULL_HANDLE;
            vkDestroyShaderModule(dev, mod, nullptr);
            return;
        }
        vkDestroyShaderModule(dev, mod, nullptr);
        std::vector<VkDescriptorPoolSize> ps;
        for (uint32_t i = 0; i < nb; ++i) {
            bool found = false;
            for (auto& s : ps)
                if (s.type == binds[i].descriptorType) {
                    s.descriptorCount++;
                    found = true;
                }
            if (!found) ps.push_back({binds[i].descriptorType, 1});
        }
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = static_cast<uint32_t>(ps.size());
        dpci.pPoolSizes = ps.data();
        VkDescriptorPool dpool = VK_NULL_HANDLE;
        CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dpool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &sl;
        CHECK(vkAllocateDescriptorSets(dev, &dsai, &set));
        std::vector<VkWriteDescriptorSet> wr(nb);
        std::vector<VkDescriptorBufferInfo> bi(nb);
        VkDescriptorImageInfo ii{};
        ii.sampler = samp;
        ii.imageView = image;
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        uint32_t bslot = 0;
        for (uint32_t i = 0; i < nb; ++i) {
            wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[i].dstSet = set;
            wr[i].dstBinding = binds[i].binding;
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = binds[i].descriptorType;
            if (binds[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
                binds[i].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                if (binds[i].binding == image_binding) wr[i].pImageInfo = &ii;
            } else {
                bi[bslot].buffer = bufs[bslot];
                bi[bslot].offset = 0;
                bi[bslot].range = sizes[bslot];
                wr[i].pBufferInfo = &bi[bslot];
                bslot++;
            }
        }
        vkUpdateDescriptorSets(dev, nb, wr.data(), 0, nullptr);
        (void)dpool;
        (void)sl;
    };

    // Packed int8 sample. Exact when the file carries act scales, otherwise max-abs/127.
    std::cout << "note i8_path i8vec4_dot4 " << (ts.act_int8() ? "exact_qat" : "approx_maxabs") << "\n";
    std::cout << "note i8_dot glslang_15_has_no_GL_EXT_shader_integer_dot_product\n";
    if (!int8) {
        std::cout << "bench sample_i8 skipped shaderInt8\n";
    } else {
        VkDescriptorSetLayoutBinding binds[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkBuffer bufs[3] = {desc.buffer, blob.buffer, outb.buffer};
        VkDeviceSize sizes[3] = {desc.size, blob.size, outb.size};
        VkPipeline pipe = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        make_compute(shader_dir + "/ntc_sample_i8.comp.spv", binds, 3, bufs, sizes, VK_NULL_HANDLE, VK_NULL_HANDLE, 99,
                     pipe, layout, set);
        if (!pipe) {
            std::cout << "bench sample_i8 skipped pipeline\n";
        } else {
            auto draw = [&](VkCommandBuffer c, uint32_t w, uint32_t h) {
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
                vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
                uint32_t pc[4] = {0, 0, w, h};
                vkCmdPushConstants(c, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
                vkCmdDispatch(c, (w + 7) / 8, (h + 7) / 8, 1);
            };
            auto one = [&](uint32_t w, uint32_t h, const char* name, bool read) {
                if (cpu_dev && (w > 512 || h > 512)) {
                    std::cout << "bench " << name << " skipped cpu_device\n";
                    return;
                }
                PassStats st = time_records([&](VkCommandBuffer c) { draw(c, w, h); }, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                report(name, st, double(w) * h, cpu_dev);
                if (!read) return;
                VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                CHECK(vkResetCommandBuffer(cmd, 0));
                CHECK(vkBeginCommandBuffer(cmd, &begin));
                draw(cmd, w, h);
                VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                                     nullptr, 0, nullptr);
                CHECK(vkEndCommandBuffer(cmd));
                submit(cmd);
                auto cpu = ntc::decode_mip(ts, 0);
                std::vector<float> gpu(cpu.size());
                std::memcpy(gpu.data(), outb.mapped, gpu.size() * 4);
                std::cout << (ts.act_int8() ? "i8_exact_psnr " : "i8_approx_psnr ") << psnr_of(gpu, cpu) << "\n";
            };
            one(base_w, base_h, "sample_i8_native", true);
            one(1920, 1080, "sample_i8_1920x1080", false);
            one(2560, 1440, "sample_i8_2560x1440", false);
            one(3840, 2160, "sample_i8_3840x2160", false);
        }
    }

    if (!coopmat) {
        std::cout << "bench sample_coopmat skipped no_cooperative_matrix\n";
    } else {
        std::cout << "bench sample_coopmat skipped host_checks_native_match\n";
        VkDescriptorSetLayoutBinding binds[4]{};
        for (uint32_t i = 0; i < 3; ++i) {
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        binds[3].binding = 6;
        binds[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[3].descriptorCount = 1;
        binds[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkBuffer bufs[4] = {desc.buffer, blob.buffer, outb.buffer, scratch.buffer};
        VkDeviceSize sizes[4] = {desc.size, blob.size, outb.size, scratch.size};
        VkPipeline pipe = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        make_compute(shader_dir + "/ntc_coopmat.comp.spv", binds, 4, bufs, sizes, VK_NULL_HANDLE, VK_NULL_HANDLE, 99, pipe,
                     layout, set);
        if (!pipe || ts.group_size != 0) {
            std::cout << "bench sample_coopmat_native skipped " << (pipe ? "grouped_weights" : "pipeline") << "\n";
        } else {
            auto draw = [&](VkCommandBuffer c, uint32_t w, uint32_t h) {
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
                vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
                uint32_t pc[4] = {0, 0, w, h};
                vkCmdPushConstants(c, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
                vkCmdDispatch(c, (w + 15) / 16, 1, 1);
            };
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            CHECK(vkResetCommandBuffer(cmd, 0));
            CHECK(vkBeginCommandBuffer(cmd, &begin));
            draw(cmd, base_w, base_h);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                                 nullptr, 0, nullptr);
            CHECK(vkEndCommandBuffer(cmd));
            submit(cmd);
            auto cpu = ntc::decode_mip(ts, 0);
            std::vector<float> gpu(cpu.size());
            std::memcpy(gpu.data(), outb.mapped, std::min(gpu.size() * 4, static_cast<size_t>(outb.size)));
            double err = 0;
            for (size_t i = 0; i < cpu.size(); ++i) err = std::max(err, std::abs(double(gpu[i] - cpu[i])));
            std::cout << "coopmat_max_abs " << err << "\n";
            if (err > 2e-2) {
                std::cout << "bench sample_coopmat skipped mismatch\n";
            } else if (cpu_dev) {
                std::cout << "bench sample_coopmat_3840x2160 skipped cpu_device\n";
            } else {
                for (auto wh : res) {
                    std::string name = "sample_coopmat_" + std::to_string(wh[0]) + "x" + std::to_string(wh[1]);
                    PassStats st = time_records([&](VkCommandBuffer c) { draw(c, wh[0], wh[1]); },
                                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                    report(name.c_str(), st, double(wh[0]) * wh[1], cpu_dev);
                }
            }
        }
    }
    if (!coopvec) std::cout << "bench sample_coopvec skipped no_extension\n";
    else std::cout << "bench sample_coopvec skipped toolchain_has_no_GL_NV_cooperative_vector\n";

    // Fragment fullscreen into a color attachment. The compute BC rows stay.
    std::cout << "note fs_pass one_triangle color_attachment timestamp_color_output\n";
    if (!graphics) {
        std::cout << "bench sample_fs skipped no_graphics_queue\n";
        std::cout << "bench bc7_fs skipped no_graphics_queue\n";
    } else {
        // NTC fragment.
        VkDescriptorSetLayoutBinding binds[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayout sl = VK_NULL_HANDLE;
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = 2;
        lci.pBindings = binds;
        CHECK(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &sl));
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        pcr.size = 16;
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &sl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &layout));
        VkShaderModule vs = shader_mod(dev, shader_dir + "/fullscreen.vert.spv");
        VkShaderModule fs = shader_mod(dev, shader_dir + "/ntc_fs.frag.spv");
        VkPipeline pipe = VK_NULL_HANDLE;
        if (vs && fs) {
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vs;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fs;
            stages[1].pName = "main";
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vp.viewportCount = 1;
            vp.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = VK_CULL_MODE_NONE;
            rs.lineWidth = 1.f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineColorBlendAttachmentState cba{};
            cba.colorWriteMask = 0xF;
            VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cb.attachmentCount = 1;
            cb.pAttachments = &cba;
            VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dys{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dys.dynamicStateCount = 2;
            dys.pDynamicStates = dyns;
            VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
            VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            ri.colorAttachmentCount = 1;
            ri.pColorAttachmentFormats = &fmt;
            VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            gp.pNext = &ri;
            gp.stageCount = 2;
            gp.pStages = stages;
            gp.pVertexInputState = &vi;
            gp.pInputAssemblyState = &ia;
            gp.pViewportState = &vp;
            gp.pRasterizationState = &rs;
            gp.pMultisampleState = &ms;
            gp.pColorBlendState = &cb;
            gp.pDynamicState = &dys;
            gp.layout = layout;
            if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
        }
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        VkDescriptorPool dpool = VK_NULL_HANDLE;
        CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dpool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &sl;
        VkDescriptorSet set = VK_NULL_HANDLE;
        CHECK(vkAllocateDescriptorSets(dev, &dsai, &set));
        VkDescriptorBufferInfo dbi{desc.buffer, 0, desc.size};
        VkDescriptorBufferInfo bbi{blob.buffer, 0, blob.size};
        VkWriteDescriptorSet wr[2]{};
        wr[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[0].dstSet = set;
        wr[0].dstBinding = 0;
        wr[0].descriptorCount = 1;
        wr[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[0].pBufferInfo = &dbi;
        wr[1] = wr[0];
        wr[1].dstBinding = 1;
        wr[1].pBufferInfo = &bbi;
        vkUpdateDescriptorSets(dev, 2, wr, 0, nullptr);

        auto fs_target = [&](uint32_t w, uint32_t h, VkImage& image, VkImageView& view, VkDeviceMemory& mem) {
            VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ii.imageType = VK_IMAGE_TYPE_2D;
            ii.format = VK_FORMAT_R8G8B8A8_UNORM;
            ii.extent = {w, h, 1};
            ii.mipLevels = 1;
            ii.arrayLayers = 1;
            ii.samples = VK_SAMPLE_COUNT_1_BIT;
            ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
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
            CHECK(vkAllocateMemory(dev, &ai, nullptr, &mem));
            CHECK(vkBindImageMemory(dev, image, mem, 0));
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = image;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = VK_FORMAT_R8G8B8A8_UNORM;
            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vi.subresourceRange.levelCount = 1;
            vi.subresourceRange.layerCount = 1;
            CHECK(vkCreateImageView(dev, &vi, nullptr, &view));
        };
        if (!pipe) {
            std::cout << "bench sample_fs skipped pipeline\n";
        } else {
            for (auto wh : res) {
                if (cpu_dev) {
                    std::cout << "bench sample_fs_" << wh[0] << "x" << wh[1] << " skipped cpu_device\n";
                    continue;
                }
                VkImage image = VK_NULL_HANDLE;
                VkImageView view = VK_NULL_HANDLE;
                VkDeviceMemory mem = VK_NULL_HANDLE;
                fs_target(wh[0], wh[1], image, view, mem);
                auto record = [&](VkCommandBuffer c) {
                    VkImageMemoryBarrier to{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                    to.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    to.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    to.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    to.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    to.image = image;
                    to.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    to.subresourceRange.levelCount = 1;
                    to.subresourceRange.layerCount = 1;
                    to.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                         0, 0, nullptr, 0, nullptr, 1, &to);
                    VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                    att.imageView = view;
                    att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    att.clearValue.color.float32[3] = 1.f;
                    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
                    ri.renderArea = {{0, 0}, {wh[0], wh[1]}};
                    ri.layerCount = 1;
                    ri.colorAttachmentCount = 1;
                    ri.pColorAttachments = &att;
                    vkCmdBeginRendering(c, &ri);
                    VkViewport viewport{};
                    viewport.width = float(wh[0]);
                    viewport.height = float(wh[1]);
                    viewport.maxDepth = 1.f;
                    VkRect2D scissor{{0, 0}, {wh[0], wh[1]}};
                    vkCmdSetViewport(c, 0, 1, &viewport);
                    vkCmdSetScissor(c, 0, 1, &scissor);
                    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, nullptr);
                    uint32_t pc[4] = {0, 0, wh[0], wh[1]};
                    vkCmdPushConstants(c, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, pc);
                    vkCmdDraw(c, 3, 1, 0, 0);
                    vkCmdEndRendering(c);
                };
                std::string name = "sample_fs_" + std::to_string(wh[0]) + "x" + std::to_string(wh[1]);
                PassStats st = time_records(record, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
                report(name.c_str(), st, double(wh[0]) * wh[1], cpu_dev);
                vkDestroyImageView(dev, view, nullptr);
                vkDestroyImage(dev, image, nullptr);
                vkFreeMemory(dev, mem, nullptr);
            }
        }
        BlockFile bf;
        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(phys, VK_FORMAT_BC7_UNORM_BLOCK, &fp);
        const bool bc7_ok = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
        if (!load_blocks(bc7_path, bf) || bf.format != 1 || !bc7_ok) {
            std::cout << "bench bc7_fs skipped " << (bc7_ok ? "no_file" : "format") << "\n";
        } else {
            std::cout << "note bc7_fs one_triangle linear_sample color_attachment upload_outside_timestamp\n";
            Buf staging = make_buf(dev, phys, bf.payload.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, bf.payload.data(), false);
            VkImage bcimg = VK_NULL_HANDLE;
            VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ii.imageType = VK_IMAGE_TYPE_2D;
            ii.format = VK_FORMAT_BC7_UNORM_BLOCK;
            ii.extent = {bf.w, bf.h, 1};
            ii.mipLevels = 1;
            ii.arrayLayers = 1;
            ii.samples = VK_SAMPLE_COUNT_1_BIT;
            ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            CHECK(vkCreateImage(dev, &ii, nullptr, &bcimg));
            VkMemoryRequirements req{};
            vkGetImageMemoryRequirements(dev, bcimg, &req);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = req.size;
            try {
                ai.memoryTypeIndex = find_mem(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            } catch (const std::runtime_error&) {
                ai.memoryTypeIndex = find_mem(phys, req.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            }
            VkDeviceMemory bcmem = VK_NULL_HANDLE;
            CHECK(vkAllocateMemory(dev, &ai, nullptr, &bcmem));
            CHECK(vkBindImageMemory(dev, bcimg, bcmem, 0));
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = bcimg;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = VK_FORMAT_BC7_UNORM_BLOCK;
            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vi.subresourceRange.levelCount = 1;
            vi.subresourceRange.layerCount = 1;
            VkImageView bcview = VK_NULL_HANDLE;
            CHECK(vkCreateImageView(dev, &vi, nullptr, &bcview));
            VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            sci.magFilter = VK_FILTER_LINEAR;
            sci.minFilter = VK_FILTER_LINEAR;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            VkSampler samp = VK_NULL_HANDLE;
            CHECK(vkCreateSampler(dev, &sci, nullptr, &samp));
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            CHECK(vkResetCommandBuffer(cmd, 0));
            CHECK(vkBeginCommandBuffer(cmd, &begin));
            VkImageMemoryBarrier to{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            to.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to.image = bcimg;
            to.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            to.subresourceRange.levelCount = 1;
            to.subresourceRange.layerCount = 1;
            to.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                                 nullptr, 1, &to);
            VkBufferImageCopy cp{};
            cp.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            cp.imageSubresource.layerCount = 1;
            cp.imageExtent = {bf.w, bf.h, 1};
            vkCmdCopyBufferToImage(cmd, staging.buffer, bcimg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
            VkImageMemoryBarrier to2 = to;
            to2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            to2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &to2);
            CHECK(vkEndCommandBuffer(cmd));
            submit(cmd);
            VkDescriptorSetLayoutBinding bb{};
            bb.binding = 0;
            bb.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bb.descriptorCount = 1;
            bb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayout bsl = VK_NULL_HANDLE;
            VkDescriptorSetLayoutCreateInfo blci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            blci.bindingCount = 1;
            blci.pBindings = &bb;
            CHECK(vkCreateDescriptorSetLayout(dev, &blci, nullptr, &bsl));
            VkPushConstantRange bpcr{};
            bpcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bpcr.size = 16;
            VkPipelineLayoutCreateInfo bplci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            bplci.setLayoutCount = 1;
            bplci.pSetLayouts = &bsl;
            bplci.pushConstantRangeCount = 1;
            bplci.pPushConstantRanges = &bpcr;
            VkPipelineLayout blayout = VK_NULL_HANDLE;
            CHECK(vkCreatePipelineLayout(dev, &bplci, nullptr, &blayout));
            VkShaderModule bvs = shader_mod(dev, shader_dir + "/fullscreen.vert.spv");
            VkShaderModule bfs = shader_mod(dev, shader_dir + "/bc_fs.frag.spv");
            VkPipeline bpipe = VK_NULL_HANDLE;
            if (bvs && bfs) {
                VkPipelineShaderStageCreateInfo stages[2]{};
                stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
                stages[0].module = bvs;
                stages[0].pName = "main";
                stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
                stages[1].module = bfs;
                stages[1].pName = "main";
                VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
                VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
                ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
                VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
                vp.viewportCount = 1;
                vp.scissorCount = 1;
                VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
                rs.polygonMode = VK_POLYGON_MODE_FILL;
                rs.cullMode = VK_CULL_MODE_NONE;
                rs.lineWidth = 1.f;
                VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
                ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
                VkPipelineColorBlendAttachmentState cba{};
                cba.colorWriteMask = 0xF;
                VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
                cb.attachmentCount = 1;
                cb.pAttachments = &cba;
                VkDynamicState dyns[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
                VkPipelineDynamicStateCreateInfo dys{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
                dys.dynamicStateCount = 2;
                dys.pDynamicStates = dyns;
                VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM;
                VkPipelineRenderingCreateInfo pri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
                pri.colorAttachmentCount = 1;
                pri.pColorAttachmentFormats = &cfmt;
                VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
                gp.pNext = &pri;
                gp.stageCount = 2;
                gp.pStages = stages;
                gp.pVertexInputState = &vi;
                gp.pInputAssemblyState = &ia;
                gp.pViewportState = &vp;
                gp.pRasterizationState = &rs;
                gp.pMultisampleState = &ms;
                gp.pColorBlendState = &cb;
                gp.pDynamicState = &dys;
                gp.layout = blayout;
                if (vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, nullptr, &bpipe) != VK_SUCCESS) bpipe = VK_NULL_HANDLE;
            }
            VkDescriptorPoolSize bps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
            VkDescriptorPoolCreateInfo bdp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            bdp.maxSets = 1;
            bdp.poolSizeCount = 1;
            bdp.pPoolSizes = &bps;
            VkDescriptorPool bpool = VK_NULL_HANDLE;
            CHECK(vkCreateDescriptorPool(dev, &bdp, nullptr, &bpool));
            VkDescriptorSetAllocateInfo bai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            bai.descriptorPool = bpool;
            bai.descriptorSetCount = 1;
            bai.pSetLayouts = &bsl;
            VkDescriptorSet bset = VK_NULL_HANDLE;
            CHECK(vkAllocateDescriptorSets(dev, &bai, &bset));
            VkDescriptorImageInfo di{};
            di.sampler = samp;
            di.imageView = bcview;
            di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkWriteDescriptorSet bwr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            bwr.dstSet = bset;
            bwr.dstBinding = 0;
            bwr.descriptorCount = 1;
            bwr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bwr.pImageInfo = &di;
            vkUpdateDescriptorSets(dev, 1, &bwr, 0, nullptr);
            if (!bpipe) {
                std::cout << "bench bc7_fs skipped pipeline\n";
            } else {
                auto time_bc = [&](uint32_t w, uint32_t h, const char* name) {
                    if (cpu_dev && (w > 512 || h > 512)) {
                        std::cout << "bench " << name << " skipped cpu_device\n";
                        return;
                    }
                    VkImage image = VK_NULL_HANDLE;
                    VkImageView view = VK_NULL_HANDLE;
                    VkDeviceMemory mem = VK_NULL_HANDLE;
                    fs_target(w, h, image, view, mem);
                    auto record = [&](VkCommandBuffer c) {
                        VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                        bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                        bar.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        bar.image = image;
                        bar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                        bar.subresourceRange.levelCount = 1;
                        bar.subresourceRange.layerCount = 1;
                        bar.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &bar);
                        VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                        att.imageView = view;
                        att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                        att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
                        ri.renderArea = {{0, 0}, {w, h}};
                        ri.layerCount = 1;
                        ri.colorAttachmentCount = 1;
                        ri.pColorAttachments = &att;
                        vkCmdBeginRendering(c, &ri);
                        VkViewport viewport{};
                        viewport.width = float(w);
                        viewport.height = float(h);
                        viewport.maxDepth = 1.f;
                        VkRect2D scissor{{0, 0}, {w, h}};
                        vkCmdSetViewport(c, 0, 1, &viewport);
                        vkCmdSetScissor(c, 0, 1, &scissor);
                        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, bpipe);
                        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, blayout, 0, 1, &bset, 0, nullptr);
                        uint32_t pc[4] = {w, h, 0, 0};
                        vkCmdPushConstants(c, blayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, pc);
                        vkCmdDraw(c, 3, 1, 0, 0);
                        vkCmdEndRendering(c);
                    };
                    PassStats st = time_records(record, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
                    report(name, st, double(w) * h, cpu_dev);
                    vkDestroyImageView(dev, view, nullptr);
                    vkDestroyImage(dev, image, nullptr);
                    vkFreeMemory(dev, mem, nullptr);
                };
                time_bc(bf.w, bf.h, "bc7_fs_native");
                time_bc(1920, 1080, "bc7_fs_1920x1080");
                time_bc(2560, 1440, "bc7_fs_2560x1440");
                time_bc(3840, 2160, "bc7_fs_3840x2160");
            }
            vkDestroyPipeline(dev, bpipe, nullptr);
            if (bvs) vkDestroyShaderModule(dev, bvs, nullptr);
            if (bfs) vkDestroyShaderModule(dev, bfs, nullptr);
            vkDestroyPipelineLayout(dev, blayout, nullptr);
            vkDestroyDescriptorSetLayout(dev, bsl, nullptr);
            vkDestroyDescriptorPool(dev, bpool, nullptr);
            vkDestroySampler(dev, samp, nullptr);
            vkDestroyImageView(dev, bcview, nullptr);
            vkDestroyImage(dev, bcimg, nullptr);
            vkFreeMemory(dev, bcmem, nullptr);
            destroy_buf(dev, staging);
        }
        vkDestroyPipeline(dev, pipe, nullptr);
        if (vs) vkDestroyShaderModule(dev, vs, nullptr);
        if (fs) vkDestroyShaderModule(dev, fs, nullptr);
        vkDestroyPipelineLayout(dev, layout, nullptr);
        vkDestroyDescriptorSetLayout(dev, sl, nullptr);
        vkDestroyDescriptorPool(dev, dpool, nullptr);
    }

    // Tile cache. 16x16 tiles, RGBA8 atlas the size of the material, hardware
    // linear filter on the shade. A sliding window of half the texture.
    if (base_w < 32 || base_h < 32) {
        std::cout << "bench tile_cache skipped small_texture\n";
    } else {
        const uint32_t tiles_x = (base_w + 15) / 16;
        const uint32_t tiles_y = (base_h + 15) / 16;
        const uint32_t ntiles = tiles_x * tiles_y;
        const uint32_t win_w = std::max(16u, base_w / 2);
        const uint32_t win_h = std::max(16u, base_h / 2);
        Buf req = make_buf(dev, phys, ntiles * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           nullptr, false);
        Buf resident = make_buf(dev, phys, ntiles * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                nullptr, false);
        VkImage atlas = VK_NULL_HANDLE;
        VkImageView atlas_view = VK_NULL_HANDLE;
        VkDeviceMemory atlas_mem = VK_NULL_HANDLE;
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = VK_FORMAT_R8G8B8A8_UNORM;
        ii.extent = {base_w, base_h, 1};
        ii.mipLevels = 1;
        ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        CHECK(vkCreateImage(dev, &ii, nullptr, &atlas));
        VkMemoryRequirements mreq{};
        vkGetImageMemoryRequirements(dev, atlas, &mreq);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mreq.size;
        try {
            ai.memoryTypeIndex = find_mem(phys, mreq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        } catch (const std::runtime_error&) {
            ai.memoryTypeIndex =
                find_mem(phys, mreq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        CHECK(vkAllocateMemory(dev, &ai, nullptr, &atlas_mem));
        CHECK(vkBindImageMemory(dev, atlas, atlas_mem, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = atlas;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_R8G8B8A8_UNORM;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        CHECK(vkCreateImageView(dev, &vi, nullptr, &atlas_view));
        std::cout << "tile_cache_bytes " << (mreq.size + ntiles * 8) << "\n";
        std::cout << "note tile_cache rgba8_atlas tile_16 window_half feedback_then_fill_then_hw_filter\n";
        const uint32_t shade_w = cpu_dev ? 128u : 1920u;
        const uint32_t shade_h = cpu_dev ? 128u : 1080u;
        VkSampler tsamp = VK_NULL_HANDLE;
        VkSamplerCreateInfo tsci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        tsci.magFilter = VK_FILTER_LINEAR;
        tsci.minFilter = VK_FILTER_LINEAR;
        tsci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        tsci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        CHECK(vkCreateSampler(dev, &tsci, nullptr, &tsamp));
        auto pipe_for = [&](const std::string& spv, VkDescriptorSetLayoutBinding* binds, uint32_t nb, VkPipeline& p,
                            VkPipelineLayout& pl, VkDescriptorSet& set, VkDescriptorBufferInfo* infos, uint32_t nbuf,
                            VkImageView iv, VkImageLayout layout_img, uint32_t image_binding, bool sampled) {
            p = VK_NULL_HANDLE;
            VkShaderModule mod = shader_mod(dev, spv);
            if (!mod) return;
            VkDescriptorSetLayout sl = VK_NULL_HANDLE;
            VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            lci.bindingCount = nb;
            lci.pBindings = binds;
            CHECK(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &sl));
            VkPushConstantRange pcr{};
            pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcr.size = 16;
            VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &sl;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pcr;
            CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));
            VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            st.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            st.module = mod;
            st.pName = "main";
            VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpi.stage = st;
            cpi.layout = pl;
            if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &p) != VK_SUCCESS) p = VK_NULL_HANDLE;
            vkDestroyShaderModule(dev, mod, nullptr);
            VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8},
                                          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                                          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}};
            VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            dp.maxSets = 1;
            dp.poolSizeCount = 3;
            dp.pPoolSizes = ps;
            VkDescriptorPool dpool = VK_NULL_HANDLE;
            CHECK(vkCreateDescriptorPool(dev, &dp, nullptr, &dpool));
            VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dai.descriptorPool = dpool;
            dai.descriptorSetCount = 1;
            dai.pSetLayouts = &sl;
            CHECK(vkAllocateDescriptorSets(dev, &dai, &set));
            VkWriteDescriptorSet wr[8]{};
            VkDescriptorImageInfo ii{};
            ii.sampler = sampled ? tsamp : VK_NULL_HANDLE;
            ii.imageView = iv;
            ii.imageLayout = layout_img;
            uint32_t bi = 0;
            for (uint32_t i = 0; i < nb; ++i) {
                wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wr[i].dstSet = set;
                wr[i].dstBinding = binds[i].binding;
                wr[i].descriptorCount = 1;
                wr[i].descriptorType = binds[i].descriptorType;
                if (binds[i].binding == image_binding) wr[i].pImageInfo = &ii;
                else {
                    wr[i].pBufferInfo = &infos[bi++];
                }
            }
            vkUpdateDescriptorSets(dev, nb, wr, 0, nullptr);
            (void)nbuf;
            (void)dpool;
            (void)sl;
        };
        VkPipeline req_p = VK_NULL_HANDLE, fill_p = VK_NULL_HANDLE, shade_p = VK_NULL_HANDLE;
        VkPipelineLayout req_l = VK_NULL_HANDLE, fill_l = VK_NULL_HANDLE, shade_l = VK_NULL_HANDLE;
        VkDescriptorSet req_s = VK_NULL_HANDLE, fill_s = VK_NULL_HANDLE, shade_s = VK_NULL_HANDLE;
        VkDescriptorSetLayoutBinding rb{};
        rb.binding = 0;
        rb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        rb.descriptorCount = 1;
        rb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorBufferInfo rinfo{req.buffer, 0, req.size};
        pipe_for(shader_dir + "/ntc_tile_request.comp.spv", &rb, 1, req_p, req_l, req_s, &rinfo, 1, VK_NULL_HANDLE,
                 VK_IMAGE_LAYOUT_GENERAL, 99, false);
        VkDescriptorSetLayoutBinding fb[5]{};
        uint32_t fbind[5] = {0, 1, 3, 4, 5};
        VkDescriptorType ftype[5] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
        for (int i = 0; i < 5; ++i) {
            fb[i].binding = fbind[i];
            fb[i].descriptorType = ftype[i];
            fb[i].descriptorCount = 1;
            fb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorBufferInfo finfo[4] = {{desc.buffer, 0, desc.size},
                                           {blob.buffer, 0, blob.size},
                                           {req.buffer, 0, req.size},
                                           {resident.buffer, 0, resident.size}};
        pipe_for(shader_dir + "/ntc_tile_fill.comp.spv", fb, 5, fill_p, fill_l, fill_s, finfo, 4, atlas_view,
                 VK_IMAGE_LAYOUT_GENERAL, 5, false);
        VkDescriptorSetLayoutBinding sb[2]{};
        sb[0].binding = 0;
        sb[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sb[0].descriptorCount = 1;
        sb[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        sb[1].binding = 1;
        sb[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sb[1].descriptorCount = 1;
        sb[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorBufferInfo sinfo[1] = {{outb.buffer, 0, outb.size}};
        pipe_for(shader_dir + "/ntc_tile_shade.comp.spv", sb, 2, shade_p, shade_l, shade_s, sinfo, 1, atlas_view,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, true);
        if (!req_p || !fill_p || !shade_p) {
            std::cout << "bench tile_cache_cold skipped pipeline\n";
            std::cout << "bench tile_cache_steady skipped pipeline\n";
        } else {
            auto frame = [&](VkCommandBuffer c, uint32_t origin_x, bool to_sample, bool keep_atlas) {
                vkCmdFillBuffer(c, req.buffer, 0, VK_WHOLE_SIZE, 0);
                VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                bb.buffer = req.buffer;
                bb.size = VK_WHOLE_SIZE;
                vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                                     1, &bb, 0, nullptr);
                uint32_t rpc[4] = {shade_w | (tiles_x << 16), shade_h, win_w | (origin_x << 16), win_h};
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, req_p);
                vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, req_l, 0, 1, &req_s, 0, nullptr);
                vkCmdPushConstants(c, req_l, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, rpc);
                vkCmdDispatch(c, (shade_w + 7) / 8, (shade_h + 7) / 8, 1);
                VkBufferMemoryBarrier br = bb;
                br.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                br.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                                     nullptr, 1, &br, 0, nullptr);
                VkImageMemoryBarrier ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                ib.oldLayout = keep_atlas ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
                ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                ib.image = atlas;
                ib.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                ib.subresourceRange.levelCount = 1;
                ib.subresourceRange.layerCount = 1;
                ib.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                                     nullptr, 0, nullptr, 1, &ib);
                uint32_t fpc[4] = {tiles_x, 16, base_w, base_h};
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, fill_p);
                vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, fill_l, 0, 1, &fill_s, 0, nullptr);
                vkCmdPushConstants(c, fill_l, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, fpc);
                vkCmdDispatch(c, ntiles, 1, 1);
                VkImageMemoryBarrier is = ib;
                is.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                is.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                is.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                is.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                                     nullptr, 0, nullptr, 1, &is);
                if (to_sample) {
                    uint32_t spc[4] = {shade_w, shade_h, win_w | (origin_x << 16), win_h};
                    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, shade_p);
                    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, shade_l, 0, 1, &shade_s, 0, nullptr);
                    vkCmdPushConstants(c, shade_l, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, spc);
                    vkCmdDispatch(c, (shade_w + 7) / 8, (shade_h + 7) / 8, 1);
                }
                (void)tiles_y;
            };
            vkCmdFillBuffer(cmd, resident.buffer, 0, VK_WHOLE_SIZE, 0);
            // resident clear must be submitted before the cold frame. Do it inside the timed cold record.
            auto cold = [&](VkCommandBuffer c) {
                vkCmdFillBuffer(c, resident.buffer, 0, VK_WHOLE_SIZE, 0);
                VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                bb.buffer = resident.buffer;
                bb.size = VK_WHOLE_SIZE;
                vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                                     1, &bb, 0, nullptr);
                frame(c, 0, true, false);
            };
            PassStats cold_st = time_records(cold, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            report("tile_cache_cold", cold_st, double(shade_w) * shade_h, cpu_dev);
            std::vector<double> steady;
            for (int f = 1; f <= std::max(iters, 4); ++f) {
                uint32_t ox = uint32_t((f * 16) % std::max(1, int(base_w - win_w)));
                auto rec = [&](VkCommandBuffer c) { frame(c, ox, true, true); };
                // One iteration each so the resident mask carries across frames.
                VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                CHECK(vkResetCommandBuffer(cmd, 0));
                CHECK(vkBeginCommandBuffer(cmd, &begin));
                vkCmdResetQueryPool(cmd, qpool, 0, 2);
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, 0);
                rec(cmd);
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, 1);
                CHECK(vkEndCommandBuffer(cmd));
                submit(cmd);
                uint64_t stamps[2] = {};
                vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(stamps), stamps, 8,
                                      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
                if (stamps[1] > stamps[0] && period > 0.f) steady.push_back(double(stamps[1] - stamps[0]) * double(period) / 1e6);
            }
            report("tile_cache_steady", summarize(steady), double(shade_w) * shade_h, cpu_dev);
            std::cout << "tile_cache_shade " << shade_w << " " << shade_h << " window " << win_w << " " << win_h << "\n";
        }
        vkDestroySampler(dev, tsamp, nullptr);
        vkDestroyImageView(dev, atlas_view, nullptr);
        vkDestroyImage(dev, atlas, nullptr);
        vkFreeMemory(dev, atlas_mem, nullptr);
        destroy_buf(dev, req);
        destroy_buf(dev, resident);
    }

    destroy_buf(dev, scratch);
    destroy_buf(dev, outb);
    destroy_buf(dev, blob);
    destroy_buf(dev, desc);
    vkDestroyQueryPool(dev, qpool, nullptr);
    vkDestroyCommandPool(dev, pool, nullptr);
}
