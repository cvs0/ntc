// Vulkan 1.3 compute (and optional dynamic-rendering) host.
// Required decode path: fp32 dots, no vendor extension.
// Optional fp16 dot is used only when shaderFloat16 is reported.
// Cooperative matrix is queried and not used: one texel does not fill a tile.
// Subgroup size is read from the device, never assumed.

#include "ntc/ntc.hpp"

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(expr)                                                                  \
    do {                                                                             \
        VkResult _r = (expr);                                                        \
        if (_r != VK_SUCCESS) {                                                      \
            throw std::runtime_error(std::string(#expr) + " -> " + std::to_string(_r)); \
        }                                                                            \
    } while (0)

static std::vector<uint8_t> read_bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static uint32_t find_mem(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags) return i;
    }
    throw std::runtime_error("no memory type");
}

struct Buf {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

static Buf make_buf(VkDevice dev, VkPhysicalDevice phys, VkDeviceSize size, VkBufferUsageFlags usage,
                    const void* src) {
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
    ai.memoryTypeIndex =
        find_mem(phys, req.memoryTypeBits,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.memory));
    CHECK(vkBindBufferMemory(dev, b.buffer, b.memory, 0));
    CHECK(vkMapMemory(dev, b.memory, 0, req.size, 0, &b.mapped));
    if (src) std::memcpy(b.mapped, src, static_cast<size_t>(size));
    else std::memset(b.mapped, 0, static_cast<size_t>(size));
    return b;
}

static void destroy_buf(VkDevice dev, Buf& b) {
    if (b.mapped) vkUnmapMemory(dev, b.memory);
    if (b.buffer) vkDestroyBuffer(dev, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(dev, b.memory, nullptr);
}

static VkShaderModule shader_mod(VkDevice dev, const std::string& path) {
    auto code = read_bin(path);
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule m = VK_NULL_HANDLE;
    CHECK(vkCreateShaderModule(dev, &ci, nullptr, &m));
    return m;
}

static float max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: ntc_vk file.ntc shader_dir\n";
        return 2;
    }
    try {
        const std::string ntc_path = argv[1];
        const std::string dir = argv[2];
        ntc::TextureSet ts = ntc::load_file(ntc_path);
        auto cpu = ntc::decode_mip(ts, 0);
        auto up = ntc::make_gpu_upload(ts, 0, ntc::kFilterPoint, 4);
        const uint32_t w = ts.width;
        const uint32_t h = ts.height;
        const uint32_t c = static_cast<uint32_t>(ts.num_channels());
        const VkDeviceSize out_bytes = static_cast<VkDeviceSize>(w) * h * c * 4u;

        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "ntc_vk";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        VkInstance instance = VK_NULL_HANDLE;
        CHECK(vkCreateInstance(&ici, nullptr, &instance));

        uint32_t ndev = 0;
        CHECK(vkEnumeratePhysicalDevices(instance, &ndev, nullptr));
        std::vector<VkPhysicalDevice> devs(ndev);
        CHECK(vkEnumeratePhysicalDevices(instance, &ndev, devs.data()));
        VkPhysicalDevice phys = VK_NULL_HANDLE;
        int best = -1;
        for (auto d : devs) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(d, &p);
            int score = 0;
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score = 3;
            else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score = 2;
            else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) score = 1;
            if (score > best) {
                best = score;
                phys = d;
            }
        }
        if (!phys) throw std::runtime_error("no device");

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys, &props);
        VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sg;
        vkGetPhysicalDeviceProperties2(phys, &p2);

        uint32_t next = 0;
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &next, nullptr);
        std::vector<VkExtensionProperties> exts(next);
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &next, exts.data());
        bool has_coop = false;
        bool has_f16ext = false;
        for (const auto& e : exts) {
            if (std::strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) has_coop = true;
            if (std::strcmp(e.extensionName, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) == 0) has_f16ext = true;
        }
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceShaderFloat16Int8Features f16q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDeviceVulkan13Features v13q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        v13q.pNext = &f16q;
        void* tail = &v13q;
        if (has_coop) {
            f16q.pNext = &cm;
        }
        VkPhysicalDeviceFeatures2 fq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        fq.pNext = tail;
        vkGetPhysicalDeviceFeatures2(phys, &fq);
        const bool float16 = f16q.shaderFloat16 == VK_TRUE;
        const bool int8 = f16q.shaderInt8 == VK_TRUE;
        const bool coop = has_coop && cm.cooperativeMatrix == VK_TRUE;
        const bool cpu_dev = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;

        std::cout << "device " << props.deviceName << " vendor 0x" << std::hex << props.vendorID << std::dec
                  << " type " << props.deviceType << " api " << VK_VERSION_MAJOR(props.apiVersion) << "."
                  << VK_VERSION_MINOR(props.apiVersion) << " driver " << props.driverVersion
                  << " subgroup " << sg.subgroupSize << " float16 " << float16 << " int8 " << int8
                  << " coopmat " << coop << " cpu " << cpu_dev << "\n";

        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());
        uint32_t qfam = 0;
        bool found_q = false;
        for (uint32_t i = 0; i < nq; ++i) {
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                qfam = i;
                found_q = true;
                break;
            }
        }
        if (!found_q) throw std::runtime_error("no compute queue");

        float prio = 1.f;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = qfam;
        qi.queueCount = 1;
        qi.pQueuePriorities = &prio;

        VkPhysicalDeviceShaderFloat16Int8Features f16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        f16.shaderFloat16 = float16 ? VK_TRUE : VK_FALSE;
        f16.shaderInt8 = VK_FALSE;
        VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        v13.dynamicRendering = v13q.dynamicRendering;
        v13.pNext = has_f16ext ? static_cast<void*>(&f16) : nullptr;
        VkPhysicalDeviceFeatures2 feat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        feat.pNext = &v13;
        feat.features.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;

        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        di.pNext = &feat;
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        VkDevice dev = VK_NULL_HANDLE;
        CHECK(vkCreateDevice(phys, &di, nullptr, &dev));
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(dev, qfam, 0, &queue);

        Buf desc = make_buf(dev, phys, up.desc.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, up.desc.data());
        Buf blob = make_buf(dev, phys, up.blob.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, up.blob.data());
        Buf outb = make_buf(dev, phys, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr);

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
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        CHECK(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &layout));

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = 16;
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VkPipelineLayout pl = VK_NULL_HANDLE;
        CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

        VkShaderModule mod = shader_mod(dev, dir + "/ntc_decompress.comp.spv");
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = stage;
        cpi.layout = pl;
        VkPipeline pipe = VK_NULL_HANDLE;
        CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));

        VkDescriptorPoolSize ps{};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = 6;
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 2;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &pool));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        CHECK(vkAllocateDescriptorSets(dev, &dsai, &set));
        VkDescriptorBufferInfo infos[3] = {
            {desc.buffer, 0, desc.size},
            {blob.buffer, 0, blob.size},
            {outb.buffer, 0, outb.size},
        };
        VkWriteDescriptorSet writes[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(dev, 3, writes, 0, nullptr);

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

        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        VkQueryPool qpool = VK_NULL_HANDLE;
        CHECK(vkCreateQueryPool(dev, &qpci, nullptr, &qpool));

        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vkBeginCommandBuffer(cmd, &begin));
        vkCmdResetQueryPool(cmd, qpool, 0, 2);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, nullptr);
        uint32_t pc[4] = {0, 0, 0, 0};
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, 0);
        vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, 1);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
        CHECK(vkEndCommandBuffer(cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
        CHECK(vkQueueWaitIdle(queue));

        std::vector<float> gpu(cpu.size());
        std::memcpy(gpu.data(), outb.mapped, static_cast<size_t>(out_bytes));
        const float err = max_abs(cpu, gpu);
        std::cout << "compute_max_abs " << err << "\n";

        uint64_t stamps[2] = {};
        vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(stamps), stamps, sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT);
        double ms = 0.0;
        if (stamps[1] > stamps[0] && props.limits.timestampPeriod > 0.f) {
            ms = double(stamps[1] - stamps[0]) * double(props.limits.timestampPeriod) / 1e6;
        }
        if (cpu_dev) std::cout << "timestamp_ms_lavapipe " << ms << " not_a_performance_claim\n";
        else std::cout << "timestamp_ms " << ms << "\n";

        if (err > 1e-3f) {
            std::cerr << "compute mismatch\n";
            return 1;
        }

        if (float16) {
            const std::string fp16_path = dir + "/ntc_decompress_fp16.comp.spv";
            std::ifstream probe(fp16_path);
            if (probe) {
                VkShaderModule m16 = shader_mod(dev, fp16_path);
                stage.module = m16;
                cpi.stage = stage;
                VkPipeline p16 = VK_NULL_HANDLE;
                CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &p16));
                CHECK(vkResetCommandBuffer(cmd, 0));
                CHECK(vkBeginCommandBuffer(cmd, &begin));
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p16);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, nullptr);
                vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
                vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                                     &mb, 0, nullptr, 0, nullptr);
                CHECK(vkEndCommandBuffer(cmd));
                CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
                CHECK(vkQueueWaitIdle(queue));
                std::vector<float> g16(cpu.size());
                std::memcpy(g16.data(), outb.mapped, static_cast<size_t>(out_bytes));
                std::cout << "fp16_max_abs " << max_abs(cpu, g16) << "\n";
                vkDestroyPipeline(dev, p16, nullptr);
                vkDestroyShaderModule(dev, m16, nullptr);
            }
        }

        std::cout << "ok\n";
        vkDestroyQueryPool(dev, qpool, nullptr);
        vkDestroyCommandPool(dev, cpool, nullptr);
        vkDestroyDescriptorPool(dev, pool, nullptr);
        vkDestroyPipeline(dev, pipe, nullptr);
        vkDestroyShaderModule(dev, mod, nullptr);
        vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyDescriptorSetLayout(dev, layout, nullptr);
        destroy_buf(dev, desc);
        destroy_buf(dev, blob);
        destroy_buf(dev, outb);
        vkDestroyDevice(dev, nullptr);
        vkDestroyInstance(instance, nullptr);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
    return 0;
}
