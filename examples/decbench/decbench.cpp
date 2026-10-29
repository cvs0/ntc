// ntc_decbench: decode microbenchmark and correctness check (Vulkan compute).
//
//   ntc_decbench [options] FILE.ntc [FILE.ntc ...]
//
// For every file:
//   1. check: each decoder variant decodes texel centers of mips 0..2 and is
//      compared against ntc::decode_mip on the CPU (max abs, PSNR).
//   2. bench: each variant decodes one sample per pixel at each screen size
//      and writes one rgba8 word (4 B). The BC baselines (one BC7 texture;
//      BC7 + BC5 + BC7 material) do one bilinear fetch per texture and write
//      the same 4 B. Timing is GPU timestamps around each dispatch.
//
// Variants: legacy (shaders/ntc_decode.glsl interpreter), exact, fma, fp16,
// i8 (shaders/ntc_fast.glsl), bc7, bc_mat.
//
// Options:
//   --device N          physical device index (default: first discrete, else 0)
//   --sizes LIST        WxH,WxH (default 1920x1080,2560x1440,3840x2160)
//   --iters N           timed dispatches per case (default 50)
//   --warmup N          untimed dispatches per case (default 10)
//   --uv-scale S        texels per pixel at mip 0, or auto (= 1.0)
//   --variants LIST     comma list (default all)
//   --check-only        skip timing
//   --check-mips N      mips to check against the CPU (default 3, 0 skips)
//   --json PATH         also write results as JSON
//   --shader-dir DIR    SPIR-V directory (default: next to the executable, then build/shaders)

#include "ntc/fast.hpp"
#include "ntc/ntc.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

#define VKC(x)                                                                          \
    do {                                                                                \
        VkResult r_ = (x);                                                              \
        if (r_ != VK_SUCCESS) {                                                         \
            std::fprintf(stderr, "vulkan error %d at %s:%d: %s\n", r_, __FILE__, __LINE__, #x); \
            std::exit(2);                                                               \
        }                                                                               \
    } while (0)

struct Ctx {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mem{};
    bool fp16 = false;
    bool dot = false;
    bool dot_accel = false;
    bool exec_props = false;
    bool bc = false;
    PFN_vkGetPipelineExecutablePropertiesKHR get_exec = nullptr;
    PFN_vkGetPipelineExecutableStatisticsKHR get_stats = nullptr;
};

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* map = nullptr;
};

uint32_t find_mem(const Ctx& c, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c.mem.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (c.mem.memoryTypes[i].propertyFlags & want) == want) return i;
    std::fprintf(stderr, "no memory type\n");
    std::exit(2);
}

Buf make_buf(const Ctx& c, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) {
    Buf b;
    b.size = std::max<VkDeviceSize>(size, 16);
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = b.size;
    bi.usage = usage;
    VKC(vkCreateBuffer(c.dev, &bi, nullptr, &b.buf));
    VkMemoryRequirements r;
    vkGetBufferMemoryRequirements(c.dev, b.buf, &r);
    VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.allocationSize = r.size;
    alloc_info.memoryTypeIndex = find_mem(c, r.memoryTypeBits, flags);
    VKC(vkAllocateMemory(c.dev, &alloc_info, nullptr, &b.mem));
    VKC(vkBindBufferMemory(c.dev, b.buf, b.mem, 0));
    if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) VKC(vkMapMemory(c.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
    return b;
}

void free_buf(const Ctx& c, Buf& b) {
    if (b.map) vkUnmapMemory(c.dev, b.mem);
    if (b.buf) vkDestroyBuffer(c.dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(c.dev, b.mem, nullptr);
    b = Buf{};
}

const VkMemoryPropertyFlags kHost = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

VkCommandBuffer begin_cmd(const Ctx& c) {
    VkCommandBufferAllocateInfo alloc_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc_info.commandPool = c.pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    VkCommandBuffer cb;
    VKC(vkAllocateCommandBuffers(c.dev, &alloc_info, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKC(vkBeginCommandBuffer(cb, &bi));
    return cb;
}

void end_cmd(const Ctx& c, VkCommandBuffer cb) {
    VKC(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VKC(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
    VKC(vkQueueWaitIdle(c.queue));
    vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
}

// Device-local buffer filled through a staging copy.
Buf upload_buf(const Ctx& c, const void* src, VkDeviceSize size, VkBufferUsageFlags usage) {
    Buf b = make_buf(c, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf s = make_buf(c, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHost);
    std::memset(s.map, 0, static_cast<size_t>(s.size));
    if (src) std::memcpy(s.map, src, static_cast<size_t>(size));
    VkCommandBuffer cb = begin_cmd(c);
    VkBufferCopy rg{0, 0, b.size};
    vkCmdCopyBuffer(cb, s.buf, b.buf, 1, &rg);
    end_cmd(c, cb);
    free_buf(c, s);
    return b;
}

struct Img {
    VkImage img = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

Img upload_bc(const Ctx& c, VkFormat fmt, int w, int h, const std::vector<uint8_t>& blocks) {
    Img im;
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = fmt;
    ii.extent = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VKC(vkCreateImage(c.dev, &ii, nullptr, &im.img));
    VkMemoryRequirements r;
    vkGetImageMemoryRequirements(c.dev, im.img, &r);
    VkMemoryAllocateInfo alloc_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_info.allocationSize = r.size;
    alloc_info.memoryTypeIndex = find_mem(c, r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VKC(vkAllocateMemory(c.dev, &alloc_info, nullptr, &im.mem));
    VKC(vkBindImageMemory(c.dev, im.img, im.mem, 0));
    Buf s = make_buf(c, blocks.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHost);
    std::memcpy(s.map, blocks.data(), blocks.size());
    VkCommandBuffer cb = begin_cmd(c);
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im.img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy rg{};
    rg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    rg.imageExtent = ii.extent;
    vkCmdCopyBufferToImage(cb, s.buf, im.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rg);
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    end_cmd(c, cb);
    free_buf(c, s);
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im.img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VKC(vkCreateImageView(c.dev, &vi, nullptr, &im.view));
    return im;
}

void free_img(const Ctx& c, Img& im) {
    if (im.view) vkDestroyImageView(c.dev, im.view, nullptr);
    if (im.img) vkDestroyImage(c.dev, im.img, nullptr);
    if (im.mem) vkFreeMemory(c.dev, im.mem, nullptr);
    im = Img{};
}

std::vector<uint32_t> read_spv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const auto n = static_cast<size_t>(f.tellg());
    std::vector<uint32_t> w(n / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(w.data()), static_cast<std::streamsize>(w.size() * 4));
    return w;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

struct Pc {
    uint32_t w, h, mip, pad;
    float sx, sy, ox, oy;
};

struct Pipe {
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorPool dp = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::string stats;  // register / instruction statistics when available
};

void free_pipe(const Ctx& c, Pipe& p) {
    if (p.pipe) vkDestroyPipeline(c.dev, p.pipe, nullptr);
    if (p.pl) vkDestroyPipelineLayout(c.dev, p.pl, nullptr);
    if (p.dp) vkDestroyDescriptorPool(c.dev, p.dp, nullptr);
    if (p.dsl) vkDestroyDescriptorSetLayout(c.dev, p.dsl, nullptr);
    p = Pipe{};
}

std::string pipeline_stats(const Ctx& c, VkPipeline pipe) {
    if (!c.exec_props || !c.get_exec || !c.get_stats) return "";
    VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pi.pipeline = pipe;
    uint32_t n = 0;
    if (c.get_exec(c.dev, &pi, &n, nullptr) != VK_SUCCESS || n == 0) return "";
    std::ostringstream os;
    VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
    ei.pipeline = pipe;
    ei.executableIndex = 0;
    uint32_t ns = 0;
    if (c.get_stats(c.dev, &ei, &ns, nullptr) != VK_SUCCESS) return "";
    std::vector<VkPipelineExecutableStatisticKHR> st(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
    if (c.get_stats(c.dev, &ei, &ns, st.data()) != VK_SUCCESS) return "";
    bool first = true;
    for (const auto& s : st) {
        std::string name = s.name;
        std::string lname = name;
        for (auto& ch : lname) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        const char* keys[] = {"gpr", "register", "spill", "scratch", "instruction", "occupancy", "wave", "local", "private"};
        bool keep = false;
        for (const char* k : keys) keep = keep || lname.find(k) != std::string::npos;
        if (!keep) continue;
        os << (first ? "" : "; ") << name << "=";
        first = false;
        switch (s.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: os << (s.value.b32 ? "true" : "false"); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: os << s.value.i64; break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: os << s.value.u64; break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: os << s.value.f64; break;
            default: os << "?";
        }
    }
    return os.str();
}

// types: one VkDescriptorType per binding.
Pipe make_pipe(const Ctx& c, const std::string& spv_path, const std::vector<VkDescriptorType>& types,
               const std::vector<int32_t>& spec, uint32_t pc_size) {
    Pipe p;
    std::vector<VkDescriptorSetLayoutBinding> binds(types.size());
    std::map<VkDescriptorType, uint32_t> counts;
    for (size_t i = 0; i < types.size(); ++i) {
        binds[i].binding = static_cast<uint32_t>(i);
        binds[i].descriptorType = types[i];
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        counts[types[i]]++;
    }
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = static_cast<uint32_t>(binds.size());
    lci.pBindings = binds.data();
    VKC(vkCreateDescriptorSetLayout(c.dev, &lci, nullptr, &p.dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &p.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VKC(vkCreatePipelineLayout(c.dev, &plci, nullptr, &p.pl));

    const auto code = read_spv(spv_path);
    if (code.empty()) {
        std::fprintf(stderr, "missing %s\n", spv_path.c_str());
        std::exit(2);
    }
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = code.size() * 4;
    mi.pCode = code.data();
    VkShaderModule mod;
    VKC(vkCreateShaderModule(c.dev, &mi, nullptr, &mod));
    std::vector<VkSpecializationMapEntry> me(spec.size());
    for (size_t i = 0; i < spec.size(); ++i)
        me[i] = {static_cast<uint32_t>(i), static_cast<uint32_t>(i * 4), 4};
    VkSpecializationInfo si{};
    si.mapEntryCount = static_cast<uint32_t>(me.size());
    si.pMapEntries = me.data();
    si.dataSize = spec.size() * 4;
    si.pData = spec.data();
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = mod;
    ci.stage.pName = "main";
    ci.stage.pSpecializationInfo = spec.empty() ? nullptr : &si;
    ci.layout = p.pl;
    if (c.exec_props) ci.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    VKC(vkCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipe));
    vkDestroyShaderModule(c.dev, mod, nullptr);
    p.stats = pipeline_stats(c, p.pipe);

    std::vector<VkDescriptorPoolSize> ps;
    for (auto& kv : counts) ps.push_back({kv.first, kv.second});
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;
    dpi.poolSizeCount = static_cast<uint32_t>(ps.size());
    dpi.pPoolSizes = ps.data();
    VKC(vkCreateDescriptorPool(c.dev, &dpi, nullptr, &p.dp));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = p.dp;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &p.dsl;
    VKC(vkAllocateDescriptorSets(c.dev, &dai, &p.set));
    return p;
}

void bind_buf(const Ctx& c, Pipe& p, uint32_t binding, VkDescriptorType t, VkBuffer b, VkDeviceSize range) {
    VkDescriptorBufferInfo bi{b, 0, range};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = p.set;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = t;
    w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(c.dev, 1, &w, 0, nullptr);
}

void bind_img(const Ctx& c, Pipe& p, uint32_t binding, VkImageView v, VkSampler s) {
    VkDescriptorImageInfo ii{s, v, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = p.set;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(c.dev, 1, &w, 0, nullptr);
}

void barrier(VkCommandBuffer cb) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0,
                         nullptr);
}

// One dispatch, then copy `bytes` of `out` back to the host.
std::vector<uint8_t> run_once(const Ctx& c, const Pipe& p, const void* pc, uint32_t pc_size, uint32_t w, uint32_t h,
                              const Buf& out, VkDeviceSize bytes) {
    Buf rb = make_buf(c, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, kHost);
    VkCommandBuffer cb = begin_cmd(c);
    vkCmdFillBuffer(cb, out.buf, 0, VK_WHOLE_SIZE, 0x7FC00000u);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pl, 0, 1, &p.set, 0, nullptr);
    vkCmdPushConstants(cb, p.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size, pc);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    barrier(cb);
    VkBufferCopy rg{0, 0, bytes};
    vkCmdCopyBuffer(cb, out.buf, rb.buf, 1, &rg);
    end_cmd(c, cb);
    std::vector<uint8_t> v(static_cast<size_t>(bytes));
    std::memcpy(v.data(), rb.map, v.size());
    free_buf(c, rb);
    return v;
}

struct Timing {
    double median_ms = 0, p95_ms = 0, min_ms = 0;
};

Timing time_pipe(const Ctx& c, const Pipe& p, const void* pc, uint32_t pc_size, uint32_t w, uint32_t h, int warmup,
                 int iters) {
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = static_cast<uint32_t>(2 * iters);
    VkQueryPool qp;
    VKC(vkCreateQueryPool(c.dev, &qi, nullptr, &qp));
    VkCommandBuffer cb = begin_cmd(c);
    vkCmdResetQueryPool(cb, qp, 0, qi.queryCount);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pl, 0, 1, &p.set, 0, nullptr);
    vkCmdPushConstants(cb, p.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size, pc);
    for (int i = 0; i < warmup + iters; ++i) {
        const int t = i - warmup;
        if (t >= 0) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, static_cast<uint32_t>(2 * t));
        vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
        if (t >= 0) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, static_cast<uint32_t>(2 * t + 1));
        barrier(cb);
    }
    end_cmd(c, cb);
    std::vector<uint64_t> ts(qi.queryCount);
    VKC(vkGetQueryPoolResults(c.dev, qp, 0, qi.queryCount, ts.size() * 8, ts.data(), 8,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    vkDestroyQueryPool(c.dev, qp, nullptr);
    std::vector<double> ms(iters);
    for (int i = 0; i < iters; ++i)
        ms[i] = static_cast<double>(ts[2 * i + 1] - ts[2 * i]) * c.props.limits.timestampPeriod * 1e-6;
    std::sort(ms.begin(), ms.end());
    Timing t;
    t.min_ms = ms.front();
    t.median_ms = ms[ms.size() / 2];
    t.p95_ms = ms[std::min(ms.size() - 1, static_cast<size_t>(std::ceil(0.95 * ms.size())) - 1)];
    return t;
}

Ctx make_ctx(int want_dev) {
    Ctx c;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "ntc_decbench";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &c.inst) != VK_SUCCESS) {
        std::fprintf(stderr, "no vulkan 1.3 instance\n");
        std::exit(3);
    }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(c.inst, &n, nullptr);
    if (n == 0) {
        std::fprintf(stderr, "no vulkan device\n");
        std::exit(3);
    }
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(c.inst, &n, devs.data());
    int pick = want_dev;
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        std::printf("device %u %s api %u.%u.%u%s\n", i, p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion),
                    VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion),
                    p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? " discrete" : "");
        if (pick < 0 && p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) pick = static_cast<int>(i);
    }
    if (pick < 0 || pick >= static_cast<int>(n)) pick = 0;
    c.phys = devs[pick];
    vkGetPhysicalDeviceProperties(c.phys, &c.props);
    vkGetPhysicalDeviceMemoryProperties(c.phys, &c.mem);
    if (c.props.apiVersion < VK_API_VERSION_1_3) {
        std::fprintf(stderr, "device %s is below Vulkan 1.3\n", c.props.deviceName);
        std::exit(3);
    }

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &qn, qf.data());
    c.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < qn; ++i)
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].timestampValidBits > 0) {
            c.qfam = i;
            break;
        }
    if (c.qfam == UINT32_MAX) {
        std::fprintf(stderr, "no compute queue with timestamps\n");
        std::exit(3);
    }

    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(c.phys, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> exts(en);
    vkEnumerateDeviceExtensionProperties(c.phys, nullptr, &en, exts.data());
    bool has_exec = false;
    for (auto& e : exts)
        if (std::strcmp(e.extensionName, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME) == 0) has_exec = true;

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR exq{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features v13q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features v12q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12q.pNext = &v13q;
    if (has_exec) v13q.pNext = &exq;
    VkPhysicalDeviceFeatures2 fq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    fq.pNext = &v12q;
    vkGetPhysicalDeviceFeatures2(c.phys, &fq);
    c.fp16 = v12q.shaderFloat16 == VK_TRUE;
    c.dot = v13q.shaderIntegerDotProduct == VK_TRUE;
    c.exec_props = has_exec && exq.pipelineExecutableInfo == VK_TRUE;
    c.bc = fq.features.textureCompressionBC == VK_TRUE;

    VkPhysicalDeviceShaderIntegerDotProductProperties dp{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &dp;
    vkGetPhysicalDeviceProperties2(c.phys, &p2);
    c.dot_accel = dp.integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE;

    float prio = 1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = c.qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR exon{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    exon.pipelineExecutableInfo = c.exec_props ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    v13.shaderIntegerDotProduct = c.dot ? VK_TRUE : VK_FALSE;
    if (c.exec_props) v13.pNext = &exon;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.shaderFloat16 = c.fp16 ? VK_TRUE : VK_FALSE;
    v12.pNext = &v13;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.features.textureCompressionBC = c.bc ? VK_TRUE : VK_FALSE;
    f2.pNext = &v12;
    const char* ext_names[] = {VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = c.exec_props ? 1u : 0u;
    dci.ppEnabledExtensionNames = ext_names;
    VKC(vkCreateDevice(c.phys, &dci, nullptr, &c.dev));
    vkGetDeviceQueue(c.dev, c.qfam, 0, &c.queue);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = c.qfam;
    VKC(vkCreateCommandPool(c.dev, &pci, nullptr, &c.pool));
    if (c.exec_props) {
        c.get_exec = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
            vkGetDeviceProcAddr(c.dev, "vkGetPipelineExecutablePropertiesKHR"));
        c.get_stats = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
            vkGetDeviceProcAddr(c.dev, "vkGetPipelineExecutableStatisticsKHR"));
    }
    std::printf("using %s driver %u fp16 %d int_dot %d int_dot_accel %d exec_stats %d bc %d max_ubo %u\n",
                c.props.deviceName, c.props.driverVersion, c.fp16, c.dot, c.dot_accel, c.exec_props, c.bc,
                c.props.limits.maxUniformBufferRange);
    return c;
}

struct Variant {
    std::string name;
    std::string spv;  // empty for BC
    bool ntc_fast = false;
    bool exact = false;   // must be bit-identical
    bool tight = false;   // must be within tolerance
};

std::vector<std::pair<int, int>> parse_sizes(const std::string& s) {
    std::vector<std::pair<int, int>> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        int w = 0, h = 0;
        if (std::sscanf(tok.c_str(), "%dx%d", &w, &h) == 2 && w > 0 && h > 0) out.push_back({w, h});
    }
    return out;
}

std::string exe_dir(const char* argv0) {
    std::string s = argv0;
    const auto k = s.find_last_of("/\\");
    return k == std::string::npos ? std::string(".") : s.substr(0, k);
}

bool file_exists(const std::string& p) { return std::ifstream(p).good(); }

double psnr(const std::vector<float>& a, const std::vector<float>& b) {
    double se = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        se += d * d;
    }
    const double mse = se / std::max<size_t>(1, a.size());
    return mse <= 0 ? 999.0 : 10.0 * std::log10(1.0 / mse);
}

// Synthetic BC data: random endpoints and indices, mode 6 for BC7 (bit 6
// set, 0x40) so every block takes the same decode path.
std::vector<uint8_t> synth_blocks(int w, int h, bool bc7, uint32_t seed) {
    const int bw = (w + 3) / 4, bh = (h + 3) / 4;
    std::vector<uint8_t> b(static_cast<size_t>(bw) * bh * 16);
    std::mt19937 rng(seed);
    for (auto& x : b) x = static_cast<uint8_t>(rng());
    if (bc7)
        for (size_t i = 0; i < b.size(); i += 16) b[i] = 0x40;
    return b;
}

}  // namespace

int main(int argc, char** argv) {
    int want_dev = -1, iters = 50, warmup = 10;
    std::string sizes_s = "1920x1080,2560x1440,3840x2160", uv_s = "auto", variants_s, json_path, shader_dir;
    bool check_only = false;
    int check_mips_max = 3;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--device") want_dev = std::atoi(next().c_str());
        else if (a == "--sizes") sizes_s = next();
        else if (a == "--iters") iters = std::max(1, std::atoi(next().c_str()));
        else if (a == "--warmup") warmup = std::max(0, std::atoi(next().c_str()));
        else if (a == "--uv-scale") uv_s = next();
        else if (a == "--variants") variants_s = next();
        else if (a == "--check-only") check_only = true;
        else if (a == "--check-mips") check_mips_max = std::max(0, std::atoi(next().c_str()));
        else if (a == "--json") json_path = next();
        else if (a == "--shader-dir") shader_dir = next();
        else if (a == "-h" || a == "--help") {
            std::printf("usage: ntc_decbench [--device N] [--sizes WxH,...] [--iters N] [--warmup N] [--uv-scale S|auto]\n"
                        "                    [--variants legacy,exact,fma,fp16,i8,bc7,bc_mat] [--check-only] [--check-mips N]\n"
                        "                    [--json PATH]\n"
                        "                    [--shader-dir DIR] FILE.ntc ...\n");
            return 0;
        } else files.push_back(a);
    }
    if (files.empty()) {
        std::fprintf(stderr, "no .ntc file given (bench/make_bench_ntc.py writes synthetic ones)\n");
        return 1;
    }
    if (shader_dir.empty()) {
        const std::string d = exe_dir(argv[0]);
        for (const std::string& cand : {d + "/shaders", d + "/../shaders", d, std::string("build/shaders")})
            if (file_exists(cand + "/ntc_fast_fma.comp.spv")) {
                shader_dir = cand;
                break;
            }
        if (shader_dir.empty()) shader_dir = "build/shaders";
    }
    const auto sizes = parse_sizes(sizes_s);
    const double uv_scale = uv_s == "auto" ? 1.0 : std::atof(uv_s.c_str());

    Ctx c = make_ctx(want_dev);
    std::vector<Variant> all = {
        {"legacy", "ntc_legacy_bench.comp.spv", false, false, true},
        {"exact", "ntc_fast_exact.comp.spv", true, true, true},
        {"fma", "ntc_fast_fma.comp.spv", true, false, true},
        {"fp16", "ntc_fast_fp16.comp.spv", true, false, false},
        {"i8", "ntc_fast_i8.comp.spv", true, false, false},
        {"bc7", "", false, false, false},
        {"bc_mat", "", false, false, false},
    };
    std::vector<Variant> vars;
    for (const auto& v : all) {
        if (!variants_s.empty() && ("," + variants_s + ",").find("," + v.name + ",") == std::string::npos) continue;
        if (v.name == "fp16" && !c.fp16) {
            std::printf("skip fp16: shaderFloat16 unsupported\n");
            continue;
        }
        if (v.name == "i8" && !c.dot) {
            std::printf("skip i8: shaderIntegerDotProduct unsupported\n");
            continue;
        }
        if ((v.name == "bc7" || v.name == "bc_mat") && !c.bc) {
            std::printf("skip %s: textureCompressionBC unsupported\n", v.name.c_str());
            continue;
        }
        vars.push_back(v);
    }

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.maxLod = 0.f;
    VkSampler sampler;
    VKC(vkCreateSampler(c.dev, &sci, nullptr, &sampler));

    int failures = 0;
    std::ostringstream js;
    js << "{\"device\":\"" << c.props.deviceName << "\",\"driver\":" << c.props.driverVersion
       << ",\"int_dot_accel\":" << (c.dot_accel ? "true" : "false") << ",\"files\":[";
    bool first_file = true;

    for (const std::string& path : files) {
        const auto bytes = read_file(path);
        if (bytes.empty()) {
            std::fprintf(stderr, "cannot read %s\n", path.c_str());
            ++failures;
            continue;
        }
        ntc::TextureSet ts;
        try {
            ts = ntc::parse(bytes.data(), bytes.size());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: %s\n", path.c_str(), e.what());
            ++failures;
            continue;
        }
        ntc::FastUpload fu = ntc::make_fast_upload(ts);
        const ntc::GpuUpload lu = ntc::make_gpu_upload(ts, 0, ntc::kFilterPoint, 4);
        const ntc::SampleCost lc = ntc::legacy_sample_cost(ts);
        const ntc::SampleCost fc = ntc::fast_sample_cost(ts);
        const int nch = ts.num_channels();
        std::printf("\nfile %s %ux%u ch %d grids %zu layers %zu mips %u file_bytes %zu\n", path.c_str(), ts.width,
                    ts.height, nch, ts.grids.size(), ts.layers.size(), ts.num_mips, bytes.size());
        std::printf("cost legacy: macs %d loads %d (latent %d weight %d scale %d) load_bytes %d alu~%d\n", lc.macs,
                    lc.latent_loads + lc.weight_loads + lc.scale_loads, lc.latent_loads, lc.weight_loads,
                    lc.scale_loads, lc.load_bytes, lc.alu);
        std::printf("cost fast:   macs %d loads %d (latent) uniform_words %d load_bytes %d alu~%d\n", fc.macs,
                    fc.latent_loads, fc.uniform_words, fc.load_bytes, fc.alu);
        std::printf("resident: legacy blob %zu B, fast latents %u B + params %zu B, bc7 %d B, bc_mat %d B\n",
                    lu.blob.size(), fu.latent_bytes, fu.params.size() * 4,
                    ((ts.width + 3) / 4) * ((ts.height + 3) / 4) * 16,
                    3 * ((ts.width + 3) / 4) * ((ts.height + 3) / 4) * 16);
        if (!fu.ok) std::printf("fast layout unavailable: %s\n", fu.why.c_str());
        if (!first_file) js << ",";
        first_file = false;
        js << "{\"path\":\"" << path << "\",\"width\":" << ts.width << ",\"height\":" << ts.height
           << ",\"channels\":" << nch << ",\"macs\":" << fc.macs << ",\"legacy_loads\":"
           << (lc.latent_loads + lc.weight_loads + lc.scale_loads) << ",\"fast_loads\":" << fc.latent_loads
           << ",\"latent_bytes_fast\":" << fu.latent_bytes << ",\"blob_bytes_legacy\":" << lu.blob.size()
           << ",\"check\":[";

        // Buffers.
        const uint32_t ubo_range = std::min<uint32_t>(65536u, c.props.limits.maxUniformBufferRange);
        std::vector<uint32_t> params = fu.params;
        params.resize(ubo_range / 4, 0u);
        Buf b_params{}, b_lat{}, b_desc{}, b_blob{};
        if (fu.ok && fu.params.size() * 4 <= ubo_range) {
            b_params = upload_buf(c, params.data(), params.size() * 4, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            b_lat = upload_buf(c, fu.latents.data(), fu.latents.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        } else if (fu.ok) {
            fu.ok = false;
            fu.why = "params exceed maxUniformBufferRange";
        }
        b_desc = upload_buf(c, lu.desc.data(), lu.desc.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::vector<uint8_t> blob = lu.blob;
        blob.resize((blob.size() + 3) & ~size_t(3), 0);
        b_blob = upload_buf(c, blob.data(), blob.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        size_t max_px = static_cast<size_t>(ts.width) * ts.height * nch * 4;
        for (auto& s : sizes) max_px = std::max(max_px, static_cast<size_t>(s.first) * s.second * 4);
        Buf b_out = make_buf(c, max_px, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        auto fast_spec = [&](int mode) {
            std::vector<int32_t> s(fu.spec.begin(), fu.spec.end());
            s.push_back(mode);
            return s;
        };
        const std::vector<VkDescriptorType> fast_types = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
        const std::vector<VkDescriptorType> leg_types(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        auto bind_fast = [&](Pipe& p) {
            bind_buf(c, p, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, b_params.buf, ubo_range);
            bind_buf(c, p, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, b_lat.buf, VK_WHOLE_SIZE);
            bind_buf(c, p, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, b_out.buf, VK_WHOLE_SIZE);
        };
        auto bind_leg = [&](Pipe& p) {
            bind_buf(c, p, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, b_desc.buf, VK_WHOLE_SIZE);
            bind_buf(c, p, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, b_blob.buf, VK_WHOLE_SIZE);
            bind_buf(c, p, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, b_out.buf, VK_WHOLE_SIZE);
        };

        // 1. Correctness against the CPU reference.
        const int check_mips = std::min<int>(check_mips_max, static_cast<int>(ts.num_mips));
        std::vector<std::vector<float>> ref(check_mips);
        for (int m = 0; m < check_mips; ++m) ref[m] = ntc::decode_mip(ts, m);
        bool first_check = true;
        for (const auto& v : vars) {
            if (v.name == "bc7" || v.name == "bc_mat") continue;
            if (v.ntc_fast && !fu.ok) continue;
            Pipe p;
            if (v.ntc_fast) {
                p = make_pipe(c, shader_dir + "/" + v.spv, fast_types, fast_spec(0), sizeof(Pc));
                bind_fast(p);
            } else {
                p = make_pipe(c, shader_dir + "/ntc_decompress.comp.spv", leg_types, {}, 16);
                bind_leg(p);
            }
            for (int m = 0; m < check_mips; ++m) {
                const uint32_t w = std::max(1u, ts.width >> m), h = std::max(1u, ts.height >> m);
                std::vector<uint8_t> got;
                if (v.ntc_fast) {
                    Pc pc{w, h, static_cast<uint32_t>(m), 0, 1.f, 1.f, 0.f, 0.f};
                    got = run_once(c, p, &pc, sizeof(pc), w, h, b_out, static_cast<VkDeviceSize>(w) * h * nch * 4);
                } else {
                    uint32_t pc[4] = {static_cast<uint32_t>(m), 0, 0, 0};
                    got = run_once(c, p, pc, 16, w, h, b_out, static_cast<VkDeviceSize>(w) * h * nch * 4);
                }
                std::vector<float> g(got.size() / 4);
                std::memcpy(g.data(), got.data(), got.size());
                double max_abs = 0, ref_max = 0;
                size_t diff = 0;
                for (size_t i = 0; i < g.size(); ++i) {
                    const double d = std::fabs(static_cast<double>(g[i]) - ref[m][i]);
                    if (!(d == d)) {
                        max_abs = INFINITY;
                    } else {
                        max_abs = std::max(max_abs, d);
                    }
                    ref_max = std::max(ref_max, std::fabs(static_cast<double>(ref[m][i])));
                    std::uint32_t a, b;
                    std::memcpy(&a, &g[i], 4);
                    std::memcpy(&b, &ref[m][i], 4);
                    diff += a != b;
                }
                const double db = psnr(g, ref[m]);
                const char* verdict = "info";
                if (v.exact) verdict = diff == 0 ? "PASS" : "FAIL";
                else if (v.tight) verdict = max_abs <= 1e-4 * std::max(1.0, ref_max) ? "PASS" : "FAIL";
                if (std::strcmp(verdict, "FAIL") == 0) ++failures;
                std::printf("check %-6s mip %d %ux%u max_abs %.3g diff_values %zu psnr_vs_cpu %.2f dB %s\n",
                            v.name.c_str(), m, w, h, max_abs, diff, db, verdict);
                js << (first_check ? "" : ",") << "{\"variant\":\"" << v.name << "\",\"mip\":" << m
                   << ",\"max_abs\":" << (std::isfinite(max_abs) ? max_abs : 1e30) << ",\"diff_values\":" << diff
                   << ",\"psnr\":" << db << ",\"verdict\":\"" << verdict << "\"}";
                first_check = false;
            }
            free_pipe(c, p);
        }
        js << "],\"bench\":[";

        // 2. Timing.
        if (!check_only) {
            const int bw = (static_cast<int>(ts.width) + 3) & ~3, bh = (static_cast<int>(ts.height) + 3) & ~3;
            Img t_bc7a{}, t_bc5{}, t_bc7b{};
            bool bc_ready = false;
            for (const auto& v : vars)
                if (v.name == "bc7" || v.name == "bc_mat") bc_ready = true;
            if (bc_ready) {
                t_bc7a = upload_bc(c, VK_FORMAT_BC7_UNORM_BLOCK, bw, bh, synth_blocks(bw, bh, true, 1));
                t_bc5 = upload_bc(c, VK_FORMAT_BC5_UNORM_BLOCK, bw, bh, synth_blocks(bw, bh, false, 2));
                t_bc7b = upload_bc(c, VK_FORMAT_BC7_UNORM_BLOCK, bw, bh, synth_blocks(bw, bh, true, 3));
            }
            std::printf("\n%-8s %-10s %10s %10s %10s %10s %10s %8s  %s\n", "variant", "size", "median_ms", "p95_ms",
                        "ns/sample", "Gsample/s", "model_GB/s", "vs_bcmat", "stats");
            bool first_b = true;
            for (const auto& sz : sizes) {
                const uint32_t W = static_cast<uint32_t>(sz.first), H = static_cast<uint32_t>(sz.second);
                const float sx = static_cast<float>(uv_scale * W / ts.width);
                const float sy = static_cast<float>(uv_scale * H / ts.height);
                std::map<std::string, double> med;
                std::vector<std::string> lines;
                for (const auto& v : vars) {
                    if (v.ntc_fast && !fu.ok) continue;
                    Pipe p;
                    Pc pc{W, H, 0, 0, sx, sy, 0.f, 0.f};
                    const void* pcp = &pc;
                    uint32_t pcs = sizeof(pc);
                    uint32_t lpc[4] = {static_cast<uint32_t>(sx * 65536.f), static_cast<uint32_t>(sy * 65536.f), W, H};
                    int model_bytes = 4;
                    if (v.ntc_fast) {
                        p = make_pipe(c, shader_dir + "/" + v.spv, fast_types, fast_spec(1), sizeof(Pc));
                        bind_fast(p);
                        model_bytes += fc.load_bytes;
                    } else if (v.name == "legacy") {
                        p = make_pipe(c, shader_dir + "/" + v.spv, leg_types, {}, 16);
                        bind_leg(p);
                        pcp = lpc;
                        pcs = 16;
                        model_bytes += lc.load_bytes;
                    } else {
                        const int ntex = v.name == "bc7" ? 1 : 3;
                        p = make_pipe(c, shader_dir + "/bc_bench.comp.spv",
                                      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                                      {ntex}, sizeof(Pc));
                        bind_img(c, p, 0, t_bc7a.view, sampler);
                        bind_img(c, p, 1, t_bc5.view, sampler);
                        bind_img(c, p, 2, t_bc7b.view, sampler);
                        bind_buf(c, p, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, b_out.buf, VK_WHOLE_SIZE);
                        model_bytes += 16 * ntex;  // one 16 B block per texture per sample at 1 texel/pixel
                    }
                    const Timing t = time_pipe(c, p, pcp, pcs, W, H, warmup, iters);
                    const double samples = static_cast<double>(W) * H;
                    const double ns = t.median_ms * 1e6 / samples;
                    const double gbps = samples * model_bytes / (t.median_ms * 1e-3) / 1e9;
                    med[v.name] = t.median_ms;
                    char line[512];
                    std::snprintf(line, sizeof(line), "%-8s %-10s %10.4f %10.4f %10.4f %10.3f %10.1f", v.name.c_str(),
                                  (std::to_string(W) + "x" + std::to_string(H)).c_str(), t.median_ms, t.p95_ms, ns,
                                  samples / (t.median_ms * 1e-3) / 1e9, gbps);
                    lines.push_back(std::string(line) + "\t" + p.stats);
                    js << (first_b ? "" : ",") << "{\"variant\":\"" << v.name << "\",\"w\":" << W << ",\"h\":" << H
                       << ",\"median_ms\":" << t.median_ms << ",\"p95_ms\":" << t.p95_ms << ",\"min_ms\":" << t.min_ms
                       << ",\"ns_per_sample\":" << ns << ",\"model_gbps\":" << gbps << ",\"stats\":\"" << p.stats
                       << "\"}";
                    first_b = false;
                    free_pipe(c, p);
                }
                for (size_t i = 0; i < lines.size(); ++i) {
                    const std::string& l = lines[i];
                    const auto tab = l.find('\t');
                    std::string name = l.substr(0, l.find(' '));
                    std::string ratio = "-";
                    if (med.count("bc_mat") && med["bc_mat"] > 0) {
                        char r[32];
                        std::snprintf(r, sizeof(r), "%.2fx", med[name] / med["bc_mat"]);
                        ratio = r;
                    }
                    std::printf("%s %8s  %s\n", l.substr(0, tab).c_str(), ratio.c_str(), l.substr(tab + 1).c_str());
                }
            }
            free_img(c, t_bc7a);
            free_img(c, t_bc5);
            free_img(c, t_bc7b);
        }
        js << "]}";
        free_buf(c, b_params);
        free_buf(c, b_lat);
        free_buf(c, b_desc);
        free_buf(c, b_blob);
        free_buf(c, b_out);
    }
    js << "],\"failures\":" << failures << "}\n";
    if (!json_path.empty()) {
        std::ofstream(json_path) << js.str();
        std::printf("\nwrote %s\n", json_path.c_str());
    }
    vkDestroySampler(c.dev, sampler, nullptr);
    vkDestroyCommandPool(c.dev, c.pool, nullptr);
    vkDestroyDevice(c.dev, nullptr);
    vkDestroyInstance(c.inst, nullptr);
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "OK", failures);
    return failures ? 1 : 0;
}
