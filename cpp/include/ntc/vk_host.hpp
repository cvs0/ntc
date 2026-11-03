// Instance and device setup shared by the Vulkan hosts.
//
// MoltenVK is a portability driver. Vulkan 1.3.216 and later hide it unless
// the instance enables VK_KHR_portability_enumeration and sets
// VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR. The device then requires
// VK_KHR_portability_subset or vkCreateDevice fails. Desktop loaders do not
// advertise the enumeration extension, so those bits stay off there.

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
#define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME "VK_KHR_portability_enumeration"
#endif
#ifndef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
#define VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR 0x00000001u
#endif
#ifndef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
#define VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME "VK_KHR_portability_subset"
#endif

namespace ntc::vkhost {

inline bool ext_listed(const std::vector<VkExtensionProperties>& exts, const char* name) {
    for (const auto& e : exts)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

inline std::vector<VkExtensionProperties> instance_extensions() {
    uint32_t n = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr) != VK_SUCCESS || n == 0) return {};
    std::vector<VkExtensionProperties> exts(n);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &n, exts.data()) != VK_SUCCESS) return {};
    exts.resize(n);
    return exts;
}

inline std::vector<VkExtensionProperties> device_extensions(VkPhysicalDevice phys) {
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr) != VK_SUCCESS || n == 0) return {};
    std::vector<VkExtensionProperties> exts(n);
    if (vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, exts.data()) != VK_SUCCESS) return {};
    exts.resize(n);
    return exts;
}

struct Instance {
    VkInstance handle = VK_NULL_HANDLE;
    bool portability = false;
    std::string error;
};

// MoltenVK compiles MSL with Metal fast-math unless this is 0. fast-math
// contracts a multiply and a following add even when SPIR-V marked them
// NoContraction, so the exact variant disagreed with the CPU by a few ulps
// (7e-9 on tiny.ntc, about 5e-7 on a 2K file). Other drivers ignore the
// variable. Set it before the loader opens the ICD.
inline void force_moltenvk_ieee() {
#if defined(_WIN32)
    _putenv_s("MVK_CONFIG_FAST_MATH_ENABLED", "0");
#else
    setenv("MVK_CONFIG_FAST_MATH_ENABLED", "0", 1);
#endif
}

// `extra` holds surface extensions the caller already decided to enable.
// Pointers in `extra` must outlive the call.
inline Instance create_instance(const char* app_name, uint32_t api, const std::vector<const char*>& extra = {}) {
    force_moltenvk_ieee();
    Instance out;
    const auto have = instance_extensions();
    std::vector<const char*> names = extra;
    out.portability = ext_listed(have, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    if (out.portability) names.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = app_name;
    app.apiVersion = api;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = static_cast<uint32_t>(names.size());
    ici.ppEnabledExtensionNames = names.empty() ? nullptr : names.data();
    if (out.portability) ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

    const VkResult rc = vkCreateInstance(&ici, nullptr, &out.handle);
    if (rc != VK_SUCCESS) {
        out.handle = VK_NULL_HANDLE;
        out.error = std::string("vkCreateInstance -> ") + std::to_string(static_cast<int>(rc));
        if (!out.portability)
            out.error += " (loader has no VK_KHR_portability_enumeration; MoltenVK will not be listed)";
    }
    return out;
}

// Features the hosts actually branch on. Missing bits stay false and the
// matching shader is skipped. The fp32 storage-buffer path does not need them.
struct Caps {
    bool portability_subset = false;
    bool shader_float16 = false;
    bool shader_int8 = false;
    bool shader_int_dot = false;
    bool int_dot_4x8_accel = false;
    bool storage_buffer_dynamic_index = false;
    bool dynamic_rendering = false;
    bool subgroup_size_control = false;
    bool compute_full_subgroups = false;
    bool texture_compression_bc = false;
    bool shader_int64 = false;
    uint32_t subgroup_size = 0;
    uint32_t max_compute_invocations = 0;
    uint32_t max_shared_memory = 0;
    uint32_t timestamp_valid_bits = 0;
    uint32_t timestamp_queue = 0;
};

inline Caps query_caps(VkPhysicalDevice phys) {
    Caps c;
    const auto exts = device_extensions(phys);
    c.portability_subset = ext_listed(exts, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);

    VkPhysicalDeviceFeatures base{};
    vkGetPhysicalDeviceFeatures(phys, &base);
    c.storage_buffer_dynamic_index = base.shaderStorageBufferArrayDynamicIndexing == VK_TRUE;
    c.texture_compression_bc = base.textureCompressionBC == VK_TRUE;
    c.shader_int64 = base.shaderInt64 == VK_TRUE;

    VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext = &sg;
    vkGetPhysicalDeviceProperties2(phys, &props);
    c.subgroup_size = sg.subgroupSize;
    c.max_compute_invocations = props.properties.limits.maxComputeWorkGroupInvocations;
    c.max_shared_memory = props.properties.limits.maxComputeSharedMemorySize;

    VkPhysicalDeviceShaderFloat16Int8Features f16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.pNext = &v13;
    v13.pNext = &f16;
    VkPhysicalDeviceFeatures2 feat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feat.pNext = &v12;
    vkGetPhysicalDeviceFeatures2(phys, &feat);
    c.shader_float16 = f16.shaderFloat16 == VK_TRUE || v12.shaderFloat16 == VK_TRUE;
    c.shader_int8 = f16.shaderInt8 == VK_TRUE;
    c.shader_int_dot = v13.shaderIntegerDotProduct == VK_TRUE;
    c.dynamic_rendering = v13.dynamicRendering == VK_TRUE;
    c.subgroup_size_control = v13.subgroupSizeControl == VK_TRUE;
    c.compute_full_subgroups = v13.computeFullSubgroups == VK_TRUE;

    VkPhysicalDeviceShaderIntegerDotProductProperties dots{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
    VkPhysicalDeviceProperties2 dot_props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    dot_props.pNext = &dots;
    vkGetPhysicalDeviceProperties2(phys, &dot_props);
    c.int_dot_4x8_accel = dots.integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE;

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    if (nq) vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());
    for (uint32_t i = 0; i < nq; ++i) {
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
        if (qf[i].timestampValidBits > c.timestamp_valid_bits) {
            c.timestamp_valid_bits = qf[i].timestampValidBits;
            c.timestamp_queue = i;
        }
    }
    return c;
}

// Keeps the portability-subset feature struct alive across vkCreateDevice.
// Enable every subset bit the device reported. Enabling one it does not
// support makes vkCreateDevice fail.
struct DeviceExtras {
    std::vector<const char*> extensions;
    bool subset = false;
#ifdef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR
    VkPhysicalDevicePortabilitySubsetFeaturesKHR subset_feat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR};
#endif
    void* chain(void* next) {
#ifdef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR
        if (!subset) return next;
        subset_feat.pNext = next;
        return &subset_feat;
#else
        (void)subset;
        return next;
#endif
    }
};

inline DeviceExtras device_extras(VkPhysicalDevice phys, std::vector<const char*> extra = {}) {
    DeviceExtras out;
    out.extensions = std::move(extra);
    const auto exts = device_extensions(phys);
    if (!ext_listed(exts, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) return out;
    out.subset = true;
    out.extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
#ifdef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR
    VkPhysicalDevicePortabilitySubsetFeaturesKHR q{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 feat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feat.pNext = &q;
    vkGetPhysicalDeviceFeatures2(phys, &feat);
    out.subset_feat = q;
    out.subset_feat.pNext = nullptr;
#endif
    return out;
}

// On-disk VkPipelineCache. MoltenVK's first compile of the specialized
// fast shaders is minutes. A later run of the same device and driver
// reuses this file. A cache from another GPU is ignored by the driver.
struct DiskPipelineCache {
    VkPipelineCache handle = VK_NULL_HANDLE;
    std::string path;

    void save(VkDevice dev) const {
        if (!handle || path.empty()) return;
        size_t n = 0;
        if (vkGetPipelineCacheData(dev, handle, &n, nullptr) != VK_SUCCESS || n == 0) return;
        std::vector<uint8_t> data(n);
        if (vkGetPipelineCacheData(dev, handle, &n, data.data()) != VK_SUCCESS) return;
        data.resize(n);
        const std::filesystem::path dest(path);
        const std::filesystem::path tmp = dest.string() + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary);
            if (!out) return;
            out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
            if (!out) return;
        }
        std::error_code ec;
        std::filesystem::rename(tmp, dest, ec);
    }

    void destroy(VkDevice dev) {
        save(dev);
        if (handle && dev) vkDestroyPipelineCache(dev, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
};

inline DiskPipelineCache open_pipeline_cache(VkDevice dev, const VkPhysicalDeviceProperties& props) {
    DiskPipelineCache cache;
    std::filesystem::path root;
#if defined(_WIN32)
    if (const char* local = std::getenv("LOCALAPPDATA")) root = local;
#else
    if (const char* home = std::getenv("HOME")) root = std::filesystem::path(home) / ".cache";
#endif
    if (root.empty()) root = std::filesystem::temp_directory_path();
    root /= "ntc";
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    char leaf[96];
    std::snprintf(leaf, sizeof(leaf), "pipelines-%08x-%08x-%u.bin", props.vendorID, props.deviceID,
                  props.driverVersion);
    cache.path = (root / leaf).string();

    std::vector<uint8_t> initial;
    {
        std::ifstream in(cache.path, std::ios::binary);
        if (in) {
            in.seekg(0, std::ios::end);
            const auto bytes = in.tellg();
            if (bytes > 0 && bytes < 64 * 1024 * 1024) {
                initial.resize(static_cast<size_t>(bytes));
                in.seekg(0);
                in.read(reinterpret_cast<char*>(initial.data()), static_cast<std::streamsize>(initial.size()));
                if (!in) initial.clear();
            }
        }
    }
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = initial.size();
    ci.pInitialData = initial.empty() ? nullptr : initial.data();
    if (vkCreatePipelineCache(dev, &ci, nullptr, &cache.handle) != VK_SUCCESS) {
        ci.initialDataSize = 0;
        ci.pInitialData = nullptr;
        cache.handle = VK_NULL_HANDLE;
        vkCreatePipelineCache(dev, &ci, nullptr, &cache.handle);
    }
    return cache;
}

}  // namespace ntc::vkhost
