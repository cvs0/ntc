#pragma once

#include <vulkan/vulkan.h>

// Same rule as ntc.gpu_policy.required_subgroup_size. ``req`` must stay
// alive through vkCreate*Pipelines. A device that cannot require 32 leaves
// the stage unchanged, so a driver whose subgroup is already 32 or 64 still
// creates the pipeline.
inline void ntc_attach_subgroup_32(VkPipelineShaderStageCreateInfo& stage,
                                   VkPipelineShaderStageRequiredSubgroupSizeCreateInfo& req,
                                   VkPhysicalDevice phys) {
    VkPhysicalDeviceVulkan13Features feat{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &feat;
    vkGetPhysicalDeviceFeatures2(phys, &features);
    if (feat.subgroupSizeControl != VK_TRUE) return;
    VkPhysicalDeviceVulkan13Properties props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &props;
    vkGetPhysicalDeviceProperties2(phys, &properties);
    if (props.minSubgroupSize > 32u || props.maxSubgroupSize < 32u) return;
    if ((props.requiredSubgroupSizeStages & stage.stage) == 0u) return;
    req.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    req.pNext = const_cast<void*>(stage.pNext);
    req.requiredSubgroupSize = 32u;
    stage.pNext = &req;
}
