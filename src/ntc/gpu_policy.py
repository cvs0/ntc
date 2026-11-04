"""Subgroup size shared by the Vulkan pipeline helper.

RDNA reports subgroupSize 64 even when the hardware wave is 32. Decode and
training kernels ask for 32 on both vendors when the device allows it.
Workgroups in the compute shaders are 32, 64, or 256 invocations, all
multiples of 32.
"""

from __future__ import annotations

REQUIRED_SUBGROUP = 32


def required_subgroup_size(
    min_size: int,
    max_size: int,
    stage_mask: int,
    stage_bit: int,
    control: bool,
) -> int | None:
    """Return 32, or None when the device cannot require it for this stage."""
    if not control:
        return None
    if int(min_size) > REQUIRED_SUBGROUP or int(max_size) < REQUIRED_SUBGROUP:
        return None
    if (int(stage_mask) & int(stage_bit)) == 0:
        return None
    return REQUIRED_SUBGROUP
