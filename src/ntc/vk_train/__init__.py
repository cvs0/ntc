"""Vulkan training step. Off unless something calls ``reference`` directly.

``bench/train_max_v2.py`` does not import this package. The laptop queue
keeps the PyTorch loop. PyTorch eager is the oracle: ``adam_step`` and
``gather_taps`` match it within 1e-5 relative. The GLSL kernels are the
same formulas, compiled to SPIR-V for both vendors.
"""

from ntc.vk_train.reference import adam_step, gather_taps

__all__ = ["adam_step", "gather_taps"]
