"""Neural texture compression for PBR materials.

Per-material multi-resolution latent grids plus a small MLP.
The container and the reference decoder are defined so a GLSL shader
and a C++20 decoder implement the same function.
"""

__version__ = "1.0.0"
