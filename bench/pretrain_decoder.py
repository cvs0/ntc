"""Train a shared MLP prior and write a small weight file.

The procedure is fixed. Seed 0. Eight images. Each image starts from the
previous image's MLP. Latents are PCA of that image, then the MLP is restored
and Adam runs ``--steps`` steps. CC0 directories are used when ``--materials``
points at folders that contain ``manifest.json``. Otherwise the eight images
are procedural and the file is reproducible from this script.

Windows, from the repo root:

    python bench/pretrain_decoder.py --preset small --out src/ntc/priors/decoder_small.pt
    python bench/pretrain_decoder.py --preset g16 --out src/ntc/priors/decoder_g16.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ntc.encode import TrainConfig, _pca_init
from ntc.model import NtcNet
from ntc.train_fast import StackedTrain, _loss_from_mips, _prep_mips


def _procedural(index: int, size: int, channels: int) -> torch.Tensor:
    torch.manual_seed(1000 + index)
    ys = torch.linspace(0, 1, size)
    xs = torch.linspace(0, 1, size)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    phase = index * 0.7
    planes = []
    for c in range(channels):
        freq = 2.0 + (index + c) % 5
        plane = 0.5 + 0.5 * torch.sin(gx * freq + phase) * torch.cos(gy * (freq * 0.5) + c)
        if c % 3 == 1:
            plane = plane * 0.5 + gy * 0.5
        elif c % 3 == 2:
            plane = plane * 0.3 + ((gx + gy) * 0.5)
        planes.append(plane.clamp(0, 1))
    return torch.stack(planes, 0).float()


def _load_cc0(root: Path, size: int, channels: int) -> list[torch.Tensor]:
    from ntc.materials import load_material

    images = []
    if not root.is_dir():
        return images
    for folder in sorted(p for p in root.iterdir() if (p / "manifest.json").is_file()):
        arr, _sems, _info = load_material(folder, size)
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).float()
        if tensor.shape[0] < channels:
            pad = tensor[-1:].repeat(channels - tensor.shape[0], 1, 1)
            tensor = torch.cat([tensor, pad], 0)
        images.append(tensor[:channels])
        if len(images) >= 8:
            break
    return images


def _preset(name: str) -> dict:
    if name == "g16":
        return {
            "size": 32,
            "channels": 9,
            "hidden": (32,),
            "grids": [(32, 32, 4, 4), (8, 8, 2, 2)],
            "steps": 8,
        }
    return {
        "size": 24,
        "channels": 3,
        "hidden": (16,),
        "grids": [(12, 12, 4, 4)],
        "steps": 12,
    }


def train_prior(preset: str, materials: Path | None, steps: int | None) -> dict:
    spec = _preset(preset)
    if steps is not None:
        spec["steps"] = steps
    images = _load_cc0(materials, spec["size"], spec["channels"]) if materials else []
    source = "cc0" if images else "procedural"
    if not images:
        images = [_procedural(i, spec["size"], spec["channels"]) for i in range(8)]
    mlp = None
    torch.manual_seed(0)
    cfg = TrainConfig(
        steps=spec["steps"],
        hidden=spec["hidden"],
        grids=spec["grids"],
        mip_levels=2,
        seed_pool=1,
        adaptive_pool=1,
        with_ssim=False,
        log_every=10**9,
        compile_step="off",
        cuda_graph="off",
        fast=True,
    )
    for image in images:
        net = NtcNet(
            image_channels=spec["channels"],
            grids=spec["grids"],
            hidden=list(spec["hidden"]),
            activation=1,
            pe_mode=0,
            pe_octaves=0,
            weight_bits=8,
            group_size=0,
            quant_proxy="ste",
            activation_requant=False,
            qat_latents=True,
            qat_weights=True,
        )
        if mlp is not None:
            _restore_mlp(net, mlp)
        _pca_init(net, image)
        if mlp is not None:
            _restore_mlp(net, mlp)
        stacked = StackedTrain([net])
        opt = torch.optim.Adam(
            [
                {"params": list(stacked.latents.parameters()) + list(stacked.log_scales.parameters()), "lr": 1e-2, "name": "latent"},
                {"params": list(stacked.weights.parameters()) + list(stacked.biases.parameters()), "lr": 5e-3, "name": "mlp"},
            ]
        )
        mips = _prep_mips(image, cfg, torch.device("cpu"))
        for _step in range(spec["steps"]):
            opt.zero_grad(set_to_none=True)
            loss = _loss_from_mips(stacked, mips, cfg, list(range(spec["channels"])), None, 0, _step).sum()
            loss.backward()
            opt.step()
        stacked.write_seed(0, net)
        mlp = _capture_mlp(net)
    latent_channels = int(sum(g[2] for g in spec["grids"]))
    return {
        "weights": mlp["weights"],
        "biases": mlp["biases"],
        "hidden": list(spec["hidden"]),
        "grids": [list(g) for g in spec["grids"]],
        "latent_channels": latent_channels,
        "out_channels": spec["channels"],
        "source": source,
        "images": len(images),
        "steps_per_image": spec["steps"],
        "seed": 0,
        "preset": preset,
    }


def _capture_mlp(net: NtcNet) -> dict:
    return {
        "weights": [lin.weight.detach().cpu().contiguous().clone() for lin in net.linears],
        "biases": [lin.bias.detach().cpu().contiguous().clone() for lin in net.linears],
    }


def _restore_mlp(net: NtcNet, mlp: dict) -> None:
    with torch.no_grad():
        for lin, weight, bias in zip(net.linears, mlp["weights"], mlp["biases"], strict=True):
            lin.weight.copy_(weight)
            lin.bias.copy_(bias)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=("small", "g16"), default="small")
    p.add_argument("--materials", type=Path, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    blob = train_prior(args.preset, args.materials, args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(
        f"wrote {args.out} source {blob['source']} images {blob['images']} "
        f"latent_channels {blob['latent_channels']} out {blob['out_channels']}"
    )


if __name__ == "__main__":
    main()
