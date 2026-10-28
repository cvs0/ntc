"""Cloud 5090 profile stays off the laptop job list."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from bench.train_max_v2 import (
    CLOUD_BATCH,
    CLOUD_BPP,
    CLOUD_FULL_STEPS,
    CLOUD_HOLDOUT,
    CLOUD_SEED_FACTOR,
    CLOUD_STEPS_PER_S,
    CLOUD_TRAIN,
    _precision_material,
    assign_shards,
    build_cloud_jobs,
    build_jobs,
    cloud_queue_gpu_hours,
    is_dense_2k,
    ladder_step_budget,
)
from ntc.encode import TrainConfig
from ntc.materials import load_material
from ntc.train_fast import StackedTrain, _build_nets, _sync_quant_schedule, encode_arrays_fast


def test_cloud_labels_are_fit_once_per_image():
    """Ten labels share one pass. The old per-job probe was the idle GPU."""
    import time

    from bench.train_max_v2 import CLOUD_BPP, cloud_latent_setups

    rng = np.random.default_rng(2)
    n = 96
    img = rng.random((6, n, n), dtype=np.float32)
    started = time.perf_counter()
    table = cloud_latent_setups(img, list(range(6)), n, 4)
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0, elapsed
    assert set(table) == {round(float(bpp), 4) for bpp in CLOUD_BPP}
    again = time.perf_counter()
    second = cloud_latent_setups(img, list(range(6)), n, 4)
    assert time.perf_counter() - again < 0.05
    assert second is table


def test_precision_probe_uses_a_cloud_material_when_metal_plate_is_absent(tmp_path: Path):
    """The cloud tree has no ``metal_plate``. The probe must not require it."""
    cloud = tmp_path / "MetalPlates013_2K"
    cloud.mkdir()
    (cloud / "manifest.json").write_text("{}")
    assert _precision_material(tmp_path) == cloud
    laptop = tmp_path / "metal_plate"
    laptop.mkdir()
    (laptop / "manifest.json").write_text("{}")
    assert _precision_material(tmp_path) == laptop
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _precision_material(empty) is None


def test_laptop_sdk_job_is_unchanged():
    jobs = build_jobs(Path("data/materials"), None)
    sdk = next(job for job in jobs if job["name"] == "sdk_MetalPlates013_2K_b2.5")
    assert sdk["steps"] == 4000
    assert sdk["batch_pixels"] == 256 * 1024
    assert sdk["batch_mip"] == "area"
    assert sdk["seed_pool"] == 1
    assert "quant_freeze_at" not in sdk
    names = {job["name"] for job in jobs}
    assert not any(name.startswith("cloud_") for name in names)


def test_cloud_queue_shape_and_hours():
    jobs = build_cloud_jobs(Path("data/materials"))
    assert len(CLOUD_TRAIN) == 12
    assert len(CLOUD_HOLDOUT) == 3
    assert "Gravel022" in CLOUD_HOLDOUT
    assert "Gravel001" not in CLOUD_HOLDOUT
    assert set(CLOUD_TRAIN).isdisjoint(set(CLOUD_HOLDOUT))
    assert len(jobs) == len(CLOUD_TRAIN) * len(CLOUD_BPP)
    assert len({job["name"] for job in jobs}) == len(jobs)
    low = next(job for job in jobs if job["name"] == "cloud_Tiles135D_2K_b1")
    assert low["steps"] == CLOUD_FULL_STEPS == 8000
    mid = next(job for job in jobs if job["name"] == "cloud_Tiles135D_2K_b2.5")
    assert mid["steps"] == 4000
    assert low["batch_pixels"] == CLOUD_BATCH == 2 * 1024 * 1024
    assert low["batch_mip"] == "flat"
    assert low["seed_pool"] == 3
    assert low["halve_steps"] == 500
    assert low["quant_freeze_at"] == 0.90
    assert low["entropy_rate"] is True
    assert low["latent_rate_scale"] == 0.0
    high = next(job for job in jobs if job["name"] == "cloud_Tiles135D_2K_b20")
    assert high["steps"] == 4000
    assert ladder_step_budget(len(CLOUD_BPP), CLOUD_FULL_STEPS)[0] == 8000
    held = {f"cloud_{name}_2K_b2.5" for name in CLOUD_HOLDOUT}
    assert held.isdisjoint({job["name"] for job in jobs})
    assert all(is_dense_2k(job) for job in jobs)
    hours = cloud_queue_gpu_hours(jobs)
    steps = sum(int(job["steps"]) for job in jobs)
    assert hours == pytest.approx(steps * CLOUD_SEED_FACTOR / CLOUD_STEPS_PER_S / 3600.0)
    assert 8 < hours < 12
    assign = assign_shards(jobs, 5)
    assert sorted(set(assign)) == [0, 1, 2, 3, 4]


def test_opacity_map_is_an_extra_channel(tmp_path: Path):
    d = tmp_path / "leaf"
    d.mkdir()
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[..., 1] = 200
    Image.fromarray(rgb, mode="RGB").save(d / "color.jpg")
    Image.fromarray(np.full((8, 8), 128, np.uint8), mode="L").save(d / "opacity.jpg")
    (d / "manifest.json").write_text(json.dumps({"albedo": "color.jpg", "opacity": "opacity.jpg"}))
    images, sems, _info = load_material(d, size=8)
    assert images.shape[0] == 4
    assert sems[-1] == 10
    assert float(images[-1].mean()) == pytest.approx(128 / 255)


def test_quant_tail_freezes_earlier_on_the_cloud_fraction():
    net = _build_nets(
        torch.zeros(3, 8, 8),
        TrainConfig(steps=100, hidden=(4,), grids=[(4, 4, 2, 4)], pca_init=False, quant_schedule=True),
        [(4, 4, 2, 4)],
        [0],
        torch.device("cpu"),
    )
    stacked = StackedTrain(net)
    early = TrainConfig(steps=100, quant_schedule=True, quant_freeze_at=0.90)
    late = TrainConfig(steps=100, quant_schedule=True, quant_freeze_at=0.95)
    _sync_quant_schedule(stacked, early, 91)
    assert stacked.freeze_latents is True
    _sync_quant_schedule(stacked, late, 91)
    assert stacked.freeze_latents is False
    _sync_quant_schedule(stacked, late, 96)
    assert stacked.freeze_latents is True


def test_three_seeds_keep_a_finite_file_on_a_small_image():
    """A/B on a 16 image, 8 steps. The pool must stay finite and not fall apart."""
    rng = np.random.default_rng(0)
    img = rng.random((3, 16, 16), dtype=np.float32)
    common = dict(
        hidden=(8,),
        grids=[(8, 8, 2, 4)],
        fast=True,
        adaptive_pool=1,
        with_ssim=False,
        log_every=10**9,
        mip_levels=2,
        compile_step="off",
        cuda_graph="off",
        precision="fp32",
        pca_init=True,
        batch_pixels=64,
        batch_mip="flat",
        steps=8,
        seed=0,
    )
    _b, single = encode_arrays_fast(img, [0, 1, 2], TrainConfig(seed_pool=1, halve_steps=0, **common))
    _b, pool = encode_arrays_fast(img, [0, 1, 2], TrainConfig(seed_pool=3, halve_steps=4, **common))
    assert np.isfinite(single["metrics"]["psnr_u8"])
    assert np.isfinite(pool["metrics"]["psnr_u8"])
    assert pool["metrics"]["psnr_u8"] >= single["metrics"]["psnr_u8"] - 1.0


def test_collect_keeps_a_finished_json_over_a_later_partial(tmp_path: Path):
    from bench.cloud.collect_results import collect

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "cloud_Metal003_2K_b1.json").write_text(json.dumps({
        "name": "cloud_Metal003_2K_b1", "done": True, "psnr_u8": 30.0, "total_bpp": 1.1,
    }))
    (b / "cloud_Metal003_2K_b1.json").write_text(json.dumps({
        "name": "cloud_Metal003_2K_b1", "done": False,
    }))
    (b / "device.json").write_text("{}")
    rows = collect([a, b], tmp_path / "out")
    assert len(rows) == 1
    assert rows[0]["psnr_u8"] == 30.0
    saved = json.loads((tmp_path / "out" / "cloud_Metal003_2K_b1.json").read_text())
    assert saved["done"] is True


def test_cloud_entry_points_run_as_files_without_pythonpath(tmp_path: Path):
    """``python bench/x.py`` must import ``bench`` with a clean environment.

    That invocation puts ``bench/`` on sys.path. The pod failed in
    ``download_materials.py --preset cloud`` with ``No module named 'bench'``.
    """
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    script = (repo / "bench/cloud/setup_pod.sh").read_text()
    assert 'export PYTHONPATH="$ROOT/src:$ROOT"' in script
    assert "curl -m 10 -sI https://github.com" in script
    assert "no outbound internet" in script
    assert "cloud_split.json" in script
    assert "skip download" in script
    lines = script.splitlines()

    def at(text: str) -> int:
        for i, line in enumerate(lines):
            if text in line:
                return i
        raise AssertionError(text)

    # Pull stays inside the setup lock. The trainer must not inherit fd 9.
    assert at("flock 9") < at("git pull --ff-only")
    assert "HEAD matches origin/$BRANCH" in script
    assert at("exec 9>&-") < at("nohup python")
    launch = "\n".join(lines[at("nohup python") : at("nohup python") + 8])
    assert "9>&-" in launch
    syntax = subprocess.run(["bash", "-n", str(repo / "bench/cloud/setup_pod.sh")], check=False)
    assert syntax.returncode == 0

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args],
            cwd=repo,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    listed = run(["bench/download_materials.py", "--preset", "cloud", "--dry-run"])
    assert listed.returncode == 0, listed.stderr
    assert "cloud train 12 holdout 3" in listed.stdout

    planned = run(
        [
            "bench/train_max_v2.py",
            "--profile",
            "cloud5090",
            "--dry-run",
            "--shard",
            "0/5",
            "--out",
            str(tmp_path / "out"),
            "--materials",
            str(tmp_path / "materials"),
        ]
    )
    assert planned.returncode == 0, planned.stderr
    assert "cloud_" in planned.stdout
