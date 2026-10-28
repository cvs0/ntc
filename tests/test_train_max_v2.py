"""Shard partition, work queue, and lock files for train_max_v2. No training."""

import argparse
import json
import os
from pathlib import Path

import bench.train_max_v2 as v2
from bench.train_max_v2 import (
    _job_blocked,
    assign_shards,
    build_jobs,
    claim_job,
    claim_next,
    job_cost,
    owns_index,
    parse_shard,
    release_job,
)
from ntc.train_fast import mip_sample_count


def test_shard_covers_each_job_once():
    jobs = build_jobs(Path("data/materials"), None)
    names = [job["name"] for job in jobs]
    assert names[0] == "metal_g16_s20000"
    assert "metal_b12_s10000" in names
    assert "sdk_MetalPlates013_2K_b2.5" in names
    assert len(names) == len(set(names))
    owned = []
    for shard in range(3):
        owned.extend(i for i in range(len(jobs)) if owns_index(i, shard, 3))
    assert sorted(owned) == list(range(len(jobs)))
    assert parse_shard("1/3") == (1, 3)
    # 256 jobs stay full-frame unless the override is set.
    assert jobs[0]["batch_pixels"] == 0
    overridden = build_jobs(Path("data/materials"), 32768)
    assert overridden[0]["batch_pixels"] == 32768
    assert overridden[0]["lr_batch_ref"] == 256 * 256
    big = next(job for job in overridden if job["size"] == 2048)
    assert big["batch_pixels"] == 256 * 1024
    assert big["batch_mip"] == "area"
    assert big["lr_batch_ref"] == 0


def test_claim_is_exclusive_and_a_dead_pid_is_stale(tmp_path: Path):
    assert claim_job(tmp_path, "metal_b12_s10000", stale_s=3600)
    assert not claim_job(tmp_path, "metal_b12_s10000", stale_s=3600)
    release_job(tmp_path, "metal_b12_s10000")
    lock = tmp_path / "locks" / "metal_b12_s10000.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": 2**31 - 1, "name": "metal_b12_s10000"}))
    old = lock.stat().st_mtime
    os.utime(lock, (old, old))
    assert claim_job(tmp_path, "metal_b12_s10000", stale_s=3600)
    payload = json.loads(lock.read_text())
    assert payload["pid"] == os.getpid()


def test_area_batch_shrinks_coarser_mips():
    assert mip_sample_count(1048576, 1, 1048576, "flat") == 0
    assert mip_sample_count(262144, 0, 2048 * 2048, "area") == 262144
    assert mip_sample_count(262144, 1, 1024 * 1024, "area") == 65536
    assert mip_sample_count(262144, 2, 512 * 512, "area") == 16384


def test_job_cost_and_cost_balanced_shards():
    jobs = build_jobs(Path("data/materials"), None)
    by_name = {job["name"]: job for job in jobs}
    assert job_cost(by_name["metal_g16_s20000"]) == 256 * 256 * 20000
    assert job_cost(by_name["metal_b4_s10000"]) == 256 * 256 * 10000
    sdk = by_name["sdk_MetalPlates013_2K_b2.5"]
    assert sdk["batch_pixels"] == 256 * 1024
    assert sdk["batch_mip"] == "area"
    assert sdk["lr_batch_ref"] == 0
    assert sdk["min_latent_bits"] == 4
    assert sdk["latent_rate_scale"] == 0.5
    assert job_cost(sdk) == (256 * 1024) * 4000
    assert by_name["wood_floor_2k_b16_s1000"]["batch_pixels"] == 256 * 1024
    assert by_name["wood_floor_2k_b16_s1000"]["min_latent_bits"] == 0
    assign = assign_shards(jobs, 3)
    assert len(assign) == len(jobs)
    sdk_shards = {assign[i] for i, job in enumerate(jobs) if job["name"].startswith("sdk_")}
    assert sdk_shards == {0, 1, 2}
    loads = [0.0, 0.0, 0.0]
    for index, job in enumerate(jobs):
        loads[assign[index]] += job_cost(job)
    heaviest = max(job_cost(job) for job in jobs)
    assert max(loads) - min(loads) <= heaviest
    assert assign != [i % 3 for i in range(len(jobs))]
    # Index mod is still a partition. The runner does not use it.
    owned = []
    for shard in range(3):
        owned.extend(i for i in range(len(jobs)) if owns_index(i, shard, 3))
    assert sorted(owned) == list(range(len(jobs)))


def _tiny(name: str, steps: int, material: str) -> dict:
    return {
        "name": name,
        "material": material,
        "size": 32,
        "steps": steps,
        "seed_pool": 1,
        "batch_pixels": 0,
        "grids": None,
    }


def test_queue_claims_heaviest_then_another(tmp_path: Path):
    jobs = build_jobs(Path("data/materials"), None)
    first = claim_next(jobs, tmp_path, 3600, pack=False)
    second = claim_next(jobs, tmp_path, 3600, pack=False)
    heaviest = max(job_cost(job) for job in jobs)
    assert job_cost(first[0]) == heaviest
    assert job_cost(first[0]) > job_cost(second[0])
    assert [job["name"] for job in first] == ["wood_floor_1k_b16_s4000"]
    assert [job["name"] for job in second] == ["metal_g16_s20000"]


def _packable(name: str, steps: int) -> dict:
    return {
        "name": name,
        "material": "m",
        "size": 16,
        "steps": steps,
        "seed_pool": 1,
        "batch_pixels": 0,
        "schedule": "cosine",
        "mip_levels": 4,
        "warmup": 200,
        "grids": [(4, 4, 4)],
    }


def test_queue_packs_a_matching_pair(tmp_path: Path):
    jobs = [_packable("light", 1), _packable("a", 40), _packable("b", 40)]
    claimed = claim_next(jobs, tmp_path, 3600, pack=True)
    assert {job["name"] for job in claimed} == {"a", "b"}
    release_job(tmp_path, "a")
    release_job(tmp_path, "b")
    one = claim_next(jobs, tmp_path, 3600, pack=False)
    assert [job["name"] for job in one] == ["a"]


def test_collapsed_json_is_not_claimed_again(tmp_path: Path):
    mat = tmp_path / "mat"
    mat.mkdir()
    (mat / "manifest.json").write_text("{}")
    out = tmp_path / "out"
    out.mkdir()
    job = _tiny("stuck", 20, str(mat))
    (out / "stuck.json").write_text(json.dumps({"name": "stuck", "done": False, "collapsed": True}))
    assert _job_blocked(out, job)
    assert claim_next([job], out, 3600, pack=False) == []


def test_error_json_blocks_only_while_the_material_is_missing(tmp_path: Path):
    mat = tmp_path / "mat"
    out = tmp_path / "out"
    out.mkdir()
    job = _tiny("bad", 50, str(mat))
    other = _tiny("ok", 10, str(mat))
    (out / "bad.json").write_text(json.dumps({"name": "bad", "done": False, "error": "missing"}))
    assert _job_blocked(out, job)
    claimed = claim_next([job, other], out, 3600, pack=False)
    assert [row["name"] for row in claimed] == ["ok"]
    release_job(out, "ok")
    mat.mkdir()
    (mat / "manifest.json").write_text("{}")
    assert not _job_blocked(out, job)
    again = claim_next([job, other], out, 3600, pack=False)
    assert [row["name"] for row in again] == ["bad"]
    (out / "done.json").write_text(json.dumps({"done": True}))
    done = _tiny("done", 1000, str(mat))
    assert _job_blocked(out, done)


def test_workers_spawn_a_queue(monkeypatch, tmp_path: Path):
    calls = []

    class _Proc:
        def wait(self):
            return 0

    def fake_popen(cmd, **kwargs):
        calls.append(list(cmd))
        return _Proc()

    monkeypatch.setattr(v2.subprocess, "Popen", fake_popen)
    args = argparse.Namespace(
        workers=3,
        out=tmp_path,
        materials=Path("data/materials"),
        probe_steps=400,
        stale_lock_s=10,
        batch_pixels=None,
        dry_run=False,
        pack=True,
    )
    assert v2._spawn_workers(args) == 0
    assert len(calls) == 3
    for cmd in calls:
        assert "--queue" in cmd
        assert "--shard" not in cmd
        assert "--no-pack" not in cmd
    args.pack = False
    calls.clear()
    assert v2._spawn_workers(args) == 0
    assert "--no-pack" in calls[0]
