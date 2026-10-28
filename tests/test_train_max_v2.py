"""Shard partition and lock files for train_max_v2. No training."""

import json
import os
from pathlib import Path

from bench.train_max_v2 import build_jobs, claim_job, owns_index, parse_shard, release_job


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
    assert big["batch_pixels"] == 1024 * 1024
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
