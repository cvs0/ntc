"""Short cloud profile and side-by-side dense workers on one large GPU."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import bench.train_max_v2 as v2
from bench.train_max_v2 import (
    CLOUD_BPP,
    CLOUD_FAST_BPP,
    CLOUD_FAST_TRAIN,
    CLOUD_FULL_STEPS,
    CLOUD_HOLDOUT,
    CLOUD_TRAIN,
    assign_chains,
    build_cloud_jobs,
    build_jobs,
    is_cloud_profile,
    material_chains,
    planned_workers,
    worker_env,
)


def test_cloudfast_job_list(tmp_path: Path):
    assert set(CLOUD_FAST_TRAIN) <= set(CLOUD_TRAIN)
    assert not set(CLOUD_FAST_TRAIN) & set(CLOUD_HOLDOUT)
    assert CLOUD_FAST_BPP == (1.0, 2.5, 5.0, 10.0, 20.0)
    assert 5.0 not in CLOUD_BPP
    jobs = build_jobs(tmp_path, None, "cloudfast")
    assert jobs == build_cloud_jobs(tmp_path, "cloudfast")
    assert len(jobs) == 20
    names = [job["name"] for job in jobs]
    expected = [
        f"cloud_{name}_2K_b{label}"
        for name in CLOUD_FAST_TRAIN
        for label in ("1", "2.5", "5", "10", "20")
    ]
    assert names == expected
    for job in jobs:
        assert job["profile"] == "cloudfast"
        assert is_cloud_profile(job["profile"])
        assert job["size"] == 2048
        assert job["rate"] == "latent"
        first = float(job["bpp"]) == 1.0
        assert job["steps"] == (CLOUD_FULL_STEPS if first else CLOUD_FULL_STEPS // 2)
        assert job["warmup"] == (800 if first else 400)
    assert sum(job["steps"] for job in jobs) == 4 * (8000 + 4 * 4000)
    full = build_cloud_jobs(tmp_path)
    assert len(full) == len(CLOUD_TRAIN) * len(CLOUD_BPP)
    assert {job["profile"] for job in full} == {"cloud5090"}


def test_planned_workers_allows_dense_override(tmp_path: Path, monkeypatch):
    jobs = build_cloud_jobs(tmp_path, "cloudfast")
    monkeypatch.setattr(v2, "is_dense_2k", lambda job: True)
    assert planned_workers(4, jobs, tmp_path) == 1
    assert planned_workers(4, jobs, tmp_path, dense_ok=True) == 4
    assert planned_workers(1, jobs, tmp_path, dense_ok=True) == 1


def test_chains_keep_ladder_order(tmp_path: Path):
    jobs = build_cloud_jobs(tmp_path, "cloudfast")
    shuffled = list(reversed(jobs))
    chains = material_chains(shuffled)
    assert len(chains) == 4
    for chain in chains:
        assert len({job["material"] for job in chain}) == 1
        assert [job["bpp"] for job in chain] == list(CLOUD_FAST_BPP)
        assert chain[0]["steps"] == CLOUD_FULL_STEPS
    plan = assign_chains(jobs, 4)
    assert [len(worker) for worker in plan] == [1, 1, 1, 1]
    owned = [chain[0]["material"] for worker in plan for chain in worker]
    assert len(set(owned)) == 4
    two = assign_chains(jobs, 2)
    assert [len(worker) for worker in two] == [2, 2]
    for worker in two:
        for chain in worker:
            assert [job["bpp"] for job in chain] == list(CLOUD_FAST_BPP)


def test_chain_loop_runs_each_chain_once_in_order(tmp_path: Path, monkeypatch):
    jobs = build_cloud_jobs(tmp_path, "cloudfast")
    ran = []
    monkeypatch.setattr(v2, "_run", lambda job, *a, **k: ran.append(job["name"]))
    # Another worker already holds the Tiles chain.
    assert v2.claim_job(tmp_path, "chain_Tiles135D_2K", 86400.0)
    v2._chain_loop(jobs, tmp_path, "fp32", 86400.0, "worker_0")
    expected = [job["name"] for job in jobs if "Tiles135D" not in job["name"]]
    assert ran == expected
    assert not (tmp_path / "locks" / "chain_MetalPlates013_2K.lock").exists()


def test_worker_env_has_private_caches(tmp_path: Path):
    envs = [worker_env(i, base={}, root=str(tmp_path)) for i in range(4)]
    inductor = {env["TORCHINDUCTOR_CACHE_DIR"] for env in envs}
    triton = {env["TRITON_CACHE_DIR"] for env in envs}
    assert len(inductor) == 4
    assert len(triton) == 4
    assert not inductor & triton
    for env in envs:
        assert env["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
        assert env["NTC_DENSE_LOCK"] == "0"
        assert env["TORCHINDUCTOR_CACHE_DIR"].startswith(str(tmp_path))


def test_dense_lock_is_skipped_for_chain_workers(monkeypatch):
    monkeypatch.delenv("NTC_DENSE_LOCK", raising=False)
    assert v2.dense_lock_enabled()
    monkeypatch.setenv("NTC_DENSE_LOCK", "0")
    assert not v2.dense_lock_enabled()


def test_spawn_chain_workers(monkeypatch, tmp_path: Path):
    calls = []
    sleeps = []

    class _Proc:
        def wait(self):
            return 0

    def fake_popen(cmd, env=None, **kwargs):
        calls.append((list(cmd), env))
        return _Proc()

    monkeypatch.setenv("NTC_ROOT", str(tmp_path))
    monkeypatch.setattr(v2.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(v2.time, "sleep", lambda s: sleeps.append(s))
    args = argparse.Namespace(
        workers=4,
        out=tmp_path / "out",
        materials=Path("data/materials"),
        probe_steps=800,
        stale_lock_s=10,
        batch_pixels=None,
        dry_run=False,
        pack=True,
        profile="cloudfast",
        chains=True,
        stagger_s=20.0,
    )
    assert v2._spawn_workers(args) == 0
    assert len(calls) == 4
    assert sleeps == [20.0, 20.0, 20.0]
    caches = set()
    for cmd, env in calls:
        assert "--chain-queue" in cmd
        assert "--queue" not in cmd
        assert cmd[cmd.index("--profile") + 1] == "cloudfast"
        assert env["NTC_DENSE_LOCK"] == "0"
        assert Path(env["TORCHINDUCTOR_CACHE_DIR"]).is_dir()
        caches.add(env["TORCHINDUCTOR_CACHE_DIR"])
    assert len(caches) == 4


def test_cloudfast_dense_workers_dry_run(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo / 'src'}{os.pathsep}{repo}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "bench/train_max_v2.py",
            "--profile",
            "cloudfast",
            "--dense-workers",
            "4",
            "--dry-run",
            "--out",
            str(tmp_path / "out"),
            "--materials",
            str(tmp_path / "materials"),
        ],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "would spawn 4 chain workers" in result.stdout
    assert result.stdout.count("plan worker") == 4


def test_setup_pod_workers_mode():
    repo = Path(__file__).resolve().parents[1]
    script = (repo / "bench/cloud/setup_pod.sh").read_text()
    assert 'PROFILE="${NTC_PROFILE:-cloud5090}"' in script
    assert 'WORKERS="${NTC_WORKERS:-1}"' in script
    assert 'OUT="${NTC_OUT:-$ROOT/results/$PROFILE}"' in script
    assert '--dense-workers "$WORKERS"' in script
    assert 'LOG="$OUT/logs/workers.log"' in script
    assert "props.total_memory < 24 * (1024 ** 3)" in script
    lines = script.splitlines()
    launches = [i for i, line in enumerate(lines) if "nohup python" in line]
    assert len(launches) == 2
    for start in launches:
        assert "9>&-" in "\n".join(lines[start : start + 8])
