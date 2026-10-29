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


def test_backfill_jobs_skip_core_and_holdout(tmp_path: Path):
    extra = v2.cloud_backfill_jobs(tmp_path)
    materials = {Path(job["material"]).name for job in extra}
    assert materials == {f"{name}_2K" for name in CLOUD_TRAIN if name not in CLOUD_FAST_TRAIN}
    assert not materials & {f"{name}_2K" for name in CLOUD_HOLDOUT}
    assert len(extra) == (len(CLOUD_TRAIN) - len(CLOUD_FAST_TRAIN)) * len(CLOUD_FAST_BPP)
    core = {job["name"] for job in build_cloud_jobs(tmp_path, "cloudfast")}
    assert not core & {job["name"] for job in extra}
    for chain in material_chains(extra):
        assert [job["bpp"] for job in chain] == list(CLOUD_FAST_BPP)
        assert chain[0]["steps"] == CLOUD_FULL_STEPS
        assert {job["batch_pixels"] for job in chain} == {v2.CLOUD_BATCH}


def test_chain_loop_backfills_after_core(tmp_path: Path, monkeypatch):
    jobs = build_cloud_jobs(tmp_path, "cloudfast")
    extra = v2.cloud_backfill_jobs(tmp_path)
    ran = []
    monkeypatch.setattr(v2, "_run", lambda job, *a, **k: ran.append(job["name"]))
    v2._chain_loop(jobs, tmp_path, "fp32", 86400.0, "worker_0", extra)
    assert ran[: len(jobs)] == [job["name"] for job in jobs]
    assert len(ran) == len(jobs) + len(extra)


def test_compile_attempts(monkeypatch):
    from ntc.train_fast import compile_attempts

    monkeypatch.delenv("NTC_MAX_GPU", raising=False)
    monkeypatch.delenv("NTC_COMPILE_MODE", raising=False)
    assert compile_attempts("cpu") == ["default"]
    assert compile_attempts("cuda") == ["reduce-overhead", "default"]
    monkeypatch.setenv("NTC_MAX_GPU", "1")
    assert compile_attempts("cuda") == ["reduce-overhead", "max-autotune-no-cudagraphs", "default"]
    assert compile_attempts("cpu") == ["default"]
    monkeypatch.setenv("NTC_COMPILE_MODE", "max-autotune-no-cudagraphs")
    assert compile_attempts("cuda") == ["max-autotune-no-cudagraphs", "reduce-overhead", "default"]


def test_max_gpu_worker_env(tmp_path: Path):
    env = worker_env(0, base={}, root=str(tmp_path), max_gpu=True)
    assert env["NTC_MAX_GPU"] == "1"
    assert "NTC_MAX_GPU" not in worker_env(0, base={}, root=str(tmp_path))


def _dry_run(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess:
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo / 'src'}{os.pathsep}{repo}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("NTC_BACKFILL", None)
    env.pop("NTC_EXIT_WHEN_DONE", None)
    return subprocess.run(
        [
            sys.executable,
            "bench/train_max_v2.py",
            "--profile",
            "cloudfast",
            "--max-gpu",
            "--dry-run",
            "--out",
            str(tmp_path / "out"),
            "--materials",
            str(tmp_path / "materials"),
            *extra,
        ],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_max_gpu_dry_run_defaults_to_one_worker_per_chain(tmp_path: Path):
    result = _dry_run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "would spawn 4 chain workers" in result.stdout
    # Backfill is opt-in: --max-gpu alone trains only the planned chains.
    assert "plan backfill" not in result.stdout
    result = _dry_run(tmp_path, "--backfill")
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("plan backfill") == len(CLOUD_TRAIN) - len(CLOUD_FAST_TRAIN)


def test_spawn_passes_backfill(monkeypatch, tmp_path: Path):
    calls = []

    class _Proc:
        def wait(self):
            return 0

    monkeypatch.setattr(v2.subprocess, "Popen", lambda cmd, **_k: calls.append(list(cmd)) or _Proc())
    args = argparse.Namespace(
        workers=2, out=tmp_path, materials=tmp_path, probe_steps=800, stale_lock_s=10,
        batch_pixels=None, dry_run=False, pack=True, profile="cloudfast", chains=True,
        stagger_s=0.0, max_gpu=True, backfill=False,
    )
    v2._spawn_workers(args)
    assert all("--backfill" not in cmd for cmd in calls)
    calls.clear()
    args.backfill = True
    v2._spawn_workers(args)
    assert all("--backfill" in cmd for cmd in calls)


def test_completion_marker(tmp_path: Path):
    jobs = [{"name": "a_b1", "bpp": 1.0}, {"name": "a_b2", "bpp": 2.0}, {"name": "a_b3", "bpp": 3.0}]
    (tmp_path / "a_b1.json").write_text(
        '{"done": true, "psnr_u8": 30.0, "total_bpp": 0.99, "target_bpp": 1.0, "wall": 12.5}'
    )
    (tmp_path / "a_b2.json").write_text('{"done": false, "error": "boom"}')
    assert not v2.write_completion(tmp_path, jobs)
    assert (tmp_path / v2.INCOMPLETE_MARKER).is_file()
    assert not (tmp_path / v2.DONE_MARKER).is_file()
    (tmp_path / "a_b3.json").write_text(
        '{"done": true, "psnr_u8": 33.0, "total_bpp": 3.1, "target_bpp": 3.0, "wall": 7.5}'
    )
    assert v2.write_completion(tmp_path, jobs)
    assert not (tmp_path / v2.INCOMPLETE_MARKER).is_file()
    import json

    summary = json.loads((tmp_path / v2.DONE_MARKER).read_text())
    assert summary["done"] == 2 and summary["planned"] == 3
    assert [f["name"] for f in summary["failed"]] == ["a_b2"]
    assert summary["over_label"] == ["a_b3"]
    assert summary["wall_sum_s"] == 20.0


def test_setup_pod_max_gpu():
    repo = Path(__file__).resolve().parents[1]
    script = (repo / "bench/cloud/setup_pod.sh").read_text()
    assert 'MAX_GPU="${NTC_MAX_GPU:-0}"' in script
    assert "EXTRA+=(--max-gpu)" in script
    assert 'WORKERS="${NTC_WORKERS:-4}"' in script


def test_setup_pod_exit_when_done_and_backfill_opt_in():
    repo = Path(__file__).resolve().parents[1]
    script = (repo / "bench/cloud/setup_pod.sh").read_text()
    assert 'EXIT_WHEN_DONE="${NTC_EXIT_WHEN_DONE:-1}"' in script
    assert "EXTRA+=(--exit-when-done)" in script
    assert '"${NTC_BACKFILL:-0}" == "1"' in script
    assert "EXTRA+=(--backfill)" in script
