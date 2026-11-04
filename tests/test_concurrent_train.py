"""Concurrent trainers must each reach a step line.

Three shards on one pod used to share ``/tmp/torchinductor_<user>``. Inductor
forked a compile-worker pool after CUDA was already initialized, the workers
blocked on that cache lock, and only the last shard to start ever trained.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import ntc.train_fast as train_fast
from bench.train_max_v2 import watch_first_step
from ntc.train_fast import configure_inductor_process


def test_inductor_cache_is_per_process_and_compile_is_in_process(tmp_path, monkeypatch):
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.setenv("NTC_ROOT", str(tmp_path))
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "32")
    first = configure_inductor_process("0/5")
    assert first.endswith(os.path.join(".cache", "shard_0_5", "inductor"))
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    assert os.environ["TRITON_CACHE_DIR"].endswith(os.path.join(".cache", "shard_0_5", "triton"))
    # A directory the launcher already chose is kept.
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "kept"))
    os.makedirs(tmp_path / "kept", exist_ok=True)
    assert configure_inductor_process("1/5") == str(tmp_path / "kept")
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "1"


def test_watchdog_dumps_stacks_when_no_step_line_arrives(monkeypatch):
    dumped = {}

    def dump(*_args, **_kwargs):
        dumped["hit"] = True

    monkeypatch.setattr("faulthandler.dump_traceback", dump)
    train_fast.last_step_at = 0.0
    stop = watch_first_step("hung-job", timeout_s=0.4)
    time.sleep(1.0)
    stop.set()
    assert dumped.get("hit") is True


def test_watchdog_stays_quiet_after_a_step_line(monkeypatch):
    dumped = {}
    monkeypatch.setattr("faulthandler.dump_traceback", lambda *_a, **_k: dumped.setdefault("hit", True))
    train_fast.last_step_at = 0.0
    stop = watch_first_step("live-job", timeout_s=1.5)
    train_fast.last_step_at = time.monotonic()
    time.sleep(0.6)
    stop.set()
    assert "hit" not in dumped


def test_three_trainers_each_reach_step_1(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    script = r"""
import os
import sys

root = os.environ["ROOT"]
sys.path[:0] = [root + "/src", root]
import numpy as np
from ntc.encode import TrainConfig
from ntc.train_fast import encode_arrays_fast

i = int(os.environ["I"])
img = np.zeros((3, 16, 16), np.float32)
img[0] = 0.2 + 0.01 * i
encode_arrays_fast(
    img,
    [0, 1, 2],
    TrainConfig(
        steps=2,
        hidden=(4,),
        grids=[(8, 8, 2, 4)],
        fast=True,
        adaptive_pool=1,
        seed_pool=1,
        with_ssim=False,
        log_every=1,
        mip_levels=1,
        seed=i,
        compile_step="on",
        cuda_graph="off",
        precision="fp32",
        device="cpu",
        batch_pixels=16,
        batch_mip="flat",
        quant_schedule=False,
        entropy_rate=False,
        pca_init=False,
        job_name=f"p{i}",
        eval_every=0,
        early_stop=0,
    ),
)
print("REACHED", i, flush=True)
"""
    procs = []
    for i in range(3):
        env = os.environ.copy()
        for key in (
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_COMPILE_THREADS",
        ):
            env.pop(key, None)
        env["ROOT"] = str(repo)
        env["I"] = str(i)
        env["NTC_ROOT"] = str(tmp_path / f"p{i}")
        env["PYTHONPATH"] = f"{repo / 'src'}{os.pathsep}{repo}"
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", script],
                cwd=repo,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    failures = []
    for i, proc in enumerate(procs):
        try:
            out, _ = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            failures.append(f"p{i} timed out\n{out}")
            continue
        if proc.returncode != 0 or "step 1/" not in out:
            failures.append(f"p{i} rc {proc.returncode}\n{out}")
    assert not failures, "\n".join(failures)
