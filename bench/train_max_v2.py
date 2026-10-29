"""Fast training sweep. Writes ``results/train_max_v2`` and does not open ``results/train_max``.

The job list matches ``bench/train_max.py``. Each seed pool is one stacked run
with successive halving at 250 steps, then the surviving seed continues to
the same step count. Precision is chosen by PSNR at equal wall time, not by
PSNR at equal steps. ``TrainConfig.fast`` stays off in ``train_max.py``.

``--workers N`` starts N processes on one shared lock pool. Each process
claims the next undone job (highest pixels * steps first) and
an idle process takes whatever is still unlocked. ``--shard I/N`` is that
same cost order, split longest-job-first, and does not steal.
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import faulthandler
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

# Running ``python bench/train_max_v2.py`` puts bench/ on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT), str(_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from bench.train_max import G16  # noqa: E402
from ntc.device import device_report, pick_device  # noqa: E402
from ntc.encode import TrainConfig  # noqa: E402
from ntc.materials import load_material  # noqa: E402
from ntc.train_fast import RATE_OVERSHOOT, choose_precision, encode_arrays_fast  # noqa: E402


def parse_shard(text: str) -> tuple[int, int]:
    """``I/N`` with ``0 <= I < N``."""
    shard_s, n_s = text.split("/")
    shard, n = int(shard_s), int(n_s)
    if n < 1 or shard < 0 or shard >= n:
        raise SystemExit(f"bad shard {text}")
    return shard, n


def owns_index(index: int, shard: int | None, n: int | None) -> bool:
    if shard is None or n is None:
        return True
    return index % n == shard


def dense_retry_batch(
    trained: object,
    init: object,
    effective: int,
    pixels: int,
    fallback: bool,
) -> int | None:
    """Flat quarter-frame batch for one retry, or None when the run should stop.

    The b031669 guard compared the requested 262144 batch, so a full-resolution
    job that had already been raised to 1048576 flat was trained a second time
    at that same batch. ``sdk_MetalPlates013_2K_b15`` scored 11.70 dB against
    an init of 13.57 on that first dense pass. Retry only when the batch the
    trainer actually used is still under a quarter of the frame.
    """
    if not isinstance(trained, (int, float)) or not isinstance(init, (int, float)):
        return None
    if not math.isfinite(float(trained)) or not math.isfinite(float(init)):
        return None
    # Any gain over the init is a finished low-rate point. The 4 dB band
    # retried PavingStones b2.5 (27.13 vs 24.73 on the area batch) onto a
    # flat quarter frame, which added 0.31 dB and was then marked collapsed.
    if float(trained) >= float(init):
        return None
    if fallback or pixels <= 0:
        return None
    if not (0 < int(effective) < int(pixels) // 4):
        return None
    return int(pixels) // 4


def below_init(trained: object, init: object) -> bool:
    """True when training lost to the PCA file.

    A gain under 4 dB is still a fit. PavingStones b2.5 finished at 27.44
    against an init of 24.73 and had already plateaued. That file is kept.
    """
    if not isinstance(trained, (int, float)) or not isinstance(init, (int, float)):
        return False
    if not math.isfinite(float(trained)) or not math.isfinite(float(init)):
        return False
    return float(trained) < float(init)


def score_gap_blocks_collapse(loop: object, trained: object, init: object) -> bool:
    """True when the live forward still beats the init and the file does not.

    PavingStones b2.5 on 27cbd68 logged 25.26 dB. The file scored 10.32, under
    the 23.04 init, so the job was collapsed and the checkpoint was deleted.
    A gap over 1 dB is not that below-init rule.
    """
    if not isinstance(loop, (int, float)) or not isinstance(trained, (int, float)):
        return False
    if not isinstance(init, (int, float)):
        return False
    if not math.isfinite(float(loop)) or not math.isfinite(float(trained)) or not math.isfinite(float(init)):
        return False
    if abs(float(loop) - float(trained)) <= 1.0:
        return False
    return float(loop) >= float(init) and float(trained) < float(init)


def collapse_reason(trained: float, init: float, effective: int, pixels: int) -> str:
    """Why a below-init json is left ``done`` false."""
    if int(effective) >= max(int(pixels) // 4, 1):
        return (
            f"trained PSNR {float(trained):.2f} fell below init {float(init):.2f} "
            f"on dense batch {int(effective)}; not retrying the same run"
        )
    return (
        f"trained PSNR {float(trained):.2f} fell below init {float(init):.2f}"
    )


def _rate_family(name: str) -> tuple[str, float] | None:
    if "_b" not in name:
        return None
    stem, bpp_s = name.rsplit("_b", 1)
    try:
        return stem, float(bpp_s)
    except ValueError:
        return None


def _lower_rows(out: Path, job: dict) -> list[dict]:
    parsed = _rate_family(str(job.get("name") or ""))
    if parsed is None or not out.is_dir():
        return []
    stem, bpp = parsed
    rows = []
    for path in out.glob(f"{stem}_b*.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not row.get("done") or not row.get("grids"):
            continue
        other = _rate_family(str(row.get("name") or path.stem))
        if other is None or other[0] != stem or other[1] >= bpp - 1e-6:
            continue
        score = row.get("psnr_u8")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            continue
        rows.append({
            "name": str(row.get("name") or path.stem),
            "psnr": float(score),
            "bpp": other[1],
            "grids": row["grids"],
            "checkpoint": row.get("checkpoint"),
            "total_bpp": row.get("total_bpp"),
        })
    return rows


def best_lower_sibling(out: Path, job: dict) -> dict | None:
    """The next-lower finished label for ``job``, the neighbor to warm-start from."""
    rows = _lower_rows(out, job)
    if not rows:
        return None
    neighbor_bpp = max(row["bpp"] for row in rows)
    neighbors = [row for row in rows if abs(row["bpp"] - neighbor_bpp) < 1e-6]
    return max(neighbors, key=lambda row: row["psnr"])


def cheaper_better(out: Path, job: dict, psnr: float) -> dict | None:
    """Finished lower-bitrate sibling of ``job`` whose file PSNR beats ``psnr``.

    ``b5`` at 33.45 dB beat ``b10`` at 33.07 dB. The higher label is retrained
    from a channel-grown copy of the cheaper grid so step 0 matches that file.
    """
    rows = [row for row in _lower_rows(out, job) if row["psnr"] > float(psnr) + 0.05]
    if not rows:
        return None
    return max(rows, key=lambda row: row["psnr"])


def ladder_step_budget(count: int, full: int) -> list[int]:
    """First label keeps ``full`` steps. Each later label gets half.

    Five labels at 4000 steps were 20000 steps. The budget is 12000, which
    is 40 percent less. The first point is trained from scratch. The rest
    warm-start, which measured +1.6 dB and 1.95 bpp less at equal steps.
    """
    if count <= 0:
        return []
    half = max(1, int(full) // 2)
    return [int(full)] + [half] * (count - 1)


def _higher_rows(out: Path, job: dict) -> list[dict]:
    parsed = _rate_family(str(job.get("name") or ""))
    if parsed is None or not out.is_dir():
        return []
    stem, bpp = parsed
    rows = []
    for path in out.glob(f"{stem}_b*.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not row.get("done") or not row.get("grids"):
            continue
        other = _rate_family(str(row.get("name") or path.stem))
        if other is None or other[0] != stem or other[1] <= bpp + 1e-6:
            continue
        score = row.get("psnr_u8")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            continue
        rows.append({
            "name": str(row.get("name") or path.stem),
            "psnr": float(score),
            "bpp": other[1],
            "checkpoint": row.get("checkpoint"),
        })
    return rows


def best_higher_sibling(out: Path, job: dict) -> dict | None:
    rows = _higher_rows(out, job)
    if not rows:
        return None
    neighbor_bpp = min(row["bpp"] for row in rows)
    neighbors = [row for row in rows if abs(row["bpp"] - neighbor_bpp) < 1e-6]
    return max(neighbors, key=lambda row: row["psnr"])


def apply_rate_neighbor(job: dict, out: Path) -> dict:
    """Point a latent job at its lower-rate neighbor before the first step.

    The grid keeps that neighbor's lattice and adds channels until the target
    raw rate. The checkpoint is loaded when the neighbor file kept its trained
    weights, so the new run starts at the neighbor's PSNR.
    """
    job = dict(job)
    if job.get("_neighbor_applied") or job.get("_rd_retry") or job.get("_fallback"):
        return job
    job["_neighbor_applied"] = True
    if str(job.get("rate") or "") != "latent":
        return job
    lower = best_lower_sibling(out, job)
    if lower is None or not lower.get("grids"):
        higher = best_higher_sibling(out, job)
        if higher is None:
            return job
        prior = out / f"{higher['name']}.pt"
        if prior.is_file() and higher.get("checkpoint") == "trained" and not job.get("warm_start"):
            job["warm_start"] = str(prior)
            print(f"warm start {job['name']} downward from {higher['name']}", flush=True)
        return job
    from ntc.grids import grow_grids

    scale = float(job.get("latent_rate_scale") or 1.0)
    target = float(job["bpp"])
    if 0 < scale < 1:
        target = target / scale
    # A calibrated grid on this label is replaced. Higher labels keep the
    # lower neighbor's lattice and only add channels.
    if is_cloud_profile(job.get("profile")):
        target = target * cloud_grid_headroom(float(job["bpp"]))
    grown = grow_grids(
        [tuple(g) for g in lower["grids"]],
        int(job["size"]),
        int(job["size"]),
        target,
    )
    calibrated = job.get("grids")
    headroom = cloud_grid_headroom(float(job["bpp"])) if is_cloud_profile(job.get("profile")) else 1.0
    if calibrated and (headroom > 1.0 or not grown_reaches(grown, int(job["size"]), target)):
        # The lower lattice tops out near 5 bpp at 16 channels. The label's
        # own finer lattice reaches the rate, so it trains from scratch on
        # the full step budget instead. From b10 up the grown chain also
        # landed 8 to 19 percent under the label (Tiles b10 at 256: 32.48 dB
        # at 8.83 bpp chained, 33.73 dB at 10.00 bpp on its own lattice).
        full = int(job.get("full_steps") or 0)
        if full > int(job["steps"]):
            job["steps"] = full
            job["warmup"] = int(job.get("full_warmup") or job.get("warmup") or 200)
        print(
            f"own lattice {job['name']} {calibrated}, {lower['name']} grown {grown} "
            f"is under {target:.2f} raw bpp",
            flush=True,
        )
        return job
    job["grids"] = [list(g) for g in grown]
    prior = out / f"{lower['name']}.pt"
    if prior.is_file() and lower.get("checkpoint") == "trained" and not job.get("warm_start"):
        job["warm_start"] = str(prior)
    print(f"warm start {job['name']} from {lower['name']}", flush=True)
    return job


def grown_reaches(grids, size: int, target_raw_bpp: float) -> bool:
    """True when ``grids`` store at least 95 percent of ``target_raw_bpp``."""
    from ntc.grids import latent_payload_bits

    pixels = float(max(int(size) * int(size), 1))
    have = latent_payload_bits([tuple(g) for g in grids]) / pixels
    return have >= 0.95 * float(target_raw_bpp)


# First train plus one tighter grid. A miss after that is the result.
MAX_RATE_ATTEMPTS = 2


def _missing_material_error(row: dict) -> bool:
    err = row.get("error")
    return isinstance(err, str) and (err == "missing" or err.startswith("missing "))


def _result_terminal(row: dict, material: Path | None = None) -> bool:
    """True when this json must not be trained again.

    A missing-directory error is retried once that directory has a manifest.
    ``done``, a collapse, a rate miss, or any other error is finished. A
    ``done`` false json with none of those was claimed again, so a high-rate
    miss retrained until the process was killed.
    """
    if not isinstance(row, dict):
        return False
    if row.get("done") or row.get("collapsed") or row.get("rate_miss"):
        return True
    if row.get("error"):
        if _missing_material_error(row):
            if material is None:
                return True
            return not (Path(material) / "manifest.json").is_file()
        return True
    return False


def _bpp_ratio(row: dict) -> float | None:
    bpp = row.get("total_bpp")
    target = row.get("target_bpp")
    if not isinstance(bpp, (int, float)) or not isinstance(target, (int, float)):
        return None
    if not math.isfinite(float(bpp)) or not math.isfinite(float(target)) or float(target) <= 0:
        return None
    return float(bpp) / float(target)


def rate_over_label(row: dict, band: float = RATE_OVERSHOOT) -> bool:
    """True when the file is past the grid-selection band over its label."""
    ratio = _bpp_ratio(row)
    return ratio is not None and ratio > float(band)


def _attempt_rank(row: dict) -> tuple:
    """Lower is better: a real on-target file, then the closest miss, then a broken file.

    A score-gap or collapsed file is not a rate point. Keeping it ahead of a
    healthy miss is how a trim that fell under the init used to replace the fit.
    """
    ratio = _bpp_ratio(row)
    psnr = row.get("psnr_u8")
    psnr_v = float(psnr) if isinstance(psnr, (int, float)) and math.isfinite(float(psnr)) else -1e9
    if row.get("collapsed") or row.get("score_gap"):
        return (3, 0.0, -psnr_v)
    if ratio is None:
        return (2, 0.0, -psnr_v)
    if ratio <= RATE_OVERSHOOT:
        return (0, abs(ratio - 1.0), -psnr_v)
    return (1, ratio, -psnr_v)


def prefer_rate_attempt(prev: dict | None, new: dict) -> dict:
    if prev is None:
        return new
    if _attempt_rank(new) < _attempt_rank(prev):
        return new
    return prev


def settle_rate_attempt(
    prev: dict | None, new: dict, max_attempts: int = MAX_RATE_ATTEMPTS
) -> tuple[dict, bool]:
    """Keep the closer file. One retry while it is over the label, then stop.

    The row is always terminal. ``retry`` is true only when another train
    is still allowed to replace it. A killed process leaves ``done`` set,
    so the queue does not start that train over.
    """
    chosen = dict(prefer_rate_attempt(prev, new))
    attempts = int(new.get("rate_attempts") or 0)
    if isinstance(prev, dict):
        attempts = max(attempts, int(prev.get("rate_attempts") or 0))
    if attempts < 1:
        attempts = 1
    chosen["rate_attempts"] = attempts
    over = rate_over_label(chosen)
    chosen["done"] = True
    chosen["rate_miss"] = bool(over)
    if over:
        chosen["bpp_ratio"] = _bpp_ratio(chosen)
    else:
        chosen.pop("bpp_ratio", None)
    return chosen, bool(over and attempts < int(max_attempts))


def tighter_latent_grids(grids, size: int, label_bpp: float, file_bpp: float) -> list | None:
    """A smaller latent whose raw size matches ``label * raw / file``.

    The 1.15 headroom is not applied again. None when nothing smaller exists,
    which ends the retry instead of training the same grid.
    """
    from ntc.grids import latent_payload_bits, select_grids

    if not grids or float(label_bpp) <= 0 or float(file_bpp) <= float(label_bpp) * RATE_OVERSHOOT:
        return None
    current = [tuple(g) for g in grids]
    pixels = float(max(int(size) * int(size), 1))
    raw = latent_payload_bits(current) / pixels
    if raw <= 0:
        return None
    want = float(label_bpp) * raw / float(file_bpp)
    picked = select_grids(
        int(size),
        int(size),
        float(want),
        [32],
        9,
        0,
        0,
        8,
        0,
        adaptive_pool=1,
        rate="latent",
        min_bits=4,
    )[0]
    if latent_payload_bits(picked) >= latent_payload_bits(current):
        return None
    return [list(g) for g in picked]


def _ckpt_bytes(ckpt: Path) -> bytes | None:
    if not ckpt.is_file():
        return None
    try:
        return ckpt.read_bytes()
    except OSError:
        return None


def _write_attempt(bundle: dict, dest: Path, ntc_path: Path, ckpt: Path) -> None:
    dest.write_text(json.dumps(bundle["row"], indent=2))
    if bundle.get("blob") is not None:
        ntc_path.write_bytes(bundle["blob"])
    data = bundle.get("ckpt")
    if data is not None:
        ckpt.write_bytes(data)
    elif ckpt.is_file():
        ckpt.unlink()


def _restore_if_better(job: dict, row: dict, blob: bytes, dest: Path, ntc_path: Path, ckpt: Path) -> dict:
    """Write ``row`` unless a saved attempt is closer to the label."""
    prev = job.get("_best_bundle")
    prev_row = prev.get("row") if isinstance(prev, dict) else None
    if not isinstance(prev_row, dict) or _attempt_rank(row) < _attempt_rank(prev_row):
        row = dict(row)
        if rate_over_label(row):
            row["rate_miss"] = True
            row["bpp_ratio"] = _bpp_ratio(row)
        _write_attempt({"row": row, "blob": blob, "ckpt": _ckpt_bytes(ckpt)}, dest, ntc_path, ckpt)
        return row
    restored = dict(prev_row)
    restored["rate_attempts"] = max(
        int(restored.get("rate_attempts") or 0), int(row.get("rate_attempts") or 0)
    )
    restored["done"] = True
    if rate_over_label(restored):
        restored["rate_miss"] = True
        restored["bpp_ratio"] = _bpp_ratio(restored)
    _write_attempt(
        {"row": restored, "blob": prev.get("blob"), "ckpt": prev.get("ckpt")},
        dest,
        ntc_path,
        ckpt,
    )
    return restored


def _force_terminal(out: Path, job: dict) -> None:
    """Mark a finished-but-unsettled json done so the queue cannot claim it.

    Missing-material errors stay retryable. Everything else that is still
    ``done`` false becomes a recorded miss. The bytes already on disk stay.
    """
    dest = out / f"{job['name']}.json"
    if not dest.is_file():
        return
    try:
        row = json.loads(dest.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return
    if _result_terminal(row, Path(job["material"])) or _missing_material_error(row):
        return
    row = dict(row)
    row["done"] = True
    row["rate_miss"] = True
    ratio = _bpp_ratio(row)
    if ratio is not None:
        row["bpp_ratio"] = ratio
    row["rate_attempts"] = max(int(row.get("rate_attempts") or 0), MAX_RATE_ATTEMPTS)
    dest.write_text(json.dumps(row, indent=2))
    print(f"settle {job['name']} attempts exhausted", flush=True)


def upper_hull(points: list[tuple[float, float, str]]) -> set[str]:
    """Names on the upper convex hull of ``(rate, psnr, name)``.

    A point under the chord between its neighbors is dropped. A point that
    does not beat every cheaper rate is dropped. Endpoints stay when they
    are not duplicated at the same rate by a higher score.
    """
    ordered = sorted(points, key=lambda item: (item[0], item[1]))
    dedup: list[tuple[float, float, str]] = []
    for rate, psnr, name in ordered:
        if dedup and abs(dedup[-1][0] - rate) <= 1e-9:
            if psnr >= dedup[-1][1]:
                dedup[-1] = (rate, psnr, name)
            continue
        dedup.append((rate, psnr, name))
    hull: list[tuple[float, float, str]] = []
    for point in dedup:
        while len(hull) >= 2:
            origin, prev = hull[-2], hull[-1]
            cross = (prev[0] - origin[0]) * (point[1] - origin[1]) - (prev[1] - origin[1]) * (
                point[0] - origin[0]
            )
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(point)
    return {item[2] for item in hull}


def refresh_rate_hull(out: Path, name: str) -> None:
    """Write ``on_hull`` for every finished bitrate of this material."""
    parsed = _rate_family(name)
    if parsed is None or not out.is_dir():
        return
    stem, _bpp = parsed
    points = []
    files = []
    for path in sorted(out.glob(f"{stem}_b*.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not row.get("done"):
            continue
        rate = row.get("total_bpp")
        score = row.get("psnr_u8")
        label = str(row.get("name") or path.stem)
        if not isinstance(rate, (int, float)) or not isinstance(score, (int, float)):
            continue
        points.append((float(rate), float(score), label))
        files.append((path, row, label))
    keep = upper_hull(points)
    for path, row, label in files:
        flag = label in keep
        if row.get("on_hull") is flag:
            continue
        row["on_hull"] = flag
        path.write_text(json.dumps(row, indent=2))


def job_cost(job: dict) -> float:
    """Estimated work: pixels per step times steps.

    A ``batch_pixels`` below the frame counts as the batch. A batch of 0, or a
    batch at least as large as the frame, counts as the full frame. The 256
    sampler ignores a batch larger than the frame. A seed pool is one stacked
    step, so it is not a second factor.
    """
    size = int(job["size"])
    full = size * size
    batch = int(job.get("batch_pixels") or 0)
    pixels = batch if 0 < batch < full else full
    return float(pixels) * float(job["steps"])


def assign_shards(jobs: list[dict], n: int) -> list[int]:
    """Longest-processing-time-first. One shard id per job, in job order."""
    if n < 1:
        raise SystemExit("shard count must be >= 1")
    order = sorted(range(len(jobs)), key=lambda i: job_cost(jobs[i]), reverse=True)
    load = [0.0] * n
    assign = [0] * len(jobs)
    for index in order:
        shard = min(range(n), key=lambda s: (load[s], s))
        assign[index] = shard
        load[shard] += job_cost(jobs[index])
    return assign


_SDK_DIRS = ("MetalPlates013_2K", "PavingStones070_2K", "Tiles135D_2K")

# ambientCG 2K names. Train and held-out are disjoint. Held-out is downloaded
# and never given a job, so a config search cannot fit those three.
CLOUD_TRAIN = (
    "MetalPlates013",
    "Metal003",
    "PavingStones070",
    "Rock023",
    "Ground054",
    "Tiles135D",
    "Tiles074",
    "Fabric048",
    "Carpet001",
    "WoodFloor051",
    "Concrete034",
    "Bark001",
)
CLOUD_HOLDOUT = ("Leather011", "LeafSet017", "Gravel022")
# 1 and 2 sit under BC1. 4 is BC1. 8 is BC7 and ASTC 4x4. 2.5 stays so the
# laptop b2.5 points still have a neighbor. 8000 on the first label: the
# 5070 MetalPlates b2.5 run was still above its init at 4000 and had not
# flattened. Tiles flattened at 3000 under a raw-rate penalty that the
# file-unit fix removes, so the extra steps are there for that climb.
CLOUD_BPP = (1.0, 2.0, 2.5, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)
CLOUD_FULL_STEPS = 8000
# 2,097,152 flat samples. A 1,048,576 dense step was about 6.5 GB. Twice
# that fits a 32 GB card and leaves room for Adam and the compile workspace.
# A 4,194,304 frame would sit near 26 GB and is not the default.
CLOUD_BATCH = 2 * 1024 * 1024
# Budget, not a measured 5090 number. A 5070 laptop did about 20 steps/s
# at 1,048,576. This uses that rate for a 2,097,152 batch, so the hour
# figure is high if the 5090 is faster and low if the larger batch is slower.
# The seed pool is three wide for 500 steps and one wide after two halvings.
CLOUD_STEPS_PER_S = 20.0
CLOUD_SEED_FACTOR = 1.25
# Short cloud run: four train materials, five labels. 5 sits between 2.5
# and 10 so the curve has a point near BC1 and BC7 without the full ladder.
CLOUD_FAST_TRAIN = ("MetalPlates013", "PavingStones070", "Tiles135D", "Bark001")
CLOUD_FAST_BPP = (1.0, 2.5, 5.0, 10.0, 20.0)
# Every profile that trains the cloud 2K latent ladder.
CLOUD_PROFILES = ("cloud5090", "cloudfast")
# A card at or above this size runs several dense 2K processes at once.
# One dense 2K process at a 2,097,152 batch held about 6 GB.
DENSE_WORKERS_MIN_VRAM_GB = 80.0


def is_cloud_profile(profile: str | None) -> bool:
    return str(profile or "") in CLOUD_PROFILES


def cloud_backfill_jobs(root: Path) -> list[dict]:
    """The cloudfast ladder on the other cloud train materials.

    A chain worker with ``--max-gpu`` takes these once the four cloudfast
    chains are claimed, so the card does not idle. Held-out names are never
    jobs.
    """
    extra = tuple(name for name in CLOUD_TRAIN if name not in CLOUD_FAST_TRAIN)
    return _cloud_ladder(root, extra, CLOUD_FAST_BPP, "cloudfast")


def cloud_profile_bpp(profile: str | None) -> tuple[float, ...]:
    return CLOUD_FAST_BPP if str(profile or "") == "cloudfast" else CLOUD_BPP


_CLOUD_FIT: dict[tuple, dict[float, tuple]] = {}

# Grids are sized for this much over the label. The PCA coder ratio
# overestimates the trained file, so a grid fit to the label itself landed
# 4 to 20 percent under it (Tiles b5 at 256: 4.56 bpp). The hinge and the
# export trim bring a larger grid down onto the label.
# Low labels keep the plain fit: a 1.25 grid trimmed onto b1 lost 0.6 to
# 1.0 dB at 256, more than the 3 percent rate it recovered.
CLOUD_GRID_HEADROOM = 1.15
CLOUD_HEADROOM_FROM_BPP = 10.0


def cloud_grid_headroom(bpp: float) -> float:
    return CLOUD_GRID_HEADROOM if float(bpp) >= CLOUD_HEADROOM_FROM_BPP - 1e-9 else 1.0


def cloud_latent_setups(
    images,
    semantics: list[int],
    size: int,
    min_bits: int = 4,
    labels: tuple[float, ...] | None = None,
) -> dict[float, tuple]:
    """Fit every cloud label for one image. Later jobs read this table.

    The per-label probe used to range-code a full 1024 grid on one CPU
    core between jobs. One pass fills the table. A second call is a lookup.
    """
    from ntc.train_fast import _image_rate_key, fit_latent_grids

    labels = CLOUD_BPP if labels is None else tuple(labels)
    key = (
        _image_rate_key(images),
        int(size),
        int(min_bits),
        int(images.shape[0]),
        tuple(round(float(b), 4) for b in labels),
    )
    found = _CLOUD_FIT.get(key)
    if found is not None:
        return found
    table: dict[float, tuple] = {}
    started = time.perf_counter()
    for bpp in labels:
        grids, ratio = fit_latent_grids(
            images,
            semantics,
            int(size),
            int(size),
            float(bpp) * cloud_grid_headroom(float(bpp)),
            [32],
            int(images.shape[0]),
            int(min_bits),
        )
        table[round(float(bpp), 4)] = (grids, float(ratio))
    _CLOUD_FIT[key] = table
    print(
        f"latent setups {len(labels)} labels {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return table


def install_fault_hooks() -> None:
    """``kill -USR1 <pid>`` dumps every thread into the shard log."""
    faulthandler.enable(file=sys.stderr, all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)


def step_watch_timeout() -> float:
    raw = os.environ.get("NTC_STEP_WATCH_S", "300")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 300.0


def watch_first_step(job_name: str, timeout_s: float | None = None) -> threading.Event:
    """Dump stacks if this job prints no step line before ``timeout_s``.

    The hang after ``latent rate scale`` never reaches the step printer, so
    the line itself cannot report it. Default timeout is 5 minutes.
    """
    import ntc.train_fast as train_fast

    timeout = step_watch_timeout() if timeout_s is None else float(timeout_s)
    baseline = float(train_fast.last_step_at)
    stop = threading.Event()

    def run() -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if stop.is_set() or float(train_fast.last_step_at) > baseline:
                return
            stop.wait(0.25)
        if stop.is_set() or float(train_fast.last_step_at) > baseline:
            return
        print(
            f"watchdog: no step line within {timeout:.0f}s of {job_name}; dumping stacks",
            flush=True,
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    threading.Thread(target=run, name=f"step-watch-{job_name}", daemon=True).start()
    return stop


def planned_steps(job: dict) -> int:
    """Steps a ladder job runs when nothing stops it early."""
    steps = int(job["steps"])
    full = int(job.get("full_steps") or 0)
    if full and cloud_grid_headroom(float(job.get("bpp") or 0)) > 1.0:
        return max(steps, full)
    return steps


def cloud_queue_gpu_hours(jobs: list[dict] | None = None) -> float:
    """GPU-hours for the cloud queue on the 30 steps/s budget.

    Labels from ``CLOUD_HEADROOM_FROM_BPP`` up always train their own
    lattice for ``full_steps``. A lower label also does when its grown
    lattice cannot reach the rate, which is decided at run time, so it is
    counted at its chain budget here.
    """
    if jobs is None:
        jobs = build_cloud_jobs(Path("."))
    steps = sum(planned_steps(job) for job in jobs)
    return steps * CLOUD_SEED_FACTOR / CLOUD_STEPS_PER_S / 3600.0


def _sdk_missing(root: Path) -> list[Path]:
    return [root / name for name in _SDK_DIRS if not (root / name / "manifest.json").is_file()]


def ensure_sdk_materials(root: Path) -> None:
    """Download the RTXNTC 2K set when a manifest is missing, then stop if it still is.

    This is ``download_materials --preset rtxntc``. Workers are not started
    while one of the three directories is absent, so a process cannot exit
    early on a job list that is only those materials. A lock file keeps two
    processes from unzipping the same archive.
    """
    if not _sdk_missing(root):
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".rtxntc_download.lock"
    while _sdk_missing(root):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_stale(lock, 3600):
                try:
                    lock.unlink()
                except OSError:
                    pass
            else:
                time.sleep(1.0)
            continue
        try:
            os.write(fd, json.dumps({"pid": os.getpid(), "name": "rtxntc"}).encode())
        finally:
            os.close(fd)
        try:
            gap = _sdk_missing(root)
            if not gap:
                return
            print("download rtxntc", " ".join(path.name for path in gap), flush=True)
            try:
                from bench.download_materials import sdk_materials

                sdk_materials(root)
            except Exception as exc:
                raise SystemExit(f"download_materials --preset rtxntc failed: {exc}") from exc
            gap = _sdk_missing(root)
            if gap:
                raise SystemExit("missing SDK materials after download: " + ", ".join(str(path) for path in gap))
        finally:
            try:
                lock.unlink()
            except OSError:
                pass
        return


def precision_identity(info: dict | None = None) -> dict:
    """Device name and backend that ``precision.json`` must match."""
    if info is None:
        info = device_report(pick_device("auto"))
    return {
        "device_name": str(info.get("name") or ""),
        "backend": str(info.get("backend") or ""),
    }


def precision_matches(report: dict, identity: dict) -> bool:
    """True when this process may reuse a stored precision probe.

    A file written on the RX 9070 must not choose bf16 for an RTX 5070.
    A file with no device key is the old format and is not reused.
    """
    if not isinstance(report, dict) or not isinstance(identity, dict):
        return False
    if report.get("device_name") != identity.get("device_name"):
        return False
    if report.get("backend") != identity.get("backend"):
        return False
    return report.get("equal_wall_choice") in ("fp32", "bf16", "fp16")


def nonfinite_score(row: dict) -> str | None:
    """Name of the first PSNR field that cannot be ranked, or None."""
    for key in ("psnr_u8", "psnr_u8_init", "psnr_u8_trained", "best_val_psnr"):
        value = row.get(key)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return f"non-finite {key}"
    return None


def _job_blocked(out: Path, job: dict) -> bool:
    """Skip a finished json. Skip an error json only while its material is still absent.

    A previous run records ``error`` for a missing directory. Once that
    directory has a manifest, the next process trains it. Any other error,
    a collapse, or a rate miss is finished. Inside one process the caller
    also passes a skip set so a crash is not claimed again.
    """
    dest = out / f"{job['name']}.json"
    if not dest.is_file():
        return False
    try:
        prev = json.loads(dest.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return _result_terminal(prev, Path(job["material"]))


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _lock_stale(lock: Path, stale_s: float) -> bool:
    try:
        payload = json.loads(lock.read_text())
        pid = int(payload.get("pid", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        return True
    age = time.time() - lock.stat().st_mtime
    if age > stale_s:
        return True
    return not _pid_alive(pid)


def claim_job(out: Path, name: str, stale_s: float) -> bool:
    """Atomic create of ``locks/<name>.lock``. False if a live worker holds it."""
    lock = out / "locks" / f"{name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        if not _lock_stale(lock, stale_s):
            return False
        try:
            lock.unlink()
        except OSError:
            return False
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock), flags)
    except FileExistsError:
        return False
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "name": name, "time": time.time()}).encode())
    finally:
        os.close(fd)
    return True


def release_job(out: Path, name: str) -> None:
    lock = out / "locks" / f"{name}.lock"
    try:
        lock.unlink()
    except OSError:
        pass


GPU_DENSE_LOCK = "gpu_dense"
# Chain workers on a large card set this to 0 so dense 2K jobs train side
# by side. The per-job lock files still stop a job from running twice.
_DENSE_LOCK_ENV = "NTC_DENSE_LOCK"


def dense_lock_enabled() -> bool:
    return os.environ.get(_DENSE_LOCK_ENV, "1") != "0"


def _select_job_grids(job: dict) -> list:
    grids = job.get("grids")
    if grids:
        return [tuple(g) for g in grids]
    from ntc.grids import select_grids

    bpp = float(job["bpp"])
    # An unmeasured scale is 0 until the PCA probe. The GPU lock still uses
    # the historical 0.5 estimate so a b5-sized grid is not treated as small.
    raw_scale = job.get("latent_rate_scale")
    scale = 0.5 if raw_scale is None or float(raw_scale) <= 0 else float(raw_scale)
    if str(job.get("rate") or "payload") == "latent" and 0 < scale < 1:
        bpp = bpp / scale
    size = int(job["size"])
    return select_grids(
        size,
        size,
        bpp,
        [32],
        9,
        0,
        0,
        8,
        0,
        rate=str(job.get("rate") or "payload"),
        min_bits=int(job.get("min_latent_bits") or 0),
    )[0]


def is_dense_2k(job: dict) -> bool:
    """True when this job trains at a flat quarter-frame batch.

    One such step already fills the RX 9070. A second process on the same
    GPU made each job about 3x slower and did not raise throughput.
    """
    size = int(job.get("size") or 0)
    if size < 2048:
        return False
    from ntc.train_fast import fullres_batch_floor

    grids = _select_job_grids(job)
    batch, mode = fullres_batch_floor(
        size,
        size,
        grids,
        int(job.get("batch_pixels") or 0),
        str(job.get("batch_mip") or "flat"),
    )
    return mode == "flat" and batch >= (size * size) // 4


def _unfinished_jobs(jobs: list[dict], out: Path) -> list[dict]:
    pending = []
    for job in jobs:
        dest = out / f"{job['name']}.json"
        if dest.is_file():
            try:
                if _result_terminal(json.loads(dest.read_text()), Path(job["material"])):
                    continue
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        pending.append(job)
    return pending


def planned_workers(requested: int, jobs: list[dict], out: Path, dense_ok: bool = False) -> int:
    """Use one process when every remaining job is a dense 2K run.

    ``dense_ok`` is set when the card has room for several dense processes
    (``--dense-workers`` or a card with at least 80 GB). Then ``requested``
    is kept.
    """
    if requested <= 1:
        return requested
    if dense_ok:
        return requested
    pending = _unfinished_jobs(jobs, out)
    if pending and all(is_dense_2k(job) for job in pending):
        return 1
    return requested


def _wait_dense_gpu(out: Path, stale_s: float) -> None:
    while not claim_job(out, GPU_DENSE_LOCK, stale_s):
        time.sleep(0.5)


def _run(job: dict, out: Path, precision: str, shard_label: str, stale_s: float) -> dict:
    dest = out / f"{job['name']}.json"
    ntc_path = out / f"{job['name']}.ntc"
    ckpt = out / f"{job['name']}.pt"
    if dest.is_file():
        prev = json.loads(dest.read_text())
        if _result_terminal(prev, Path(job["material"])):
            print("skip", job["name"], flush=True)
            return prev
    if not claim_job(out, job["name"], stale_s):
        print("locked", job["name"], flush=True)
        return {"name": job["name"], "done": False, "locked": True}
    try:
        return _run_claimed(job, out, precision, shard_label, dest, ntc_path, ckpt, stale_s)
    finally:
        release_job(out, job["name"])


def _run_claimed(
    job: dict,
    out: Path,
    precision: str,
    shard_label: str,
    dest: Path,
    ntc_path: Path,
    ckpt: Path,
    stale_s: float = 86400.0,
) -> dict:
    # Retries copy the job dict, so the clock survives them and ``wall``
    # covers the latent fit, the lock wait and every retry.
    job.setdefault("_started", time.perf_counter())
    if dest.is_file() and not job.get("_replace_done"):
        prev = json.loads(dest.read_text())
        if _result_terminal(prev, Path(job["material"])):
            print("skip", job["name"], flush=True)
            return prev
    mat = Path(job["material"])
    if not (mat / "manifest.json").is_file():
        row = {"name": job["name"], "done": False, "error": f"missing {mat}"}
        dest.write_text(json.dumps(row, indent=2))
        print("missing", job["name"], mat, flush=True)
        return row
    loaded = job.get("_loaded")
    if loaded is None:
        images, sems, _info = load_material(mat, int(job["size"]))
    else:
        images, sems, _info = loaded
    if (
        str(job.get("rate") or "") == "latent"
        and job.get("grids") is None
        and float(job.get("latent_rate_scale") or 0) <= 0
    ):
        bits = int(job.get("min_latent_bits") or 0)
        if is_cloud_profile(job.get("profile")):
            table = cloud_latent_setups(
                images, sems, int(job["size"]), bits, cloud_profile_bpp(job.get("profile"))
            )
            grids, ratio = table[round(float(job["bpp"]), 4)]
        else:
            from ntc.train_fast import fit_latent_grids

            grids, ratio = fit_latent_grids(
                images,
                sems,
                int(job["size"]),
                int(job["size"]),
                float(job["bpp"]),
                [32],
                int(images.shape[0]),
                bits,
            )
        job = dict(job)
        job["latent_rate_scale"] = ratio
        job["grids"] = [list(g) for g in grids]
        print(f"latent rate scale {job['name']} {ratio:.3f} grids {grids}", flush=True)
    if not job.get("_gpu_held") and dense_lock_enabled() and is_dense_2k(job):
        _wait_dense_gpu(out, stale_s)
        held = dict(job)
        held["_gpu_held"] = True
        held["_loaded"] = (images, sems, _info)
        try:
            return _run_claimed(held, out, precision, shard_label, dest, ntc_path, ckpt, stale_s)
        finally:
            release_job(out, GPU_DENSE_LOCK)
    job = apply_rate_neighbor(job, out)
    cfg = TrainConfig(
        steps=int(job["steps"]),
        target_bpp=float(job["bpp"]),
        grids=None if job.get("grids") is None else [tuple(g) for g in job["grids"]],
        seed=int(job["seed"]),
        adaptive_pool=int(job.get("adaptive_pool", 1)),
        seed_pool=int(job.get("seed_pool", 1)),
        latent_deflate=True,
        latent_coder="best",
        rate=str(job.get("rate", "payload")),
        mip_levels=int(job.get("mip_levels", 4)),
        mip_loss="footprint",
        warmup_steps=int(job.get("warmup", 200)),
        schedule=str(job.get("schedule", "cosine")),
        precision=precision,
        eval_every=int(job.get("eval_every", 500)),
        early_stop=int(job.get("early_stop", 8)),
        batch_pixels=int(job.get("batch_pixels", 0)),
        quant_freeze_at=float(job.get("quant_freeze_at", 0.95)),
        lr_batch_ref=int(job.get("lr_batch_ref", 0)),
        min_latent_bits=int(job.get("min_latent_bits", 0)),
        batch_mip=str(job.get("batch_mip", "flat")),
        latent_rate_scale=float(job.get("latent_rate_scale", 1.0)),
        entropy_rate=bool(job.get("entropy_rate", False)),
        quant_schedule=bool(job.get("quant_schedule", False)),
        resume=str(ckpt) if ckpt.is_file() else "",
        ckpt=str(ckpt),
        log_every=100,
        job_name=str(job["name"]),
        warm_start=str(job.get("warm_start") or ""),
        device="auto",
        with_ssim=False,
        fast=True,
        halve_steps=int(job.get("halve_steps", 250 if int(job.get("seed_pool", 1)) > 1 else 0)),
        compile_step="auto",
        cuda_graph="auto",
    )
    t0 = time.perf_counter()
    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} start {job['name']} "
        f"steps {job['steps']} batch {job.get('batch_pixels')} {job.get('batch_mip')}",
        flush=True,
    )
    stop_watch = watch_first_step(str(job["name"]))
    try:
        blob, stats = encode_arrays_fast(images, sems, cfg)
    finally:
        stop_watch.set()
    wall = time.perf_counter() - t0
    row = {
        "name": job["name"],
        "done": True,
        "fast": True,
        "wall_s": wall,
        # End to end for this label, the latent fit and any retry included.
        "wall": time.perf_counter() - float(job["_started"]),
        "psnr_u8": stats["metrics"]["psnr_u8"],
        # Achieved file rate, the block_baseline meaning. The label is target_bpp.
        "bpp": stats["total_bpp"],
        "total_bpp": stats["total_bpp"],
        "rate_trim": stats.get("rate_trim"),
        "rate_overhead_bpp": stats.get("rate_overhead_bpp"),
        "latent_bpp_mip0": stats.get("latent_bpp_mip0"),
        "file_bpp_chain": stats.get("file_bpp_chain"),
        "bppc": stats.get("bppc"),
        "latent_stored_bytes": stats.get("latent_stored_bytes"),
        "latent_plain_bytes": stats.get("latent_plain_bytes"),
        "flags_coder": stats.get("latent_coder"),
        "checkpoint": stats.get("checkpoint"),
        "steps_ran": stats.get("steps_ran"),
        "psnr_u8_init": stats.get("psnr_u8_init"),
        "psnr_u8_trained": stats.get("psnr_u8_trained"),
        "psnr_u8_loop": stats.get("psnr_u8_loop"),
        "score_gap": bool(stats.get("score_gap")),
        "seed": stats.get("seed"),
        "grids": stats.get("grids"),
        "stopped_early": stats.get("stopped_early"),
        "best_val_psnr": stats.get("best_val_psnr"),
        "precision": stats.get("precision", precision),
        "steps": job["steps"],
        "size": job["size"],
        "seconds_train": stats.get("seconds_train"),
        "steps_per_s": stats.get("steps_per_s"),
        "shard": shard_label,
        "worker_pid": os.getpid(),
        "batch_pixels": int(stats.get("batch_pixels", job.get("batch_pixels", 0))),
        "lr_batch_ref": int(job.get("lr_batch_ref", 0)),
        "batch_mip": str(stats.get("batch_mip", job.get("batch_mip", "flat"))),
        "target_bpp": float(job["bpp"]),
        "latent_rate_scale": float(stats.get("latent_rate_scale", job.get("latent_rate_scale", 1.0))),
        "min_latent_bits": int(job.get("min_latent_bits", 0)),
        "compiled": stats.get("compiled"),
        "compile_mode": stats.get("compile_mode"),
        "cuda_graph": stats.get("cuda_graph"),
        "halving": stats.get("halving"),
        "device": stats.get("device"),
    }
    reason = nonfinite_score(row)
    if reason:
        raise RuntimeError(reason)
    ntc_path.write_bytes(blob)
    trained = row.get("psnr_u8_trained")
    init = row.get("psnr_u8_init")
    loop = row.get("psnr_u8_loop")
    pixels = int(job["size"]) * int(job["size"])
    effective = int(row.get("batch_pixels") or 0)
    if score_gap_blocks_collapse(loop, trained, init):
        # The checkpoint stays. Marking this not done made the queue train
        # the same job again, and a finished checkpoint was then deleted.
        row["done"] = True
        row["collapsed"] = False
        row["score_gap"] = True
        row["rate_attempts"] = int(job.get("_rate_attempts") or 0) + 1
        row["error"] = (
            f"loop PSNR {float(loop):.2f} and file PSNR {float(trained):.2f} disagree; "
            f"init {float(init):.2f}; checkpoint kept"
        )
        print("score gap", job["name"], row["error"], flush=True)
        row = _restore_if_better(job, row, blob, dest, ntc_path, ckpt)
        return row
    retry_batch = dense_retry_batch(trained, init, effective, pixels, bool(job.get("_fallback")))
    if retry_batch is not None:
        print(
            f"near init {job['name']} trained {float(trained):.2f} init {float(init):.2f}, "
            f"retry flat {retry_batch}",
            flush=True,
        )
        if ckpt.is_file():
            ckpt.unlink()
        if ntc_path.is_file():
            ntc_path.unlink()
        if dest.is_file():
            dest.unlink()
        fallback = dict(job)
        fallback["batch_pixels"] = retry_batch
        fallback["batch_mip"] = "flat"
        fallback["_fallback"] = True
        return _run_claimed(fallback, out, precision, shard_label, dest, ntc_path, ckpt, stale_s)
    near_init = below_init(trained, init)
    if near_init:
        row["done"] = False
        row["collapsed"] = True
        row["error"] = collapse_reason(float(trained), float(init), effective, pixels)
        # The json is set aside. The checkpoint must not be resumed later.
        if ckpt.is_file():
            ckpt.unlink()
        print("collapsed", job["name"], row["error"], flush=True)
        row = _restore_if_better(job, row, blob, dest, ntc_path, ckpt)
        _print_finish(job, row, wall, shard_label)
        return row
    if row.get("done") and not job.get("_rd_retry"):
        lower = cheaper_better(out, job, float(row["psnr_u8"]))
        if lower is not None:
            from ntc.grids import grow_grids

            scale = float(job.get("latent_rate_scale") or 1.0)
            target = float(job["bpp"])
            if str(job.get("rate") or "") == "latent" and 0 < scale < 1:
                target = target / scale
            grown = grow_grids(
                [tuple(g) for g in lower["grids"]],
                int(job["size"]),
                int(job["size"]),
                target,
            )
            trained_grids = [tuple(g) for g in (row.get("grids") or [])]
            if (
                grown
                and [tuple(g) for g in grown] != trained_grids
                and grown_reaches(grown, int(job["size"]), target)
            ):
                print(
                    f"rd {job['name']} {float(row['psnr_u8']):.2f} under {lower['name']} "
                    f"{lower['psnr']:.2f}, retry grown {grown}",
                    flush=True,
                )
                kept = dict(row)
                kept["done"] = True
                kept["rate_attempts"] = int(job.get("_rate_attempts") or 0) + 1
                if rate_over_label(kept):
                    kept["rate_miss"] = True
                    kept["bpp_ratio"] = _bpp_ratio(kept)
                bundle = {"row": kept, "blob": blob, "ckpt": _ckpt_bytes(ckpt)}
                _write_attempt(bundle, dest, ntc_path, ckpt)
                fallback = dict(job)
                fallback["grids"] = [list(g) for g in grown]
                fallback["_rd_retry"] = True
                fallback["_replace_done"] = True
                fallback["_neighbor_applied"] = True
                fallback["_best_bundle"] = bundle
                fallback["_suppress_finish"] = True
                fallback["_rate_attempts"] = int(kept["rate_attempts"])
                prior = out / f"{lower['name']}.pt"
                if prior.is_file() and lower.get("checkpoint") == "trained":
                    fallback["warm_start"] = str(prior)
                return _run_claimed(
                    fallback, out, precision, shard_label, dest, ntc_path, ckpt, stale_s
                )
    if row.get("done"):
        row = _finish_rate(
            job, row, blob, out, precision, shard_label, dest, ntc_path, ckpt, stale_s
        )
    else:
        dest.write_text(json.dumps(row, indent=2))
    if not job.get("_suppress_finish"):
        _print_finish(job, row, wall, shard_label)
    return row


def _finish_rate(
    job: dict,
    row: dict,
    blob: bytes,
    out: Path,
    precision: str,
    shard_label: str,
    dest: Path,
    ntc_path: Path,
    ckpt: Path,
    stale_s: float,
) -> dict:
    """Keep the closer file. Retrain once on a smaller grid, then stop."""
    prev = job.get("_best_bundle")
    prev_row = prev.get("row") if isinstance(prev, dict) else None
    row = dict(row)
    row["rate_attempts"] = max(int(job.get("_rate_attempts") or 0) + 1, int(row.get("rate_attempts") or 0))
    chosen, retry = settle_rate_attempt(prev_row, row)
    if isinstance(prev_row, dict) and _attempt_rank(row) >= _attempt_rank(prev_row):
        bundle = {"row": chosen, "blob": prev.get("blob"), "ckpt": prev.get("ckpt")}
    else:
        bundle = {"row": chosen, "blob": blob, "ckpt": _ckpt_bytes(ckpt)}
    _write_attempt(bundle, dest, ntc_path, ckpt)
    if chosen.get("done"):
        refresh_rate_hull(out, str(job["name"]))
    if not retry or job.get("_rate_retry_done"):
        return chosen
    tighter = tighter_latent_grids(
        chosen.get("grids") or row.get("grids"),
        int(job["size"]),
        float(job["bpp"]),
        float(chosen.get("total_bpp") or 0),
    )
    if not tighter:
        return chosen
    print(
        f"rate miss {job['name']} {float(chosen.get('total_bpp')):.2f} bpp "
        f"over label {float(job['bpp']):g}, retry {tighter}",
        flush=True,
    )
    nxt = dict(job)
    nxt["grids"] = tighter
    nxt["_rate_attempts"] = int(chosen["rate_attempts"])
    nxt["_rate_retry_done"] = True
    nxt["_replace_done"] = True
    nxt["_neighbor_applied"] = True
    nxt["_rd_retry"] = True
    nxt["_best_bundle"] = bundle
    nxt["_suppress_finish"] = True
    nxt.pop("warm_start", None)
    return _run_claimed(nxt, out, precision, shard_label, dest, ntc_path, ckpt, stale_s)


def _print_finish(job: dict, row: dict, wall: float, shard_label: str) -> None:
    shown = row.get("wall_s")
    if not isinstance(shown, (int, float)):
        shown = wall
    sps = row.get("steps_per_s")
    sps_txt = f"{sps:.2f}" if isinstance(sps, (int, float)) else "na"
    psnr = row.get("psnr_u8")
    psnr_txt = f"{float(psnr):.4f}" if isinstance(psnr, (int, float)) else "na"
    bpp = row.get("total_bpp")
    bpp_txt = f"{float(bpp):.4f}" if isinstance(bpp, (int, float)) else "na"
    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} finish {job['name']} wall {float(shown):.1f}s "
        f"worker {shard_label or '0/1'} pid {os.getpid()} "
        f"steps/s {sps_txt} steps_ran {row.get('steps_ran')} ckpt {row.get('checkpoint')} "
        f"psnr {psnr_txt} init {row.get('psnr_u8_init')} trained {row.get('psnr_u8_trained')} "
        f"total_bpp {bpp_txt} stopped {row.get('stopped_early')} "
        f"rate_miss {row.get('rate_miss')} attempts {row.get('rate_attempts')}",
        flush=True,
    )


def build_cloud_jobs(root: Path, profile: str = "cloud5090") -> list[dict]:
    """2K latent ladder for 32 GB Blackwell. Does not include the 256 sweep.

    Held-out names are not jobs. Laptop ``build_jobs`` is a different list.
    ``cloudfast`` is the same ladder on ``CLOUD_FAST_TRAIN`` and
    ``CLOUD_FAST_BPP``.
    """
    if profile == "cloudfast":
        train, labels = CLOUD_FAST_TRAIN, CLOUD_FAST_BPP
    elif profile == "cloud5090":
        train, labels = CLOUD_TRAIN, CLOUD_BPP
    else:
        raise SystemExit(f"unknown cloud profile {profile}")
    return _cloud_ladder(root, train, labels, profile)


def _cloud_ladder(root: Path, train: tuple, labels: tuple, profile: str) -> list[dict]:
    missing = [name for name in train if name not in CLOUD_TRAIN]
    if missing:
        raise RuntimeError(f"not a cloud train material: {missing}")
    held = {f"{name}_2K" for name in CLOUD_HOLDOUT}
    budgets = ladder_step_budget(len(labels), CLOUD_FULL_STEPS)
    jobs = []
    for name in train:
        folder = f"{name}_2K"
        if folder in held:
            raise RuntimeError(f"train name is held out: {name}")
        for steps, bpp in zip(budgets, labels, strict=True):
            label = int(bpp) if float(bpp).is_integer() else bpp
            jobs.append({
                "name": f"cloud_{folder}_b{label}",
                "material": str(root / folder),
                "size": 2048,
                "steps": int(steps),
                "full_steps": CLOUD_FULL_STEPS,
                "full_warmup": 800,
                "bpp": float(bpp),
                "seed": 0,
                "warmup": 800 if int(steps) >= CLOUD_FULL_STEPS else 400,
                "schedule": "cosine",
                "eval_every": 500,
                "early_stop": 16,
                "batch_pixels": CLOUD_BATCH,
                "mip_levels": 12,
                "adaptive_pool": 1,
                "grids": None,
                "rate": "latent",
                "seed_pool": 3,
                "halve_steps": 500,
                "lr_batch_ref": 0,
                "min_latent_bits": 4,
                "batch_mip": "flat",
                "latent_rate_scale": 0.0,
                "entropy_rate": True,
                "quant_schedule": True,
                "quant_freeze_at": 0.90,
                "profile": profile,
            })
    return jobs


def build_jobs(root: Path, batch_pixels: int | None, profile: str = "laptop") -> list[dict]:
    """Laptop sweep when ``profile`` is ``laptop``. ``cloud5090`` is separate.

    ``batch_pixels`` overrides 256 jobs only. It does not change the cloud list.
    """
    if is_cloud_profile(profile):
        return build_cloud_jobs(root, profile)
    if profile != "laptop":
        raise SystemExit(f"unknown profile {profile}")
    metal = root / "metal_plate"
    paving = root / "PavingStones070"

    def base(name, material, steps, bpp, seed, **kw):
        row = {
            "name": name,
            "material": str(material),
            "size": kw.pop("size", 256),
            "steps": steps,
            "bpp": bpp,
            "seed": seed,
            "warmup": kw.pop("warmup", 200),
            "schedule": kw.pop("schedule", "cosine"),
            "eval_every": kw.pop("eval_every", 500),
            "early_stop": kw.pop("early_stop", 8),
            "batch_pixels": kw.pop("batch_pixels", 0),
            "mip_levels": kw.pop("mip_levels", 4),
            "adaptive_pool": kw.pop("adaptive_pool", 1),
            "grids": kw.pop("grids", G16),
            "rate": kw.pop("rate", "payload"),
            "seed_pool": kw.pop("seed_pool", 1),
            "halve_steps": kw.pop("halve_steps", 0),
            "lr_batch_ref": 0,
            "min_latent_bits": kw.pop("min_latent_bits", 0),
            "batch_mip": kw.pop("batch_mip", "flat"),
            "latent_rate_scale": kw.pop("latent_rate_scale", 1.0),
        }
        row.update(kw)
        if batch_pixels is not None and row["size"] == 256 and int(row["batch_pixels"]) == 0:
            row["batch_pixels"] = int(batch_pixels)
            row["lr_batch_ref"] = 256 * 256
        return row

    jobs = [
        base("metal_g16_s20000", metal, 20000, 16, 0, grids=G16, seed_pool=8, halve_steps=250),
        base("paving_g16_s20000", paving, 20000, 16, 0, grids=G16, seed_pool=8, halve_steps=250),
    ]
    for bpp in (4, 6, 8, 10, 12):
        jobs.append(base(f"metal_b{bpp}_s10000", metal, 10000, bpp, 0, grids=None, adaptive_pool=4, seed_pool=4, halve_steps=250))
        jobs.append(base(f"paving_b{bpp}_s10000", paving, 10000, bpp, 0, grids=None, adaptive_pool=4, seed_pool=4, halve_steps=250))
    jobs.append(base("metal_g16_step_s10000", metal, 10000, 16, 3, schedule="step"))
    for name in ("wood_floor", "denim_fabric", "aerial_rocks_01", "plastered_wall"):
        jobs.append(base(f"{name}_b16_s10000", root / name, 10000, 16, 0, grids=None, adaptive_pool=4, seed_pool=4, halve_steps=250))
    jobs.append(base("wood_floor_1k_b16_s4000", root / "wood_floor", 4000, 16, 0, grids=None, size=1024, adaptive_pool=4, seed_pool=2, halve_steps=250))
    jobs.append(base("wood_floor_2k_b16_s1000", root / "wood_floor_2k", 1000, 16, 0, grids=None, size=2048, batch_pixels=1024 * 1024, adaptive_pool=2, seed_pool=2, halve_steps=250))
    jobs.append(base("aerial_rocks_2k_b16_s1000", root / "aerial_rocks_01_2k", 1000, 16, 0, grids=None, size=2048, batch_pixels=1024 * 1024, adaptive_pool=2, seed_pool=2, halve_steps=250))
    for name in ("MetalPlates013_2K", "PavingStones070_2K", "Tiles135D_2K"):
        budgets = ladder_step_budget(5, 4000)
        for steps, bpp in zip(budgets, (2.5, 5, 10, 15, 20), strict=True):
            jobs.append(base(
                f"sdk_{name}_b{bpp}",
                root / name,
                steps,
                bpp,
                0,
                size=2048,
                batch_pixels=1024 * 1024,
                mip_levels=12,
                rate="latent",
                adaptive_pool=1,
                grids=None,
                warmup=400,
                schedule="cosine",
                eval_every=500,
                early_stop=6,
            ))
    # Half-resolution 2048 jobs stay on a 256K area batch. A latent whose
    # grid is as large as the image is raised to a flat quarter-frame batch
    # inside the trainer. 256K area samples left those 2048 grids near the init.
    for row in jobs:
        if int(row["size"]) >= 2048 and int(row.get("batch_pixels") or 0) >= 1024 * 1024:
            row["batch_pixels"] = 256 * 1024
            row["batch_mip"] = "area"
        if row.get("rate") == "latent":
            # Drop 2-bit ties. Scale 0 measures stored/plain on this
            # material's PCA codes when the job is claimed. A process that
            # already built this list with 0.5 keeps that list until it exits.
            row["min_latent_bits"] = 4
            row["latent_rate_scale"] = 0.0
            row["entropy_rate"] = True
            row["quant_schedule"] = True
    return jobs


def _read_precision(path: Path, identity: dict) -> str | None:
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not precision_matches(report, identity):
        return None
    return str(report["equal_wall_choice"])


def _precision_material(materials: Path) -> Path | None:
    """Manifest folder for the bf16 versus fp32 probe.

    Laptop trees ship ``metal_plate``. The cloud download does not, and the
    probe was printing ``missing .../metal_plate`` and staying on fp32.
    The first cloud train folder that has a manifest is the stand-in.
    """
    preferred = materials / "metal_plate"
    if (preferred / "manifest.json").is_file():
        return preferred
    for name in CLOUD_TRAIN:
        folder = materials / f"{name}_2K"
        if (folder / "manifest.json").is_file():
            return folder
    return None


def _ensure_precision(args: argparse.Namespace) -> str:
    path = args.out / "precision.json"
    identity = precision_identity()
    choice = _read_precision(path, identity)
    if choice is not None:
        print("precision", choice, "from", path, identity["device_name"], flush=True)
        return choice
    if path.is_file():
        print(
            "precision re-probe",
            identity["backend"],
            identity["device_name"],
            "file",
            path,
            flush=True,
        )
    metal = _precision_material(args.materials)
    if metal is None:
        print("missing", args.materials / "metal_plate")
        return "fp32"
    images, _sems, _info = load_material(metal, 256)
    probe_cfg = TrainConfig(
        steps=args.probe_steps,
        grids=G16,
        hidden=(32,),
        seed=3,
        adaptive_pool=1,
        seed_pool=1,
        with_ssim=False,
        mip_levels=4,
        mip_loss="footprint",
        device="auto",
    )
    # One lock so shard workers do not each train the probe.
    if not claim_job(args.out, "precision", stale_s=args.stale_lock_s):
        for _ in range(60):
            time.sleep(1.0)
            choice = _read_precision(path, identity)
            if choice is not None:
                return choice
        print("precision lock held, using fp32", flush=True)
        return "fp32"
    try:
        choice = _read_precision(path, identity)
        if choice is not None:
            return choice
        precision_report = choose_precision(images, _sems, probe_cfg, args.probe_steps)
        precision_report["device_name"] = identity["device_name"]
        precision_report["backend"] = identity["backend"]
        path.write_text(json.dumps(precision_report, indent=2))
        precision = precision_report["equal_wall_choice"]
        print(
            "precision",
            precision,
            identity["backend"],
            identity["device_name"],
            "historical",
            precision_report["historical_equal_steps_choice"],
            flush=True,
        )
        return precision
    finally:
        release_job(args.out, "precision")


def worker_env(index: int, base: dict | None = None, root: str | None = None, max_gpu: bool = False) -> dict:
    """Environment for chain worker ``index``: private compile caches.

    Workers that share one Inductor cache and its compile pool deadlock, so
    each gets its own directory and compiles in its own process.
    """
    env = dict(os.environ if base is None else base)
    top = root or env.get("NTC_ROOT") or os.path.join(tempfile.gettempdir(), "ntc")
    cache = os.path.join(top, ".cache", f"worker_{int(index)}")
    env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache, "inductor")
    env["TRITON_CACHE_DIR"] = os.path.join(cache, "triton")
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    env["NTC_SHARD"] = f"worker_{int(index)}"
    env[_DENSE_LOCK_ENV] = "0"
    if max_gpu:
        env["NTC_MAX_GPU"] = "1"
    return env


def material_chains(jobs: list[dict]) -> list[list[dict]]:
    """One chain per material, rates ascending so each label warm-starts.

    Chains keep the order of first appearance in ``jobs``.
    """
    chains: dict[str, list[dict]] = {}
    for job in jobs:
        chains.setdefault(str(job["material"]), []).append(job)
    return [sorted(chain, key=lambda job: float(job.get("bpp") or 0)) for chain in chains.values()]


def assign_chains(jobs: list[dict], workers: int) -> list[list[list[dict]]]:
    """Static plan: chain ``i`` goes to worker ``i % workers``.

    The live queue claims chains through lock files instead, so a worker
    that finishes early takes the next one. This is the dry-run view.
    """
    if workers < 1:
        raise SystemExit("workers must be >= 1")
    plan: list[list[list[dict]]] = [[] for _ in range(workers)]
    for index, chain in enumerate(material_chains(jobs)):
        plan[index % workers].append(chain)
    return plan


def _chain_lock(chain: list[dict]) -> str:
    return "chain_" + Path(str(chain[0]["material"])).name


def _chain_loop(
    jobs: list[dict],
    out: Path,
    precision: str,
    stale_s: float,
    label: str,
    backfill: list[dict] | None = None,
) -> None:
    """Claim one whole material chain at a time and train it in rate order.

    ``backfill`` chains are claimed only after every chain in ``jobs`` is
    taken, so the main set always starts first.
    """
    chains = material_chains(jobs)
    if backfill:
        chains = chains + material_chains(backfill)
    main_count = len(material_chains(jobs))
    for index, chain in enumerate(chains):
        if index == main_count:
            print("main chains claimed; backfill", label, flush=True)
        lock = _chain_lock(chain)
        if not claim_job(out, lock, stale_s):
            continue
        try:
            print("chain", lock, " ".join(job["name"] for job in chain), flush=True)
            for job in chain:
                try:
                    _run(job, out, precision, label, stale_s)
                    _force_terminal(out, job)
                except Exception as exc:
                    traceback.print_exc()
                    dest = out / f"{job['name']}.json"
                    keep = False
                    if dest.is_file():
                        try:
                            keep = bool(json.loads(dest.read_text()).get("done"))
                        except (OSError, ValueError, json.JSONDecodeError):
                            keep = False
                    if not keep:
                        dest.write_text(json.dumps({"name": job["name"], "done": False, "error": str(exc)}, indent=2))
                    print("fail", job["name"], exc, flush=True)
        finally:
            release_job(out, lock)


DONE_MARKER = "ALL_DONE"
INCOMPLETE_MARKER = "INCOMPLETE"


def planned_jobs(args: argparse.Namespace) -> list[dict]:
    """Every label this run means to train: the profile, plus backfill if on."""
    jobs = build_jobs(args.materials, args.batch_pixels, args.profile)
    if getattr(args, "backfill", False) and args.profile == "cloudfast":
        jobs = jobs + cloud_backfill_jobs(args.materials)
    return jobs


def completion_summary(out: Path, jobs: list[dict]) -> dict:
    """Per-label outcome from the result jsons in ``out``.

    A label is settled when its json is done, collapsed, or records an
    error. Missing or unreadable jsons are pending.
    """
    rows, failed, pending = [], [], []
    for job in jobs:
        name = str(job["name"])
        path = out / f"{name}.json"
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            pending.append(name)
            continue
        if row.get("done"):
            rows.append({
                "name": name,
                "psnr_u8": row.get("psnr_u8"),
                "total_bpp": row.get("total_bpp"),
                "target_bpp": row.get("target_bpp", job.get("bpp")),
                "wall": row.get("wall", row.get("wall_s")),
                "checkpoint": row.get("checkpoint"),
            })
        elif row.get("error") or row.get("collapsed"):
            failed.append({"name": name, "error": row.get("error"), "collapsed": bool(row.get("collapsed"))})
        else:
            pending.append(name)
    walls = [float(r["wall"]) for r in rows if isinstance(r.get("wall"), (int, float))]
    over = [r["name"] for r in rows if isinstance(r.get("total_bpp"), (int, float))
            and isinstance(r.get("target_bpp"), (int, float)) and r["total_bpp"] > r["target_bpp"]]
    return {
        "planned": len(jobs),
        "done": len(rows),
        "failed": failed,
        "pending": pending,
        "over_label": over,
        "wall_sum_s": sum(walls),
        "rows": rows,
        "written": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }


def write_completion(out: Path, jobs: list[dict]) -> bool:
    """Write ``ALL_DONE`` when every planned label is settled, else ``INCOMPLETE``.

    Returns True for ``ALL_DONE``. The other marker is removed so a reader
    sees one state.
    """
    summary = completion_summary(out, jobs)
    finished = not summary["pending"]
    name, other = (DONE_MARKER, INCOMPLETE_MARKER) if finished else (INCOMPLETE_MARKER, DONE_MARKER)
    (out / name).write_text(json.dumps(summary, indent=2))
    stale = out / other
    if stale.is_file():
        stale.unlink()
    print(
        f"{name} {summary['done']}/{summary['planned']} done, {len(summary['failed'])} failed, "
        f"{len(summary['pending'])} pending -> {out / name}",
        flush=True,
    )
    return finished


def _gpu_total_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _spawn_workers(args: argparse.Namespace) -> int:
    """N processes share the lock pool. They do not get a fixed ``--shard``.

    With ``args.chains`` each process claims whole material chains, gets its
    own compile caches, skips the dense GPU lock, and starts
    ``args.stagger_s`` after the previous one.
    """
    script = str(Path(__file__).resolve())
    chains = bool(getattr(args, "chains", False))
    stagger = float(getattr(args, "stagger_s", 0.0) or 0.0)
    procs = []
    for i in range(args.workers):
        cmd = [
            sys.executable,
            script,
            "--out", str(args.out),
            "--materials", str(args.materials),
            "--probe-steps", str(args.probe_steps),
            "--stale-lock-s", str(args.stale_lock_s),
            "--chain-queue" if chains else "--queue",
            "--profile", args.profile,
        ]
        if chains and getattr(args, "max_gpu", False):
            cmd.append("--max-gpu")
        if getattr(args, "backfill", False):
            cmd.append("--backfill")
        if args.batch_pixels is not None:
            cmd.extend(["--batch-pixels", str(args.batch_pixels)])
        if not args.pack:
            cmd.append("--no-pack")
        if args.dry_run:
            cmd.append("--dry-run")
        if chains:
            if i and stagger > 0:
                time.sleep(stagger)
            env = worker_env(i, max_gpu=bool(getattr(args, "max_gpu", False)))
            os.makedirs(env["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
            os.makedirs(env["TRITON_CACHE_DIR"], exist_ok=True)
            procs.append(subprocess.Popen(cmd, env=env))
        else:
            procs.append(subprocess.Popen(cmd))
    rc = 0
    for proc in procs:
        code = proc.wait()
        if code:
            rc = code
    return rc


def work_units(jobs: list[dict], pack: bool) -> list[list[dict]]:
    """One unit per job, or one unit per pack group when ``pack`` is set."""
    if not pack:
        return [[job] for job in jobs]
    groups: dict[tuple, list[dict]] = {}
    units: list[list[dict]] = []
    for job in jobs:
        key = _pack_key(job)
        if key is None:
            units.append([job])
            continue
        bucket = groups.get(key)
        if bucket is None:
            bucket = []
            groups[key] = bucket
            units.append(bucket)
        bucket.append(job)
    return units


def claim_next(jobs: list[dict], out: Path, stale_s: float, pack: bool, skip: set[str] | None = None) -> list[dict]:
    """Claim the heaviest unlocked unit. A live lock is left for its owner.

    Returns the jobs this process now holds. Empty when every remaining job
    is finished, already erred with a missing material, or locked by a live pid.
    """
    blocked = skip or set()
    units = work_units(jobs, pack)
    # Heaviest first. Equal cost keeps the lower bitrate first so a higher
    # rate can warm-start from a finished cheaper file instead of landing
    # under it.
    units.sort(
        key=lambda unit: (
            sum(job_cost(job) for job in unit),
            -min(float(job.get("bpp") or 0) for job in unit),
        ),
        reverse=True,
    )
    for unit in units:
        claimed: list[dict] = []
        try:
            for job in unit:
                if job["name"] in blocked or _job_blocked(out, job):
                    continue
                if not claim_job(out, job["name"], stale_s):
                    continue
                if _job_blocked(out, job):
                    release_job(out, job["name"])
                    continue
                claimed.append(job)
        except Exception:
            for job in claimed:
                release_job(out, job["name"])
            raise
        if claimed:
            return claimed
    return []


def _queue_loop(jobs: list[dict], out: Path, precision: str, stale_s: float, pack: bool) -> None:
    """Steal from the lock pool until nothing is left to claim."""
    failed: set[str] = set()
    label = "queue"
    while True:
        claimed = claim_next(jobs, out, stale_s, pack, skip=failed)
        if not claimed:
            return
        try:
            if len(claimed) == 1:
                job = claimed[0]
                _run_claimed(
                    job,
                    out,
                    precision,
                    label,
                    out / f"{job['name']}.json",
                    out / f"{job['name']}.ntc",
                    out / f"{job['name']}.pt",
                )
            else:
                _train_claimed_pack(claimed, out, precision, label)
            for job in claimed:
                _force_terminal(out, job)
                dest = out / f"{job['name']}.json"
                settled = False
                if dest.is_file():
                    try:
                        settled = _result_terminal(
                            json.loads(dest.read_text()), Path(job["material"])
                        )
                    except (OSError, ValueError, json.JSONDecodeError):
                        settled = False
                if not settled:
                    failed.add(job["name"])
        except Exception as exc:
            traceback.print_exc()
            for job in claimed:
                failed.add(job["name"])
                dest = out / f"{job['name']}.json"
                keep = False
                if dest.is_file():
                    try:
                        keep = bool(json.loads(dest.read_text()).get("done"))
                    except (OSError, ValueError, json.JSONDecodeError):
                        keep = False
                if not keep:
                    dest.write_text(json.dumps({"name": job["name"], "done": False, "error": str(exc)}, indent=2))
            print("fail", " ".join(job["name"] for job in claimed), exc, flush=True)
        finally:
            for job in claimed:
                release_job(out, job["name"])


def _pack_key(job: dict) -> tuple | None:
    grids = job.get("grids")
    if not grids:
        return None
    return (
        int(job["size"]),
        int(job["steps"]),
        str(job["schedule"]),
        int(job["mip_levels"]),
        int(job["seed_pool"]),
        int(job["warmup"]),
        tuple(tuple(int(x) for x in g) for g in grids),
    )


def _train_claimed_pack(claimed: list[dict], out: Path, precision: str, shard_label: str) -> None:
    """Train jobs whose locks this process already holds. Does not release them."""
    from ntc.train_fast import encode_bank

    buckets: dict[tuple, list] = {}
    for job in claimed:
        mat = Path(job["material"])
        if not (mat / "manifest.json").is_file():
            row = {"name": job["name"], "done": False, "error": f"missing {mat}"}
            (out / f"{job['name']}.json").write_text(json.dumps(row, indent=2))
            print("missing", job["name"], mat, flush=True)
            continue
        images, sems, _info = load_material(mat, int(job["size"]))
        buckets.setdefault((images.shape[0], tuple(sems)), []).append((job, images, sems))
    for (_channels, sems), items in buckets.items():
        if len(items) == 1:
            job = items[0][0]
            _run_claimed(
                job,
                out,
                precision,
                shard_label,
                out / f"{job['name']}.json",
                out / f"{job['name']}.ntc",
                out / f"{job['name']}.pt",
            )
            continue
        first = items[0][0]
        cfg = TrainConfig(
            steps=int(first["steps"]),
            target_bpp=float(first["bpp"]),
            grids=[tuple(g) for g in first["grids"]],
            seed=int(first["seed"]),
            adaptive_pool=1,
            seed_pool=int(first["seed_pool"]),
            latent_deflate=True,
            latent_coder="best",
            rate=str(first.get("rate", "payload")),
            mip_levels=int(first["mip_levels"]),
            mip_loss="footprint",
            warmup_steps=int(first["warmup"]),
            schedule=str(first["schedule"]),
            precision=precision,
            eval_every=0,
            early_stop=0,
            batch_pixels=int(first.get("batch_pixels", 0)),
            lr_batch_ref=int(first.get("lr_batch_ref", 0)),
            log_every=100,
            job_name=str(first["name"]),
            device="auto",
            with_ssim=False,
            fast=True,
            halve_steps=0,
            compile_step="auto",
            cuda_graph="auto",
        )
        packed = encode_bank(
            [image for _job, image, _sems in items],
            list(sems),
            cfg,
            seeds_per_image=[int(job["seed"]) for job, _image, _sems in items],
        )
        for (job, _image, _sems), (blob, stats) in zip(items, packed, strict=True):
            sps = stats.get("steps_per_s")
            row = {
                "name": job["name"],
                "done": True,
                "fast": True,
                "packed_jobs": stats.get("packed_jobs"),
                "wall_s": stats.get("seconds_train"),
                "psnr_u8": stats["metrics"]["psnr_u8"],
                "total_bpp": stats["total_bpp"],
                "latent_bpp_mip0": stats.get("latent_bpp_mip0"),
                "file_bpp_chain": stats.get("file_bpp_chain"),
                "bppc": stats.get("bppc"),
                "latent_stored_bytes": stats.get("latent_stored_bytes"),
                "latent_plain_bytes": stats.get("latent_plain_bytes"),
                "flags_coder": stats.get("latent_coder"),
                "checkpoint": stats.get("checkpoint"),
                "steps_ran": stats.get("steps_ran"),
                "psnr_u8_init": stats.get("psnr_u8_init"),
                "psnr_u8_trained": stats.get("psnr_u8_trained"),
                "seed": stats.get("seed"),
                "grids": stats.get("grids"),
                "stopped_early": False,
                "precision": precision,
                "steps": job["steps"],
                "size": job["size"],
                "seconds_train": stats.get("seconds_train"),
                "steps_per_s": sps,
                "compiled": stats.get("compiled"),
                "compile_mode": stats.get("compile_mode"),
                "cuda_graph": stats.get("cuda_graph"),
                "shard": shard_label,
                "worker_pid": os.getpid(),
                "batch_pixels": int(job.get("batch_pixels", 0)),
                "lr_batch_ref": int(job.get("lr_batch_ref", 0)),
            }
            reason = nonfinite_score(row)
            if reason:
                raise RuntimeError(reason)
            (out / f"{job['name']}.ntc").write_bytes(blob)
            (out / f"{job['name']}.json").write_text(json.dumps(row, indent=2))
            sps_txt = f"{sps:.2f}" if isinstance(sps, (int, float)) else "na"
            packed_wall = float(stats.get("seconds_train") or 0.0)
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} finish {job['name']} wall {packed_wall:.1f}s "
                f"worker {shard_label or '0/1'} pid {os.getpid()} "
                f"steps/s {sps_txt} packed {stats.get('packed_jobs')} "
                f"psnr {row['psnr_u8']:.4f} total_bpp {row['total_bpp']:.4f}",
                flush=True,
            )


def _run_group(group: list[dict], out: Path, precision: str, shard_label: str, stale_s: float) -> None:
    live = []
    for job in group:
        dest = out / f"{job['name']}.json"
        if dest.is_file() and json.loads(dest.read_text()).get("done"):
            print("skip", job["name"], flush=True)
            continue
        live.append(job)
    if len(live) <= 1:
        for job in live:
            _run(job, out, precision, shard_label, stale_s)
        return
    claimed: list[dict] = []
    try:
        for job in live:
            if claim_job(out, job["name"], stale_s):
                claimed.append(job)
            else:
                print("locked", job["name"], flush=True)
        if len(claimed) <= 1:
            for job in claimed:
                release_job(out, job["name"])
            held = claimed
            claimed = []
            for job in held:
                _run(job, out, precision, shard_label, stale_s)
            return
        _train_claimed_pack(claimed, out, precision, shard_label)
    finally:
        for job in claimed:
            release_job(out, job["name"])


def _run_packed(jobs: list[dict], out: Path, precision: str, shard_label: str, stale_s: float) -> None:
    groups: dict[tuple, list[dict]] = {}
    order: list[tuple | None] = []
    for job in jobs:
        key = _pack_key(job)
        if key is None:
            order.append(None)
            groups.setdefault(None, []).append(job)
        else:
            if key not in groups:
                order.append(key)
            groups.setdefault(key, []).append(job)
    seen = set()
    for job in jobs:
        key = _pack_key(job)
        token = ("single", job["name"]) if key is None else ("group", key)
        if token in seen:
            continue
        seen.add(token)
        if key is None:
            _run(job, out, precision, shard_label, stale_s)
        else:
            _run_group(groups[key], out, precision, shard_label, stale_s)


def main() -> None:
    install_fault_hooks()
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=("laptop", *CLOUD_PROFILES), default="laptop")
    p.add_argument("--out", type=Path, default=Path("results/train_max_v2"))
    p.add_argument("--materials", type=Path, default=Path("data/materials"))
    p.add_argument("--probe-steps", type=int, default=400)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--shard", default="", help="I/N. Split by pixels*steps, longest job first. Does not steal.")
    p.add_argument("--workers", type=int, default=0, help="Spawn N processes on one shared lock queue and wait. Dense 2K jobs stay at 1 process.")
    p.add_argument("--queue", action="store_true", help="Claim the next undone job from the lock pool until none remain.")
    p.add_argument(
        "--dense-workers",
        type=int,
        default=0,
        help="Spawn N processes that each train whole material chains side by side on one large GPU. "
        "No shared dense lock. Same as --workers N on a card with >= 80 GB.",
    )
    p.add_argument("--chain-queue", action="store_true", help="Internal: claim whole material chains until none remain.")
    p.add_argument("--stagger-s", type=float, default=20.0, help="Seconds between dense worker starts.")
    p.add_argument(
        "--max-gpu",
        action="store_true",
        default=os.environ.get("NTC_MAX_GPU") == "1",
        help="Cards >= 80 GB (or with --dense-workers): one chain worker per material by default "
        "and a max-autotune fallback if CUDA graphs fail. Also NTC_MAX_GPU=1.",
    )
    p.add_argument(
        "--backfill",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("NTC_BACKFILL") == "1",
        help="cloudfast chain workers that run out of main chains train the other cloud "
        "materials on the same ladder. Off by default. Also NTC_BACKFILL=1.",
    )
    p.add_argument(
        "--exit-when-done",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("NTC_EXIT_WHEN_DONE") == "1",
        help=f"With --workers or --dense-workers: once every planned label has a result, write "
        f"OUT/{DONE_MARKER} with a summary and exit 0 (and exit at once if it is already "
        f"done). Also NTC_EXIT_WHEN_DONE=1.",
    )
    p.add_argument("--batch-pixels", type=int, default=None, help="Override the pixel batch on 256 jobs. LR scales by batch/(256*256).")
    p.add_argument("--pack", action=argparse.BooleanOptionalAction, default=True, help="Stack jobs that share grid shape, size, and step count.")
    p.add_argument("--stale-lock-s", type=float, default=86400.0)
    args = p.parse_args()
    if is_cloud_profile(args.profile) and int(args.probe_steps) == 400:
        args.probe_steps = 800
    args.out.mkdir(parents=True, exist_ok=True)
    args.chains = False
    if args.dense_workers:
        if args.workers:
            raise SystemExit("pass --workers or --dense-workers, not both")
        if args.dense_workers < 1:
            raise SystemExit("--dense-workers must be >= 1")
        args.workers = int(args.dense_workers)
        args.chains = True
    if args.max_gpu and args.profile == "cloudfast" and not (args.workers or args.chain_queue or args.shard or args.queue):
        args.workers = len(CLOUD_FAST_TRAIN)
        args.chains = True
    if args.max_gpu and not (args.dry_run or args.chain_queue):
        total = _gpu_total_gb()
        if total < DENSE_WORKERS_MIN_VRAM_GB and not args.dense_workers:
            print(f"gpu {total:.0f} GB is under {DENSE_WORKERS_MIN_VRAM_GB:.0f} GB: --max-gpu is off", flush=True)
            args.max_gpu = False
            if args.chains and args.workers == len(CLOUD_FAST_TRAIN):
                args.chains = False
    if args.max_gpu:
        os.environ["NTC_MAX_GPU"] = "1"
    if args.chain_queue and (args.queue or args.shard or args.workers):
        raise SystemExit("--chain-queue runs alone")
    if args.workers and args.shard:
        raise SystemExit("pass --workers or --shard, not both")
    if args.workers and args.queue:
        raise SystemExit("pass --workers alone; it starts the queue")
    if args.queue and args.shard:
        raise SystemExit("pass --queue or --shard, not both")
    if args.workers:
        if args.workers < 1:
            raise SystemExit("--workers must be >= 1")
        jobs = build_jobs(args.materials, args.batch_pixels, args.profile)
        if not args.chains and int(args.workers) > 1 and not args.dry_run:
            total = _gpu_total_gb()
            if total >= DENSE_WORKERS_MIN_VRAM_GB:
                print(f"gpu {total:.0f} GB: dense workers run side by side", flush=True)
                args.chains = True
        planned = planned_workers(int(args.workers), jobs, args.out, dense_ok=args.chains)
        if planned != int(args.workers):
            print(
                f"dense 2K jobs run one at a time on this GPU "
                f"(workers {args.workers} -> {planned})",
                flush=True,
            )
            args.workers = planned
        if args.dry_run and args.chains:
            for index, worker in enumerate(assign_chains(jobs, int(args.workers))):
                for chain in worker:
                    print(
                        f"plan worker {index} {_chain_lock(chain)} "
                        + " ".join(job["name"] for job in chain),
                        flush=True,
                    )
            if args.backfill and args.profile == "cloudfast":
                for chain in material_chains(cloud_backfill_jobs(args.materials)):
                    print(f"plan backfill {_chain_lock(chain)}", flush=True)
            print(f"would spawn {args.workers} chain workers", flush=True)
            return
        if args.dry_run:
            jobs = build_jobs(args.materials, args.batch_pixels, args.profile)
            for job in sorted(jobs, key=job_cost, reverse=True):
                print(
                    f"plan queue cost {job_cost(job):.0f} {job['name']} batch {job['batch_pixels']} lr_ref {job['lr_batch_ref']}",
                    flush=True,
                )
            print(f"would spawn {args.workers} queue workers", flush=True)
            return
        if args.exit_when_done and write_completion(args.out, planned_jobs(args)):
            print("every planned label is done; nothing to train", flush=True)
            raise SystemExit(0)
        # Download before the probe so a missing 2K set stops the parent.
        if is_cloud_profile(args.profile):
            from bench.download_materials import cloud_materials

            cloud_materials(args.materials)
        else:
            ensure_sdk_materials(args.materials)
        device = pick_device("auto")
        (args.out / "device.json").write_text(json.dumps(device_report(device), indent=2))
        _ensure_precision(args)
        rc = _spawn_workers(args)
        if args.exit_when_done:
            if write_completion(args.out, planned_jobs(args)):
                raise SystemExit(0)
        raise SystemExit(rc)
    shard_i, shard_n = (None, None)
    shard_label = "queue" if args.queue else ("chain" if args.chain_queue else "")
    if args.shard:
        shard_i, shard_n = parse_shard(args.shard)
        shard_label = args.shard
    from ntc.train_fast import configure_inductor_process

    inherited = os.environ.get("NTC_SHARD") if args.chain_queue else None
    os.environ["NTC_SHARD"] = inherited or shard_label or str(os.getpid())
    configure_inductor_process(os.environ["NTC_SHARD"])
    if not args.dry_run:
        if is_cloud_profile(args.profile):
            from bench.download_materials import cloud_materials

            cloud_materials(args.materials)
        else:
            ensure_sdk_materials(args.materials)
    device = pick_device("auto")
    info = device_report(device)
    (args.out / "device.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({"shard": shard_label or "0/1", "pid": os.getpid(), **info}), flush=True)
    jobs = build_jobs(args.materials, args.batch_pixels, args.profile)
    if args.dry_run:
        if args.queue:
            for job in sorted(jobs, key=job_cost, reverse=True):
                print(
                    f"plan queue cost {job_cost(job):.0f} {job['name']} batch {job['batch_pixels']} lr_ref {job['lr_batch_ref']}",
                    flush=True,
                )
            return
        assignment = assign_shards(jobs, shard_n) if shard_n else [0] * len(jobs)
        width = shard_n or 1
        for index, job in enumerate(jobs):
            if shard_n is None or assignment[index] == shard_i:
                print(
                    f"plan {index} shard {assignment[index]}/{width} cost {job_cost(job):.0f} "
                    f"{job['name']} batch {job['batch_pixels']} lr_ref {job['lr_batch_ref']}",
                    flush=True,
                )
        return
    precision = _ensure_precision(args)
    if args.chain_queue:
        backfill = None
        if args.backfill and args.profile == "cloudfast":
            backfill = cloud_backfill_jobs(args.materials)
        _chain_loop(
            jobs,
            args.out,
            precision,
            args.stale_lock_s,
            os.environ.get("NTC_SHARD") or "chain",
            backfill,
        )
        print("wrote", args.out, "chain", flush=True)
        return
    if args.queue:
        _queue_loop(jobs, args.out, precision, args.stale_lock_s, args.pack)
        print("wrote", args.out, "queue", flush=True)
        return
    if shard_n:
        assignment = assign_shards(jobs, shard_n)
        selected = [job for index, job in enumerate(jobs) if assignment[index] == shard_i]
    else:
        selected = list(jobs)
    if args.pack:
        _run_packed(selected, args.out, precision, shard_label, args.stale_lock_s)
    else:
        for job in selected:
            _run(job, args.out, precision, shard_label, args.stale_lock_s)
    print("wrote", args.out, "shard", shard_label or "0/1", flush=True)


if __name__ == "__main__":
    main()
