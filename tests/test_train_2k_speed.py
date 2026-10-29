"""The 2K speed script stays out of the train_max_v2 queue."""

import json
from pathlib import Path

from bench.train_2k_speed import B10_GRID, DENSE_BATCH, main
from ntc.grids import select_grids


def test_b10_grid_matches_the_ranked_search():
    picked = select_grids(2048, 2048, 20.0, [32], 9, 0, 0, 8, 0, rate="latent", min_bits=4)[0]
    assert picked == B10_GRID
    assert DENSE_BATCH == 1024 * 1024


def test_speed_script_writes_outside_the_v2_queue(tmp_path: Path):
    out = tmp_path / "train_2k_speed.json"
    main(
        [
            "--size",
            "16",
            "--warmup",
            "1",
            "--timed-steps",
            "2",
            "--profile-steps",
            "1",
            "--modes",
            "eager,compile_default,hip_graph,triton_gather",
            "--out",
            str(out),
        ]
    )
    report = json.loads(out.read_text())
    names = [row["name"] for row in report["modes"]]
    assert names == ["eager", "compile_default", "hip_graph", "triton_gather"]
    eager = report["modes"][0]
    assert eager["error"] is None
    assert eager["steps_per_s"] > 0
    assert report["modes"][1]["error"]
    assert report["torch"]
    assert report["device"]["device"]
    assert len(report["eager_top_kernels"]) >= 1
    assert "train_max_v2" not in out.parts
    assert not (tmp_path / "locks").exists()
