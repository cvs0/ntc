import numpy as np

from ntc.container import FLAG_ACT_INT8, parse_texture_set
from ntc.decode_ref import decode_mip
from ntc.encode import TrainConfig, encode_arrays


def test_act_int8_roundtrip_and_is_off_by_default():
    rng = np.random.default_rng(0)
    img = rng.random((3, 16, 16), dtype=np.float32)
    sems = [0, 1, 2]
    plain, _ = encode_arrays(img, sems, TrainConfig(steps=2, target_bpp=8, grids=[(8, 8, 4, 4)], hidden=[8], seed=1, log_every=10**9))
    base = parse_texture_set(plain)
    assert not base.act_int8
    assert base.act_scales == []
    blob, stats = encode_arrays(
        img,
        sems,
        TrainConfig(steps=3, target_bpp=8, grids=[(8, 8, 4, 4)], hidden=[8], seed=1, act_int8=True, log_every=10**9),
    )
    ts = parse_texture_set(blob)
    assert ts.flags & FLAG_ACT_INT8
    assert len(ts.act_scales) == 2
    assert all(s > 0 for s in ts.act_scales)
    dec = decode_mip(ts, 0)
    assert dec.shape == (16, 16, 3)
    assert np.isfinite(dec).all()
    assert stats["metrics"]["psnr_u8"] > 0
