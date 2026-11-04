import numpy as np
import torch

from ntc.container import (
    ACT_RELU,
    ChannelRec,
    build_texture_set,
    parse_texture_set,
)
from ntc.decode_ref import (
    apply_layer,
    bilinear_codes,
    box_down,
    decode_mip,
    latent_mid,
)
from ntc.model import bilinear_codes_torch


def test_box_down_known_values():
    codes = np.array([[[0, 0, 4, 4], [0, 0, 4, 4], [8, 8, 12, 12], [8, 8, 12, 12]]], dtype=np.uint8)
    out = box_down(codes)
    assert out.shape == (1, 2, 2)
    np.testing.assert_array_equal(out[0], [[0, 4], [8, 12]])


def test_bilinear_matches_torch_and_wraps():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 256, size=(3, 5, 6), dtype=np.uint8)
    uv = np.array([[0.0, 0.0], [0.5, 0.5], [1.0 - 1e-6, 0.25], [0.1, 0.9]], np.float32)
    got = bilinear_codes(codes, uv)
    t = bilinear_codes_torch(torch.from_numpy(codes.astype(np.float32)), torch.from_numpy(uv))
    np.testing.assert_allclose(got, t.numpy(), atol=1e-5)


def test_decoder_matches_hand_mlp():
    # one 8-bit latent channel, identity-sized linear, no activation on the only layer
    codes = np.zeros((1, 4, 4), np.uint8)
    codes[0, 0, 0] = 255
    scale = np.array([1.0 / 127.5], np.float32)
    # weight copies the single feature to two outputs with scales 1
    w = np.array([[1], [2]], np.int16)
    blob = build_texture_set(
        width=4,
        height=4,
        num_mips=3,
        activation=ACT_RELU,
        pe_mode=0,
        pe_octaves=0,
        flags=0,
        weight_bits=8,
        group_size=0,
        channels=[ChannelRec(0), ChannelRec(1)],
        grids=[{"codes": codes, "bits": 8, "scale": scale}],
        layers=[
            {
                "codes": w,
                "bias": np.array([0.0, 0.1], np.float32),
                "scale": np.array([[1.0], [1.0]], np.float32),
                "group_size": 0,
            }
        ],
    )
    ts = parse_texture_set(blob)
    img = decode_mip(ts, 0)
    assert img.shape == (4, 4, 2)
    # texel (0,0) center uv = (0.5/4, 0.5/4) = (0.125, 0.125)
    # grid coord = uv * 4 - 0.5 = 0.0 exactly, so the sample is code 255
    mid = float(latent_mid(8))
    value = (255.0 - mid) * float(scale[0])
    # scale is stored as fp16, so the reconstructed endpoint is not binary32-exact
    np.testing.assert_allclose(img[0, 0, 0], value, atol=2e-4)
    np.testing.assert_allclose(img[0, 0, 1], value * 2.0 + 0.1, atol=2e-3)


def test_apply_layer_group_scale():
    class L:
        weight = np.array([[1.0, 2.0, 3.0, 4.0]], np.float32)
        bias = np.array([0.5], np.float32)
        scale = np.array([[0.1, 0.2]], np.float32)
        group_size = 2

    x = np.array([[1.0, 0.0, 1.0, 1.0]], np.float32)
    y = apply_layer(x, L)
    # group0: (1*1 + 2*0) * 0.1 = 0.1
    # group1: (3*1 + 4*1) * 0.2 = 1.4
    # + bias 0.5
    np.testing.assert_allclose(y, [[2.0]], atol=1e-6)


def test_tiled_forward_matches_reference():
    import math

    import torch

    from ntc.encode import export_network
    from ntc.model import NtcNet

    torch.manual_seed(1)
    net = NtcNet(
        image_channels=2,
        grids=[(8, 8, 2, 4)],
        hidden=(4,),
        activation=1,
        pe_mode=0,
        pe_octaves=0,
        weight_bits=8,
        group_size=0,
        quant_proxy="ste",
        activation_requant=False,
        qat_latents=True,
        qat_weights=True,
        tile_size=4,
    )
    net.latents[0].data.uniform_(0, 15)
    net.tile_log_scales[0].data.fill_(math.log(0.08))
    net.tile_offsets[0].data.uniform_(-0.2, 0.2)
    net.eval()
    blob = export_network(net, [0, 1], 8, 8, 2)
    ts = parse_texture_set(blob)
    ref = decode_mip(ts, 0)
    ys = torch.arange(8)
    xs = torch.arange(8)
    yg, xg = torch.meshgrid(ys, xs, indexing="ij")
    uv = torch.stack(((xg.reshape(-1) + 0.5) / 8, (yg.reshape(-1) + 0.5) / 8), 1)
    with torch.no_grad():
        pred = net(uv, 0, 1).reshape(8, 8, 2).numpy()
    np.testing.assert_allclose(pred, ref, atol=2e-4)
