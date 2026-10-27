import torch

from ntc.quant import quantize_latent, quantize_weight


def test_latent_roundtrip_endpoints():
    scale = torch.tensor([0.5, 0.25]).view(2, 1, 1)
    # value = (q - 7.5) * scale for 4-bit, mid = 7.5
    bits = 4
    mid = 7.5
    codes = torch.tensor([0, 15], dtype=torch.float32).view(2, 1, 1)
    x = (codes - mid) * scale
    deq, q = quantize_latent(x, scale, bits)
    assert int(q[0, 0, 0]) == 0
    assert int(q[1, 0, 0]) == 15
    assert torch.allclose(deq, x, atol=1e-5)


def test_latent_ste_grad():
    scale = torch.ones(1, 1, 1)
    x = torch.tensor([[[0.2]]], requires_grad=True)
    deq, _ = quantize_latent(x, scale, 8)
    deq.sum().backward()
    assert x.grad is not None
    assert float(x.grad) == 1.0


def test_weight_group_int4_and_int8():
    torch.manual_seed(0)
    w = torch.randn(3, 8)
    for bits, gs in ((8, 0), (8, 4), (4, 4)):
        deq, codes, scales = quantize_weight(w, bits, gs)
        qmax = 127 if bits == 8 else 7
        assert int(codes.abs().max()) <= qmax
        # reconstruction from codes and scales matches deq
        if gs <= 0:
            rec = codes.float() * scales[:, :1]
        else:
            rec = torch.empty_like(w)
            for g in range(scales.shape[1]):
                sl = slice(g * gs, (g + 1) * gs)
                rec[:, sl] = codes[:, sl].float() * scales[:, g : g + 1]
        assert torch.allclose(rec, deq, atol=1e-4)
        # fp16 scale is exactly representable as fp16
        assert torch.equal(scales, scales.half().float())
