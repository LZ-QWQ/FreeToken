import pytest
import torch

from freetoken.kernel.triton import sampling
from freetoken.utils.arch import get_rocm_gfx_arch


def test_rocm_exact_sampling_uses_one_cta(monkeypatch):
    monkeypatch.setattr(sampling, "get_rocm_gfx_arch", lambda: "gfx1100")

    assert sampling._fused_plan(1, 248_320, "cuda") == (1, 248_320)


def test_cuda_exact_sampling_keeps_multi_cta_plan(monkeypatch):
    monkeypatch.setattr(sampling, "get_rocm_gfx_arch", lambda: None)
    monkeypatch.setattr(sampling, "_num_sm", lambda _device: 48)

    assert sampling._fused_plan(1, 248_320, "cuda") == (61, 4_071)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a GPU is required")
def test_gfx1100_exact_sampling_matches_torch():
    if get_rocm_gfx_arch() != "gfx1100":
        pytest.skip("this regression covers the gfx1100 single-CTA path")

    generator = torch.Generator(device="cuda").manual_seed(7)
    probs = torch.rand((1, 248_320), device="cuda", generator=generator)
    probs /= probs.sum(dim=-1, keepdim=True)

    topk = sampling.top_k_renorm_probs(probs, 20)
    values, indices = torch.topk(probs, 20, dim=-1)
    topk_ref = torch.zeros_like(probs).scatter(-1, indices, values)
    topk_ref /= topk_ref.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(topk, topk_ref)

    topp = sampling.top_p_renorm_probs(probs, 0.95)
    values, indices = torch.sort(probs, dim=-1, descending=True)
    kept = torch.where(values.cumsum(dim=-1) - values <= 0.95, values, 0.0)
    topp_ref = torch.zeros_like(probs).scatter(-1, indices, kept)
    topp_ref /= topp_ref.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(topp, topp_ref, rtol=1e-5, atol=3e-6)

    token = sampling.top_k_top_p_sampling_from_probs(
        probs, top_k=20, top_p=0.95, seed=7, offset=0
    )
    assert 0 <= token.item() < probs.size(-1)
