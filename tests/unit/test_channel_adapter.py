import torch

from custom_sam_peft.models.sam3 import _build_channel_adapter


def test_rgb_builds_no_adapter():
    assert _build_channel_adapter(channels=3, channel_semantics="rgb") is None


def test_freeform_3ch_builds_learned_adapter_not_passthrough():
    adapter = _build_channel_adapter(channels=3, channel_semantics="freeform")
    assert adapter is not None
    assert isinstance(adapter, torch.nn.Conv2d)
    assert adapter.in_channels == 3 and adapter.out_channels == 3
    # average_broadcast init for N=3 => weight == 1/3 everywhere, bias == 0
    assert torch.allclose(adapter.weight, torch.full_like(adapter.weight, 1.0 / 3.0))
    assert torch.allclose(adapter.bias, torch.zeros_like(adapter.bias))
    assert adapter.weight.requires_grad and adapter.bias.requires_grad


def test_C1_average_broadcast_mean_of_stack():
    adapter = _build_channel_adapter(channels=4, channel_semantics="freeform")
    assert torch.allclose(adapter.weight, torch.full_like(adapter.weight, 1.0 / 4.0))
    assert torch.allclose(adapter.bias, torch.zeros_like(adapter.bias))
    x = torch.randn(2, 4, 5, 6)
    out = adapter(x)
    expected_mean = x.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    assert torch.allclose(out, expected_mean, atol=1e-5)


def test_C2_grayscale_triplication_identity():
    adapter = _build_channel_adapter(channels=1, channel_semantics="grayscale")
    x = torch.randn(2, 1, 5, 6)
    out = adapter(x)
    assert torch.allclose(out, torch.cat([x, x, x], dim=1), atol=1e-6)


def test_C3b_rgba_identity_passthrough_drops_alpha():
    adapter = _build_channel_adapter(channels=4, channel_semantics="rgba")
    w = adapter.weight  # (3,4,1,1)
    for o in range(3):
        assert torch.isclose(w[o, o, 0, 0], torch.tensor(1.0))
    assert torch.allclose(w[:, 3, 0, 0], torch.zeros(3))  # alpha column zero
    assert torch.allclose(adapter.bias, torch.zeros_like(adapter.bias))
    x = torch.randn(2, 4, 5, 6)
    out = adapter(x)
    assert torch.allclose(out, x[:, :3], atol=1e-6)  # first 3 (RGB) exactly, alpha dropped
