import torch
import pytest

from runtime import build_generator, load_generator_checkpoint, resolve_device


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA smoke tests require a CUDA-capable test environment",
)


def test_resolve_device_cuda():
    device = resolve_device("cuda")

    assert device.type == "cuda"


def test_tscnet_cuda_forward_shape():
    device = resolve_device("cuda")
    model = build_generator(device, num_channel=128, mask_mode="add", module="conformer")
    model.eval()
    sample = torch.randn(1, 2, 201, 201, device=device)

    with torch.no_grad():
        real, imag = model(sample)

    assert tuple(real.shape) == (1, 1, 201, 201)
    assert tuple(imag.shape) == (1, 1, 201, 201)


def test_generator_checkpoint_roundtrip(tmp_path):
    device = resolve_device("cuda")
    model = build_generator(device, num_channel=128, mask_mode="add", module="conformer")
    checkpoint_path = tmp_path / "generator.pt"
    torch.save(model.state_dict(), checkpoint_path)

    reloaded = build_generator(device, num_channel=128, mask_mode="add", module="conformer")
    load_generator_checkpoint(reloaded, checkpoint_path, device)

    key = "dense_encoder1.conv_1.0.weight"
    assert torch.equal(model.state_dict()[key].cpu(), reloaded.state_dict()[key].cpu())
