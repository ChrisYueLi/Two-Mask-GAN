import argparse

import torch

from models import generator


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def resolve_device(requested="cuda"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def build_generator(
    device,
    n_fft=400,
    num_channel=128,
    mask_mode="add",
    module="conformer",
):
    model = generator.TSCNet(
        num_channel=num_channel,
        num_features=n_fft // 2 + 1,
        mask_mode=mask_mode,
        module=module,
    )
    return model.to(device)


def get_model_state_dict(model):
    module = model.module if hasattr(model, "module") else model
    return module.state_dict()


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_generator_checkpoint(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(_extract_state_dict(checkpoint))
    return checkpoint
