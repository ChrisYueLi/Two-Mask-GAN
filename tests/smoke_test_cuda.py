import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataloader import DemandDataset
from runtime import build_generator, load_generator_checkpoint, resolve_device


def main():
    device = resolve_device("cuda")
    fixture_root = REPO_ROOT / "tests" / "fixtures" / "audio"
    dataset = DemandDataset(str(fixture_root), cut_len=16000)
    print(f"fixture dataset size: {len(dataset)}")

    model = build_generator(device, num_channel=128, mask_mode="add", module="conformer")
    model.eval()
    sample = torch.randn(1, 2, 201, 201, device=device)
    with torch.no_grad():
        real, imag = model(sample)
    print(f"forward shapes: {tuple(real.shape)}, {tuple(imag.shape)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "generator.pt"
        torch.save(model.state_dict(), checkpoint_path)
        reloaded = build_generator(device, num_channel=128, mask_mode="add", module="conformer")
        load_generator_checkpoint(reloaded, checkpoint_path, device)
    print("checkpoint roundtrip ok")


if __name__ == "__main__":
    main()
