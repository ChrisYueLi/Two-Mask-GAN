import argparse

import torch
from torchsummary import summary

from runtime import build_generator, resolve_device


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda", help="torch device")
parser.add_argument("--num_channel", type=int, default=128, help="TSCNet channel width")
parser.add_argument("--mask_mode", type=str, default="add", choices=["add", "mul"], help="masking mode")
parser.add_argument("--module", type=str, default="conformer", choices=["conformer", "mamba"], help="sequence module")
args = parser.parse_args()

device = resolve_device(args.device)
model = build_generator(
    device,
    n_fft=400,
    num_channel=args.num_channel,
    mask_mode=args.mask_mode,
    module=args.module,
)
sample = torch.rand(1, 2, 201, 201, device=device)
real, imag = model(sample)
print([tuple(real.shape), tuple(imag.shape)])
summary(model, (2, 201, 201), batch_size=1)
