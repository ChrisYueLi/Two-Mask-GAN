import torch
import torch.utils.data
import torchaudio
import os
from utils import *
import random
from natsort import natsorted

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from torch.utils.data.distributed import DistributedSampler


class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000 * 3):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, "clean")
        self.noisy_dir = os.path.join(data_dir, "noisy")
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)

    def _repeat_to_min_len(self, wav, min_len):
        length = wav.numel()
        if length >= min_len:
            return wav
        units = min_len // length
        rem = min_len % length
        chunks = [wav] * units
        if rem:
            chunks.append(wav[:rem])
        return torch.cat(chunks, dim=-1)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            clean_ds = self._repeat_to_min_len(clean_ds, self.cut_len)
            noisy_ds = self._repeat_to_min_len(noisy_ds, self.cut_len)
            length = self.cut_len

        clean_segs = []
        noisy_segs = []
        if length >= 2 * self.cut_len:
            mid = length // 2
            max_start1 = mid - self.cut_len
            start1 = random.randint(0, max_start1)
            max_start2 = length - self.cut_len
            start2 = random.randint(mid, max_start2)
            clean_segs.append(clean_ds[start1 : start1 + self.cut_len])
            noisy_segs.append(noisy_ds[start1 : start1 + self.cut_len])
            clean_segs.append(clean_ds[start2 : start2 + self.cut_len])
            noisy_segs.append(noisy_ds[start2 : start2 + self.cut_len])
        else:
            max_start = length - self.cut_len
            start = random.randint(0, max_start)
            clean_segs.append(clean_ds[start : start + self.cut_len])
            noisy_segs.append(noisy_ds[start : start + self.cut_len])

        clean_batch = torch.stack(clean_segs, dim=0)
        noisy_batch = torch.stack(noisy_segs, dim=0)

        return clean_batch, noisy_batch, length


def _pair_collate(batch):
    clean_list = []
    noisy_list = []
    lengths = []
    for clean, noisy, length in batch:
        clean_list.append(clean)
        noisy_list.append(noisy)
        lengths.extend([length] * clean.size(0))
    clean = torch.cat(clean_list, dim=0)
    noisy = torch.cat(noisy_list, dim=0)
    lengths = torch.tensor(lengths)
    return clean, noisy, lengths


def load_data(ds_dir, batch_size, n_cpu, cut_len, val_split=0.1, seed=42, distributed=False):
    train_dir = os.path.join(ds_dir, "train")
    test_dir = os.path.join(ds_dir, "valid")

    has_split_dirs = os.path.isdir(train_dir) and os.path.isdir(test_dir)
    if has_split_dirs:
        train_ds = DemandDataset(train_dir, cut_len)
        test_ds = DemandDataset(test_dir, cut_len)
    else:
        train_full = DemandDataset(ds_dir, cut_len)
        dataset_len = len(train_full)
        if dataset_len < 2:
            raise ValueError("Dataset too small to split into train/test.")
        val_len = max(1, int(dataset_len * val_split))
        train_len = dataset_len - val_len
        if train_len < 1:
            val_len = dataset_len - 1
            train_len = 1
        generator = torch.Generator().manual_seed(seed)
        train_ds, test_ds = torch.utils.data.random_split(
            train_full, [train_len, val_len], generator=generator
        )

    train_dataset = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        collate_fn=_pair_collate,
        pin_memory=True,
        shuffle=not distributed,
        sampler=DistributedSampler(train_ds) if distributed else None,
        drop_last=True,
        num_workers=n_cpu,
    )
    test_dataset = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        collate_fn=_pair_collate,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(test_ds) if distributed else None,
        drop_last=False,
        num_workers=n_cpu,
    )

    return train_dataset, test_dataset
