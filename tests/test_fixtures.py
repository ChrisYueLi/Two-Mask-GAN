from pathlib import Path

import torch
import torchaudio

from data.dataloader import DemandDataset


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "audio"
EXPECTED_FILES = [f"{idx:05d}.wav" for idx in range(5)]


def test_fixture_audio_pairs_exist():
    clean_files = sorted(path.name for path in (FIXTURE_ROOT / "clean").glob("*.wav"))
    noisy_files = sorted(path.name for path in (FIXTURE_ROOT / "noisy").glob("*.wav"))

    assert clean_files == EXPECTED_FILES
    assert noisy_files == EXPECTED_FILES


def test_fixture_audio_pairs_have_matching_shape_and_rate():
    for filename in EXPECTED_FILES:
        clean, clean_sr = torchaudio.load(FIXTURE_ROOT / "clean" / filename)
        noisy, noisy_sr = torchaudio.load(FIXTURE_ROOT / "noisy" / filename)

        assert clean_sr == 16000
        assert noisy_sr == clean_sr
        assert clean.shape == noisy.shape
        assert clean.numel() > 0
        assert noisy.numel() > 0


def test_demand_dataset_reads_fixture_pairs():
    dataset = DemandDataset(str(FIXTURE_ROOT), cut_len=16000)

    clean, noisy, length = dataset[0]

    assert len(dataset) == 5
    assert clean.shape == noisy.shape
    assert clean.shape[-1] == 16000
    assert isinstance(length, int)
    assert torch.isfinite(clean).all()
    assert torch.isfinite(noisy).all()
