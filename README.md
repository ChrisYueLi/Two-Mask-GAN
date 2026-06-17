# Two-Mask-GAN

Experiment source code for the Two-Mask-GAN / CMGAN-style speech enhancement and post-filtering experiments used in the HRI technical-venue study.

This release is a CUDA-first, locally testable code package. It is verified against the project development environment:

- Python: `E:\conda_env\envs\dccrn\python.exe`
- PyTorch: `2.7.0+cu128`
- GPU: NVIDIA GeForce RTX 5070 Ti

CPU-only machines can still inspect the code and run some import/help checks, but full training and model smoke tests target CUDA.

## Repository Contents

- `src/train.py`: single-GPU CUDA training by default, with optional DDP via `--distributed`.
- `src/evaluate_folder.py`, `src/evaluate_file.py`, `src/evaluation.py`, `src/single_file_evaluation.py`, `src/streaming_input_evaluation.py`: enhancement and evaluation scripts.
- `src/models/`: generator, discriminator, Conformer, and local Mamba-style modules.
- `src/data/dataloader.py`: paired clean/noisy audio dataloader.
- `tests/`: local pytest checks and a CUDA smoke script.
- `tests/fixtures/audio/`: five paired clean/noisy `.wav` files used only for smoke tests.

Raw participant recordings, full training datasets, generated outputs, logs, checkpoints, and local analysis tables are intentionally excluded from version control.

## Installation

Install a PyTorch build matching your CUDA setup first, then install the remaining dependencies:

```bash
E:\conda_env\envs\dccrn\python.exe -m pip install -r requirements.txt
```

For the local smoke-test environment, the key packages are `pytest`, `pesq`, `pystoi`, and `jiwer` in addition to PyTorch and torchaudio.

## Local Verification

Run all local checks from the repository root:

```bash
E:\conda_env\envs\dccrn\python.exe -m compileall src tests
E:\conda_env\envs\dccrn\python.exe -c "import sys; sys.path.insert(0, 'src'); from models.generator import TSCNet; from data.dataloader import DemandDataset; print('ok')"
E:\conda_env\envs\dccrn\python.exe -m pytest -q
E:\conda_env\envs\dccrn\python.exe tests\smoke_test_cuda.py
```

The pytest suite checks fixture audio pairing, CUDA model forward shapes, checkpoint loading, and command-line help for the main entry points.

## Data Layout

Training expects paired clean and noisy `.wav` files. Use one of these layouts:

```text
dataset/
  train/
    clean/
    noisy/
  valid/
    clean/
    noisy/
```

or a single folder with `clean/` and `noisy/`; the dataloader can create an internal validation split. File names in `clean/` and `noisy/` must match.

## Training

Single-GPU CUDA training:

```bash
cd src
E:\conda_env\envs\dccrn\python.exe train.py --data_dir /path/to/dataset --save_model_dir ../checkpoints --log_dir ../logs --device cuda --num_channel 128 --mask_mode add --module conformer
```

Multi-GPU DDP is still available when the platform supports it:

```bash
cd src
E:\conda_env\envs\dccrn\python.exe train.py --data_dir /path/to/dataset --distributed --ddp_backend nccl
```

## Enhancement

Enhance a folder of noisy waveforms with a trained checkpoint:

```bash
cd src
E:\conda_env\envs\dccrn\python.exe evaluate_folder.py --model_path /path/to/checkpoint --test_dir /path/to/noisy_wavs --save_dir ../results/enhanced --streaming 1 --device cuda --num_channel 128 --mask_mode add --module conformer
```

`--streaming 1` uses chunked streaming-style inference; set `--streaming 0` for full-utterance inference.

## Metrics

Feature-level PESQ/STOI/SNR evaluation:

```bash
cd src
E:\conda_env\envs\dccrn\python.exe Feature_evaluation.py --enhanced_dir ../results/enhanced --clean_dir /path/to/clean_refs --output_csv ../results/evaluation_results.csv
```

ASR WER evaluation:

```bash
cd src
E:\conda_env\envs\dccrn\python.exe ASR_eval.py --asr_model /path/or/hf-model-name --data_dir /path/to/data --folder test --subfolder result --output_csv ../results/cmgan_result.csv
```

## Availability Notes

This public release includes code, lightweight documentation, and five paired audio fixtures for smoke testing. Checkpoints, raw participant audio, complete datasets, and generated experimental results should be distributed separately only when licensing, consent, ethics, and privacy constraints allow it.
