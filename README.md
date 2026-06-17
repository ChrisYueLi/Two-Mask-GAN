# Two-Mask-GAN

Experiment source code for the Two-Mask-GAN / CMGAN-style speech enhancement and post-filtering experiments used in the HRI technical-venue study.

## Repository Contents

This repository contains the shareable experiment code only:

- `src/train.py`: distributed training entry point.
- `src/evaluation.py`, `src/evaluate_folder.py`, `src/evaluate_file.py`, `src/single_file_evaluation.py`, `src/streaming_input_evaluation.py`: enhancement and evaluation scripts.
- `src/models/`: generator, discriminator, Conformer, and local Mamba-style modules.
- `src/data/dataloader.py`: paired clean/noisy audio dataloader.
- `src/tools/`: metric and TensorBoard helper utilities.

Raw participant recordings, generated audio outputs, logs, checkpoints, and local analysis tables are intentionally excluded from version control because they are large and may contain identifiable or non-shareable data.

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

or a single folder with `clean/` and `noisy/`; the dataloader can create an internal split.

File names in `clean/` and `noisy/` should match so that each noisy utterance has the corresponding clean reference.

## Installation

Create an environment with a PyTorch build matching your CUDA or CPU setup, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The exact PyTorch install command depends on the target platform and CUDA version. See the official PyTorch installation instructions for the appropriate wheel.

## Training

From the repository root:

```bash
cd src
python train.py --data_dir /path/to/dataset --save_model_dir ../checkpoints --log_dir ../logs
```

The training script uses PyTorch distributed training. On a multi-GPU machine, set the CUDA environment as needed before launching.

## Enhancement

Enhance a folder of noisy waveforms with a trained checkpoint:

```bash
cd src
python evaluate_folder.py --model_path /path/to/checkpoint --test_dir /path/to/noisy_wavs --save_dir ../results/enhanced --streaming 1
```

`--streaming 1` uses chunked streaming-style inference; set `--streaming 0` for full-utterance inference.

## Metrics

Feature-level PESQ/STOI/SNR evaluation:

```bash
cd src
python Feature_evaluation.py --enhanced_dir ../results/enhanced --clean_dir /path/to/clean_refs --output_csv ../results/evaluation_results.csv
```

ASR WER evaluation:

```bash
cd src
python ASR_eval.py --asr_model /path/or/hf-model-name --data_dir /path/to/data --folder test --subfolder result --output_csv ../results/cmgan_result.csv
```

## Availability Notes

The public release includes code and lightweight documentation. Checkpoints, raw audio, and generated experimental results should be distributed separately only when licensing, consent, ethics, and privacy constraints allow it.
