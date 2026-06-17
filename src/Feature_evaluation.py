import argparse
import os

import librosa
import numpy as np
import pandas as pd
from pesq import pesq
from pystoi import stoi


def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio, sr


def compute_pesq(noisy, clean, sr):
    return pesq(sr, clean, noisy, "wb")


def compute_stoi(noisy, clean, sr):
    return stoi(clean, noisy, sr, extended=False)


def compute_snr(noisy, clean):
    noise = noisy - clean
    snr = 10 * np.log10(np.sum(clean ** 2) / np.sum(noise ** 2))
    return snr


def evaluate_audio_files(enhanced_dir, clean_dir, output_csv):
    results = []
    for file_name in os.listdir(enhanced_dir):
        if file_name.endswith("_enhance.wav"):
            base_name = file_name.replace("_enhance.wav", "-target.wav")
            noisy_path = os.path.join(enhanced_dir, file_name)
            clean_path = os.path.join(clean_dir, base_name)
            
            noisy, sr_noisy = load_audio(noisy_path)
            clean, sr_clean = load_audio(clean_path)
            
            if sr_noisy != sr_clean:
                raise ValueError("Sampling rates do not match for noisy and clean files.")
            min_length = min(len(noisy), len(clean))
            noisy = noisy[:min_length]
            clean = clean[:min_length]

            pesq_score = compute_pesq(noisy, clean, sr_noisy)
            stoi_score = compute_stoi(noisy, clean, sr_noisy)
            snr_score = compute_snr(noisy, clean)
            
            results.append(
                {
                    "file": base_name,
                    "PESQ": pesq_score,
                    "STOI": stoi_score,
                    "SNR": snr_score,
                }
            )
            print(f"{base_name}: PESQ {pesq_score}, STOI {stoi_score}, SNR {snr_score}")
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enhanced_dir", type=str, required=True, help="folder containing *_enhance.wav files")
    parser.add_argument("--clean_dir", type=str, required=True, help="folder containing clean *-target.wav files")
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv")
    args = parser.parse_args()
    evaluate_audio_files(args.enhanced_dir, args.clean_dir, args.output_csv)
