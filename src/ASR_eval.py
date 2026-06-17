import argparse
import os

import jiwer
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def evaluate(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(args.asr_model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.asr_model,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        return_timestamps=True,
        dtype=torch_dtype,
        device=device,
    )

    pred_dir = os.path.join(args.data_dir, args.folder, args.subfolder)
    gt_dir = os.path.join(args.data_dir, args.folder, args.clean_subfolder)
    print(f"Target folder: {pred_dir}")

    normalizer = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation()])
    result = {"filename": [], "gt": [], "pred": [], "wer": []}

    print("Start recognition")
    for filename in tqdm(os.listdir(pred_dir), desc="ASR evaluating"):
        base_filename = filename.split("_")[0] + ".wav"
        pred = pipe(
            os.path.join(pred_dir, filename),
            generate_kwargs={"language": args.language},
        )["text"]
        gt = pipe(
            os.path.join(gt_dir, base_filename),
            generate_kwargs={"language": args.language},
        )["text"]

        result["filename"].append(base_filename)
        result["gt"].append(gt)
        result["pred"].append(pred)
        result["wer"].append(jiwer.wer(normalizer(gt), normalizer(pred)))

    df = pd.DataFrame.from_dict(result)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df.to_csv(args.output_csv, sep=",", index=False)
    print(f"Result saved to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_model", type=str, required=True, help="Hugging Face model name or local ASR model path")
    parser.add_argument("--data_dir", type=str, required=True, help="root directory containing evaluation audio")
    parser.add_argument("--folder", type=str, default="test", help="evaluation split or folder")
    parser.add_argument("--subfolder", type=str, default="result", help="subfolder containing enhanced audio")
    parser.add_argument("--clean_subfolder", type=str, default="clean", help="subfolder containing clean reference audio")
    parser.add_argument("--output_csv", type=str, default="cmgan_result.csv", help="path to write ASR WER results")
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    evaluate(args)
