import argparse
import json
import os
from pathlib import Path

import evaluate
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoImageProcessor, VisionEncoderDecoderModel


def load_run_info(output_dir: Path):
    run_info_path = output_dir / "run_info.json"
    if run_info_path.exists():
        with open(run_info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def resolve_model_source(model_name_or_path: str) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    try:
        return snapshot_download(repo_id=model_name_or_path, local_files_only=True)
    except Exception:
        return model_name_or_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB/outputs/vitgpt2_lora",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/home/swapnik/MyStuff/ImageCaptioning/CapData/flickr30k_images",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="/home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB/data/processed/val.csv",
    )
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    run_info = load_run_info(model_dir)
    base_model = run_info.get("model_source") or run_info.get(
        "base_model", "nlpconnect/vit-gpt2-image-captioning"
    )
    model_source = resolve_model_source(base_model)
    instruction_prefix = run_info.get("instruction_prefix", "")
    adapter_dir = Path(run_info.get("adapter_dir", str(model_dir / "decoder_lora")))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    image_processor = AutoImageProcessor.from_pretrained(model_source, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(model_source, local_files_only=True)
    if adapter_dir.exists():
        model.decoder = PeftModel.from_pretrained(model.decoder, adapter_dir)
    else:
        print(f"Adapter not found at {adapter_dir}; evaluating base model only.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    df = pd.read_csv(args.val_csv).head(args.max_samples)

    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")

    preds = []
    refs = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = Path(args.images_dir) / row["image_name"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)

        if instruction_prefix:
            prompt_ids = tokenizer(
                instruction_prefix, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
        else:
            prompt_ids = None

        with torch.no_grad():
            out_ids = model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=prompt_ids,
                num_beams=5,
                min_new_tokens=18,
                max_new_tokens=60,
                length_penalty=1.1,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        pred = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        preds.append(pred)
        refs.append([str(row["caption"])])

    bleu_score = bleu.compute(predictions=preds, references=refs)
    rouge_score = rouge.compute(predictions=preds, references=[r[0] for r in refs])

    print("Evaluation summary:")
    print({"sacrebleu": bleu_score["score"], "rougeL": rouge_score["rougeL"]})


if __name__ == "__main__":
    main()
