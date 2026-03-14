import argparse
import json
import os
import re
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


def clean_caption(text: str, prefix: str = "", max_words: int = 26) -> str:
    if prefix and text.lower().startswith(prefix.lower()):
        text = text[len(prefix):].strip()
    if text.lower().startswith("detailed caption:"):
        text = text.split(":", 1)[1].strip()

    words = text.strip().split()
    if not words:
        return text.strip()

    words = words[:max_words]
    dangling = {
        "with",
        "and",
        "that",
        "which",
        "who",
        "while",
        "in",
        "on",
        "of",
        "to",
        "for",
        "as",
        "a",
        "an",
        "the",
    }
    while words and words[-1].lower().strip(".,") in dangling:
        words.pop()

    tail2 = " ".join(w.lower().strip(".,") for w in words[-2:]) if len(words) >= 2 else ""
    if tail2 in {"while one", "while a", "while an", "while the"}:
        words = words[:-2]

    if not words:
        return text.strip()

    cleaned = " ".join(words).strip(" ,.")
    cleaned = re.sub(r"^ian\s+", "an ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:an|a)\s+people\b", "people", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    cleaned = cleaned[:1].upper() + cleaned[1:]
    if not cleaned.endswith("."):
        cleaned += "."
    return cleaned


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
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--min_new_tokens", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=18)
    parser.add_argument("--length_penalty", type=float, default=2.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=5)
    parser.add_argument("--repetition_penalty", type=float, default=1.35)
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
    meteor = None
    cider = None
    meteor_error = None
    cider_error = None
    try:
        meteor = evaluate.load("meteor")
    except Exception as exc:
        meteor_error = str(exc)
    try:
        cider = evaluate.load("cider")
    except Exception as exc:
        cider_error = str(exc)

    preds = []
    refs_nested = []
    refs_single = []

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
                num_beams=args.num_beams,
                min_new_tokens=args.min_new_tokens,
                max_new_tokens=args.max_new_tokens,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                renormalize_logits=True,
                early_stopping=True,
            )
        pred = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
        pred = clean_caption(pred, prefix=instruction_prefix)
        ref = clean_caption(str(row["caption"]), prefix=instruction_prefix)

        preds.append(pred)
        refs_nested.append([ref])
        refs_single.append(ref)

    bleu_score = bleu.compute(predictions=preds, references=refs_nested)
    rouge_score = rouge.compute(predictions=preds, references=refs_single)
    meteor_score = None
    cider_score = None
    if meteor is not None:
        meteor_score = meteor.compute(predictions=preds, references=refs_nested)["meteor"]
    if cider is not None:
        cider_score = cider.compute(predictions=preds, references=refs_nested)["cider"]

    print("Evaluation summary:")
    summary = {
        "sacrebleu": float(bleu_score["score"]),
        "rougeL": float(rouge_score["rougeL"]),
        "meteor": None if meteor_score is None else float(meteor_score),
        "cider": None if cider_score is None else float(cider_score),
    }
    print(summary)
    if meteor_error:
        print(f"METEOR unavailable: {meteor_error}")
    if cider_error:
        print(f"CIDEr unavailable: {cider_error}")


if __name__ == "__main__":
    main()
