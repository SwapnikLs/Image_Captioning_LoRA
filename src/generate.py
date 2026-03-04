import argparse
import json
import os
import re
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from PIL import Image
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
        "with", "and", "that", "which", "who", "while", "in", "on", "of", "to", "for",
        "as", "a", "an", "the"
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
        "--image_path",
        type=str,
        default="/home/swapnik/MyStuff/ImageCaptioning/CapData/flickr30k_images/793558.jpg",
    )
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
        print(f"Adapter not found at {adapter_dir}; using base model only.")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    image = Image.open(args.image_path).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)

    if instruction_prefix:
        prompt_ids = tokenizer(
            instruction_prefix,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
    else:
        prompt_ids = None

    with torch.no_grad():
        generated_ids = model.generate(
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

    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    caption = clean_caption(caption, prefix=instruction_prefix)
    print("Generated caption:")
    print(caption)


if __name__ == "__main__":
    main()
