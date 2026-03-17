import os
import re
import uuid
from pathlib import Path

import torch
from flask import Flask, render_template, request
from huggingface_hub import snapshot_download
from peft import PeftModel
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
    VisionEncoderDecoderModel,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "outputs" / "vitgpt2_lora"
FALLBACK_MODEL_DIR = PROJECT_DIR / "outputs" / "vitgpt2_lora_smoke"
UPLOAD_DIR = PROJECT_DIR / "uploads"
TEMPLATE_DIR = PROJECT_DIR / "frontend" / "templates"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

MODEL = None
TOKENIZER = None
IMAGE_PROCESSOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INSTRUCTION_PREFIX = ""
BAD_WORDS_IDS = None
BLIP_MODEL = None
BLIP_PROCESSOR = None
BLIP_MODEL_ID = os.environ.get("BLIP_MODEL_ID", "Salesforce/blip-image-captioning-base")
DEFAULT_MODE = os.environ.get("CAPTION_MODE", "blip").strip().lower()


def load_run_info(output_dir: Path) -> dict:
    run_info_path = output_dir / "run_info.json"
    if run_info_path.exists():
        import json

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


def load_blip():
    global BLIP_MODEL, BLIP_PROCESSOR
    BLIP_PROCESSOR = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
    BLIP_MODEL.to(DEVICE)
    BLIP_MODEL.eval()


def build_bad_words_ids(tokenizer):
    blocked_phrases = [
        "iced",
        " iced",
        "iced latte",
        "iced tea",
        "iannis",
        "iaanis",
    ]
    bad_words_ids = []
    for phrase in blocked_phrases:
        ids = tokenizer(phrase, add_special_tokens=False).input_ids
        if ids:
            bad_words_ids.append(ids)
    return bad_words_ids


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
    cleaned = re.sub(r"^ike\s+", "a ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(iannis|iaanis|ianis|iains|ian)\b\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:an|a)\s+people\b", "people", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    cleaned = cleaned[:1].upper() + cleaned[1:]
    if not cleaned.endswith("."):
        cleaned += "."
    return cleaned


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def init_lora_model():
    global MODEL, TOKENIZER, IMAGE_PROCESSOR, INSTRUCTION_PREFIX, BAD_WORDS_IDS

    model_dir_env = os.environ.get("MODEL_DIR", "").strip()
    if model_dir_env:
        model_dir = Path(model_dir_env)
    else:
        model_dir = DEFAULT_MODEL_DIR if DEFAULT_MODEL_DIR.exists() else FALLBACK_MODEL_DIR

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found. Checked: {DEFAULT_MODEL_DIR} and {FALLBACK_MODEL_DIR}"
        )

    run_info = load_run_info(model_dir)
    base_model = run_info.get("model_source") or run_info.get(
        "base_model", "nlpconnect/vit-gpt2-image-captioning"
    )
    model_source = resolve_model_source(base_model)
    INSTRUCTION_PREFIX = run_info.get("instruction_prefix", "")
    adapter_dir = Path(run_info.get("adapter_dir", str(model_dir / "decoder_lora")))

    TOKENIZER = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained(model_source, local_files_only=True)
    MODEL = VisionEncoderDecoderModel.from_pretrained(model_source, local_files_only=True)
    BAD_WORDS_IDS = build_bad_words_ids(TOKENIZER)

    if adapter_dir.exists():
        MODEL.decoder = PeftModel.from_pretrained(MODEL.decoder, adapter_dir)

    MODEL.eval()
    MODEL.to(DEVICE)


def generate_caption_lora(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    pixel_values = IMAGE_PROCESSOR(images=image, return_tensors="pt").pixel_values.to(DEVICE)

    if INSTRUCTION_PREFIX:
        prompt_ids = TOKENIZER(
            INSTRUCTION_PREFIX,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(DEVICE)
    else:
        prompt_ids = None

    with torch.no_grad():
        generated_ids = MODEL.generate(
            pixel_values=pixel_values,
            decoder_input_ids=prompt_ids,
            num_beams=2,
            min_new_tokens=8,
            max_new_tokens=18,
            length_penalty=2.0,
            no_repeat_ngram_size=5,
            repetition_penalty=1.35,
            renormalize_logits=True,
            early_stopping=True,
            bad_words_ids=BAD_WORDS_IDS if BAD_WORDS_IDS else None,
        )

    text = TOKENIZER.decode(generated_ids[0], skip_special_tokens=True).strip()
    return clean_caption(text, prefix=INSTRUCTION_PREFIX)


def generate_caption_blip(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = BLIP_PROCESSOR(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_ids = BLIP_MODEL.generate(
            **inputs,
            num_beams=4,
            min_new_tokens=8,
            max_new_tokens=28,
            length_penalty=1.2,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            early_stopping=True,
        )
    text = BLIP_PROCESSOR.decode(out_ids[0], skip_special_tokens=True).strip()
    return clean_caption(text, prefix="")


def init_models():
    # Deadline-safe behavior:
    # 1) Try BLIP first (better caption quality)
    # 2) Always try LoRA too for experimental comparison
    try:
        load_blip()
        print(f"Loaded BLIP model: {BLIP_MODEL_ID}")
    except Exception as exc:
        print(f"BLIP load failed, continuing without BLIP: {exc}")

    try:
        init_lora_model()
        print("Loaded LoRA caption model.")
    except Exception as exc:
        print(f"LoRA load failed, continuing without LoRA: {exc}")

    if BLIP_MODEL is None and MODEL is None:
        raise RuntimeError("No caption model could be loaded.")


def generate_caption(image_path: str, mode: str) -> str:
    mode = (mode or DEFAULT_MODE).lower()
    if mode == "lora" and MODEL is not None:
        return generate_caption_lora(image_path)
    if mode == "blip" and BLIP_MODEL is not None:
        return generate_caption_blip(image_path)
    # graceful fallback
    if BLIP_MODEL is not None:
        return generate_caption_blip(image_path)
    return generate_caption_lora(image_path)


@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    error = None
    image_rel_path = None
    selected_mode = DEFAULT_MODE if DEFAULT_MODE in {"blip", "lora"} else "blip"

    if request.method == "POST":
        selected_mode = request.form.get("mode", selected_mode).strip().lower()
        if selected_mode not in {"blip", "lora"}:
            selected_mode = "blip"

        if "image" not in request.files:
            error = "Please choose an image file."
            return render_template(
                "index.html",
                caption=caption,
                error=error,
                image_path=image_rel_path,
                mode=selected_mode,
                blip_ready=BLIP_MODEL is not None,
                lora_ready=MODEL is not None,
            )

        file = request.files["image"]
        if file.filename == "":
            error = "No file selected."
            return render_template(
                "index.html",
                caption=caption,
                error=error,
                image_path=image_rel_path,
                mode=selected_mode,
                blip_ready=BLIP_MODEL is not None,
                lora_ready=MODEL is not None,
            )

        if not allowed_file(file.filename):
            error = "Unsupported file type. Use png/jpg/jpeg/webp."
            return render_template(
                "index.html",
                caption=caption,
                error=error,
                image_path=image_rel_path,
                mode=selected_mode,
                blip_ready=BLIP_MODEL is not None,
                lora_ready=MODEL is not None,
            )

        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = Path(app.config["UPLOAD_FOLDER"]) / unique_name
        file.save(save_path)

        try:
            caption = generate_caption(str(save_path), selected_mode)
            image_rel_path = f"/uploads/{unique_name}"
        except Exception as exc:
            error = f"Caption generation failed: {exc}"

    return render_template(
        "index.html",
        caption=caption,
        error=error,
        image_path=image_rel_path,
        mode=selected_mode,
        blip_ready=BLIP_MODEL is not None,
        lora_ready=MODEL is not None,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    from flask import send_from_directory

    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    init_models()
    app.run(host="0.0.0.0", port=7860, debug=False)
