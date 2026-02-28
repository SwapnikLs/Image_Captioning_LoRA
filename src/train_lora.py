import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from huggingface_hub import snapshot_download
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class CaptionSample:
    image_name: str
    caption: str


class FlickrCaptionDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        image_processor,
        tokenizer: AutoTokenizer,
        max_target_length: int,
        instruction_prefix: str = "",
    ):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.instruction_prefix = instruction_prefix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.images_dir, row["image_name"])
        caption = f"{self.instruction_prefix}{row['caption']}".strip()

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values[0]

        labels = self.tokenizer(
            caption,
            padding=False,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids[0]

        return {"pixel_values": pixel_values, "labels": labels}


class DataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, torch.Tensor]]):
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        labels = [x["labels"] for x in batch]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
        "--config",
        type=str,
        default="/home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB/configs/train_config.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["base_model"]
    model_source = resolve_model_source(model_name)
    local_only = bool(cfg.get("local_files_only", True))
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_only)
    image_processor = AutoImageProcessor.from_pretrained(model_source, local_files_only=local_only)
    model = VisionEncoderDecoderModel.from_pretrained(model_source, local_files_only=local_only)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.use_cache = False

    # Keep the image encoder frozen for memory and stability on 4GB VRAM.
    for param in model.encoder.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=list(cfg["lora_target_modules"]),
        bias="none",
    )
    model.decoder = get_peft_model(model.decoder, lora_config)
    model.decoder.print_trainable_parameters()

    if bool(cfg["gradient_checkpointing"]):
        model.decoder.gradient_checkpointing_enable()

    train_dataset = FlickrCaptionDataset(
        csv_path=cfg["train_csv"],
        images_dir=cfg["images_dir"],
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_target_length=int(cfg["max_target_length"]),
        instruction_prefix=cfg.get("instruction_prefix", ""),
    )
    val_dataset = FlickrCaptionDataset(
        csv_path=cfg["val_csv"],
        images_dir=cfg["images_dir"],
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_target_length=int(cfg["max_target_length"]),
        instruction_prefix=cfg.get("instruction_prefix", ""),
    )

    collator = DataCollator(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg["train_batch_size"]),
        per_device_eval_batch_size=int(cfg["eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        dataloader_num_workers=int(cfg.get("num_workers", 0)),
        dataloader_pin_memory=torch.cuda.is_available(),
        fp16=bool(cfg["fp16"]) and torch.cuda.is_available(),
        max_steps=int(cfg.get("max_steps", -1)),
        logging_steps=100,
        save_strategy=str(cfg.get("save_strategy", "epoch")),
        eval_strategy=str(cfg.get("eval_strategy", "epoch")),
        predict_with_generate=True,
        generation_num_beams=int(cfg["num_beams"]),
        generation_max_length=int(cfg["max_new_tokens"]),
        remove_unused_columns=False,
        report_to=[],
        load_best_model_at_end=bool(cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=str(cfg.get("metric_for_best_model", "eval_loss")),
        greater_is_better=bool(cfg.get("greater_is_better", False)),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model()

    # Save adapter separately for clean loading later.
    adapter_dir = output_dir / "decoder_lora"
    model.decoder.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)
    image_processor.save_pretrained(output_dir)

    run_info = {
        "base_model": model_name,
        "model_source": model_source,
        "instruction_prefix": cfg.get("instruction_prefix", ""),
        "max_target_length": int(cfg["max_target_length"]),
        "adapter_dir": str(adapter_dir),
        "num_beams": int(cfg["num_beams"]),
        "length_penalty": float(cfg["length_penalty"]),
        "min_new_tokens": int(cfg["min_new_tokens"]),
        "max_new_tokens": int(cfg["max_new_tokens"]),
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print(f"Training complete. Artifacts saved at: {output_dir}")


if __name__ == "__main__":
    main()
