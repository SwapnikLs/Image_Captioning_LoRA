# Detailed Image Captioning (4GB GPU Friendly)

This is a separate, end-to-end project for generating more meaningful captions on:
- RAM: 16GB
- GPU: RTX 3050 4GB

It uses:
- `nlpconnect/vit-gpt2-image-captioning` as base
- frozen vision encoder (memory efficient)
- LoRA adapters on decoder attention layers
- beam-search decoding with minimum length for richer output

## 1) Setup

```bash
cd /home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you already use your alias-based env (`ic`), run with:

```bash
bash -ic 'ic; cd /home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB; pip install -r requirements.txt'
```

## 2) Prepare train/val splits

```bash
bash scripts/run_prepare.sh
```

Expected output:
- `data/processed/train.csv`
- `data/processed/val.csv`

## 3) Train LoRA model

```bash
bash scripts/run_train.sh
```

Main config is in:
- `configs/train_config.yaml`

Memory-safe defaults already set:
- batch size `1`
- gradient accumulation `16`
- fp16 enabled
- gradient checkpointing enabled

Outputs:
- `outputs/vitgpt2_lora/`
- `outputs/vitgpt2_lora/decoder_lora/`
- `outputs/vitgpt2_lora/run_info.json`

### Quick smoke run (recommended first)

```bash
bash scripts/run_smoke_train.sh
```

This runs `max_steps=1` without eval to verify everything is wired correctly.

## 4) Generate caption for one image

```bash
bash scripts/run_generate.sh
```

To run for another image:

```bash
python3 src/generate.py \
  --model_dir /home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB/outputs/vitgpt2_lora \
  --image_path /home/swapnik/MyStuff/ImageCaptioning/CapData/flickr30k_images/1000092795.jpg
```

## 5) Quick evaluation on val set

```bash
python3 src/quick_eval.py \
  --model_dir /home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB/outputs/vitgpt2_lora \
  --images_dir /home/swapnik/MyStuff/ImageCaptioning/CapData/flickr30k_images \
  --val_csv /home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB/data/processed/val.csv \
  --max_samples 500
```

This reports:
- SacreBLEU
- ROUGE-L

## Notes

- Your previous captions were heavily normalized; this pipeline keeps more detail via instruction prefixing and decoding controls.
- If you hit OOM:
  - reduce `gradient_accumulation_steps` only if training is too slow
  - keep batch size `1`
  - lower `max_target_length` from `48` to `40`
- For even richer captions later, swap base model to BLIP and keep the same LoRA training pattern.
