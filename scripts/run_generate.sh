#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1

python3 src/generate.py \
  --model_dir /home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB/outputs/vitgpt2_lora \
  --image_path /home/swapnik/MyStuff/ImageCaptioning/CapData/flickr30k_images/5722658.jpg \
  --num_beams 2 \
  --min_new_tokens 8 \
  --max_new_tokens 18 \
  --length_penalty 2.0 \
  --no_repeat_ngram_size 5 \
  --repetition_penalty 1.35
