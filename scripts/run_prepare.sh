#!/usr/bin/env bash
set -euo pipefail

python3 src/prepare_data.py \
  --captions_csv /home/swapnik/MyStuff/ImageCaptioning/CapData/cleaned_captions.csv \
  --output_dir /home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB/data/processed \
  --val_ratio 0.1 \
  --seed 42
