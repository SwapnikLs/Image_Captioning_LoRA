#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1

python3 src/train_lora.py \
  --config /home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB/configs/train_config.yaml
