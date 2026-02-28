#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export MODEL_DIR="/home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB/outputs/vitgpt2_lora"

python3 /home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB/backend/app.py
