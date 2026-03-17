#!/usr/bin/env bash
set -euo pipefail

export MODEL_DIR="/home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB/outputs/vitgpt2_lora"
export CAPTION_MODE="blip"

python3 /home/swapnik/MyStuff/ImageCaptioning/CapData/Captioning_LoRA_4GB/backend/app.py
