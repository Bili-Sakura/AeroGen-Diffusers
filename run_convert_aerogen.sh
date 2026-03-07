#!/bin/bash
# Convert Sonetto702/AeroGen to diffusers format and save to BiliSakura/AeroGen
# Requires: taming-transformers-rom1504, clip, kornia
# Note: Needs ~8GB+ RAM for full model conversion
set -e
cd "$(dirname "$0")"
conda activate aerogen 2>/dev/null || conda activate rsgen 2>/dev/null || true
python convert_to_diffusers.py \
    --config_path configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml \
    --repo_id Sonetto702/AeroGen \
    --ckpt_filename aerogen_diorr_last.ckpt \
    --output_dir /root/worksapce/models/BiliSakura/AeroGen
