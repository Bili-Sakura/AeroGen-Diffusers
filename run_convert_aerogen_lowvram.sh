#!/bin/bash
# Two-phase AeroGen conversion for low-RAM/VRAM machines.
#
# The full checkpoint is ~9.5GB; loading it requires 12GB+ RAM.
# If your machine OOMs, use the two-phase workflow:
#
# PHASE 1 (on machine with 16GB+ RAM, e.g. Colab, cloud VM):
#   ./run_convert_aerogen_lowvram.sh extract
#
# PHASE 2 (copy the extracted/ dir to your machine, then):
#   ./run_convert_aerogen_lowvram.sh convert /path/to/extracted
#
# Or run steps individually (step 1 and 6 need no checkpoint):
#   python convert_to_diffusers_lowvram.py ... --step 1
#   python convert_to_diffusers_lowvram.py ... --step 6
#
set -e
cd "$(dirname "$0")"
conda activate rsgen 2>/dev/null || conda activate aerogen 2>/dev/null || true

CONFIG="configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml"
REPO="Sonetto702/AeroGen"
CKPT_FILE="aerogen_diorr_last.ckpt"
OUTPUT="/root/worksapce/models/BiliSakura/AeroGen"
EXTRACT_DIR="${1:-./aerogen_extracted}"

case "${1:-}" in
  extract)
    echo "=== Phase 1: Extracting components (needs 16GB+ RAM) ==="
    python convert_to_diffusers_lowvram.py \
      --config_path "$CONFIG" \
      --repo_id "$REPO" \
      --ckpt_filename "$CKPT_FILE" \
      --extract_to "$EXTRACT_DIR"
    echo "Done. Copy '$EXTRACT_DIR' to your target machine."
    ;;
  convert)
    FROM="$2"
    if [ -z "$FROM" ] || [ ! -d "$FROM" ]; then
      echo "Usage: $0 convert /path/to/extracted"
      echo "  (extracted dir must contain unet.safetensors, vae.safetensors, etc.)"
      exit 1
    fi
    echo "=== Phase 2: Converting from extracted (low RAM) ==="
    python convert_to_diffusers_lowvram.py \
      --config_path "$CONFIG" \
      --output_dir "$OUTPUT" \
      --from_extracted "$FROM"
    echo "Done. Model at $OUTPUT"
    ;;
  step1)
    echo "=== Step 1 only (scheduler, no checkpoint) ==="
    python convert_to_diffusers_lowvram.py \
      --config_path "$CONFIG" \
      --output_dir "$OUTPUT" \
      --step 1
    ;;
  step6)
    echo "=== Step 6 only (metadata, no checkpoint) ==="
    python convert_to_diffusers_lowvram.py \
      --config_path "$CONFIG" \
      --output_dir "$OUTPUT" \
      --step 6
    ;;
  *)
    echo "Usage: $0 extract | convert /path/to/extracted | step1 | step6"
    echo ""
    echo "  extract    - Extract components (run on 16GB+ RAM machine)"
    echo "  convert    - Convert from extracted dir (low RAM)"
    echo "  step1      - Save scheduler only"
    echo "  step6      - Save metadata only"
    exit 1
    ;;
esac
