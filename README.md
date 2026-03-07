---
license: apache-2.0
library_name: diffusers
tags:
  - aerogen
  - remote-sensing
  - object-detection
  - latent-diffusion
  - bounding-box
  - arxiv:2411.15497
pipeline_tag: text-to-image
---

# AeroGen-Diffusers

**Convert AeroGen from checkpoint to diffusers format.** AeroGen generates aerial images conditioned on bounding boxes (horizontal or rotated) and object categories. This project provides conversion scripts and a native `AeroGenPipeline` for the [HuggingFace Diffusers](https://github.com/huggingface/diffusers) ecosystem.

> Source: [Sonetto702/AeroGen](https://huggingface.co/Sonetto702/AeroGen). Paper: [AeroGen: Enhancing Remote Sensing Object Detection with Diffusion-Driven Data Generation](https://arxiv.org/abs/2411.15497).

## Conversion

### Standard conversion (requires ~12GB+ RAM or 24GB VRAM)

The checkpoint is ~9.5GB. Use **GPU** (`--device cuda`) to load into VRAM and avoid RAM OOM:

```bash
cd projects/AeroGen-Diffusers
conda activate rsgen  # or aerogen

python convert_to_diffusers_lowvram.py \
    --config_path configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml \
    --repo_id Sonetto702/AeroGen \
    --ckpt_filename aerogen_diorr_last.ckpt \
    --output_dir /root/worksapce/models/BiliSakura/AeroGen \
    --device cuda
```

With a local checkpoint:

```bash
python convert_to_diffusers_lowvram.py \
    --config_path configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml \
    --ckpt_path ./ckpt/aerogen_diorr_last.ckpt \
    --output_dir /root/worksapce/models/BiliSakura/AeroGen \
    --device cuda
```

### Low-RAM / two-phase conversion

If loading the full checkpoint causes OOM (e.g. on machines with <12GB RAM):

1. **Phase 1** (on a machine with 16GB+ RAM): Extract component weights to safetensors:
   ```bash
   python convert_to_diffusers_lowvram.py \
       --config_path configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml \
       --repo_id Sonetto702/AeroGen \
       --extract_to ./aerogen_extracted
   ```

2. **Phase 2** (on the target machine): Convert from extracted files:
   ```bash
   python convert_to_diffusers_lowvram.py \
       --config_path configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml \
       --output_dir /root/worksapce/models/BiliSakura/AeroGen \
       --from_extracted ./aerogen_extracted
   ```

### Step-by-step conversion

Run individual steps (1=scheduler, 2=unet, 3=vae, 4=text_encoder, 5=condition_encoder, 6=metadata):

```bash
python convert_to_diffusers_lowvram.py ... --step 2
```

## Output structure (converted model)

| Component         | Path                 |
|-------------------|----------------------|
| UNet              | `unet/`              |
| VAE               | `vae/`               |
| Text encoder      | `text_encoder/`      |
| Condition encoder | `condition_encoder/` |
| Scheduler         | `scheduler/`         |
| Pipeline          | `pipeline.py` (root) |
| Config            | `model_index.json` (includes `scale_factor`) |

## Usage

The model repo is **self-contained** — no external code repo needed. Add only the model directory to `PYTHONPATH`:

```python
import sys
sys.path.insert(0, "/root/worksapce/models/BiliSakura/AeroGen")

from pipeline import AeroGenPipeline

pipe = AeroGenPipeline.from_pretrained("/root/worksapce/models/BiliSakura/AeroGen")
pipe = pipe.to("cuda")

# Conditioning: bboxes (B, N, 8) rotated, category_conditions (B, N, 768), mask_conditions, mask_vector
result = pipe(
    prompt="an aerial image with airplane parked on the ground",
    bboxes=bboxes,               # (B, N, 8) rotated bbox coords
    category_conditions=cats,     # (B, N, 768) category embeddings
    mask_conditions=masks,        # (B, N, H, W) spatial masks
    mask_vector=mask_vec,         # (B, N) valid-object indicators
    num_inference_steps=50,
    guidance_scale=7.5,
)

images = result.images  # List of PIL images
```

### Conditioning requirements

- **bboxes**: Rotated box coords `(B, N, 8)` or axis-aligned `(B, N, 4)`
- **category_conditions**: CLIP embeddings per object, `(B, N, 768)`
- **mask_conditions**: Spatial masks `(B, N, H, W)`
- **mask_vector**: Binary valid-object indicators `(B, N)`

See `demo/` and `src/inference/` in the original [AeroGen](https://github.com/Sonettoo/AeroGen) repo for layout preparation.

## Project structure

| Path                    | Description                          |
|-------------------------|--------------------------------------|
| `convert_to_diffusers.py`       | Original conversion (loads full model) |
| `convert_to_diffusers_lowvram.py` | Low-VRAM conversion (component-wise)   |
| `pipeline_aerogen.py`    | AeroGenPipeline definition            |
| `bldm/`                 | BLDM model (condition_tokenizer)      |
| `ldm/`                  | LDM modules (UNet, VAE, encoders)     |
| `configs/`              | YAML configs                          |

## Model sources

- **Source**: [Sonetto702/AeroGen](https://huggingface.co/Sonetto702/AeroGen)
- **Paper**: [AeroGen: Enhancing Remote Sensing Object Detection with Diffusion-Driven Data Generation](https://arxiv.org/abs/2411.15497)
- **Original repo**: [Sonettoo/AeroGen](https://github.com/Sonettoo/AeroGen)

## Citation

```bibtex
@article{tang2024aerogen,
  title={AeroGen: Enhancing Remote Sensing Object Detection with Diffusion-Driven Data Generation},
  author={Tang, Datao and Cao, Xiangyong and Wu, Xuan and Li, Jialin and Yao, Jing and Bai, Xueru and Meng, Deyu},
  journal={arXiv preprint arXiv:2411.15497},
  year={2024}
}
```
