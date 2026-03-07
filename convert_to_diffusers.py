#!/usr/bin/env python3
"""
Convert AeroGen from original checkpoint format to diffusers-style format.

Usage:
    python convert_to_diffusers.py \\
        --repo_id Sonetto702/AeroGen \\
        --config_path configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml \\
        --output_dir /root/worksapce/models/BiliSakura/AeroGen

Or with local checkpoint:
    python convert_to_diffusers.py \\
        --config_path configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml \\
        --ckpt_path ./ckpt/aerogen_diorr_last.ckpt \\
        --output_dir /root/worksapce/models/BiliSakura/AeroGen
"""

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add project root for imports
_script_dir = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(_script_dir))

from safetensors.torch import save_file as save_safetensors


def save_custom_component(component, config, save_path: Path, safe_serialization: bool = True):
    """Save a custom component in diffusers format."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save config for reconstruction
    config_path = save_path / "config.json"
    try:
        config_to_save = (
            OmegaConf.to_container(config, resolve=True)
            if hasattr(config, "to_container")
            else dict(config)
        )
    except Exception:
        config_to_save = dict(config) if not isinstance(config, dict) else config

    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)

    # Save weights
    state_dict = component.state_dict()
    if safe_serialization:
        weights_path = save_path / "diffusion_pytorch_model.safetensors"
        save_safetensors(state_dict, weights_path)
    else:
        weights_path = save_path / "diffusion_pytorch_model.bin"
        torch.save(state_dict, weights_path)


def main():
    parser = argparse.ArgumentParser(description="Convert AeroGen to diffusers format")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint. If not set, downloads from --repo_id",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Sonetto702/AeroGen",
        help="HuggingFace repo for checkpoint (default: Sonetto702/AeroGen)",
    )
    parser.add_argument(
        "--ckpt_filename",
        type=str,
        default="aerogen_diorr_last.ckpt",
        help="Checkpoint filename in HF repo (default: aerogen_diorr_last.ckpt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for diffusers-format model",
    )
    parser.add_argument("--no_safe_serialization", action="store_true", help="Use .bin instead of safetensors")
    parser.add_argument("--device", type=str, default="cpu", help="Device for loading")
    args = parser.parse_args()

    safe_serialization = not args.no_safe_serialization

    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt_path = str(ckpt_path)
    else:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.ckpt_filename,
        )
        print(f"Downloaded checkpoint: {ckpt_path}")

    config_path = Path(args.config_path)
    if not config_path.exists():
        # Try relative to script dir
        config_path = _script_dir / args.config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config_path}")
    config_path = str(config_path)

    # Patch config: FrozenCLIPEmbedder defaults to local path; use HF id
    config = OmegaConf.load(config_path)
    cond_stage = config.model.params.get("cond_stage_config")
    if cond_stage is not None:
        raw_params = cond_stage.get("params")
        if hasattr(raw_params, "to_container"):
            params = OmegaConf.to_container(raw_params, resolve=True)
        else:
            params = dict(raw_params) if raw_params else {}
        params.setdefault("version", "openai/clip-vit-large-patch14")
        config.model.params.cond_stage_config.params = OmegaConf.create(params)
        # Save patched config to temp and use it
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            OmegaConf.save(config, f.name)
            config_path = f.name

    from pipeline_aerogen import AeroGenPipeline

    print("Loading AeroGen pipeline...")
    pipe = AeroGenPipeline.from_pretrained_checkpoint(
        config_path=config_path,
        checkpoint_path=ckpt_path,
        device="cpu",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(config_path)
    model_params = config.model.params

    print(f"Saving to {output_dir}...")

    # 1. Save scheduler
    scheduler_path = output_dir / "scheduler"
    pipe.scheduler.save_pretrained(scheduler_path)
    print(f"  Saved scheduler -> {scheduler_path}")

    # 2. Save unet
    unet_config = model_params.unet_config
    unet_path = output_dir / "unet"
    save_custom_component(pipe.unet, unet_config, unet_path, safe_serialization)
    print(f"  Saved unet -> {unet_path}")
    gc.collect()

    # 3. Save vae
    vae_config = model_params.first_stage_config
    vae_path = output_dir / "vae"
    save_custom_component(pipe.vae, vae_config, vae_path, safe_serialization)
    print(f"  Saved vae -> {vae_path}")
    gc.collect()

    # 4. Save text_encoder (FrozenCLIPEmbedder)
    text_config = model_params.get("cond_stage_config")
    if text_config is None:
        text_config = {"target": "ldm.modules.encoders.modules.FrozenCLIPEmbedder", "params": {}}
    else:
        text_config = OmegaConf.to_container(text_config, resolve=True) if hasattr(text_config, "to_container") else dict(text_config)
        text_config = text_config if isinstance(text_config, dict) else {"target": str(text_config), "params": {}}
        text_config.setdefault("params", {})
        text_config["params"].setdefault("version", "openai/clip-vit-large-patch14")
    text_path = output_dir / "text_encoder"
    save_custom_component(pipe.text_encoder, text_config, text_path, safe_serialization)
    print(f"  Saved text_encoder -> {text_path}")

    # 5. Save condition_encoder (RBoxEncoder)
    cond_enc_config = model_params.get("condition_tokenizer")
    if cond_enc_config is None:
        cond_enc_config = {"target": "bldm.condition_encoder.RBoxEncoder", "params": {"in_dim": 768, "out_dim": 768}}
    cond_path = output_dir / "condition_encoder"
    save_custom_component(pipe.condition_encoder, cond_enc_config, cond_path, safe_serialization)
    print(f"  Saved condition_encoder -> {cond_path}")

    # 6. Save model_index.json
    model_index = {
        "_class_name": "AeroGenPipeline",
        "_diffusers_version": "0.25.0",
        "condition_encoder": ["pipeline_aerogen", "AeroGenPipeline"],
        "scheduler": ["diffusers", "DDIMScheduler"],
        "text_encoder": ["pipeline_aerogen", "AeroGenPipeline"],
        "unet": ["pipeline_aerogen", "AeroGenPipeline"],
        "vae": ["pipeline_aerogen", "AeroGenPipeline"],
    }
    with open(output_dir / "model_index.json", "w") as f:
        json.dump(model_index, f, indent=2)
    print(f"  Saved model_index.json")

    # 7. Copy pipeline
    pipeline_src = _script_dir / "pipeline_aerogen.py"
    pipeline_dst = output_dir / "pipeline_aerogen"
    pipeline_dst.mkdir(exist_ok=True)
    shutil.copy(pipeline_src, pipeline_dst / "pipeline_aerogen.py")
    (pipeline_dst / "__init__.py").write_text("from .pipeline_aerogen import AeroGenPipeline\n")
    print(f"  Saved pipeline_aerogen")

    # 8. Pipeline config
    pipeline_config = {"scale_factor": pipe.vae_scale_factor}
    with open(output_dir / "pipeline_config.json", "w") as f:
        json.dump(pipeline_config, f, indent=2)

    print(f"\nDone! Diffusers-format model saved to {output_dir}")
    print("Load with: AeroGenPipeline.from_pretrained('<path>')")


if __name__ == "__main__":
    main()
