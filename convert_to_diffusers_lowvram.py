#!/usr/bin/env python3
"""
Convert AeroGen from checkpoint to diffusers format with minimal GPU/RAM usage.

Processes each component separately: load checkpoint once, then for each component:
  instantiate -> load weights -> save -> delete -> gc. Never loads full BLDM.

Usage:
    # Full conversion (step by step with reporting)
    python convert_to_diffusers_lowvram.py \\
        --config_path configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml \\
        --ckpt_path <path_or_repo> \\
        --output_dir /root/worksapce/models/BiliSakura/AeroGen

    # Run single step (1=scheduler, 2=unet, 3=vae, 4=text_encoder, 5=condition_encoder, 6=metadata)
    python convert_to_diffusers_lowvram.py ... --step 2
"""

import argparse
import gc
import json
import os
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

_script_dir = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(_script_dir))

from safetensors.torch import save_file as save_safetensors


def _report(msg: str, level: str = "info"):
    """Print step report."""
    prefix = {"info": "[INFO]", "step": "[STEP]", "done": "[DONE]", "warn": "[WARN]"}.get(level, "[INFO]")
    print(f"{prefix} {msg}")


def _mem_mb():
    """Approximate current process memory usage in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0


def _load_checkpoint(ckpt_path: str, device: str = "cpu"):
    """Load checkpoint state_dict. Use device='cuda' to load into VRAM (avoids RAM OOM)."""
    _, ext = os.path.splitext(ckpt_path)
    if ext.lower() == ".safetensors":
        import safetensors.torch
        dev = device if device == "cpu" else "cuda"
        return safetensors.torch.load_file(ckpt_path, device=dev)
    load_kw = {"map_location": torch.device(device)}
    try:
        raw = torch.load(ckpt_path, **load_kw, weights_only=True)
    except (TypeError, Exception):
        raw = torch.load(ckpt_path, **load_kw, weights_only=False)
    return raw.get("state_dict", raw)


def _slice_state_dict(full_sd: dict, prefix: str) -> dict:
    """Extract keys with given prefix and strip the prefix."""
    out = {}
    n = len(prefix)
    for k, v in full_sd.items():
        if k.startswith(prefix):
            out[k[n:]] = v
    return out


def _to_json_serializable(obj):
    """Convert OmegaConf/DictConfig to plain dict for JSON."""
    try:
        from omegaconf import DictConfig, ListConfig
        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    if hasattr(obj, "to_container"):
        try:
            return OmegaConf.to_container(obj, resolve=True)
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    return obj


def save_custom_component(component, config, save_path: Path, safe_serialization: bool = True):
    """Save a custom component in diffusers format."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    config_path = save_path / "config.json"
    config_to_save = _to_json_serializable(config)

    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)

    state_dict = component.state_dict()
    if safe_serialization:
        weights_path = save_path / "diffusion_pytorch_model.safetensors"
        save_safetensors(state_dict, weights_path)
    else:
        weights_path = save_path / "diffusion_pytorch_model.bin"
        torch.save(state_dict, weights_path)


def run_extract(ckpt_path: str, extract_dir: Path, device: str = "cpu"):
    """Extract each component's state_dict to separate safetensors. Run on high-RAM machine."""
    _report("Extracting components from checkpoint (run on machine with sufficient RAM)...", "step")
    state_dict = _load_checkpoint(ckpt_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    components = [
        ("model.diffusion_model.", "unet"),
        ("first_stage_model.", "vae"),
        ("cond_stage_model.", "text_encoder"),
        ("condition_tokenizer.", "condition_encoder"),
    ]
    for prefix, name in components:
        sd = _slice_state_dict(state_dict, prefix)
        if not sd:
            _report(f"  No keys found for {name} (prefix {prefix})", "warn")
            continue
        path = extract_dir / f"{name}.safetensors"
        save_safetensors(sd, path)
        _report(f"  Saved {name} -> {path} ({len(sd)} keys)", "done")
        del sd
        gc.collect()
    _report(f"Extraction done. Use --from_extracted {extract_dir} for conversion.", "done")


def run_step(
    step: int,
    config_path: str,
    ckpt_path: str,
    output_dir: Path,
    state_dict: dict,
    safe_serialization: bool,
    extracted_dir: Path = None,
):
    """Run a single conversion step. state_dict is the full checkpoint state_dict, or None when using extracted_dir."""
    from ldm.util import instantiate_from_config

    config = OmegaConf.load(config_path)
    model_params = config.model.params

    output_dir.mkdir(parents=True, exist_ok=True)

    if step == 1:
        # Scheduler (no weights, just config)
        _report("Step 1/6: Saving scheduler (config only)", "step")
        from diffusers import DDIMScheduler

        scheduler = DDIMScheduler(
            num_train_timesteps=model_params.get("timesteps", 1000),
            beta_start=model_params.get("linear_start", 0.00085),
            beta_end=model_params.get("linear_end", 0.0120),
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type="epsilon",
        )
        sp = output_dir / "scheduler"
        scheduler.save_pretrained(sp)
        _report(f"Saved scheduler -> {sp}", "done")
        return

    if step == 2:
        # UNet: use AeroGenUNet2DConditionModel (diffusers ModelMixin wrapper)
        _report("Step 2/6: Converting UNet (AeroGenUNet2DConditionModel)", "step")
        if extracted_dir:
            import safetensors.torch
            sd = dict(safetensors.torch.load_file(str(extracted_dir / "unet.safetensors")))
        else:
            prefix = "model.diffusion_model."
            sd = _slice_state_dict(state_dict, prefix)
        _report(f"  Extracted {len(sd)} UNet keys", "info")
        unet_params = OmegaConf.to_container(model_params.unet_config.params, resolve=True)
        unet_params = dict(unet_params) if unet_params else {}
        unet_params.setdefault("legacy", False)
        from unet.unet_aerogen import AeroGenUNet2DConditionModel
        unet = AeroGenUNet2DConditionModel(**unet_params)
        # Wrap state dict: AeroGenUNet2DConditionModel has self.model = UNetModel
        wrapped_sd = {"model." + k: v for k, v in sd.items()}
        unet.load_state_dict(wrapped_sd, strict=True)
        unet.eval()
        del sd, wrapped_sd
        unet_config = {"target": "unet.unet_aerogen.AeroGenUNet2DConditionModel", "params": unet_params}
        save_custom_component(unet, unet_config, output_dir / "unet", safe_serialization)
        _report(f"Saved unet -> {output_dir / 'unet'}", "done")
        del unet
        gc.collect()
        return

    if step == 3:
        # VAE: use diffusers AutoencoderKL (native diffusers format)
        _report("Step 3/6: Converting VAE (diffusers AutoencoderKL)", "step")
        from diffusers import AutoencoderKL
        from diffusers.loaders.single_file_utils import (
            create_vae_diffusers_config_from_ldm,
            convert_ldm_vae_checkpoint,
        )

        # Build checkpoint dict for diffusers utils (expects "first_stage_model." prefix)
        if extracted_dir:
            import safetensors.torch
            vae_sd = dict(safetensors.torch.load_file(str(extracted_dir / "vae.safetensors")))
            checkpoint = {"first_stage_model." + k: v for k, v in vae_sd.items()}
            del vae_sd
        else:
            checkpoint = state_dict

        # Original config as plain dict (for create_vae_diffusers_config_from_ldm)
        config_dict = OmegaConf.to_container(config, resolve=True)

        vae_config = create_vae_diffusers_config_from_ldm(config_dict, checkpoint)
        vae = AutoencoderKL.from_config(vae_config)
        converted_sd = convert_ldm_vae_checkpoint(checkpoint, vae_config)
        vae.load_state_dict(converted_sd, strict=True)
        vae.eval()
        del checkpoint, converted_sd
        vae.save_pretrained(output_dir / "vae", safe_serialization=safe_serialization)
        _report(f"Saved vae -> {output_dir / 'vae'} (diffusers AutoencoderKL)", "done")
        del vae
        gc.collect()
        return

    if step == 4:
        # Text encoder (AeroGenCLIPTextEncoder - no ldm)
        _report("Step 4/6: Converting text encoder (AeroGenCLIPTextEncoder)", "step")
        if extracted_dir:
            import safetensors.torch
            sd = dict(safetensors.torch.load_file(str(extracted_dir / "text_encoder.safetensors")))
        else:
            prefix = "cond_stage_model."
            sd = _slice_state_dict(state_dict, prefix)
        _report(f"  Extracted {len(sd)} text_encoder keys", "info")
        version = "openai/clip-vit-large-patch14"
        text_config = {"target": "models.clip_text_encoder.AeroGenCLIPTextEncoder", "params": {"version": version}}
        from models.clip_text_encoder import AeroGenCLIPTextEncoder
        te = AeroGenCLIPTextEncoder(version=version)
        te.load_state_dict(sd, strict=True)
        te.eval()
        del sd
        save_custom_component(te, text_config, output_dir / "text_encoder", safe_serialization)
        _report(f"Saved text_encoder -> {output_dir / 'text_encoder'}", "done")
        del te
        gc.collect()
        return

    if step == 5:
        # Condition encoder (RBoxEncoder)
        _report("Step 5/6: Converting condition encoder", "step")
        if extracted_dir:
            import safetensors.torch
            sd = dict(safetensors.torch.load_file(str(extracted_dir / "condition_encoder.safetensors")))
        else:
            prefix = "condition_tokenizer."
            sd = _slice_state_dict(state_dict, prefix)
        _report(f"  Extracted {len(sd)} condition_encoder keys", "info")
        from condition_encoder.rbox_encoder import RBoxEncoder
        cond_params = {"in_dim": 768, "out_dim": 768}
        if model_params.get("condition_tokenizer") and model_params.condition_tokenizer.get("params"):
            raw = OmegaConf.to_container(model_params.condition_tokenizer.params, resolve=True) if hasattr(model_params.condition_tokenizer.params, "to_container") else dict(model_params.condition_tokenizer.params or {})
            cond_params = {k: raw.get(k, v) for k, v in cond_params.items()}
        ce = RBoxEncoder(**cond_params)
        cond_config = {"target": "condition_encoder.rbox_encoder.RBoxEncoder", "params": cond_params}
        ce.load_state_dict(sd, strict=True)
        ce.eval()
        del sd
        save_custom_component(ce, cond_config, output_dir / "condition_encoder", safe_serialization)
        _report(f"Saved condition_encoder -> {output_dir / 'condition_encoder'}", "done")
        del ce
        gc.collect()
        return

    if step == 6:
        # Metadata: model_index.json (with scale_factor), pipeline.py at root
        _report("Step 6/6: Saving pipeline metadata and scripts", "step")
        config_loaded = OmegaConf.load(config_path)
        scale_factor = config_loaded.model.params.get("scale_factor", 0.18215)
        model_index = {
            "_class_name": ["pipeline", "AeroGenPipeline"],
            "_diffusers_version": "0.25.0",
            "condition_encoder": ["pipeline", "AeroGenPipeline"],
            "scheduler": ["diffusers", "DDIMScheduler"],
            "text_encoder": ["pipeline", "AeroGenPipeline"],
            "unet": ["pipeline", "AeroGenPipeline"],
            "vae": ["pipeline", "AeroGenPipeline"],
            "scale_factor": scale_factor,
        }
        with open(output_dir / "model_index.json", "w") as f:
            json.dump(model_index, f, indent=2)

        shutil.copy(_script_dir / "pipeline_aerogen.py", output_dir / "pipeline.py")
        if (_script_dir / "modular_pipeline.py").exists():
            shutil.copy(_script_dir / "modular_pipeline.py", output_dir / "modular_pipeline.py")
        # Bundle models, condition_encoder, unet (no ldm/bldm - self-contained)
        for subdir in ("models", "condition_encoder", "unet"):
            src = _script_dir / subdir
            if not src.exists():
                continue
            dst = output_dir / subdir
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

        _report("Saved model_index.json, pipeline.py, models/, condition_encoder/, unet/", "done")
        return

    raise ValueError(f"Unknown step: {step}")


def main():
    parser = argparse.ArgumentParser(description="Low-VRAM AeroGen to diffusers conversion")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config")
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
        help="HuggingFace repo for checkpoint",
    )
    parser.add_argument(
        "--ckpt_filename",
        type=str,
        default="aerogen_diorr_last.ckpt",
        help="Checkpoint filename in HF repo",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for diffusers-format model",
    )
    parser.add_argument("--no_safe_serialization", action="store_true", help="Use .bin instead of safetensors")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Run only this step (1-6). Default: run all.",
    )
    parser.add_argument(
        "--extract_to",
        type=str,
        default=None,
        help="Extract component weights to dir (safetensors). Run on high-RAM machine.",
    )
    parser.add_argument(
        "--from_extracted",
        type=str,
        default=None,
        help="Load from pre-extracted dir (use after --extract_to on another machine).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for loading checkpoint (cuda uses VRAM, avoids RAM OOM). Default: cuda if available.",
    )
    args = parser.parse_args()

    safe_serialization = not args.no_safe_serialization
    output_dir = Path(args.output_dir)

    # Resolve config path
    config_path = Path(args.config_path)
    if not config_path.exists():
        config_path = _script_dir / args.config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config_path}")
    config_path = str(config_path)

    # Patch cond_stage default for CLIP
    config = OmegaConf.load(config_path)
    cond_stage = config.model.params.get("cond_stage_config")
    if cond_stage is not None:
        raw = cond_stage.get("params")
        if hasattr(raw, "to_container"):
            params = OmegaConf.to_container(raw, resolve=True)
        else:
            params = dict(raw) if raw else {}
        params.setdefault("version", "openai/clip-vit-large-patch14")
        config.model.params.cond_stage_config.params = OmegaConf.create(params)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            OmegaConf.save(config, f.name)
            config_path = f.name

    extracted_dir = Path(args.from_extracted) if args.from_extracted else None

    # Extract-only mode: save components to dir and exit
    if args.extract_to:
        extract_dir = Path(args.extract_to)
        if args.ckpt_path:
            ckpt_path = str(Path(args.ckpt_path).resolve())
        else:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id=args.repo_id, filename=args.ckpt_filename)
            _report(f"Downloaded checkpoint: {ckpt_path}")
        run_extract(ckpt_path, extract_dir, device=args.device)
        return

    # Resolve checkpoint path (only for steps 2-5 when not using extracted)
    ckpt_path = None
    steps_to_run = [args.step] if args.step is not None else [1, 2, 3, 4, 5, 6]
    needs_ckpt = not extracted_dir and any(s in (2, 3, 4, 5) for s in steps_to_run)

    if needs_ckpt:
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
            _report(f"Downloaded checkpoint: {ckpt_path}")

    for s in steps_to_run:
        if s < 1 or s > 6:
            raise ValueError("--step must be 1-6")
        # Steps 1 and 6 don't need checkpoint or extracted
        if s in (1, 6):
            run_step(s, config_path, ckpt_path, output_dir, state_dict=None, safe_serialization=safe_serialization, extracted_dir=None)
        elif extracted_dir:
            # Load from pre-extracted safetensors (low memory)
            run_step(s, config_path, ckpt_path, output_dir, state_dict=None, safe_serialization=safe_serialization, extracted_dir=extracted_dir)
        else:
            # Load full checkpoint for this step
            _report(f"Loading checkpoint for step {s}...", "info")
            mem_before = _mem_mb()
            state_dict = _load_checkpoint(ckpt_path, device=args.device)
            mem_after = _mem_mb()
            _report(f"Checkpoint loaded: {len(state_dict)} keys, ~{mem_after - mem_before:.0f} MB", "done")
            run_step(s, config_path, ckpt_path, output_dir, state_dict, safe_serialization, extracted_dir=None)
            del state_dict
            gc.collect()
            _report(f"Freed checkpoint from memory", "info")
        _report(f"Memory: ~{_mem_mb():.0f} MB", "info")

    _report(f"\nConversion complete. Output: {output_dir}", "done")
    _report("Load with: AeroGenPipeline.from_pretrained('<path>')", "info")


if __name__ == "__main__":
    main()
