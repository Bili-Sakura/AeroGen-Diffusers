"""
AeroGen modular components: scheduler config, component loading, and path setup.

Self-contained - no ldm/bldm. Scheduler is created in-code (no scheduler/ folder required).
"""

import importlib
import json
import sys
from pathlib import Path
from typing import Optional, Union

from diffusers import DDIMScheduler

# Ensure model dir is on path for local module imports (unet, models, condition_encoder)
_pipeline_dir = Path(__file__).resolve().parent
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

# Default DDIM scheduler config (matches scheduler/scheduler_config.json)
DEFAULT_SCHEDULER_CONFIG = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "clip_sample": False,
    "set_alpha_to_one": False,
    "prediction_type": "epsilon",
}


def ensure_ldm_path(pretrained_model_name_or_path: Union[str, Path]) -> Path:
    """Add model repo to path so local modules can be imported. Returns resolved path."""
    path = Path(pretrained_model_name_or_path)
    if not path.exists():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(pretrained_model_name_or_path))
    path = path.resolve()
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
    return path


def ensure_ldm_path_from_config(config_path: str) -> None:
    """Walk up from config file dir to find project root and add to path."""
    d = Path(config_path).resolve().parent
    for _ in range(10):
        if (d / "pipeline.py").exists() or (d / "unet").is_dir():
            s = str(d)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        parent = d.parent
        if parent == d:
            break
        d = parent


def _get_class_from_string(target: str):
    """Resolve class from dotted path (diffusers-style, no OmegaConf)."""
    module_path, cls_name = target.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _instantiate_from_config(config: dict):
    """Instantiate from dict with 'target' and 'params' (diffusers-style, no OmegaConf)."""
    if not isinstance(config, dict) or "target" not in config:
        raise KeyError("Expected key 'target' to instantiate.")
    cls = _get_class_from_string(config["target"])
    params = dict(config.get("params") or {})
    params.pop("ckpt_path", None)
    params.pop("ignore_keys", None)
    params.pop("target", None)  # avoid passing target into constructor
    return cls(**params)


def create_scheduler(model_path: Path) -> DDIMScheduler:
    """Create DDIMScheduler from path/scheduler if exists, else from defaults."""
    scheduler_path = model_path / "scheduler"
    if scheduler_path.exists() and (scheduler_path / "scheduler_config.json").exists():
        return DDIMScheduler.from_pretrained(scheduler_path)
    return DDIMScheduler(**DEFAULT_SCHEDULER_CONFIG)


def load_component(model_path: Path, name: str):
    """Load a custom component (unet, vae, text_encoder, condition_encoder).

    VAE: Uses diffusers AutoencoderKL.from_pretrained when saved in diffusers format
         (config has down_block_types, no target). Otherwise uses target/params.
    """
    import torch
    comp_path = model_path / name
    with open(comp_path / "config.json") as f:
        cfg = json.load(f)

    # Diffusers native format (e.g. AutoencoderKL.save_pretrained): no "target" key
    if "target" not in cfg and name == "vae":
        from diffusers import AutoencoderKL
        return AutoencoderKL.from_pretrained(comp_path)

    component = _instantiate_from_config(cfg)
    safetensors_path = comp_path / "diffusion_pytorch_model.safetensors"
    bin_path = comp_path / "diffusion_pytorch_model.bin"
    if safetensors_path.exists():
        import safetensors.torch
        state = safetensors.torch.load_file(str(safetensors_path))
    elif bin_path.exists():
        try:
            state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No weights in {comp_path} "
            "(expected diffusion_pytorch_model.safetensors or .bin)"
        )
    component.load_state_dict(state, strict=True)
    component.eval()
    return component


def load_components(
    model_path: Union[str, Path],
) -> dict:
    """Load all pipeline components. Returns dict with unet, vae, text_encoder, condition_encoder, scheduler, scale_factor."""
    path = ensure_ldm_path(model_path)
    scheduler = create_scheduler(path)
    unet = load_component(path, "unet")
    vae = load_component(path, "vae")
    text_encoder = load_component(path, "text_encoder")
    condition_encoder = load_component(path, "condition_encoder")

    scale_factor = 0.18215
    model_index_path = path / "model_index.json"
    if model_index_path.exists():
        with open(model_index_path) as f:
            model_index = json.load(f)
        scale_factor = model_index.get("scale_factor", scale_factor)

    return {
        "unet": unet,
        "vae": vae,
        "text_encoder": text_encoder,
        "condition_encoder": condition_encoder,
        "scheduler": scheduler,
        "scale_factor": scale_factor,
    }
