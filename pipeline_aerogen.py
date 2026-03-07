"""
AeroGen Pipeline using native HuggingFace Diffusers.

This module provides a DiffusionPipeline subclass that wraps AeroGen's
custom UNet, condition encoder, VAE, and text encoder into a standard
diffusers pipeline interface, using DDIMScheduler for the denoising loop.

Usage:
    # Load from config + checkpoint
    pipeline = AeroGenPipeline.from_pretrained_checkpoint(
        config_path="configs/.../v1-finetune-DIOR-R.yaml",
        checkpoint_path="./ckpt/aerogen_diorr_last.ckpt",
    )

    # Load from diffusers-format (after convert_to_diffusers.py)
    pipeline = AeroGenPipeline.from_pretrained("/path/to/AeroGen")
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

# Ensure model repo is on path for trust_remote_code / custom_pipeline loading
_pipeline_dir = Path(__file__).resolve().parent
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

import einops
import numpy as np
import torch
import yaml
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import BaseOutput
from PIL import Image

from modular_pipeline import (
    ensure_ldm_path,
    ensure_ldm_path_from_config,
    load_component,
    load_components,
    create_scheduler,
    _instantiate_from_config,
)


@dataclass
class AeroGenPipelineOutput(BaseOutput):
    """Output class for AeroGen pipeline.

    Attributes:
        images: List of generated PIL images.
    """

    images: List[Image.Image]


class AeroGenPipeline(DiffusionPipeline):
    """Pipeline for AeroGen: conditional aerial image generation with
    bounding box and category controls.

    This pipeline wraps AeroGen's custom components (UNet, condition encoder,
    VAE, text encoder) and uses a native diffusers DDIMScheduler for the
    denoising loop, replacing the original custom DDIM sampler.

    Args:
        unet: The custom UNet model (openaimodel_bbox_v2.UNetModel).
        scheduler: A diffusers DDIMScheduler instance.
        vae: The VAE model (AutoencoderKL) for latent encoding/decoding.
        text_encoder: The frozen CLIP text encoder for prompt conditioning.
        condition_encoder: The RBoxEncoder or BoxEncoder for bbox conditioning.
        scale_factor: VAE latent scale factor (default: 0.18215 for SD 1.x).
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        scheduler: DDIMScheduler,
        vae: torch.nn.Module,
        text_encoder: torch.nn.Module,
        condition_encoder: torch.nn.Module,
        scale_factor: float = 0.18215,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            condition_encoder=condition_encoder,
        )
        self.vae_scale_factor = scale_factor

    @property
    def device(self) -> torch.device:
        """Return the device of the pipeline's first nn.Module parameter."""
        for module in [self.unet, self.vae, self.text_encoder, self.condition_encoder]:
            if isinstance(module, torch.nn.Module):
                params = list(module.parameters())
                if params:
                    return params[0].device
        return torch.device("cpu")

    @property
    def _execution_device(self) -> torch.device:
        return self.device

    @classmethod
    def from_pretrained_checkpoint(
        cls,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> "AeroGenPipeline":
        """Load an AeroGenPipeline from a YAML config and checkpoint.

        DEPRECATED: ldm/bldm have been removed. Use from_pretrained() with a
        diffusers-format model (converted via convert_to_diffusers_lowvram.py).
        """
        raise NotImplementedError(
            "from_pretrained_checkpoint is no longer supported (ldm/bldm removed). "
            "Use AeroGenPipeline.from_pretrained() with a diffusers-format model."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        subfolder: Optional[str] = None,
        **kwargs,
    ) -> Union["AeroGenPipeline", torch.nn.Module]:
        """Load AeroGenPipeline from a diffusers-format directory.

        Supports native diffusers loading via DiffusionPipeline.from_pretrained(..., trust_remote_code=True).
        When subfolder is provided (e.g. by diffusers for component loading), returns only that component.

        Args:
            pretrained_model_name_or_path: Path to the diffusers-format
                directory or HuggingFace repo ID.
            device: Device to load the model onto.
            subfolder: If set, load only this component (unet, vae, text_encoder, condition_encoder).

        Returns:
            An AeroGenPipeline instance, or a single component if subfolder is set.
        """
        path = ensure_ldm_path(pretrained_model_name_or_path)

        # Single-component loading (for diffusers trust_remote_code component loading)
        subfolder = kwargs.pop("subfolder", subfolder)
        if subfolder in ("unet", "vae", "text_encoder", "condition_encoder"):
            return load_component(path, subfolder)

        components = load_components(path)
        pipe = cls(
            unet=components["unet"],
            scheduler=components["scheduler"],
            vae=components["vae"],
            text_encoder=components["text_encoder"],
            condition_encoder=components["condition_encoder"],
            scale_factor=components["scale_factor"],
        )

        if device is not None:
            pipe = pipe.to(device)
        return pipe

    def _encode_prompt(self, prompt: Union[str, List[str]]) -> torch.Tensor:
        """Encode text prompt(s) using the frozen CLIP text encoder."""
        if isinstance(prompt, str):
            prompt = [prompt]
        return self.text_encoder.encode(prompt)

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representations using the VAE."""
        latents = (1.0 / self.vae_scale_factor) * latents
        image = self.vae.decode(latents)
        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        bboxes: torch.Tensor,
        category_conditions: torch.Tensor,
        mask_conditions: torch.Tensor,
        mask_vector: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.2,
        height: int = 512,
        width: int = 512,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
    ) -> AeroGenPipelineOutput:
        """Generate aerial images conditioned on bounding boxes and categories.

        Args:
            prompt: Text prompt(s) describing the aerial scene.
            bboxes: Bounding box coordinates tensor of shape (B, N, 8) for
                rotated boxes or (B, N, 4) for axis-aligned boxes.
            category_conditions: Category embedding tensor of shape
                (B, N, 768).
            mask_conditions: Spatial mask tensor of shape (B, N, H, W).
            mask_vector: Binary vector indicating valid objects, shape (B, N).
            num_inference_steps: Number of DDIM denoising steps.
            guidance_scale: Classifier-free guidance scale. Values > 1.0
                enable guidance.
            eta: DDIM eta parameter controlling stochasticity.
            height: Output image height (must be divisible by 8).
            width: Output image width (must be divisible by 8).
            num_images_per_prompt: Number of images to generate per prompt.
            generator: Optional torch.Generator for reproducibility.
            output_type: Output format, either "pil" for PIL images or
                "tensor" for raw image tensors.

        Returns:
            AeroGenPipelineOutput with the generated images.
        """
        device = self._execution_device

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        # Repeat conditions for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt = prompt * num_images_per_prompt
            bboxes = torch.cat(
                [bboxes] * num_images_per_prompt, dim=0
            )
            category_conditions = torch.cat(
                [category_conditions] * num_images_per_prompt, dim=0
            )
            mask_conditions = torch.cat(
                [mask_conditions] * num_images_per_prompt, dim=0
            )
            mask_vector = torch.cat(
                [mask_vector] * num_images_per_prompt, dim=0
            )

        total_batch = batch_size * num_images_per_prompt

        # 1. Encode text prompts
        text_embeddings = self._encode_prompt(prompt)

        # 2. Encode unconditional prompt for CFG
        if guidance_scale > 1.0:
            uncond_embeddings = self._encode_prompt([""] * total_batch)

        # 3. Move conditions to device
        bboxes = bboxes.to(device).float()
        category_conditions = category_conditions.to(device).float()
        mask_conditions = mask_conditions.to(device).float()
        mask_vector = mask_vector.to(device).float()

        # 4. Encode bbox conditions
        control = self.condition_encoder(
            text_embeddings=[category_conditions],
            masks=[mask_vector],
            boxes=[bboxes],
        )

        # 5. Prepare latent noise
        latent_shape = (
            total_batch,
            4,
            height // 8,
            width // 8,
        )
        latents = torch.randn(
            latent_shape, device=device, generator=generator
        )

        # 6. Set up scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # 7. Scale initial noise by scheduler init_noise_sigma
        latents = latents * self.scheduler.init_noise_sigma

        # 8. Denoising loop
        for t in self.scheduler.timesteps:
            timesteps = torch.full(
                (total_batch,), t, device=device, dtype=torch.long
            )

            if guidance_scale > 1.0:
                # Classifier-free guidance: run model twice
                latent_input = torch.cat([latents, latents], dim=0)
                timestep_input = torch.cat([timesteps, timesteps], dim=0)

                context_in = torch.cat(
                    [uncond_embeddings, text_embeddings], dim=0
                )
                control_in = torch.cat([control, control], dim=0)
                category_in = [
                    torch.cat(
                        [category_conditions, category_conditions], dim=0
                    )
                ]
                mask_in = [
                    torch.cat(
                        [mask_conditions, mask_conditions], dim=0
                    )
                ]

                noise_pred = self.unet(
                    x=latent_input,
                    timesteps=timestep_input,
                    context=context_in,
                    control=control_in,
                    category_control=category_in,
                    mask_control=mask_in,
                )

                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (
                    noise_text - noise_uncond
                )
            else:
                noise_pred = self.unet(
                    x=latents,
                    timesteps=timesteps,
                    context=text_embeddings,
                    control=control,
                    category_control=[category_conditions],
                    mask_control=[mask_conditions],
                )

            # Use diffusers scheduler step
            scheduler_output = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=latents,
                eta=eta,
                generator=generator,
            )
            latents = scheduler_output.prev_sample

        # 9. Decode latents
        images = self._decode_latents(latents)

        # 10. Post-process
        if output_type == "pil":
            images = (
                einops.rearrange(images, "b c h w -> b h w c") * 127.5 + 127.5
            )
            images = images.cpu().numpy().clip(0, 255).astype(np.uint8)
            images = [Image.fromarray(img) for img in images]
        elif output_type == "tensor":
            images = images.cpu()
        else:
            raise ValueError(
                f"Unknown output_type '{output_type}'. "
                "Use 'pil' or 'tensor'."
            )

        return AeroGenPipelineOutput(images=images)
