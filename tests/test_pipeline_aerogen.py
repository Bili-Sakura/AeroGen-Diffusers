"""Tests for the AeroGen diffusers pipeline."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from PIL import Image

from pipeline_aerogen import AeroGenPipeline, AeroGenPipelineOutput


# ---------------------------------------------------------------------------
# Mock components that replicate the interfaces of the real models
# ---------------------------------------------------------------------------


class MockTextEncoder(nn.Module):
    """Mimics FrozenCLIPEmbedder: encode(list[str]) -> (B, 77, 768)."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def encode(self, text):
        batch_size = len(text) if isinstance(text, list) else 1
        return torch.randn(batch_size, 77, 768, device=self.dummy.device)


class MockVAE(nn.Module):
    """Mimics AutoencoderKL: decode(z) -> image tensor (B, 3, H*8, W*8)."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def decode(self, z):
        # Return deterministic output derived from input for reproducibility
        b, c, h, w = z.shape
        out = z[:, :3].repeat(1, 1, 8, 8)
        return out


class MockConditionEncoder(nn.Module):
    """Mimics RBoxEncoder: forward(boxes, masks, text_embeddings) -> (B, N, 768)."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, boxes, masks, text_embeddings):
        B, N, _ = boxes[0].shape
        return torch.randn(B, N, 768, device=self.dummy.device)


class MockUNet(nn.Module):
    """Mimics UNetModel: forward(x, timesteps, context, control, ...) -> noise pred."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, timesteps, context, control, category_control, mask_control, **kwargs):
        # Return deterministic output based on input (for reproducibility testing)
        return torch.zeros_like(x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pipeline():
    """Build a pipeline with mock components on CPU."""
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.0120,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type="epsilon",
    )
    pipe = AeroGenPipeline(
        unet=MockUNet(),
        scheduler=scheduler,
        vae=MockVAE(),
        text_encoder=MockTextEncoder(),
        condition_encoder=MockConditionEncoder(),
        scale_factor=0.18215,
    )
    return pipe


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAeroGenPipelineOutput:
    def test_output_creation(self):
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        output = AeroGenPipelineOutput(images=[img])
        assert len(output.images) == 1
        assert isinstance(output.images[0], Image.Image)

    def test_output_multiple_images(self):
        imgs = [
            Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
            for _ in range(4)
        ]
        output = AeroGenPipelineOutput(images=imgs)
        assert len(output.images) == 4


class TestPipelineInit:
    def test_pipeline_creation(self, mock_pipeline):
        assert mock_pipeline is not None
        assert isinstance(mock_pipeline.scheduler, DDIMScheduler)
        assert isinstance(mock_pipeline.unet, MockUNet)
        assert isinstance(mock_pipeline.vae, MockVAE)
        assert isinstance(mock_pipeline.text_encoder, MockTextEncoder)
        assert isinstance(mock_pipeline.condition_encoder, MockConditionEncoder)
        assert mock_pipeline.vae_scale_factor == 0.18215

    def test_scheduler_config(self, mock_pipeline):
        scheduler = mock_pipeline.scheduler
        assert scheduler.config.num_train_timesteps == 1000
        assert scheduler.config.beta_start == 0.00085
        assert scheduler.config.beta_end == 0.0120


class TestPipelineEncode:
    def test_encode_single_prompt(self, mock_pipeline):
        embeddings = mock_pipeline._encode_prompt("test prompt")
        assert embeddings.shape == (1, 77, 768)

    def test_encode_batch_prompt(self, mock_pipeline):
        embeddings = mock_pipeline._encode_prompt(["prompt 1", "prompt 2"])
        assert embeddings.shape == (2, 77, 768)

    def test_decode_latents(self, mock_pipeline):
        latents = torch.randn(1, 4, 64, 64)
        images = mock_pipeline._decode_latents(latents)
        assert images.shape[0] == 1
        assert images.shape[1] == 3


class TestPipelineCall:
    def test_basic_call_pil(self, mock_pipeline):
        B, N = 1, 5
        result = mock_pipeline(
            prompt="an aerial image",
            bboxes=torch.randn(B, N, 8),
            category_conditions=torch.randn(B, N, 768),
            mask_conditions=torch.randn(B, N, 64, 64),
            mask_vector=torch.ones(B, N),
            num_inference_steps=2,  # minimal steps for speed
            guidance_scale=7.5,
            height=64,
            width=64,
            num_images_per_prompt=1,
            output_type="pil",
        )
        assert isinstance(result, AeroGenPipelineOutput)
        assert len(result.images) == 1
        assert isinstance(result.images[0], Image.Image)

    def test_basic_call_tensor(self, mock_pipeline):
        B, N = 1, 5
        result = mock_pipeline(
            prompt="an aerial image",
            bboxes=torch.randn(B, N, 8),
            category_conditions=torch.randn(B, N, 768),
            mask_conditions=torch.randn(B, N, 64, 64),
            mask_vector=torch.ones(B, N),
            num_inference_steps=2,
            guidance_scale=1.0,  # no CFG
            height=64,
            width=64,
            num_images_per_prompt=1,
            output_type="tensor",
        )
        assert isinstance(result, AeroGenPipelineOutput)
        assert isinstance(result.images, torch.Tensor)

    def test_multiple_images_per_prompt(self, mock_pipeline):
        B, N = 1, 5
        num_images = 3
        result = mock_pipeline(
            prompt="an aerial image",
            bboxes=torch.randn(B, N, 8),
            category_conditions=torch.randn(B, N, 768),
            mask_conditions=torch.randn(B, N, 64, 64),
            mask_vector=torch.ones(B, N),
            num_inference_steps=2,
            guidance_scale=7.5,
            height=64,
            width=64,
            num_images_per_prompt=num_images,
            output_type="pil",
        )
        assert len(result.images) == num_images

    def test_no_guidance(self, mock_pipeline):
        B, N = 1, 5
        result = mock_pipeline(
            prompt="an aerial image",
            bboxes=torch.randn(B, N, 8),
            category_conditions=torch.randn(B, N, 768),
            mask_conditions=torch.randn(B, N, 64, 64),
            mask_vector=torch.ones(B, N),
            num_inference_steps=2,
            guidance_scale=1.0,
            height=64,
            width=64,
            output_type="pil",
        )
        assert len(result.images) == 1

    def test_invalid_output_type(self, mock_pipeline):
        B, N = 1, 5
        with pytest.raises(ValueError, match="Unknown output_type"):
            mock_pipeline(
                prompt="an aerial image",
                bboxes=torch.randn(B, N, 8),
                category_conditions=torch.randn(B, N, 768),
                mask_conditions=torch.randn(B, N, 64, 64),
                mask_vector=torch.ones(B, N),
                num_inference_steps=2,
                guidance_scale=1.0,
                height=64,
                width=64,
                output_type="invalid",
            )

    def test_batch_prompts(self, mock_pipeline):
        B, N = 2, 5
        result = mock_pipeline(
            prompt=["prompt 1", "prompt 2"],
            bboxes=torch.randn(B, N, 8),
            category_conditions=torch.randn(B, N, 768),
            mask_conditions=torch.randn(B, N, 64, 64),
            mask_vector=torch.ones(B, N),
            num_inference_steps=2,
            guidance_scale=7.5,
            height=64,
            width=64,
            output_type="pil",
        )
        assert len(result.images) == 2

    def test_reproducibility_with_generator(self, mock_pipeline):
        B, N = 1, 5
        kwargs = dict(
            prompt="an aerial image",
            bboxes=torch.randn(B, N, 8),
            category_conditions=torch.randn(B, N, 768),
            mask_conditions=torch.randn(B, N, 64, 64),
            mask_vector=torch.ones(B, N),
            num_inference_steps=2,
            guidance_scale=1.0,
            height=64,
            width=64,
            output_type="tensor",
        )
        gen1 = torch.Generator().manual_seed(42)
        result1 = mock_pipeline(**kwargs, generator=gen1)

        gen2 = torch.Generator().manual_seed(42)
        result2 = mock_pipeline(**kwargs, generator=gen2)

        assert torch.equal(result1.images, result2.images)
