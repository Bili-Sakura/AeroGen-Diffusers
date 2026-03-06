# AeroGen inference using the native HuggingFace Diffusers pipeline.

import sys

if "./" not in sys.path:
    sys.path.append("./")
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pipeline_aerogen import AeroGenPipeline
from src.datasets.dataset_display import MyDataset

original_path = "demo/txt"
config_path = "configs/stable-diffusion/dual/v1-finetune-DIOR-R.yaml"
ckpt_file_path = "./ckpt/aerogen_diorr_last.ckpt"
resolution = 512
mask_size = 64
batch_size = 1
num_samples = 2
ddim_steps = 50
guidance_scale = 7.5
eta = 0.2
H = 512
W = 512

dataset = MyDataset(original_path, resolution, mask_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=32)

# Load the pipeline from checkpoint
pipeline = AeroGenPipeline.from_pretrained_checkpoint(
    config_path=config_path,
    checkpoint_path=ckpt_file_path,
    device="cuda",
)

for batch in tqdm(dataloader):
    filenames = batch["filename"]
    prompts = batch["txt"]

    result = pipeline(
        prompt=list(prompts),
        bboxes=batch["bboxes"].float(),
        category_conditions=batch["category_conditions"].float(),
        mask_conditions=batch["mask_conditions"].float(),
        mask_vector=batch["mask_vector"].float(),
        num_inference_steps=ddim_steps,
        guidance_scale=guidance_scale,
        eta=eta,
        height=H,
        width=W,
        num_images_per_prompt=num_samples,
        output_type="pil",
    )

    for i, filename in enumerate(filenames):
        images = result.images[i * num_samples : (i + 1) * num_samples]

        for idx, image in enumerate(images):
            output_dir = os.path.join("./demo/img", str(idx))
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, filename.replace("txt", "jpg"))
            image.save(output_file)

            print(f"Generation images saved at {output_file}")
            print("ok")
