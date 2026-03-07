"""Standalone CLIP text encoder for AeroGen. Uses transformers only (no ldm)."""

import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel


class AeroGenCLIPTextEncoder(nn.Module):
    """CLIP text encoder compatible with FrozenCLIPEmbedder interface.
    Uses transformers CLIPTextModel + CLIPTokenizer. No ldm dependency.
    """

    def __init__(self, version: str = "openai/clip-vit-large-patch14", device: str = "cuda", max_length: int = 77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        if isinstance(text, str):
            text = [text]
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        tokens = batch_encoding["input_ids"].to(device)
        outputs = self.transformer(input_ids=tokens)
        return outputs.last_hidden_state

    def encode(self, text):
        return self(text)
