"""
RBoxEncoder - pure PyTorch, no ldm/bldm dependency.

Encodes rotated bounding boxes (8 coords) with Fourier embedding and text embeddings.
"""

import torch
import torch.nn as nn


class FourierEmbedder:
    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)

    @torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, cat_dim)


class RBoxEncoder(nn.Module):
    """Encoder for rotated bounding boxes (8 coords) with text embeddings."""

    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 8  # 2 is sin&cos, 8 is xyxyxyxy

        self.linears = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_text_feature = nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = nn.Parameter(torch.zeros([self.position_dim]))

    def forward(self, boxes=None, masks=None, text_embeddings=None, **kwargs):
        # Pipeline passes boxes=[bboxes], masks=[mask_vector], text_embeddings=[category_conditions]
        boxes = (boxes or kwargs.get("boxes", [[]]))[0]
        masks = (masks or kwargs.get("masks", [[]]))[0]
        text_embeddings = (text_embeddings or kwargs.get("text_embeddings", [[]]))[0]

        B, N, _ = boxes.shape
        masks = masks.unsqueeze(-1)

        xyxy_embedding = self.fourier_embedder(boxes)  # B*N*8 --> B*N*C

        text_null = self.null_text_feature.view(1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)

        text_embeddings = text_embeddings * masks + (1 - masks) * text_null
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        objs = self.linears(torch.cat([text_embeddings, xyxy_embedding], dim=-1))
        assert objs.shape == torch.Size([B, N, self.out_dim])
        return objs
