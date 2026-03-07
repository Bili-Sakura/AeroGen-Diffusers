"""MaskCrossAttention - self-contained, no ldm/bldm."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat

from .diffusion_util import zero_module


def exists(val):
    return val is not None


def default(val, d):
    return val if val is not None else d


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    """Simple cross-attention for MaskCrossAttention (no control branch)."""

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) j (c)", h=h, c=sim.shape[-1])
            sim_copy = sim.clone()
            sim_copy.masked_fill_(~mask, max_neg_value)
            sim = sim_copy
        sim = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", sim, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class MaskCrossAttention(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, inner_dim=320, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.inner_dim = inner_dim
        self.norm1 = Normalize(in_channels)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.ffn = FeedForward(dim=inner_dim, dim_out=inner_dim, dropout=0.0, glu=True)
        self.proj_in = nn.Linear(in_channels, 320)
        self.crossattn = CrossAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, context_dim=context_dim)
        self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))

    def forward(self, x, category_control, mask_control, timesteps, attention_strength=0.2, ts_m=200):
        ts = timesteps[0].item()
        if ts < ts_m:
            return x
        x_in = x
        mask_control = mask_control[0]
        category_control = category_control[0]
        b, c, h, w = x.shape
        _, n, _, _ = mask_control.shape
        x = repeat(x, "b c h w -> (b n) c h w", n=n)
        mask_control_in = rearrange(mask_control, "b n h w -> (b n) h w").contiguous().bool()
        category_control = rearrange(category_control.unsqueeze(2), "b n c l -> (b n) c l").contiguous()
        x = rearrange(x, "(b n) c h w -> (b n) (h w) c", b=b, n=n).contiguous()
        x = self.proj_in(x)
        x = self.crossattn(self.norm2(x), category_control, mask_control_in) + x
        x = self.ffn(self.norm3(x)) + x
        x = self.proj_out(x)
        x = rearrange(x, "(b n) (h w) c -> b n c h w", b=b, n=n, h=h, w=w).contiguous()
        mask_control = mask_control.unsqueeze(2).expand(-1, -1, c, -1, -1)
        x = x * mask_control
        x_sum = x.sum(dim=1)
        return attention_strength * x_sum + (1 - attention_strength) * x_in
