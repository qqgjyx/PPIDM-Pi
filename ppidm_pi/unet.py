"""1-D U-Net score network for ECG diffusion.

Architecture:
  - Sinusoidal timestep embedding
  - Class-conditioned via AdaptiveGroupNorm (scale/shift)
  - Encoder: 4 downsampling stages [128, 256, 512, 1024]
  - Bottleneck with self-attention
  - Decoder: 4 upsampling stages with skip connections
  - Output: noise prediction ε_θ(x_t, t, y)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal positional embedding for diffusion timestep
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Adaptive Group Norm — class conditioning via scale/shift
# ---------------------------------------------------------------------------
class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_channels: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(cond_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.proj(cond)[:, :, None]          # [B, 2C, 1]
        scale, shift = scale_shift.chunk(2, dim=1)
        return self.norm(x) * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------
class ResBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(out_ch, cond_dim, num_groups)
        self.norm2 = AdaptiveGroupNorm(out_ch, cond_dim, num_groups)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x), cond))
        h = F.silu(self.norm2(self.conv2(h), cond))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Self-attention at bottleneck resolution
# ---------------------------------------------------------------------------
class Attention1d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # Scaled dot-product attention
        attn = torch.einsum("bhcl,bhcm->bhlm", q, k) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhlm,bhcm->bhcl", attn, v)
        out = out.reshape(B, C, L)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Downsample / Upsample
# ---------------------------------------------------------------------------
class Downsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full 1-D U-Net
# ---------------------------------------------------------------------------
class UNet1d(nn.Module):
    """1-D U-Net score network for ECG noise prediction.

    Args:
        in_channels: Number of ECG leads (default 12).
        base_channels: Channel width at first stage.
        channel_mults: Multipliers per encoder stage.
        num_classes: Number of electrolyte label classes.
        time_dim: Dimensionality of the timestep embedding.
    """

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_classes: int = 3,
        time_dim: int = 256,
    ):
        super().__init__()
        self.time_dim = time_dim
        cond_dim = time_dim

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Class embedding
        self.class_emb = nn.Embedding(num_classes, time_dim)

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = [base_channels]
        ch = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.encoder_blocks.append(ResBlock1d(ch, out_ch, cond_dim))
            channels.append(out_ch)
            self.downsamples.append(Downsample1d(out_ch))
            ch = out_ch

        # Bottleneck
        self.mid_block1 = ResBlock1d(ch, ch, cond_dim)
        self.mid_attn = Attention1d(ch)
        self.mid_block2 = ResBlock1d(ch, ch, cond_dim)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            skip_ch = channels.pop()
            self.upsamples.append(Upsample1d(ch))
            self.decoder_blocks.append(ResBlock1d(ch + skip_ch, out_ch, cond_dim))
            ch = out_ch

        # Output
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv1d(ch, in_channels, 3, padding=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Noisy ECG ``[B, C, L]``.
            t: Diffusion timestep ``[B]`` (integer or float).
            y: Class label ``[B]`` (long).

        Returns:
            Predicted noise ``[B, C, L]``.
        """
        cond = self.time_mlp(t) + self.class_emb(y)

        h = self.input_proj(x)
        skips = [h]
        for block, down in zip(self.encoder_blocks, self.downsamples):
            h = block(h, cond)
            skips.append(h)
            h = down(h)

        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        for block, up in zip(self.decoder_blocks, self.upsamples):
            h = up(h)
            s = skips.pop()
            # Handle potential size mismatch from downsampling
            if h.shape[-1] != s.shape[-1]:
                h = F.pad(h, (0, s.shape[-1] - h.shape[-1]))
            h = torch.cat([h, s], dim=1)
            h = block(h, cond)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
