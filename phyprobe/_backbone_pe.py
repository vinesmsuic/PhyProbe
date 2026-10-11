"""PE-based backbone (PE-Core, PE-Spatial) + quality head."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _import_pe():
    try:
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms
        from core.vision_encoder.config import PE_TEXT_CONFIG
        return pe, transforms, PE_TEXT_CONFIG
    except ImportError:
        raise ImportError(
            "perception_models is required but not installed.\n"
            "Install it with:\n"
            "    pip install git+https://github.com/facebookresearch/perception_models.git\n"
            "or add the local clone to sys.path before importing phyprobe."
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


class _RoPE1D(nn.Module):
    """1D RoPE buffer (non-persistent — rebuilt deterministically on load)."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_len = 0

    def _ensure_cache(self, length: int, device: torch.device) -> None:
        if length <= self._cache_len:
            return
        pos   = torch.arange(length * 2, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq.to(device))
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("_cos", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("_sin", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)
        self._cache_len = length * 2

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        self._ensure_cache(int(positions.max().item()) + 1, x.device)
        cos = self._cos[:, :, positions, :]
        sin = self._sin[:, :, positions, :]
        return x * cos + _rotate_half(x) * sin


class _RoPEAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.q_proj    = nn.Linear(d_model, d_model, bias=False)
        self.k_proj    = nn.Linear(d_model, d_model, bias=False)
        self.v_proj    = nn.Linear(d_model, d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model)
        self.rope      = _RoPE1D(self.head_dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        B, T, D = x.shape

        def _h(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q = self.rope(_h(self.q_proj(x)), positions)
        K = self.rope(_h(self.k_proj(x)), positions)
        V = _h(self.v_proj(x))
        out = F.scaled_dot_product_attention(Q, K, V)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, D))


class _RoPETransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = _RoPEAttention(d_model, num_heads)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), positions)
        x = x + self.ff(self.norm2(x))
        return x


class _RoPETransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dim_feedforward: int):
        super().__init__()
        self.layers = nn.ModuleList([
            _RoPETransformerLayer(d_model, num_heads, dim_feedforward)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, positions)
        return x


def _make_sincos_embedding(max_len: int, dim: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32)
        * (-torch.log(torch.tensor(10000.0)) / dim)
    )
    pe = torch.zeros(1, max_len, dim)
    pe[0, :, 0::2] = torch.sin(pos * div)
    pe[0, :, 1::2] = torch.cos(pos * div[: dim // 2])
    return pe


class PEModel(nn.Module):
    """
    PE backbone (PE-Core or PE-Spatial) with a temporal-aggregation quality head.

    Input:  float32 tensor [B, C, T, H, W] with pixel values in [0, 1].
    Output: raw logit [B, 1]  (higher = more violated).
    """

    def __init__(
        self,
        pe_config: str = "PE-Core-G14-448",
        proj_head_mode: str = "stats",
        quality_head_dim: int = 256,
        input_frames: Optional[int] = None,
        pos_embed_type: str = "learned",
        device: str = "cpu",
    ):
        super().__init__()

        pe, _, PE_TEXT_CONFIG = _import_pe()

        self.proj_head_mode = proj_head_mode
        self.pos_embed_type = pos_embed_type
        self._device = torch.device(device)

        # PE-Spatial configs do not have a text encoder.
        self._is_vision_only = pe_config not in PE_TEXT_CONFIG
        if self._is_vision_only:
            self.pe_model = pe.VisionTransformer.from_config(pe_config, pretrained=True)
        else:
            self.pe_model = pe.CLIP.from_config(pe_config, pretrained=True)
        self.pe_model = self.pe_model.to(self._device).to(torch.bfloat16).eval()

        for p in self.pe_model.parameters():
            p.requires_grad_(False)

        # Detect feature dimension
        if hasattr(self.pe_model, "visual") and hasattr(self.pe_model.visual, "embed_dim"):
            self.feature_dim: int = self.pe_model.visual.embed_dim
        elif hasattr(self.pe_model, "embed_dim"):
            self.feature_dim = self.pe_model.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224, device=self._device, dtype=torch.bfloat16)
                out = (
                    self.pe_model(dummy)
                    if self._is_vision_only
                    else self.pe_model.encode_image(dummy)
                )
                if out.dim() == 3:
                    out = out.mean(1)
                self.feature_dim = out.shape[-1]

        # Build temporal aggregation modules
        if proj_head_mode == "stats":
            head_in_dim = self.feature_dim * 3
        elif proj_head_mode == "attn_pool":
            head_in_dim = self.feature_dim
            self.attn_query = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
            nn.init.trunc_normal_(self.attn_query, std=0.02)
            self.attn_pool = nn.MultiheadAttention(
                self.feature_dim, num_heads=16, batch_first=True
            )
        elif proj_head_mode == "transformer":
            head_in_dim = self.feature_dim
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.feature_dim,
                nhead=16,
                dim_feedforward=self.feature_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer_head = nn.TransformerEncoder(enc_layer, num_layers=2)
        elif proj_head_mode == "temporal_transformer":
            head_in_dim = self.feature_dim
            self.temporal_cls = nn.Parameter(torch.zeros(1, 1, self.feature_dim))
            nn.init.trunc_normal_(self.temporal_cls, std=0.02)
            if pos_embed_type == "rope":
                self.temporal_transformer = _RoPETransformerEncoder(
                    d_model=self.feature_dim, num_heads=16,
                    num_layers=2, dim_feedforward=self.feature_dim * 4,
                )
            else:
                max_t = (input_frames or 64) + 1
                if pos_embed_type == "sincos":
                    self.register_buffer(
                        "temporal_pos_embed", _make_sincos_embedding(max_t, self.feature_dim)
                    )
                else:  # learned
                    self.temporal_pos_embed = nn.Parameter(
                        torch.zeros(1, max_t, self.feature_dim)
                    )
                    nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=self.feature_dim,
                    nhead=16,
                    dim_feedforward=self.feature_dim * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.temporal_transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        else:  # "mlp"
            head_in_dim = self.feature_dim

        self.quality_head = nn.Sequential(
            nn.Linear(head_in_dim, quality_head_dim),
            nn.GELU(),
            nn.Linear(quality_head_dim, quality_head_dim // 2),
            nn.GELU(),
            nn.Linear(quality_head_dim // 2, 1),
        )

    def _encode_frame(self, frames: torch.Tensor) -> torch.Tensor:
        """[B*T, 3, H, W] bfloat16 → [B*T, D] float32."""
        if self._is_vision_only:
            out = self.pe_model(frames)
            if out.dim() == 3:
                out = out.mean(dim=1)
        else:
            out = self.pe_model.encode_image(frames)
        return out.float()

    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """[B, T, D] → [B, D] (or [B, 3D] for stats)."""
        if self.proj_head_mode == "mlp":
            return features.mean(dim=1)
        if self.proj_head_mode == "stats":
            return torch.cat(
                [
                    features.mean(dim=1),
                    features.std(dim=1, unbiased=False),
                    features.max(dim=1).values,
                ],
                dim=-1,
            )
        if self.proj_head_mode == "attn_pool":
            q = self.attn_query.expand(features.size(0), -1, -1)
            out, _ = self.attn_pool(q, features, features)
            return out.squeeze(1)
        if self.proj_head_mode == "transformer":
            return self.transformer_head(features).mean(dim=1)
        if self.proj_head_mode == "temporal_transformer":
            B, T, _ = features.shape
            cls = self.temporal_cls.expand(B, -1, -1)
            x = torch.cat([cls, features], dim=1)
            if self.pos_embed_type == "rope":
                positions = torch.arange(T + 1, device=x.device)
                x = self.temporal_transformer(x, positions)
            else:
                x = x + self.temporal_pos_embed[:, : T + 1]
                x = self.temporal_transformer(x)
            return x[:, 0]
        raise ValueError(f"Unknown proj_head_mode: {self.proj_head_mode!r}")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, C, T, H, W] float32 in [0, 1]
        Returns:
            [B, 1] raw logit
        """
        B, C, T, H, W = video.shape
        frames = video.transpose(1, 2).contiguous().view(B * T, C, H, W)
        frames = frames.to(torch.bfloat16)
        with torch.no_grad():
            feats = self._encode_frame(frames)  # [B*T, D]
        feats = feats.view(B, T, -1)
        pooled = self._pool_features(feats)
        return self.quality_head(pooled)
