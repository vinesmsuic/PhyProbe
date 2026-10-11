"""
Backbone-agnostic model factory.

Adding a new backbone:
    1. Create a module _backbone_<name>.py with a nn.Module that accepts
       (config_dict, device) and exposes a standard forward(video) → [B,1].
    2. Register it in _BACKBONE_REGISTRY below.
    3. Add variant entries to _hub.VARIANTS.
"""

from __future__ import annotations

import torch.nn as nn


def _build_pe(config_dict: dict, device: str) -> nn.Module:
    from ._backbone_pe import PEModel

    return PEModel(
        pe_config=config_dict.get("pecore_variant", "PE-Core-G14-448"),
        proj_head_mode=config_dict.get("pecore_proj_head_mode", "stats"),
        quality_head_dim=config_dict.get("quality_head_dim", 256),
        input_frames=config_dict.get("cur_frames", 48),
        pos_embed_type=config_dict.get("pecore_pos_embed_type", "learned"),
        device=device,
    )


# Map model_backbone string → builder callable
_BACKBONE_REGISTRY: dict[str, callable] = {
    "pecore_proj": _build_pe,
    # Future entries:
    # "vjepa2_proj": _build_vjepa2,
    # "dinov3_proj": _build_dinov3,
}


def build_model(config_dict: dict, device: str = "cpu") -> nn.Module:
    """
    Instantiate the appropriate model from a config dict.

    The ``model_backbone`` key in ``config_dict`` determines which backbone
    class is used.  All other keys are passed through to the backbone builder.
    """
    backbone = config_dict.get("model_backbone", "pecore_proj")
    if backbone not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unsupported backbone: {backbone!r}. "
            f"Available: {list(_BACKBONE_REGISTRY)}"
        )
    return _BACKBONE_REGISTRY[backbone](config_dict, device)
