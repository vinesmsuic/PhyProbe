"""Checkpoint download and loading utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import torch

# Local checkpoints directory (phyprobe/checkpoints/ next to the package root).
_CHECKPOINTS_DIR = Path(__file__).parent.parent / "checkpoints"

# Pretrained variant registry.
# Each entry maps a variant name → model config.
# "backbone" must match a key in _model._BACKBONE_REGISTRY.
#
# Download sources are tried in order: local file → Hugging Face Hub → Google Drive.
VARIANTS: dict[str, dict] = {
    "pecore_paper": {
        # TODO: set to the Hugging Face model repo once the weights are uploaded,
        # e.g. "vinesmsuic/PhyProbe". While None, the Hub is skipped.
        "hf_repo_id": None,
        "hf_filename": "pecore_paper.pt",
        "hf_revision": None,  # pin a commit hash for reproducible downloads
        "gdrive_file_id": "1SdGpDY3TDeML5KRUviEgJ2z7yEw-5Vw2",
        "local_filename": "pecore_paper.pt",
        "model_backbone": "pecore_proj",
        "pecore_variant": "PE-Core-G14-448",
        "pecore_proj_head_mode": "stats",
        "quality_head_dim": 256,
        "cur_frames": 48,
        "cur_height": 480,
        "cur_width": 480,
    },
}

DEFAULT_VARIANT = "pecore_paper"

# Download/source keys that are not part of the model config.
_META_KEYS = {"hf_repo_id", "hf_filename", "hf_revision", "gdrive_file_id", "local_filename"}


def _hf_download(repo_id: str, filename: str, revision: Optional[str] = None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download checkpoints: pip install huggingface_hub"
        )
    print(f"Downloading checkpoint from Hugging Face Hub: {repo_id}/{filename}")
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))


def _gdrive_download(file_id: str, dest: Path) -> Path:
    try:
        import gdown
    except ImportError:
        raise ImportError("gdown is required to download checkpoints: uv add gdown")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoint from Google Drive → {dest}")
    gdown.download(id=file_id, output=str(dest), quiet=False)
    return dest


def _get_local_checkpoint(variant_cfg: dict) -> Optional[Path]:
    """Return path to the variant's checkpoint, downloading it if needed."""
    local_filename = variant_cfg.get("local_filename")
    if local_filename is not None:
        local_path = _CHECKPOINTS_DIR / local_filename
        if local_path.exists():
            return local_path
    else:
        local_path = None

    hf_repo_id = variant_cfg.get("hf_repo_id")
    hf_filename = variant_cfg.get("hf_filename")
    if hf_repo_id and hf_filename:
        return _hf_download(hf_repo_id, hf_filename, variant_cfg.get("hf_revision"))

    gdrive_file_id = variant_cfg.get("gdrive_file_id")
    if gdrive_file_id and local_path is not None:
        return _gdrive_download(gdrive_file_id, local_path)
    return None


def _safe_torch_load(path: Path, device: str):
    """Load a checkpoint without executing arbitrary pickled code."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError as e:
        raise RuntimeError(
            f"Refusing to load {path}: it contains Python objects that are not allowed "
            "in a weights-only load, which could run arbitrary code.\n"
            "If you trust this file, re-save it with only tensors and plain types, e.g.:\n"
            "    ckpt = torch.load(path, weights_only=False)\n"
            "    torch.save({'config': dict(ckpt['config']),\n"
            "                'model_state_dict': ckpt['model_state_dict']}, new_path)\n"
            f"Original error: {e}"
        ) from e


def load_checkpoint(
    variant: str = DEFAULT_VARIANT,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> tuple[dict, dict]:
    """
    Load a PhyProbe checkpoint and return (config_dict, state_dict).

    Supply ``checkpoint_path`` to load a local .pt file directly, or leave
    it as None to use the variant's checkpoint (downloading it from the
    Hugging Face Hub or Google Drive automatically if not yet present).

    Checkpoints are always loaded with ``weights_only=True``.
    """
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        fallback_config = VARIANTS.get(variant, {})
    else:
        if variant not in VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Choose from: {list(VARIANTS)}"
            )
        fallback_config = VARIANTS[variant]
        ckpt_path = _get_local_checkpoint(fallback_config)
        if ckpt_path is None:
            raise RuntimeError(
                f"No checkpoint found for variant {variant!r} and no download source configured."
            )

    ckpt = _safe_torch_load(ckpt_path, device)

    if isinstance(ckpt, dict) and "config" in ckpt:
        config_dict = ckpt["config"]
        weights_key = (
            "ema_model_state_dict"
            if "ema_model_state_dict" in ckpt
            else "model_state_dict"
        )
        state_dict = ckpt[weights_key]
    else:
        # Plain state dict — reconstruct config from variant defaults
        config_dict = {k: v for k, v in fallback_config.items() if k not in _META_KEYS}
        state_dict = ckpt

    # Strip DDP / torch.compile key prefixes
    state_dict = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in state_dict.items()
    }

    return config_dict, state_dict
