"""Checkpoint loading and download utilities."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import torch

# Pretrained weights ship inside the package, as in LPIPS.  They hold only the
# quality head; the frozen PE backbone comes from the perception_models
# pretrained download.
_WEIGHTS_DIR = Path(__file__).parent / "weights"

# Checkpoints found here are used as-is (a source checkout's checkpoints/ folder).
_REPO_CHECKPOINTS_DIR = Path(__file__).parent.parent / "checkpoints"

# Google Drive downloads are cached here. Override with $PHYPROBE_CACHE.
_CACHE_DIR = Path(
    os.environ.get("PHYPROBE_CACHE")
    or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "phyprobe"
)

# Pretrained variant registry.
# Each entry maps a variant name → model config.
# "model_backbone" must match a key in _model._BACKBONE_REGISTRY.
#
# Sources are tried in order: bundled weights → local file → Hugging Face Hub
# → Google Drive.
VARIANTS: dict[str, dict] = {
    "pecore_paper": {
        "weights_file": "pecore_paper.pt",
        "hf_repo_id": "vinesmsuic/PhyProbe",
        "hf_filename": "pecore_paper.pt",
        "hf_revision": "b71eb8b91724e2954858730ae0ea39eed54cf426",
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

# Source keys that are not part of the model config.
_META_KEYS = {
    "weights_file", "hf_repo_id", "hf_filename", "hf_revision", "gdrive_file_id",
    "local_filename",
}


def _hf_download(repo_id: str, filename: str, revision: Optional[str] = None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download checkpoints: pip install huggingface_hub"
        )
    print(f"Fetching checkpoint from Hugging Face Hub: {repo_id}/{filename}")
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))


def _gdrive_download(file_id: str, dest: Path) -> Path:
    try:
        import gdown
    except ImportError:
        raise ImportError("gdown is required to download checkpoints: pip install gdown")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoint from Google Drive → {dest}")
    gdown.download(id=file_id, output=str(dest), quiet=False)
    return dest


def _get_local_checkpoint(variant_cfg: dict, download: bool = False) -> Optional[Path]:
    """Return path to the variant's checkpoint.

    Uses the bundled weights unless ``download`` is True, then falls back to a
    local file, the Hugging Face Hub and Google Drive.
    """
    weights_file = variant_cfg.get("weights_file")
    if not download and weights_file and (_WEIGHTS_DIR / weights_file).exists():
        return _WEIGHTS_DIR / weights_file

    local_filename = variant_cfg.get("local_filename")
    if not download and local_filename is not None:
        for d in (_REPO_CHECKPOINTS_DIR, _CACHE_DIR):
            if (d / local_filename).exists():
                return d / local_filename

    gdrive_file_id = variant_cfg.get("gdrive_file_id")
    has_gdrive = bool(gdrive_file_id and local_filename)

    hf_repo_id = variant_cfg.get("hf_repo_id")
    hf_filename = variant_cfg.get("hf_filename")
    if hf_repo_id and hf_filename:
        try:
            return _hf_download(hf_repo_id, hf_filename, variant_cfg.get("hf_revision"))
        except ImportError:
            raise
        except Exception as e:
            if not has_gdrive:
                raise
            print(f"Hugging Face Hub download failed ({e}); falling back to Google Drive.")

    if has_gdrive:
        dest = _CACHE_DIR / local_filename
        return dest if dest.exists() else _gdrive_download(gdrive_file_id, dest)
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
    download: bool = False,
) -> tuple[dict, dict]:
    """
    Load a PhyProbe checkpoint and return (config_dict, state_dict).

    Supply ``checkpoint_path`` to load a local .pt file directly, or leave
    it as None to use the variant's weights bundled with the package.  With
    ``download=True`` the weights are fetched from the Hugging Face Hub
    (falling back to Google Drive) instead.

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
        ckpt_path = _get_local_checkpoint(fallback_config, download=download)
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
