"""
PhyProbe — Physical Consistency Violation Scorer for generated videos.

Quick start::

    import phyprobe

    # Load a pretrained variant
    model = phyprobe.PhyProbe("pecore_paper")       # PE-Core (default)
    model = phyprobe.PhyProbe("pespatial_paper")    # PE-Spatial

    score = model.score("my_video.mp4")             # float in [0, 1]
    # 0 = no violation, 1 = strong physical consistency violation

    # Restrict to first 3 seconds (matches training window)
    score = model.score("my_video.mp4", clip_sec=3.0)

    # Load from a local checkpoint
    model = phyprobe.PhyProbe(checkpoint="/path/to/step_10000.pt")

    # Score multiple videos
    scores = model.score_batch(["a.mp4", "b.mp4", "c.mp4"])

    # Functional one-liner
    s = phyprobe.score("clip.mp4")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from ._hub import load_checkpoint, VARIANTS, DEFAULT_VARIANT
from ._model import build_model
from ._preprocess import load_video

__version__ = "0.1.0"
__all__ = ["PhyProbe", "score", "VARIANTS"]


class PhyProbe:
    """
    PhyProbe: physical consistency violation scorer for generated videos.

    Parameters
    ----------
    variant : str
        Pretrained variant (only ``"pecore_paper"`` is supported).
        Ignored when ``checkpoint`` is supplied.
    checkpoint : str or Path, optional
        Path to a local ``.pt`` checkpoint file.
    device : str or torch.device, optional
        Compute device.  Defaults to CUDA if available, otherwise CPU.
    compile_model : bool, optional
        Apply ``torch.compile`` for faster GPU throughput (PyTorch ≥ 2.0).
        Defaults to ``True`` on GPU.
    """

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        checkpoint: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        compile_model: Optional[bool] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        config_dict, state_dict = load_checkpoint(
            variant=variant,
            checkpoint_path=str(checkpoint) if checkpoint is not None else None,
            device=str(self.device),
        )

        self._frames: int = config_dict.get("cur_frames", 48)
        self._height: int = config_dict.get("cur_height", 480)
        self._width: int = config_dict.get("cur_width", 480)
        self._fps: float = float(config_dict.get("cur_fps", 16.0))

        self._model = build_model(config_dict, device=str(self.device))
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval().to(self.device)

        if compile_model is None:
            compile_model = self.device.type == "cuda" and hasattr(torch, "compile")
        if compile_model:
            self._model = torch.compile(self._model, mode="reduce-overhead")

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def score(
        self,
        video: Union[str, Path, torch.Tensor],
        fps: Optional[float] = None,
        clip_sec: Optional[float] = None,
    ) -> float:
        """
        Score a single video for physical consistency violations.

        The video is resampled to ``fps``, then ``num_frames`` are uniformly
        sampled across the full duration (or first ``clip_sec`` seconds).

        Parameters
        ----------
        video :
            Path to a video file **or** a pre-loaded float32 tensor
            ``[1, C, T, H, W]`` in ``[0, 1]``.
        fps : float, optional
            Resample the video to this frame rate before sampling.
            Defaults to the value stored in the model config (16 fps).
        clip_sec : float, optional
            If set, restrict sampling to the first ``clip_sec`` seconds of the
            resampled video.  Useful to match the training window
            (e.g. ``clip_sec=3.0`` at 16 fps → 48 frames).

        Returns
        -------
        float
            Raw model output clamped to [0, 1].
            0 = no violation, 1 = strong physical consistency violation.
        """
        if isinstance(video, (str, Path)):
            tensor = load_video(
                video,
                num_frames=self._frames,
                height=self._height,
                width=self._width,
                fps=fps if fps is not None else self._fps,
                clip_sec=clip_sec,
            ).to(self.device)
        else:
            tensor = video.to(self.device)

        use_cuda = self.device.type == "cuda"
        ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_cuda
            else torch.amp.autocast("cpu")
        )
        with ctx:
            raw = self._model(tensor).squeeze()

        return torch.clamp(raw.float(), 0.0, 1.0).item()

    @torch.inference_mode()
    def score_batch(
        self,
        videos: list[Union[str, Path]],
        fps: Optional[float] = None,
        clip_sec: Optional[float] = None,
        batch_size: int = 1,
    ) -> list[float]:
        """
        Score a list of video paths.

        Parameters
        ----------
        videos :
            List of video file paths.
        fps : float, optional
            Resample rate.  Defaults to the model config value (16 fps).
        clip_sec : float, optional
            Restrict sampling to the first ``clip_sec`` seconds.
        batch_size :
            Number of videos per forward pass.

        Returns
        -------
        list of float
            Raw scores clamped to [0, 1], one per input video in order.
        """
        _fps = fps if fps is not None else self._fps
        scores: list[float] = []
        use_cuda = self.device.type == "cuda"
        ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_cuda
            else torch.amp.autocast("cpu")
        )

        for i in range(0, len(videos), batch_size):
            chunk = videos[i : i + batch_size]
            tensors = [
                load_video(
                    v,
                    num_frames=self._frames,
                    height=self._height,
                    width=self._width,
                    fps=_fps,
                    clip_sec=clip_sec,
                )
                for v in chunk
            ]
            batch = torch.cat(tensors, dim=0).to(self.device)

            with ctx:
                raw = self._model(batch).squeeze(-1).float()  # [B]

            for v in torch.clamp(raw, 0.0, 1.0).tolist():
                scores.append(v)

        return scores


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

_default_model: Optional[PhyProbe] = None


def score(
    video: Union[str, Path],
    fps: Optional[float] = None,
    clip_sec: Optional[float] = None,
    device: Optional[str] = None,
) -> float:
    """
    Score a video with a cached default model.

    Example::

        import phyprobe
        s = phyprobe.score("clip.mp4")
        s = phyprobe.score("clip.mp4", clip_sec=3.0)
    """
    global _default_model
    if _default_model is None:
        _default_model = PhyProbe(device=device)
    return _default_model.score(video, fps=fps, clip_sec=clip_sec)
