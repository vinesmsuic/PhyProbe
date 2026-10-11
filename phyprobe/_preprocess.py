"""Video loading and preprocessing for PhyProbe inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

DEFAULT_FPS = 16.0


def load_video(
    path: Union[str, Path],
    num_frames: int = 48,
    height: int = 480,
    width: int = 480,
    fps: float = DEFAULT_FPS,
    clip_sec: Optional[float] = None,
) -> torch.Tensor:
    """
    Load a video and return a float32 tensor [1, C, T, H, W] in [0, 1].

    The video is resampled to ``fps`` first.  ``num_frames`` frames are then
    uniformly sampled from the resampled sequence.

    Args:
        path:       Path to the video file.
        num_frames: Number of frames to return (default: 48).
        height:     Output frame height in pixels (default: 480).
        width:      Output frame width in pixels (default: 480).
        fps:        Target sampling rate in frames per second (default: 16).
        clip_sec:   If set, only use the first ``clip_sec`` seconds of the
                    resampled video before uniform sampling.
                    Useful to match the training window (e.g. 3 s → 48 frames).

    Returns:
        Float32 tensor of shape [1, 3, T, H, W] with values in [0, 1].
    """
    try:
        from decord import VideoReader, cpu
    except ImportError:
        raise ImportError(
            "decord is required: pip install decord\n"
            "On some systems: conda install -c conda-forge decord"
        )
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required: pip install opencv-python")

    reader = VideoReader(str(path), ctx=cpu(0))
    total = len(reader)
    if total == 0:
        raise ValueError(f"No frames in video: {path}")

    native_fps = float(reader.get_avg_fps()) or 30.0
    duration_sec = total / native_fps

    # Resample to target fps
    n_resampled = max(1, int(round(duration_sec * fps)))
    resampled_indices = [
        min(total - 1, int(round((i / fps) * native_fps)))
        for i in range(n_resampled)
    ]

    # Optionally restrict to first clip_sec seconds
    if clip_sec is not None:
        max_frames = max(1, int(round(clip_sec * fps)))
        resampled_indices = resampled_indices[:max_frames]

    pool = resampled_indices
    n_pool = len(pool)

    # Uniformly sample num_frames from the pool; repeat last frame if too short
    if n_pool >= num_frames:
        positions = np.linspace(0, n_pool - 1, num_frames)
        sample_indices = [pool[int(round(p))] for p in positions]
    else:
        sample_indices = pool + [pool[-1]] * (num_frames - n_pool)

    frames = reader.get_batch(sample_indices).asnumpy()  # (T, H_orig, W_orig, 3) uint8
    del reader

    if frames.shape[1] != height or frames.shape[2] != width:
        frames = np.stack([
            cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR)
            for f in frames
        ])

    # (T, H, W, 3) uint8  →  (1, 3, T, H, W) float32 [0, 1]
    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    return tensor.unsqueeze(0)
