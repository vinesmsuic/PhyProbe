"""
phyprobe-score CLI.

Videos are always resampled to 16 fps; frames are uniformly sampled across
the full duration by default.

Usage:
    phyprobe-score video.mp4
    phyprobe-score --variant pespatial_paper video.mp4
    phyprobe-score --checkpoint model.pt video.mp4
    phyprobe-score --clip-sec 3.0 video.mp4   # first 3 s only
    phyprobe-score --folder /path/to/videos/
    phyprobe-score --raw video.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}


def _find_videos(folder: str) -> list[Path]:
    return sorted(
        p for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser(
        description="Score videos for physical consistency violations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", nargs="?", default=None,
                        help="Video file to score.")
    parser.add_argument("--folder", default=None,
                        help="Score all videos found recursively under this folder.")
    parser.add_argument("--variant", default="pecore_paper",
                        help="Pretrained variant: pecore_paper (default) or pespatial_paper.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a local .pt checkpoint file.")
    parser.add_argument("--device", default=None,
                        help="Compute device (default: cuda if available, else cpu).")
    parser.add_argument("--raw", action="store_true",
                        help="Output raw unclamped logit instead of the clamped [0, 1] score.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Resample video to this frame rate before sampling "
                             "(default: 16).")
    parser.add_argument("--clip-sec", dest="clip_sec", type=float, default=None,
                        help="Only use the first N seconds of the resampled video before "
                             "uniform sampling.  Default: use full video duration.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1,
                        help="Videos per forward pass when scoring a folder (default: 1).")
    args = parser.parse_args()

    if args.video is None and args.folder is None:
        parser.error("Provide a video path or --folder.")
    if args.video and args.folder:
        parser.error("Provide either a video path or --folder, not both.")

    from . import PhyProbe

    model = PhyProbe(
        variant=args.variant,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    if args.video:
        if args.raw:
            import torch
            # Bypass clamp: call _model directly for the raw logit
            from ._preprocess import load_video
            tensor = load_video(
                args.video,
                num_frames=model._frames, height=model._height,
                width=model._width, fps=args.fps or model._fps,
                clip_sec=args.clip_sec,
            ).to(model.device)
            with torch.inference_mode():
                s = model._model(tensor).squeeze().float().item()
            label = "logit"
        else:
            s = model.score(args.video, fps=args.fps, clip_sec=args.clip_sec)
            label = "clamped"
        print(f"{s:.4f}  ({label})  {args.video}")
    else:
        videos = _find_videos(args.folder)
        if not videos:
            sys.exit(f"No video files found under: {args.folder}")

        print(f"Scoring {len(videos)} video(s) in {args.folder}\n")
        scores_list = model.score_batch(
            videos,
            fps=args.fps,
            clip_sec=args.clip_sec,
            batch_size=args.batch_size,
        )
        results = sorted(zip(scores_list, videos), key=lambda x: x[0], reverse=True)

        print(f"{'Score':>8}  Path")
        print("─" * 70)
        for s, p in results:
            print(f"{s:8.4f}  {p}")

        raw_scores = [s for s, _ in results]
        print("─" * 70)
        print(
            f"n={len(raw_scores)}  "
            f"mean={sum(raw_scores)/len(raw_scores):.4f}  "
            f"min={min(raw_scores):.4f}  "
            f"max={max(raw_scores):.4f}"
        )


if __name__ == "__main__":
    main()
