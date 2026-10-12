#!/usr/bin/env python3
"""
Compare two videos side-by-side.  The higher-scoring video has more violations.

Usage:
    python compare_videos.py video_a.mp4 video_b.mp4
    python compare_videos.py --clip-sec 3.0 video_a.mp4 video_b.mp4
"""

import argparse
import phyprobe

parser = argparse.ArgumentParser()
parser.add_argument("video_a")
parser.add_argument("video_b")
parser.add_argument("--variant", default="pecore_paper")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--device", default=None)
parser.add_argument("--clip-sec", dest="clip_sec", type=float, default=None)
args = parser.parse_args()

model = phyprobe.PhyProbe(variant=args.variant, checkpoint=args.checkpoint, device=args.device)
sa, sb = model.score_batch([args.video_a, args.video_b], clip_sec=args.clip_sec)

print(f"Video A: {sa:.4f}  {args.video_a}")
print(f"Video B: {sb:.4f}  {args.video_b}")
delta = sa - sb
more = args.video_a if delta > 0 else args.video_b
print(f"Delta  : {delta:+.4f}  →  more violated: {more}")
