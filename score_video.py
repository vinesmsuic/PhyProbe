#!/usr/bin/env python3
"""
Score a single video for physical consistency violations.

Usage:
    python score_video.py video.mp4
    python score_video.py --clip-sec 3.0 video.mp4
    python score_video.py --checkpoint path/to/model.pt video.mp4
"""

import argparse
import phyprobe

parser = argparse.ArgumentParser()
parser.add_argument("video", help="Video file to score.")
parser.add_argument("--variant", default="pecore_paper")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--device", default=None)
parser.add_argument("--clip-sec", dest="clip_sec", type=float, default=None,
                    help="Only use the first N seconds (default: full video).")
args = parser.parse_args()

model = phyprobe.PhyProbe(variant=args.variant, checkpoint=args.checkpoint, device=args.device)
score = model.score(args.video, clip_sec=args.clip_sec)
print(f"PhyProbe score: {score:.4f}  (0 = no violation, 1 = violation)  [{args.video}]")
