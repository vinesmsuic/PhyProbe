#!/usr/bin/env python3
"""
Score all videos in a folder and print a ranked list.

Usage:
    python score_folder.py /path/to/videos/
    python score_folder.py --clip-sec 3.0 /path/to/videos/
    python score_folder.py --variant pespatial_paper /path/to/videos/
"""

import argparse
from pathlib import Path
import phyprobe

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}

parser = argparse.ArgumentParser()
parser.add_argument("folder")
parser.add_argument("--variant", default="pecore_paper")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--device", default=None)
parser.add_argument("--clip-sec", dest="clip_sec", type=float, default=None)
parser.add_argument("--batch-size", type=int, default=1)
args = parser.parse_args()

videos = sorted(
    p for p in Path(args.folder).rglob("*")
    if p.is_file() and p.suffix.lower() in VIDEO_EXTS
)
if not videos:
    raise SystemExit(f"No videos found under: {args.folder}")

model = phyprobe.PhyProbe(variant=args.variant, checkpoint=args.checkpoint, device=args.device)
scores = model.score_batch(videos, clip_sec=args.clip_sec, batch_size=args.batch_size)

results = sorted(zip(scores, videos), key=lambda x: x[0], reverse=True)
print(f"{'Score':>8}  Path")
print("─" * 70)
for s, p in results:
    print(f"{s:8.4f}  {p}")

raw = [s for s, _ in results]
print("─" * 70)
print(f"n={len(raw)}  mean={sum(raw)/len(raw):.4f}  min={min(raw):.4f}  max={max(raw):.4f}")
