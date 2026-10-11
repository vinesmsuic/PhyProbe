#!/usr/bin/env python3
"""
Gradio web demo for PhyProbe.

Upload one video to score it, or two videos to compare which one violates
physics more. Lets the user pick how frames are sampled: uniformly across
the full video, or only from the first N seconds.

Usage:
    python gradio_demo.py
    python gradio_demo.py --port 7860
"""

from __future__ import annotations

import argparse

import gradio as gr

import phyprobe

UNIFORM = "Uniform sampling (full video)"
FIRST_N = "First N seconds only"

_model: phyprobe.PhyProbe | None = None


def _get_model(device: str | None) -> phyprobe.PhyProbe:
    global _model
    if _model is None:
        _model = phyprobe.PhyProbe(device=device or None)
    return _model


def _verdict(score: float) -> str:
    if score < 0.2:
        return "🟢 no to low violation"
    elif score < 0.6:
        return "🟡 moderate violation"
    else:
        return "🔴 high violation"


def run(video_a, video_b, sampling_mode, clip_sec, device):
    if video_a is None:
        raise gr.Error("Upload at least one video (Video A).")

    clip = float(clip_sec) if sampling_mode == FIRST_N else None
    model = _get_model(device.strip() if device else None)

    videos = [video_a] + ([video_b] if video_b else [])
    scores = model.score_batch(videos, clip_sec=clip)

    sampling_desc = (
        f"first {clip:g} s, uniformly sampled" if clip is not None
        else "full video, uniformly sampled"
    )
    lines = [f"_Sampling: {sampling_desc} → {model._frames} frames @ {model._fps:g} fps_", ""]
    lines.append(f"**Video A**: `{scores[0]:.4f}`  —  {_verdict(scores[0])}")
    if len(scores) == 2:
        lines.append(f"**Video B**: `{scores[1]:.4f}`  —  {_verdict(scores[1])}")
        delta = scores[0] - scores[1]
        more = "Video A" if delta > 0 else "Video B"
        lines.append(f"\n**Δ (A − B)** = `{delta:+.4f}`  →  more violated: **{more}**")
    lines.append("\n_Score range: 0 = no violation, 1 = strong physical consistency violation._")
    return "\n\n".join(lines)


def build_demo(default_device: str | None = None) -> gr.Blocks:
    with gr.Blocks(title="PhyProbe Demo") as demo:
        gr.Markdown(
            "# 🧪 PhyProbe — Physical Consistency Violation Scorer\n"
            "Upload one video to score it, or two videos to compare which one "
            "violates physics more. Higher score → more severe violation."
        )

        with gr.Row():
            video_a = gr.Video(label="Video A (required)")
            video_b = gr.Video(label="Video B (optional — for comparison)")

        with gr.Row():
            sampling_mode = gr.Radio(
                choices=[UNIFORM, FIRST_N],
                value=UNIFORM,
                label="Frame sampling",
            )
            clip_sec = gr.Number(
                value=2.0, precision=1, minimum=0.5, maximum=10.0, step=0.5,
                label="Clip length (seconds)", visible=False,
            )

        sampling_mode.change(
            lambda mode: gr.update(visible=(mode == FIRST_N)),
            inputs=sampling_mode,
            outputs=clip_sec,
        )

        device = gr.Textbox(
            value=default_device or "",
            label="Device override (optional, e.g. 'cuda' or 'cpu' — default: auto)",
        )

        submit = gr.Button("Score", variant="primary")
        output = gr.Markdown()

        submit.click(
            run,
            inputs=[video_a, video_b, sampling_mode, clip_sec, device],
            outputs=output,
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda",
                        help="Device override (e.g. cuda, cpu — default: auto).")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link.")
    args = parser.parse_args()

    # Pre-load model (downloads checkpoint if missing).
    print("Loading PhyProbe model...")
    _get_model(args.device)
    print("Model loaded.")

    demo = build_demo(default_device=args.device)
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
