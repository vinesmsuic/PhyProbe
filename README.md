# PhyProbe

Physical consistency violation scorer for AI-generated videos.

[Project page](https://research.nvidia.com/labs/cosmos-lab/phyprobe/) · [Paper](https://arxiv.org/abs/2609.38377)

Higher score → more severe physical consistency violations.

---

## Installation

```bash
pip install phyprobe
```

**Required:** `perception_models` (not on PyPI — install from source):

```bash
pip install git+https://github.com/facebookresearch/perception_models.git
```

---

## Pretrained variants

| Variant | Backbone | Notes |
|---|---|---|
| `pecore_paper` (default) | PE-Core-G14-448 | Reproduces paper results |

The trained head weights (~4 MB) ship with the package. The frozen PE-Core
backbone is downloaded by `perception_models` on first use. The weights are
also on the [Hugging Face Hub](https://huggingface.co/vinesmsuic/PhyProbe);
pass `download=True` (CLI: `--download`) to fetch them from there instead.

---

## Python API

```python
import phyprobe

# Load a pretrained variant
model = phyprobe.PhyProbe()                     # pecore_paper (default)

# Score a video  →  float in [0, 1]
# Videos are always resampled to 16 fps; frames are uniformly sampled
# across the full duration by default.
score = model.score("video.mp4")

# Restrict to the first 3 seconds before uniform sampling
score = model.score("video.mp4", clip_sec=3.0)

# Score multiple videos
scores = model.score_batch(["a.mp4", "b.mp4", "c.mp4"])
scores = model.score_batch(["a.mp4", "b.mp4"], clip_sec=3.0)

# One-liner
score = phyprobe.score("video.mp4")
```

**Score interpretation:**  
`0.0` = no physical consistency violation detected  
`1.0` = strong violation

### Load from a local checkpoint

```python
model = phyprobe.PhyProbe(checkpoint="/path/to/model.pt")
```

---

## Command-line interface

```bash
# Score a single video
phyprobe-score video.mp4

# Restrict to first 3 seconds
phyprobe-score --clip-sec 3.0 video.mp4

# Score all videos in a folder
phyprobe-score --folder /path/to/videos/

# Raw unclamped logit
phyprobe-score --raw video.mp4

# python -m also works
python -m phyprobe video.mp4
```

---

## Example scripts

```bash
python score_video.py video.mp4
python compare_videos.py video_a.mp4 video_b.mp4
python score_folder.py /path/to/videos/
```

---

## Gradio demo

Launch an interactive web UI to upload videos and score them in the browser:

```bash
pip install gradio
python gradio_demo.py
# open http://localhost:7860
```

Options:

```bash
python gradio_demo.py --port 7860          # custom port
python gradio_demo.py --device cpu         # force CPU
python gradio_demo.py --share              # create a public link (72 h)
```

Upload one video to score it, or two videos to compare which one violates physics more.

---

## Google Colab

`phyprobe_colab.ipynb` is a self-contained notebook you can upload directly to Google Colab — no local installation required.

1. Open [Google Colab](https://colab.research.google.com/) → **File → Upload notebook** → select `phyprobe_colab.ipynb`
2. Switch to a GPU runtime: **Runtime → Change runtime type → T4 GPU**
3. **Run All** (`Ctrl+F9`)

The notebook installs the only external dependency (`perception_models`), downloads the pretrained checkpoint automatically, and lets you upload videos via a file picker widget. An optional cell also launches the Gradio UI with a public share link.

---

## Adding a new backbone

1. Create `phyprobe/_backbone_<name>.py` with an `nn.Module` that accepts
   `(config_dict, device)` and returns `[B, 1]` from `forward(video)`.
2. Register a builder in `phyprobe/_model.py` under `_BACKBONE_REGISTRY`.
3. Add variant entries in `phyprobe/_hub.py` under `VARIANTS`.

---

## Citation

If you use PhyProbe in your research, please cite:

```bibtex
@misc{ku2026phyproberethinkingphysicalconsistency,
      title={PhyProbe: Rethinking Physical Consistency Evaluation in Generated Videos}, 
      author={Max Ku and Jiaojiao Fan and Zekun Hao and Francesco Ferroni and Heng Wang and Wenhu Chen and Ming-Yu Liu and Prithvijit Chattopadhyay},
      year={2026},
      eprint={2609.38377},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.38377}, 
}
```
