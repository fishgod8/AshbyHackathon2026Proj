"""Permanent web address for the workbench on Modal (both datasets, live Lookup).

    .venv-modal/bin/modal deploy deploy/modal_app.py     # build + deploy; prints the URL
    .venv-modal/bin/modal app stop pathology-assistant   # take it offline

Run the same deploy command again after the code or the data folders change (e.g. a new dataset from Mayumi:
rebuild ui/data_* first). The image bakes in the code, the data folders and the QuiltNet weights; no keys
are needed at run time (literature screening is done offline).

Cost: the app sleeps when idle (no charge) and wakes on the next visit (cold start, about half a minute to a
minute). For a demo day keep one container warm, then set it back:
    ASHBY_MIN_CONTAINERS=1 .venv-modal/bin/modal deploy deploy/modal_app.py
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
DATA = ["data_bach", "data_breakhis"]
MIN_CONTAINERS = int(os.environ.get("ASHBY_MIN_CONTAINERS", "0"))
APP_DIR = "/root/app"
QUILTNET = "hf-hub:wisdomik/QuiltNet-B-32"


def download_weights() -> None:
    """Fetch QuiltNet into the image so a cold start doesn't download it."""
    import open_clip

    open_clip.create_model_and_transforms(QUILTNET)


image = (
    modal.Image.debian_slim(python_version="3.13")
    # Same versions as ui/.venv; CPU-only torch keeps the image small.
    .pip_install("torch==2.14.1", "torchvision==0.29.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("gradio==6.30.0", "open_clip_torch==3.3.0", "timm==1.0.30", "scikit-learn==1.9.1", "joblib==1.6.0",
                 "pandas==3.0.6", "numpy==2.5.4", "pillow==12.3.0", "huggingface_hub==2.2.0", "fastapi==0.143.3",
                 "openslide-python==1.4.6", "openslide-bin==4.0.1.2")  # Slide view reads the clicked patch from the slide
    .run_function(download_weights)
    .add_local_dir(ROOT / "src", f"{APP_DIR}/src", ignore=["**/__pycache__/**"])
    .add_local_file(ROOT / "ui/app.py", f"{APP_DIR}/ui/app.py")
    .add_local_file(ROOT / "ui/main.py", f"{APP_DIR}/ui/main.py")
    .add_local_file(ROOT / "ui/assets/favicon.png", f"{APP_DIR}/ui/assets/favicon.png")
)
for name in DATA:  # data last: changing it doesn't rebuild the layers above
    image = image.add_local_dir(ROOT / "ui" / name, f"{APP_DIR}/ui/{name}", ignore=["decisions.json"])

app = modal.App("pathology-assistant", image=image)


@app.function(cpu=2, memory=6144, timeout=600, scaledown_window=20 * 60,
              min_containers=MIN_CONTAINERS, max_containers=1)  # one container: Gradio sessions live in memory
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import sys

    import gradio as gr
    from fastapi import FastAPI

    sys.path[:0] = [f"{APP_DIR}/ui", APP_DIR]
    sys.argv = ["main.py", "--data", *DATA, "--live", "--session-decisions"]
    import main  # builds one app.py per dataset and the page with the dataset switch

    return gr.mount_gradio_app(FastAPI(), main.site, path="/", **main.launch_options())
