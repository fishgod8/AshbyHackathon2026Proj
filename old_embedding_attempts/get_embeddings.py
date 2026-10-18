#!/usr/bin/env python3
"""Student quickstart for the pathology embedding server.

See STUDENT_GUIDE.md for the full walkthrough (what /embed vs /classify
return, troubleshooting, using your own images/prompts) - this script is the
runnable version of that walkthrough.

Fill in ENDPOINT and API_KEY below (the organizers will give you both), then:

    pip install requests pillow   # pillow is only needed for the placeholder
                                   # tiles this demo generates - see note below
    python example_student_usage.py

This is everything you need on your own machine: no CONCH/MUSK, no torch, no
GPU. Those all run on the server; you're just sending it HTTP requests, the
same way client.py does.
"""

import base64
import io
import json
import csv

import requests
from pathlib import Path
import numpy as np

CRC_DATASET_PATH = "/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/"
PATH_GEN_DATASET = "/projects/bfqi/shared/datasets/pathgen_dataset/"
PATH_MMU_DATASET = "/projects/bfqi/shared/datasets/pathmmu/"
QUILT_DATASET = "/work/nvme/bfqi/shared/datasets/quilt/"
# conch
ENDPOINT = "http://gpub055.delta.ncsa.illinois.edu:8000"  # from the organizers, or connection.json
API_KEY = "pathology-a635d8d6257ec250cba411a7d0015522"  # from the organizers, or connection.json
# musk: http://gpub055.delta.ncsa.illinois.edu:8001  key: pathology-b3c1ee0440811e42e9e88ee01fd7b45e
# quiltnet: http://gpub055.delta.ncsa.illinois.edu:8002  key: pathology-fb3527e02553d669c4394ab555428f06
# plip: http://gpub055.delta.ncsa.illinois.edu:8003  key: pathology-596428c144e2ac806ff2eefecad01831

OUTPUT = Path("crc_embeddings.csv")

def image_bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

def get_tile_images(path: str) -> list:
    data_dir = Path(
        path
    )
    return [i for i in data_dir.glob("**/*.tif")]


def load_your_own_image(path: str) -> str:
    """This is all you need for real data - no PIL, no preprocessing.
    The server handles decoding, resizing, and normalization."""
    with open(path, "rb") as f:
        return image_bytes_to_b64(f.read())


def make_placeholder_tiles_b64():
    """Three differently-colored synthetic stand-ins, so /classify has more
    than one image to tell apart. These are NOT real pathology images - swap
    this for [load_your_own_image(p) for p in ["tile1.png", "tile2.png"]]
    once you have real data."""
    from PIL import Image

    colors = [(180, 120, 160), (230, 210, 220), (90, 40, 60)]
    tiles = []
    for color in colors:
        tile = Image.new("RGB", (224, 224), color)
        buf = io.BytesIO()
        tile.save(buf, format="PNG")
        tiles.append(image_bytes_to_b64(buf.getvalue()))
    return tiles


def check_health():
    """Fail fast with a clear message instead of a confusing stack trace if
    the endpoint is wrong, the server isn't up yet, or the API key is stale."""
    try:
        response = requests.get(f"{ENDPOINT}/health", timeout=10)
    except requests.exceptions.ConnectionError as e:
        raise SystemExit(
            f"Could not reach {ENDPOINT} - check ENDPOINT matches what the "
            f"organizers gave you, and that the server is still running ({e})"
        ) from e
    if not response.ok:
        raise SystemExit(f"Server at {ENDPOINT} responded with HTTP {response.status_code}: {response.text}")
    info = response.json()
    print(f"Connected. Server is running model={info['model']} on device={info['device']}.\n")


def _post(path, payload):
    response = requests.post(
        f"{ENDPOINT}{path}",
        json=payload,
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=60,
    )
    if response.status_code == 401:
        raise SystemExit("Got HTTP 401 - check API_KEY matches what the organizers gave you.")
    response.raise_for_status()
    return response.json()


def embed(images_b64):
    return _post("/embed", {"images": images_b64})["embeddings"]


def classify(images_b64, prompts):
    return _post("/classify", {"images": images_b64, "prompts": prompts})["results"]


def main():
    check_health()
    image_paths = get_tile_images(CRC_DATASET_PATH)
    print(f"Found {len(image_paths)} images to embed.")

    BATCH_SIZE = 64
    all_embeddings = []

    with OUTPUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "embedding"])

        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i : i + BATCH_SIZE]
            batch_b64 = [load_your_own_image(p) for p in batch_paths]

            # Single inference pass per batch
            batch_embeddings = embed(batch_b64)
            all_embeddings.extend(batch_embeddings)

            # Write rows immediately
            for path, emb in zip(batch_paths, batch_embeddings):
                writer.writerow([str(path), path.parent.name, emb])

            processed = min(i + BATCH_SIZE, len(image_paths))
            print(f"Processed {processed}/{len(image_paths)} tiles")

    # Save complete JSON
    with open("crc_embeddings.json", "w", encoding="utf-8") as file:
        json.dump(all_embeddings, file)

    print("Embedding generation complete!")

    print(f"/embed: got {len(all_embeddings)} embeddings, each {len(all_embeddings[0])}-dimensional.")

    # prompts = {
    #     "tumor": "a histopathology image of tumor tissue",
    #     "stroma": "a histopathology image of stroma",
    #     "necrosis": "a histopathology image of necrotic tissue",
    # }
    # results = classify(tiles, prompts)
    # for i, r in enumerate(results):
    #     print(f"/classify tile {i}: predicted '{r['predicted_label']}' (score={r['score']:.3f})")


if __name__ == "__main__":
    main()
