import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model_identity import load_checkpoint

# Load the model adapter from this project.
try:
    from .models import resolve_adapter
except ImportError:  # Support ``python bach/predict_image.py``.
    from models import resolve_adapter


EXPECTED_EMBEDDING_MODEL = "openclip:wisdomik/QuiltNet-B-32"
EXPECTED_AGGREGATION = "mean of 12 patches"
EXPECTED_FEATURE_DIMENSION = 512


def _default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def load_classifier(model_path):
    """Load and validate a trusted local Mayumi checkpoint once per path."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {path}")
    saved, _ = load_checkpoint(path)
    classifier = saved["model"]
    model_classes = [str(value) for value in classifier.classes_]
    saved_classes = [str(value) for value in saved["classes"]]
    if model_classes != saved_classes:
        raise ValueError("Saved classes do not match classifier.classes_")
    if saved["embedding_model"] != EXPECTED_EMBEDDING_MODEL:
        raise ValueError(f"Unexpected embedding model: {saved['embedding_model']!r}")
    if saved["aggregation"] != EXPECTED_AGGREGATION:
        raise ValueError(f"Unexpected embedding aggregation: {saved['aggregation']!r}")
    if int(classifier.n_features_in_) != EXPECTED_FEATURE_DIMENSION:
        raise ValueError(
            f"Classifier expects {classifier.n_features_in_} features, not "
            f"{EXPECTED_FEATURE_DIMENSION}"
        )
    return classifier


@lru_cache(maxsize=2)
def load_embedding_adapter(device):
    """Cache the expensive QuiltNet adapter for repeated image inference."""
    return resolve_adapter(EXPECTED_EMBEDDING_MODEL, device=device)


def predict_image(image_path, model_path=None, device=None, embedding_batch_size=None):
    image_path = Path(image_path)

    if model_path is None:
        model_path = Path(__file__).with_name("classifier.pkl")

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    classifier = load_classifier(str(Path(model_path).resolve()))

    # Load the same pretrained QuiltNet used during training.
    device = device or _default_device()
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    adapter = load_embedding_adapter(device)

    # Use the exact same patch positions as prepare_bach.py.
    with Image.open(image_path) as original:
        image = original.convert("RGB")
        width, height = image.size

        if width < 224 or height < 224:
            raise ValueError(
                "Image must be at least 224 x 224 pixels."
            )

        patches = []

        for row in range(3):
            for col in range(4):
                x = round((width - 224) * col / 3)
                y = round((height - 224) * row / 2)

                patch = image.crop((x, y, x + 224, y + 224))
                patches.append(adapter.preprocess(patch))

    # Generate the 12 QuiltNet embeddings.
    batch = torch.stack(patches)

    # A 12-image QuiltNet batch can exceed memory on CPU-only machines. Small
    # batches produce the same per-patch embeddings while reducing peak RAM.
    if embedding_batch_size is None:
        embedding_batch_size = len(batch) if device.startswith("cuda") else 2
    if isinstance(embedding_batch_size, bool) or not isinstance(embedding_batch_size, int) or embedding_batch_size < 1:
        raise ValueError("embedding_batch_size must be a positive integer")
    with torch.no_grad():
        embeddings = np.concatenate(
            [adapter.encode_images(batch[start:start + embedding_batch_size])
             for start in range(0, len(batch), embedding_batch_size)],
            axis=0,
        )

    # Combine 12 patch embeddings into one image embedding.
    image_embedding = np.mean(embeddings, axis=0)
    if image_embedding.shape != (EXPECTED_FEATURE_DIMENSION,):
        raise ValueError(
            f"QuiltNet produced {image_embedding.shape}, expected "
            f"({EXPECTED_FEATURE_DIMENSION},) after aggregation"
        )

    # Run the classifier.
    scores = classifier.predict_proba(
        image_embedding.reshape(1, -1)
    )[0]

    classes = list(classifier.classes_)

    return {
        "predicted_class": classes[int(np.argmax(scores))],
        "scores": {
            label: float(score)
            for label, score in zip(classes, scores)
        },
        "embedding": image_embedding.tolist()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    result = predict_image(args.image)

    # Print a compact response (omit the 512-number embedding).
    print(json.dumps({
        "predicted_class": result["predicted_class"],
        "scores": result["scores"]
    }, indent=2))
