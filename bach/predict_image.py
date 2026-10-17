import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image

# Load the model adapter from this project.
from models import resolve_adapter


def predict_image(image_path, model_path=None):
    image_path = Path(image_path)

    if model_path is None:
        model_path = Path(__file__).with_name("classifier.pkl")

    saved = joblib.load(model_path)
    classifier = saved["model"]

    # Load the same pretrained QuiltNet used during training.
    adapter = resolve_adapter(
        "openclip:wisdomik/QuiltNet-B-32",
        device="cuda"
    )

    # Use the exact same patch positions as prepare_bach.py.
    with Image.open(image_path) as original:
        image = original.convert("RGB")
        width, height = image.size

        if (width, height) != (2048, 1536):
            raise ValueError(
                "This model currently expects a 2048 x 1536 "
                "BACH-style microscopy image."
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

    with torch.no_grad():
        embeddings = adapter.encode_images(batch)

    # Combine 12 patch embeddings into one image embedding.
    image_embedding = np.mean(embeddings, axis=0)

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
