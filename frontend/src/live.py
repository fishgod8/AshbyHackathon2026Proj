"""Live analysis of an uploaded tile, for the UI's upload tab (and later HF Spaces).

Same path as the precomputed cases: QuiltNet embedding -> analyze_case() with
the same adapters/stubs, so an uploaded tile gets exactly the result format the
UI already renders. Embedding mirrors the official openclip adapter
(external/ashby-hackathon-2026/models.py: encode_image + L2 normalize), so it is
comparable with the gallery embeddings computed by run_embeddings.py.

Quick check:
    python -m src.live --gallery embeddings/quiltnet_train_subset.csv data/val_subset/TUM/<tile>.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.analyze import analyze_case, load_embeddings
from src.stubs import StubClassifier, StubRetriever

QUILTNET = "hf-hub:wisdomik/QuiltNet-B-32"


class LiveAnalyzer:
    def __init__(self, gallery_csv: Path, device: str = "cpu", seed: int = 0) -> None:
        import open_clip
        import torch

        self.torch, self.device = torch, device
        paths, labels, emb = load_embeddings(gallery_csv)
        # Relative tile paths (deployed bundle) are relative to the CSV's folder.
        paths = [p if Path(p).is_absolute() else str((gallery_csv.parent / p).resolve()) for p in paths]
        self.clf = StubClassifier(emb, labels, seed=seed)
        self.retriever = StubRetriever(emb, labels, paths)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(QUILTNET)
        self.model = self.model.to(device).eval()

    def embed(self, image: Image.Image) -> np.ndarray:
        batch = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            emb = self.model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].float().cpu().numpy()

    def analyze(self, image: Image.Image) -> dict:
        """Uploaded image -> result dict (gallery image paths still point at the source tiles)."""
        return analyze_case(self.embed(image), self.clf, self.retriever)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gallery", type=Path, required=True)
    p.add_argument("image", type=Path)
    args = p.parse_args()
    result = LiveAnalyzer(args.gallery).analyze(Image.open(args.image))
    print(result["prediction"], result["risk"], result["candidate_set"], result["reason"])
    print("similar:", [(s["label"], round(s["score"], 3)) for s in result["similar_cases"]])


if __name__ == "__main__":
    main()
