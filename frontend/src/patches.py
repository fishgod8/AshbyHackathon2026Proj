"""Patch-level layer for BACH: one embedding per 224 px patch instead of one per photo.

Used by the slide view: a slide or a large photo is cut into patches, and every patch goes through the same chain
as a photo (QuiltNet -> Mayumi's classifier -> Ashley's checks -> similar confirmed patches).

Library (from the original photos, data/bach_raw/, Mayumi's split):
  each photo -> Mayumi's 12 patch positions -> QuiltNet, L2-normalized, one row per patch
  train patches -> similar-patch gallery (crops saved for display) and Ashley's reference set
  val patches   -> Ashley's limit (95th percentile of the distance to the 5th nearest train patch)
  test patches  -> a check of Mayumi's photo-level classifier applied to single patches

    python -m src.patches build --raw data/bach_raw --out ui/data_bach
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.bach import (DEFAULT_THRESHOLD, MAYUMI_BRANCH, TEAM_REPO, MayumiClassifier, QuiltEmbedder, SupportGate,
                      build_support_gate, patch_boxes, quilt_settings)

REPO = Path(__file__).resolve().parent.parent
PATCH = 224


def embed_crops(embedder: QuiltEmbedder, crops: list[Image.Image], batch: int = 64) -> np.ndarray:
    """L2-normalized QuiltNet embedding of each crop (official adapter: encode_image + L2 norm)."""
    torch, out = embedder.torch, []
    for i in range(0, len(crops), batch):
        x = torch.stack([embedder.preprocess(c.convert("RGB")) for c in crops[i:i + batch]]).to(embedder.device)
        with torch.no_grad():
            e = embedder.model.encode_image(x)
            out.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


class PatchLibrary:
    """Train patches (gallery + reference), their mean (for centered similarity) and Ashley's patch-level gate."""

    def __init__(self, data_dir: Path) -> None:
        d = data_dir / "patches"
        self.index = pd.read_csv(d / "index.csv")
        self.emb = np.load(d / "embeddings.npy")
        train = (self.index.split == "train").to_numpy()
        self.gallery = self.index[train].reset_index(drop=True)
        g = self.emb[train]
        self.mean = g.mean(axis=0)
        c = g - self.mean
        self.centered = c / np.linalg.norm(c, axis=1, keepdims=True)
        self.gate = SupportGate(d / "support")
        self.crops = d / "crops"

    def similar(self, emb: np.ndarray, k: int = 5, label: str | None = None) -> list[dict]:
        """Centered-cosine top-k train patches (stub retrieval until Felix's module does patch-to-patch)."""
        q = emb / np.linalg.norm(emb) - self.mean
        q = q / np.linalg.norm(q)
        scores = self.centered @ q
        idx = np.arange(len(scores)) if label is None else np.flatnonzero(self.gallery.label.to_numpy() == label)
        top = idx[np.argsort(-scores[idx])[:k]]
        return [{"image": str(self.crops / f"{Path(self.gallery.image_id[i]).stem}_{self.gallery.patch[i]:02d}.jpg"),
                 "label": self.gallery.label[i], "score": round(float(scores[i]), 3),
                 "detail": f"{self.gallery.image_id[i]} · patch {self.gallery.patch[i] + 1}"} for i in top]


def build(raw: Path, out: Path, verify_photos: int) -> None:
    split = pd.read_csv(raw / "split.csv")
    d = out / "patches"
    (d / "crops").mkdir(parents=True, exist_ok=True)
    import torch

    embedder = QuiltEmbedder(device="mps" if torch.backends.mps.is_available() else "cpu")
    rows, embs = [], []
    for n, r in enumerate(split.itertuples(), start=1):
        img = Image.open(raw / r.split / r.image_id).convert("RGB")
        boxes = patch_boxes(*img.size)
        crops = [img.crop(b) for b in boxes]
        embs.append(embed_crops(embedder, crops))
        for k, (b, c) in enumerate(zip(boxes, crops)):
            rows.append({"image_id": r.image_id, "label": r.label, "split": r.split, "patch": k, "x": b[0], "y": b[1]})
            if r.split == "train":  # display copies for the similar-patch view (embeddings come from the originals)
                dest = d / "crops" / f"{Path(r.image_id).stem}_{k:02d}.jpg"
                if not dest.exists():
                    c.save(dest, quality=92)
        if n % 50 == 0 or n == len(split):
            print(f"{n}/{len(split)} photos", flush=True)
    index = pd.DataFrame(rows)
    E = np.concatenate(embs).astype(np.float32)
    index.to_csv(d / "index.csv", index=False)
    np.save(d / "embeddings.npy", E)

    # Our 12 patches averaged must reproduce Mayumi's photo embeddings
    mayumi = pd.read_csv(io.BytesIO(subprocess.run(["git", "-C", str(TEAM_REPO), "show", f"{MAYUMI_BRANCH}:bach/image_embeddings.csv"],
                                                   check=True, capture_output=True).stdout))
    vec = dict(zip(mayumi.image_id, mayumi.embedding))
    cos = []
    for image_id in split.image_id[:verify_photos]:
        ours = E[(index.image_id == image_id).to_numpy()].mean(axis=0)
        theirs = np.asarray(ast.literal_eval(vec[image_id]), dtype=np.float32)
        cos.append(float(ours @ theirs / np.linalg.norm(ours) / np.linalg.norm(theirs)))
    print(f"patch means vs Mayumi's photo embeddings ({len(cos)} photos): cosine min {min(cos):.6f}")

    # Ashley's gate at patch level: reference = train patches, limit from val patches
    emb_df = pd.DataFrame({"image_id": [f"{Path(i).stem}_{p:02d}" for i, p in zip(index.image_id, index.patch)],
                           "label": index.label, "split": index.split, "vec": list(E)})
    settings = quilt_settings("n/a", source="BACH patch embeddings, one per 224 px patch (src/patches.py)")
    settings["preprocessing"]["patches"] = "one 224x224 patch (Mayumi's 3x4 positions on the photos)"
    settings["extraction"]["pooling"] = "none: one embedding per patch"
    gate = build_support_gate(emb_df, d, "n/a", settings)  # writes <data>/patches/support/

    # Mayumi's photo-level classifier on single test patches (weak check: each patch gets its photo's label)
    clf = MayumiClassifier(out / "classifier.pkl")
    te = (index.split == "test").to_numpy()
    probs = clf.model.predict_proba(E[te])
    pred = np.array(clf.classes)[probs.argmax(1)]
    acc = float(np.mean(pred == index.label[te].to_numpy()))
    conf = probs.max(1)
    summary = {"patches": len(index), "train_patches": int((index.split == "train").sum()),
               "gate_limit": gate.config["threshold"], "test_patch_accuracy": round(acc, 3),
               "test_patches_confident": round(float(np.mean(conf >= DEFAULT_THRESHOLD)), 3)}
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", type=Path, default=REPO / "data/bach_raw")
    b.add_argument("--out", type=Path, default=REPO / "ui/data_bach")
    b.add_argument("--verify", type=int, default=40, help="photos whose patch mean is checked against Mayumi's")
    args = ap.parse_args()
    build(args.raw, args.out, args.verify)


if __name__ == "__main__":
    main()
