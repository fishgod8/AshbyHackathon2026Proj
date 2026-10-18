"""BACH (ICIAR 2018 breast histology) integration: the team's real modules end to end.

Per microscopy image (BACH photos are 2048x1536 H&E; any size from 224x224 is accepted, as in
Mayumi's predict_image.py), exactly as Mayumi's prepare_bach.py / predict_image.py (team repo, branch mayumi-model):
  12 patches of 224x224 on a 3x4 grid -> QuiltNet (official openclip adapter: encode_image + L2 norm)
  -> mean of the 12 -> Mayumi's classifier.pkl (StandardScaler + LogisticRegression) -> 4 class probabilities
  -> two review checks by Ashley (vendored verbatim in src/team/):
       confidence: assess_uncertainty(probabilities, threshold)            (branch ashley-uncertainty)
       reference support: cosine distance to the 5th nearest training image,
       flagged above the 95th percentile of validation images              (main, ashley_review/)
  -> centered cosine top-k over the reference gallery (Felix's retrieval not delivered yet: CenteredRetriever)

`build` assembles a UI data folder (cases.json, meta.json, evidence.json, gallery.csv, classifier.pkl,
images/, examples/, support/ = Ashley's reference index and gate in her file formats). Photos come from the BACH Zenodo zip by range request (only the ones needed), are
checked against Mayumi's embeddings, and are stored as display JPEGs (full image, no cropping).

Example:
    python -m src.bach build --out ui/data_bach --per-class 10 \
        --test-zip ~/Downloads/ICIAR2018_BACH_Challenge_TestDataset.zip
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import random
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.analyze import case_stats, export_png, risk_coverage
from src.make_subset import HttpRangeFile, fetch_member
from src.stubs import StubRetriever
from src.team.ashley_uncertainty import assess_uncertainty

sys.path.insert(0, str(Path(__file__).resolve().parent / "team"))  # Ashley's package imports itself as ashley_review
from ashley_review import reference_support, review  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# Inside the team repo (frontend/ folder) the repo root is the parent; in a standalone copy, a clone next to it.
TEAM_REPO = REPO.parent if (REPO.parent / ".git").exists() else REPO.parent / "AshbyHackathon2026Proj"
MAYUMI_BRANCH = "origin/mayumi-model"
BACH_ZIP = "https://zenodo.org/api/records/3632035/files/ICIAR2018_BACH_Challenge.zip/content"
QUILTNET = "hf-hub:wisdomik/QuiltNet-B-32"
QUILT_DIM = 512
MIN_SIDE = 224  # one patch; Mayumi's predict_image.py accepts any size from here (mayumi-model @ 7f630c3)
CLASS_NAMES = {"Normal": "Normal tissue", "Benign": "Benign lesion",
               "InSitu": "In situ carcinoma", "Invasive": "Invasive carcinoma"}
# Confidence threshold, raised from Ashley's provisional 0.7 (agreed with her 10/18): the lowest value that sends
# every validation error to review on Mayumi's 40 validation images (0.7 caught 6 of 8, 0.9 caught 8 of 8).
DEFAULT_THRESHOLD = 0.9
SUPPORT_PERCENTILE = 95  # Ashley's gate rule: flag if reference_distance > validation 95th percentile

# Ashley's reference-support module fixes EMBEDDING_DIMENSION = 1024 (MUSK). QuiltNet gives 512, so we set the
# module constant here instead of editing her file (both modules read it at call time).
reference_support.EMBEDDING_DIMENSION = review.EMBEDDING_DIMENSION = QUILT_DIM
ATTRIBUTION = (
    'Images: BACH, ICIAR 2018 Grand Challenge (Aresta et al., Medical Image Analysis 2019; '
    '<a href="https://zenodo.org/records/3632035" target="_blank" rel="noopener">Zenodo 3632035</a>, '
    "CC BY-NC-ND). Embeddings: QuiltNet-B-32. Literature: PubMed (NCBI). "
    "Research demo; not for clinical use."
)


# ---- Image -> embedding (Mayumi's patching, official QuiltNet adapter) ----

def patch_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """The 12 patch positions of prepare_bach.py: 3 rows x 4 columns, spread over the image."""
    boxes = []
    for row in range(3):
        for col in range(4):
            x = round((width - 224) * col / 3)
            y = round((height - 224) * row / 2)
            boxes.append((x, y, x + 224, y + 224))
    return boxes


class QuiltEmbedder:
    def __init__(self, device: str = "cpu") -> None:
        import open_clip
        import torch

        self.torch, self.device = torch, device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(QUILTNET)
        self.model = self.model.to(device).eval()

    def image_embedding(self, image: Image.Image) -> np.ndarray:
        """Mean of the 12 L2-normalized patch embeddings (what Mayumi's classifier was trained on)."""
        img = image.convert("RGB")
        if min(img.size) < MIN_SIDE:
            raise ValueError(f"Image must be at least {MIN_SIDE} × {MIN_SIDE} pixels; got {img.size[0]} × {img.size[1]}.")
        batch = self.torch.stack([self.preprocess(img.crop(b)) for b in patch_boxes(*img.size)]).to(self.device)
        with self.torch.no_grad():
            emb = self.model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float().cpu().numpy().mean(axis=0)


# ---- Team modules behind small adapters ----

class MayumiClassifier:
    def __init__(self, path: Path) -> None:
        import joblib

        saved = joblib.load(path)
        self.model = saved["model"]
        self.classes = [str(c) for c in saved["classes"]]

    def predict(self, emb: np.ndarray) -> tuple[list[str], np.ndarray]:
        return self.classes, self.model.predict_proba(emb[None, :])[0]


def ashley_flag(probs: np.ndarray, classes: list[str], threshold: float) -> dict:
    confidence, needs_review = assess_uncertainty([float(p) for p in probs], threshold)
    first, second = np.argsort(probs)[::-1][:2]
    top, runner_up = classes[first], classes[second]
    if not needs_review:
        return {"risk": "confident", "candidate_set": [top],
                "reason": f"{top} scores {confidence:.0%}, at or above the review threshold of {threshold:.0%}."}
    return {"risk": "review", "candidate_set": [top, runner_up],
            "reason": (f"{top} ({confidence:.0%}) is below the review threshold of {threshold:.0%}; "
                       f"runner-up {runner_up} ({probs[second]:.0%}).")}


def support_check(emb: np.ndarray, gate: SupportGate, image_id: str | None = None) -> dict:
    """Ashley's reference-support gate: is this image far from every training image the model learned from?"""
    r = review.review_embedding(emb.astype(np.float32), gate.index, gate.config, image_id)
    d, limit = r["reference_distance"], r["distance_threshold"]
    k = gate.config["k"]
    if r["needs_review"]:
        reason = (f"Unfamiliar image: the {ordinal(k)} closest training image is at cosine distance {d:.3f}, "
                  f"beyond the review limit of {limit:.3f}.")
    else:
        reason = f"Close to the training images: the {ordinal(k)} closest is at cosine distance {d:.3f} (review limit {limit:.3f})."
    return {"flagged": bool(r["needs_review"]), "distance": round(d, 4), "threshold": round(limit, 4), "k": k,
            "percentile": gate.config["percentile"], "n_reference": len(gate.index.image_ids), "reason": reason}


def ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def analyze_embedding(emb: np.ndarray, clf: MayumiClassifier, retriever: CenteredRetriever, threshold: float,
                      gate: SupportGate | None = None, image_id: str | None = None) -> dict:
    """One image embedding -> the result format the UI reads (image paths as the gallery stores them)."""
    classes, probs = clf.predict(emb)  # the classifier sees the raw mean embedding, as in training
    flag = ashley_flag(probs, classes, threshold)
    support = support_check(emb, gate, image_id) if gate else None
    if support:  # either check sends the image to review; flagged reasons first
        reasons = [flag["reason"], support["reason"]]
        if support["flagged"] and flag["risk"] == "confident":
            reasons.reverse()
        flag = {**flag, "risk": "review" if support["flagged"] else flag["risk"], "reason": " ".join(reasons)}
    order = np.argsort(probs)[::-1]
    candidates = flag["candidate_set"]
    q = retriever.center(emb)
    rnd = lambda refs: [{**r, "score": round(r["score"], 3)} for r in refs]  # noqa: E731
    return {
        "prediction": classes[order[0]],
        "probabilities": {classes[i]: round(float(probs[i]), 3) for i in order},
        **flag,
        "similar_cases": rnd(retriever.retrieve(q, k=5)),
        "contrast_cases": rnd(retriever.retrieve(q, k=2, label=candidates[1])) if len(candidates) > 1 else [],
        "candidate_examples": {c: rnd(retriever.retrieve(q, k=3, label=c)) for c in candidates},
        **({"support": support} if support else {}),
    }


class CenteredRetriever(StubRetriever):
    """Stub retrieval (until Felix's module lands) on centered cosine.

    Raw cosine between these embeddings is always high (0.74 to 0.997 on BACH; same class 0.919 vs different
    class 0.902): every image shares one large "H&E histology" component, and the 12-patch mean shrinks the rest.
    Removing the mean of the training embeddings first makes the score readable (same class 0.135, different
    class -0.045) without changing the ranking much. Mayumi's classifier does the same through its StandardScaler.
    """

    def __init__(self, emb: np.ndarray, labels: np.ndarray, paths: list[str], mean: np.ndarray) -> None:
        self.mean = mean
        super().__init__(self.center(emb), labels, paths)

    def center(self, emb: np.ndarray) -> np.ndarray:
        v = emb / np.linalg.norm(emb, axis=-1, keepdims=True) - self.mean
        return v / np.linalg.norm(v, axis=-1, keepdims=True)


def load_gallery(data_dir: Path) -> CenteredRetriever:
    g = pd.read_csv(data_dir / "gallery.csv")
    emb = np.stack(g.embedding.map(lambda s: np.asarray(json.loads(s), dtype=np.float32)))
    return CenteredRetriever(emb, g.label.to_numpy(), g.image_path.tolist(), np.load(data_dir / "embedding_mean.npy"))


class SupportGate:
    """Ashley's reference index + gate config, loaded with her own functions from <data>/support/."""

    def __init__(self, support_dir: Path) -> None:
        self.index = reference_support.load_reference_index(
            support_dir / "embeddings.npy", support_dir / "split_manifest.csv", support_dir / "extraction_config.json")
        self.config = json.loads((support_dir / "reference_gate_config.json").read_text())


def quilt_settings(commit: str) -> dict:
    """Extractor settings in Ashley's config format (she checks reference and gate configs match)."""
    from huggingface_hub import try_to_load_from_cache

    weights = try_to_load_from_cache(QUILTNET.removeprefix("hf-hub:"), "open_clip_pytorch_model.bin")
    blob = Path(weights).resolve().name if isinstance(weights, str) else ""
    return {
        "model": "openclip:wisdomik/QuiltNet-B-32",
        "checkpoint": QUILTNET,
        "checkpoint_sha256": blob if len(blob) == 64 else "unknown",  # Hugging Face names LFS blobs by sha256
        "preprocessing": {"patches": "12 crops of 224x224 on a 3x4 grid over the image (prepare_bach.py)",
                          "transform": "open_clip QuiltNet-B-32 preprocess"},
        "extraction": {"per_patch": "encode_image, L2-normalized", "pooling": "mean of the 12 patch embeddings",
                       "output_dtype": "float32", "embedding_dimension": QUILT_DIM},
        "software_versions": {"embeddings": f"Mayumi's bach/image_embeddings.csv (mayumi-model @ {commit})"},
    }


def build_support_gate(emb_df: pd.DataFrame, out: Path, commit: str) -> SupportGate:
    """Ashley's rule on Mayumi's split: reference = her 320 training images, limit from her 40 validation images."""
    rows = emb_df[emb_df.split.isin(["train", "val"])].reset_index(drop=True)
    manifest = pd.DataFrame({"image_id": rows.image_id, "row_index": range(len(rows)), "label": rows.label,
                             "split": rows.split.map({"train": "reference", "val": "validation"})})
    matrix = np.stack(rows.vec).astype(np.float32)
    d = out / "support"
    d.mkdir(exist_ok=True)
    np.save(d / "embeddings.npy", matrix)
    manifest.to_csv(d / "split_manifest.csv", index=False)
    settings = quilt_settings(commit)
    (d / "extraction_config.json").write_text(json.dumps(settings, indent=2))

    index = reference_support.load_reference_index(d / "embeddings.npy", d / "split_manifest.csv",
                                                   d / "extraction_config.json")
    val = manifest[manifest.split == "validation"]
    scores = reference_support.score_queries(index, val.image_id.tolist(), matrix[val.row_index.to_numpy()], settings)
    dist = np.array([s.reference_distance for s in scores])
    threshold = float(np.percentile(dist, SUPPORT_PERCENTILE))  # NumPy default (linear), as Ashley's config
    config = {
        "purpose": "Ashley's reference-support gate on BACH (QuiltNet, Mayumi's split); built by src/bach.py.",
        "distance": "cosine distance = 1 - cosine similarity after L2 normalization",
        "k": reference_support.K_NEIGHBORS,
        "reference_split": "reference",
        "query_split": "validation",
        "threshold_rule": "flag if reference_distance > validation 95th percentile",
        "percentile": SUPPORT_PERCENTILE,
        "percentile_method": "NumPy percentile method='linear' (linear interpolation; NumPy default definition)",
        "threshold": threshold,
        "validation_count": len(dist),
        "flagged_count": int((dist > threshold).sum()),
        "feature_settings": {"shape": list(matrix.shape), "dtype": str(matrix.dtype), **settings},
    }
    (d / "reference_gate_config.json").write_text(json.dumps(config, indent=2))
    return SupportGate(d)


class BachLiveAnalyzer:
    """Lookup on a new image: same path as the precomputed cases."""

    def __init__(self, data_dir: Path, device: str = "cpu") -> None:
        meta = json.loads((data_dir / "meta.json").read_text())
        self.threshold = meta["bach"]["threshold"]
        self.clf = MayumiClassifier(data_dir / "classifier.pkl")
        self.retriever = load_gallery(data_dir)
        self.gate = SupportGate(data_dir / "support") if (data_dir / "support").exists() else None
        self.embedder = QuiltEmbedder(device)
        self.data_dir = data_dir

    def analyze(self, image: Image.Image) -> dict:
        result = analyze_embedding(self.embedder.image_embedding(image), self.clf, self.retriever, self.threshold,
                                   self.gate)
        # Gallery paths are relative to the data folder; the UI localizes absolute paths.
        fix = lambda refs: [{**r, "image": str(self.data_dir / r["image"])} for r in refs]  # noqa: E731
        return {**result, "similar_cases": fix(result["similar_cases"]), "contrast_cases": fix(result["contrast_cases"]),
                "candidate_examples": {k: fix(v) for k, v in result["candidate_examples"].items()}}


# ---- Build the UI data folder ----

def git_show(ref_path: str) -> bytes:
    return subprocess.run(["git", "-C", str(TEAM_REPO), "show", ref_path], check=True, capture_output=True).stdout


def save_display(img: Image.Image, dest: Path, max_width: int | None, quality: int) -> None:
    """Whole image, aspect kept; only scaled and re-encoded for the browser."""
    img = img.convert("RGB")
    if max_width and img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)), Image.LANCZOS)
    img.save(dest, quality=quality)


def build(out: Path, per_class: int, threshold: float, seed: int, test_zip: Path | None, n_examples: int,
          verify: bool, n_robust: int = 12) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    commit = subprocess.run(["git", "-C", str(TEAM_REPO), "rev-parse", "--short", MAYUMI_BRANCH],
                            check=True, capture_output=True, text=True).stdout.strip()
    (out / "classifier.pkl").write_bytes(git_show(f"{MAYUMI_BRANCH}:bach/classifier.pkl"))
    emb_df = pd.read_csv(io.BytesIO(git_show(f"{MAYUMI_BRANCH}:bach/image_embeddings.csv")))
    emb_df["vec"] = emb_df.embedding.map(lambda s: np.asarray(ast.literal_eval(s), dtype=np.float32))

    rng = random.Random(seed)
    gallery = pd.concat([
        g.loc[sorted(rng.sample(list(g.index), per_class))]
        for _, g in emb_df[emb_df.split == "train"].groupby("label")
    ])
    tests = emb_df[emb_df.split == "test"]
    wanted = pd.concat([gallery, tests])

    # Photos by range request, checked against Mayumi's embeddings, stored as display JPEGs.
    zf = zipfile.ZipFile(HttpRangeFile(BACH_ZIP))
    members = {Path(i.filename).name: i for i in zf.infolist() if "/Photos/" in i.filename and i.filename.endswith(".tif")}
    embedder = QuiltEmbedder() if verify else None
    cosines = []
    # Only photos not saved yet (a rerun resumes); every downloaded photo is checked when verify is on.
    todo = [r for r in wanted.itertuples() if not (out / "images" / f"{Path(r.image_id).stem}.jpg").exists()]

    def fetch(row):  # ~9 MB each; Zenodo gives ~1 MB/s per connection, so download 4 at a time
        return row, fetch_member(BACH_ZIP, members[row.image_id])

    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, (row, data) in enumerate(pool.map(fetch, todo), start=1):
            dest = out / "images" / f"{Path(row.image_id).stem}.jpg"
            img = Image.open(io.BytesIO(data))
            if embedder:
                ours = embedder.image_embedding(img)
                cosines.append(float(ours @ row.vec / (np.linalg.norm(ours) * np.linalg.norm(row.vec))))
            if not dest.exists():
                save_display(img, dest, max_width=1024, quality=90)
            if n % 10 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} photos" + (f", embedding cosine vs Mayumi min {min(cosines):.5f}" if cosines else ""),
                      flush=True)

    pd.DataFrame({
        "image_path": [f"images/{Path(i).stem}.jpg" for i in gallery.image_id],
        "label": gallery.label,
        "embedding": gallery.vec.map(lambda v: json.dumps(v.tolist())),
    }).to_csv(out / "gallery.csv", index=False)

    train = np.stack(emb_df[emb_df.split == "train"].vec)
    np.save(out / "embedding_mean.npy", (train / np.linalg.norm(train, axis=1, keepdims=True)).mean(axis=0))
    clf = MayumiClassifier(out / "classifier.pkl")
    retriever = load_gallery(out)
    gate = build_support_gate(emb_df, out, commit)
    order = list(tests.index)
    random.Random(seed).shuffle(order)  # case ids must not reveal the class (file names do: b/is/iv/n)
    cases, conf_only = [], []  # conf_only: the same cases with the confidence check alone, for the comparison
    for k, idx in enumerate(order, start=1):
        row = tests.loc[idx]
        case = {"case_id": f"case_{k:03d}", "image": f"images/{Path(row.image_id).stem}.jpg",
                "true_label": row.label, "source_image": row.image_id}
        cases.append({**case, **analyze_embedding(row.vec, clf, retriever, threshold, gate, row.image_id)})
        conf_only.append({**case, **analyze_embedding(row.vec, clf, retriever, threshold)})

    examples, robustness = [], None
    if test_zip:
        (out / "examples").mkdir(exist_ok=True)
        for old in (out / "examples").glob("example_*"):
            old.unlink()  # regenerated below (earlier builds stored JPEG)
        with zipfile.ZipFile(test_zip) as tz:  # unlabeled challenge test photos; WSIs are never read
            photos = sorted((i for i in tz.namelist() if "/Photos/" in i and i.endswith(".tif")),
                            key=lambda s: int(Path(s).stem.replace("test", "")))
            for n, name in enumerate(photos[:n_examples], start=1):
                with tz.open(name) as f:
                    dest = out / "examples" / f"example_{n:02d}.webp"
                    # Full size and lossless: JPEG compression shifts the embedding enough to change the result.
                    Image.open(f).convert("RGB").save(dest, lossless=True)
                    examples.append(dest.name)
            robustness = input_robustness(tz, photos[:n_robust], embedder or QuiltEmbedder(), clf, retriever,
                                          threshold, gate)

    meta = {
        "source": "BACH breast histology · team models",
        "pipeline": "bach",
        "dataset": "BACH (ICIAR 2018) microscopy images, 2048×1536 H&E",
        "model": f"QuiltNet-B-32, mean of 12 patches → Mayumi's classifier (mayumi-model @ {commit})",
        "classes": clf.classes,
        "class_names": {c: CLASS_NAMES.get(c, c) for c in clf.classes},
        "reference_noun": "images",
        "input_hint": "Upload an H&E photo of breast tissue (PNG if you can: JPEG compression can change the result)",
        "examples_label": "Try an example from the BACH challenge test set (unlabeled)",
        "attribution": ATTRIBUTION,
        "bach": {"threshold": threshold, "uncertainty": "Ashley's assess_uncertainty (ashley-uncertainty)",
                 "support": f"Ashley's reference-support gate (main), limit {gate.config['threshold']:.4f}",
                 "retrieval": "stub centered-cosine top-k (Felix's module pending)", "gallery_per_class": per_class},
        "similarity_note": ("Similarity: cosine after removing what all training images have in common "
                            "(1 = same, 0 = unrelated, below 0 = less alike than average)."),
    }
    stats = case_stats(cases)
    n_ref, n_val = len(gate.index.image_ids), gate.config["validation_count"]
    evidence = {
        "source": (f"Mayumi's held-out test images ({len(cases)}). Ashley's two review checks: confidence below "
                   f"{threshold:.0%} (provisional) or an unfamiliar image. Real team models; small test set."),
        "alpha": None,
        **stats,
        "risk_coverage": risk_coverage(cases),
        "models_title": "Review checks compared",
        "models": [{"model": f"Confidence below {threshold:.0%}", **case_stats(conf_only)},
                   {"model": "Confidence or unfamiliar image", **stats}],
        **({"robustness": robustness} if robustness else {}),
        "split_note": (f"Test cases: {len(cases)} BACH images Mayumi held out (10 per class). BACH has no patient IDs, "
                       "so her split is by image and results may be optimistic. Unfamiliar image: the "
                       f"{ordinal(gate.config['k'])} closest of her {n_ref} training images is farther than for "
                       f"{gate.config['percentile']}% of her {n_val} validation images (limit "
                       f"{gate.config['threshold']:.3f}). Reference gallery shown on screen: "
                       f"{len(gallery)} of the {n_ref} training images ({per_class} per class)."),
    }
    (out / "cases.json").write_text(json.dumps(cases, indent=2))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2))

    wrong = [c for c in cases if c["prediction"] != c["true_label"]]
    print(f"{len(cases)} cases | accuracy {stats['accuracy']:.3f} | review {sum(c['risk'] == 'review' for c in cases)} | "
          f"wrong {len(wrong)} (flagged {sum(c['risk'] == 'review' for c in wrong)}) | gallery {len(gallery)} | "
          f"examples {len(examples)}")
    print(f"support gate: limit {gate.config['threshold']:.4f} | validation flagged {gate.config['flagged_count']}/"
          f"{n_val} | test flagged {sum(c['support']['flagged'] for c in cases)}/{len(cases)} | wrong flagged by it "
          f"{sum(c['support']['flagged'] for c in wrong)} | confidence-only review {sum(c['risk'] == 'review' for c in conf_only)}")
    if cosines:
        print(f"embedding check over {len(cosines)} photos: cosine min {min(cosines):.6f}, mean {np.mean(cosines):.6f}")


def input_robustness(tz: zipfile.ZipFile, photos: list[str], embedder: QuiltEmbedder, clf: MayumiClassifier,
                     retriever: CenteredRetriever, threshold: float, gate: SupportGate) -> dict:
    """Same challenge photos, changed the way uploads often are: does the suggestion move, and is it caught?"""
    def jpeg(img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)  # an ordinary JPEG export (4:2:0 chroma subsampling)
        buf.seek(0)
        return Image.open(buf)

    variants = {"Original photo (lossless TIFF)": lambda im: im,
                "Saved as an ordinary JPEG (quality 92)": jpeg,
                "Half resolution (1024 × 768)": lambda im: im.resize((im.width // 2, im.height // 2), Image.LANCZOS)}
    results = {name: [] for name in variants}
    for name in photos:
        img = Image.open(io.BytesIO(tz.read(name))).convert("RGB")
        for v, change in variants.items():
            results[v].append(analyze_embedding(embedder.image_embedding(change(img)), clf, retriever, threshold, gate))
    base = [r["prediction"] for r in results["Original photo (lossless TIFF)"]]
    rows = [{"input": v,
             "changed": sum(r["prediction"] != b for r, b in zip(rs, base)),
             "unfamiliar": sum(r["support"]["flagged"] for r in rs),
             "review": sum(r["risk"] == "review" for r in rs)} for v, rs in results.items()]
    for r in rows:
        print(f"robustness | {r['input']}: suggestion changed {r['changed']}/{len(photos)}, "
              f"unfamiliar {r['unfamiliar']}/{len(photos)}, review {r['review']}/{len(photos)}")
    return {"n": len(photos), "rows": rows,
            "source": f"First {len(photos)} photos of the unlabeled BACH challenge test set; no ground truth needed."}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=REPO / "ui/data_bach")
    b.add_argument("--per-class", type=int, default=10, help="training images per class in the reference gallery")
    b.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--test-zip", type=Path, default=None, help="BACH challenge test zip, for Lookup examples")
    b.add_argument("--examples", type=int, default=6)
    b.add_argument("--no-verify", action="store_true", help="skip re-embedding photos to check against Mayumi's")
    args = ap.parse_args()
    build(args.out, args.per_class, args.threshold, args.seed, args.test_zip, args.examples, not args.no_verify)


if __name__ == "__main__":
    main()
