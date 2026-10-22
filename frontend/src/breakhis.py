"""BreakHis (breast histopathology, 82 patients) integration: Mayumi's embeddings, a temporary classifier.

Per image (700x460 H&E PNG, 40X to 400X), exactly as Mayumi's BreakHis embeddings (checked: cosine 1.00000):
  6 patches of 224x224 on a 2x3 grid -> QuiltNet (official openclip adapter) -> mean of the 6
  -> TEMPORARY classifier, benign vs malignant: Mayumi's BACH recipe (StandardScaler + LogisticRegression,
     C picked on her validation patients); replace with her classifier when she delivers one
  -> Ashley's two review checks (confidence threshold, reference support), as for BACH
  -> centered cosine top-k over the training images, with patient and subtype (stub until Felix's module)

Mayumi's split is by patient (54 train / 14 val / 14 test, no patient in two splits), so the test numbers are
what a new patient would see.

Example:
    python -m src.breakhis build --raw data/breakhis/raw --embeddings data/breakhis --out ui/data_breakhis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analyze import case_stats, risk_coverage
from src.bach import (DEFAULT_THRESHOLD, MayumiClassifier, QuiltEmbedder, SupportGate, analyze_embedding,
                      build_support_gate, load_gallery, ordinal, quilt_settings)

REPO = Path(__file__).resolve().parent.parent
GRID = (2, 3)
CLASS_NAMES = {"Benign": "Benign", "Malignant": "Malignant"}
SUBTYPES = {"adenosis": "Adenosis", "fibroadenoma": "Fibroadenoma", "phyllodes_tumor": "Phyllodes tumor",
            "tubular_adenoma": "Tubular adenoma", "ductal_carcinoma": "Ductal carcinoma",
            "lobular_carcinoma": "Lobular carcinoma", "mucinous_carcinoma": "Mucinous carcinoma",
            "papillary_carcinoma": "Papillary carcinoma"}
ATTRIBUTION = (
    'Images: BreakHis (Spanhol et al., IEEE Trans Biomed Eng 2016; '
    '<a href="https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/" '
    'target="_blank" rel="noopener">UFPR VRI</a>, non-commercial research use). Embeddings: QuiltNet-B-32. '
    "Literature: PubMed (NCBI). Research demo; not for clinical use."
)


def image_table(emb_dir: Path, raw: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per source image (Mayumi's 6 patch rows averaged), with a local path to the PNG."""
    m = pd.read_csv(emb_dir / "patch_manifest.csv")
    X = np.load(emb_dir / "patch_embeddings.npy")
    m["row"] = np.arange(len(m))
    img = m.groupby("image_path", sort=False).agg(
        rows=("row", list), patient=("patient_id", "first"), subtype=("subtype", "first"),
        diagnosis=("diagnosis", "first"), magnification=("magnification", "first"), split=("split", "first"),
    ).reset_index()
    img["local"] = img.image_path.map(lambda p: raw / p.split("breakhis_project/", 1)[1])
    img["image_id"] = img.image_path.map(lambda p: Path(p).name)
    img["label"] = img.diagnosis.str.title()
    E = np.stack([X[r].mean(axis=0) for r in img.rows]).astype(np.float32)
    return img, E


def fit_classifier(E: np.ndarray, img: pd.DataFrame) -> tuple[object, float, float]:
    """Mayumi's BACH recipe on the BreakHis split: regularization chosen on the validation patients only."""
    tr, va = (img.split == "train").to_numpy(), (img.split == "val").to_numpy()
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(E[tr], img.label[tr])
        acc = accuracy_score(img.label[va], model.predict(E[va]))
        if best is None or acc > best[1]:
            best = (model, acc, C)
    return best


def save_jpeg(src: Path, dest: Path, quality: int = 88) -> None:
    """Whole image, only re-encoded for the browser (700x460 stays as is)."""
    if not dest.exists():
        Image.open(src).convert("RGB").save(dest, quality=quality)


def pick_examples(cases: pd.DataFrame, test: pd.DataFrame) -> list[Path]:
    """One image per demo behavior: confident and right, confident and wrong, flagged (with a next step)..."""
    c = cases.assign(conf=cases.probabilities.map(lambda p: max(p.values())),
                     unfamiliar=cases.support.map(lambda s: s["flagged"]), ok=cases.prediction == cases.true_label)
    c = c[(c.magnification == "200X") & ~c.unfamiliar]
    wanted = [("Ductal carcinoma", "confident", True, False), ("Fibroadenoma", "confident", False, False),
              ("Phyllodes tumor", "review", True, True), ("Mucinous carcinoma", "review", True, True),
              ("Tubular adenoma", "review", False, True), ("Lobular carcinoma", "confident", True, False)]
    local = dict(zip(test.image_id, test.local))
    picks = []
    for subtype, risk, ok, low_first in wanted:
        pool = c[(c.subtype == subtype) & (c.risk == risk) & (c.ok == ok)].sort_values("conf", ascending=low_first)
        if len(pool):
            picks.append(local[pool.source_image.iloc[0]])
    return picks


def build(raw: Path, emb_dir: Path, out: Path, threshold: float, n_examples: int, verify: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    img, E = image_table(emb_dir, raw)
    assert img.local.map(Path.exists).all(), "missing BreakHis PNGs under --raw"

    if verify:  # our live patching must reproduce Mayumi's patch embeddings
        embedder = QuiltEmbedder(grid=GRID)
        sample = img.sample(verify, random_state=0)
        cos = []
        for i, r in sample.iterrows():
            ours = embedder.image_embedding(Image.open(r.local))
            cos.append(float(ours @ E[i] / (np.linalg.norm(ours) * np.linalg.norm(E[i]))))
        print(f"embedding check over {len(cos)} images: cosine min {min(cos):.6f}")

    model, val_acc, C = fit_classifier(E, img)
    joblib.dump({"model": model, "classes": list(model.classes_), "embedding_model": "openclip:wisdomik/QuiltNet-B-32",
                 "aggregation": "mean of 6 patches (2x3)", "note": "TEMPORARY: Mayumi's BACH recipe on BreakHis, "
                 f"benign vs malignant, C={C} picked on validation patients (accuracy {val_acc:.3f})"},
                out / "classifier.pkl")
    clf = MayumiClassifier(out / "classifier.pkl")

    tr = (img.split == "train").to_numpy()
    train = E[tr] / np.linalg.norm(E[tr], axis=1, keepdims=True)
    np.save(out / "embedding_mean.npy", train.mean(axis=0))
    gal = img[tr]
    for r in gal.itertuples():
        save_jpeg(r.local, out / "images" / f"{Path(r.image_id).stem}.jpg")
    pd.DataFrame({
        "image_path": [f"images/{Path(i).stem}.jpg" for i in gal.image_id],
        "label": gal.label, "patient": gal.patient, "subtype": gal.subtype.map(SUBTYPES),
        "magnification": gal.magnification, "embedding": [json.dumps(v.tolist()) for v in E[tr]],
    }).to_csv(out / "gallery.csv", index=False)
    retriever = load_gallery(out)

    emb_df = pd.DataFrame({"image_id": img.image_id, "label": img.label, "split": img.split, "vec": list(E)})
    gate = build_support_gate(emb_df, out, "n/a", quilt_settings(
        "n/a", GRID, "BreakHis patch_embeddings.csv (Mayumi, 10/18)"))

    test = img[img.split == "test"]
    cases = []
    for k, (i, r) in enumerate(test.iterrows(), start=1):
        save_jpeg(r.local, out / "images" / f"{Path(r.image_id).stem}.jpg")
        cases.append({"case_id": f"case_{k:04d}", "image": f"images/{Path(r.image_id).stem}.jpg",
                      "true_label": r.label, "source_image": r.image_id, "patient": r.patient,
                      "subtype": SUBTYPES[r.subtype], "magnification": r.magnification,
                      **analyze_embedding(E[i], clf, retriever, threshold, gate, r.image_id)})

    # Lookup examples: test patients at 200X, chosen to show each behavior (labels on the page don't reveal which)
    (out / "examples").mkdir(exist_ok=True)
    for old in (out / "examples").glob("example_*"):
        old.unlink()
    picks = pick_examples(pd.DataFrame(cases), test)[:n_examples]
    for n, src in enumerate(picks, start=1):
        Image.open(src).convert("RGB").save(out / "examples" / f"example_{n:02d}.png")

    steps = REPO / "ui/next_steps_breakhis.json"
    if steps.exists():
        (out / "next_steps.json").write_text(steps.read_text())

    stats = case_stats(cases)
    by_patient = []
    for pid, cs in pd.DataFrame(cases).groupby("patient"):
        by_patient.append({"patient": pid, "subtype": cs.subtype.iloc[0], "truth": cs.true_label.iloc[0],
                           "n": len(cs), "wrong": int((cs.prediction != cs.true_label).sum()),
                           "mean_confidence": round(float(cs.probabilities.map(lambda p: max(p.values())).mean()), 3),
                           "review": int((cs.risk == "review").sum())})
    by_patient.sort(key=lambda r: -r["wrong"] / r["n"])
    n_ref, n_val = len(gate.index.image_ids), gate.config["validation_count"]
    n_pat = img.groupby("split").patient.nunique().to_dict()
    meta = {
        "source": "BreakHis breast histology",
        "pipeline": "breakhis",
        "dataset": f"BreakHis breast tumors ({img.patient.nunique()} patients): benign or malignant",
        "dataset_label": "BreakHis",
        "model": f"QuiltNet features + linear classifier, trained on {n_pat.get('train')} patients",
        "model_detail": f"QuiltNet-B-32, mean of 6 patches → logistic regression (C={C}), Mayumi's BACH recipe",
        "classes": clf.classes,
        "class_names": {c: CLASS_NAMES.get(c, c) for c in clf.classes},
        "reference_noun": "images",
        "input_hint": "Upload an H&E photo of breast tissue (PNG if you can: JPEG compression can change the result)",
        "examples_label": "Try an example from the BreakHis test patients (not used for training)",
        "attribution": ATTRIBUTION,
        "similarity_note": ("Similarity: cosine after removing what all training images have in common "
                            "(1 = same, 0 = unrelated, below 0 = less alike than average)."),
        "bach": {"threshold": threshold, "grid": list(GRID), "uncertainty": "Ashley's assess_uncertainty",
                 "support": f"Ashley's reference-support gate, limit {gate.config['threshold']:.4f}",
                 "retrieval": "stub centered-cosine top-k with patient and subtype"},
    }
    worst = by_patient[0]
    evidence = {
        "source": (f"BreakHis test patients ({n_pat.get('test')} patients, {len(cases)} images), never seen in "
                   f"training. An image goes to a pathologist when the AI is less than {threshold:.0%} sure or the "
                   "image is unlike the training images."),
        "alpha": None, **stats,
        "risk_coverage": risk_coverage(cases),
        "models_title": "Review checks compared",
        "models": [{"model": f"Confidence below {threshold:.0%}",
                    **case_stats([{**c, "risk": "review" if max(c["probabilities"].values()) < threshold
                                   else "confident"} for c in cases])},
                   {"model": "Confidence or unfamiliar image", **stats}],
        "by_patient": by_patient,
        "by_patient_note": (f"Errors come in whole patients: {worst['subtype'].lower()} patient {worst['patient']} "
                            f"({worst['truth'].lower()}) was wrong on {worst['wrong']} of {worst['n']} images at a mean "
                            f"confidence of {worst['mean_confidence']:.0%}. Per-image confidence cannot catch this."),
        "split_note": (f"Split by patient: {n_pat.get('train')} training, {n_pat.get('val')} validation and "
                       f"{n_pat.get('test')} test patients; no patient appears in two splits. Classifier: the BACH "
                       f"recipe (regularization picked on validation patients, accuracy {val_acc:.1%}). Unfamiliar "
                       f"image: the {ordinal(gate.config['k'])} closest of {n_ref} training images is farther than "
                       f"for {gate.config['percentile']}% of {n_val} validation images (limit "
                       f"{gate.config['threshold']:.3f}). Reference gallery: all {len(gal)} training images."),
    }
    (out / "cases.json").write_text(json.dumps(cases, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2))
    wrong = [c for c in cases if c["prediction"] != c["true_label"]]
    print(f"{len(cases)} test images | accuracy {stats['accuracy']:.3f} | review {stats['review_rate']:.0%} | "
          f"error among confident {stats['error_rate_confident']} | wrong {len(wrong)} "
          f"(flagged {sum(c['risk'] == 'review' for c in wrong)}) | gallery {len(gal)} | examples {len(picks)}")


class BreakHisLiveAnalyzer:
    """Lookup on a new image: same path as the precomputed cases (2x3 patches)."""

    def __init__(self, data_dir: Path, device: str = "cpu") -> None:
        meta = json.loads((data_dir / "meta.json").read_text())
        self.threshold = meta["bach"]["threshold"]
        self.clf = MayumiClassifier(data_dir / "classifier.pkl")
        self.retriever = load_gallery(data_dir)
        self.gate = SupportGate(data_dir / "support")
        self.embedder = QuiltEmbedder(device, grid=GRID)
        self.data_dir = data_dir

    def analyze(self, image: Image.Image) -> dict:
        result = analyze_embedding(self.embedder.image_embedding(image), self.clf, self.retriever, self.threshold,
                                   self.gate)
        fix = lambda refs: [{**r, "image": str(self.data_dir / r["image"])} for r in refs]  # noqa: E731
        return {**result, "similar_cases": fix(result["similar_cases"]), "contrast_cases": fix(result["contrast_cases"]),
                "candidate_examples": {k: fix(v) for k, v in result["candidate_examples"].items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", type=Path, default=REPO / "data/breakhis/raw", help="folder holding BreaKHis_v1/")
    b.add_argument("--embeddings", type=Path, default=REPO / "data/breakhis",
                   help="patch_manifest.csv + patch_embeddings.npy (from Mayumi's CSVs)")
    b.add_argument("--out", type=Path, default=REPO / "ui/data_breakhis")
    b.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    b.add_argument("--examples", type=int, default=6)
    b.add_argument("--verify", type=int, default=20, help="images to re-embed and check against Mayumi's (0 = skip)")
    args = ap.parse_args()
    build(args.raw, args.embeddings, args.out, args.threshold, args.examples, args.verify)


if __name__ == "__main__":
    main()
