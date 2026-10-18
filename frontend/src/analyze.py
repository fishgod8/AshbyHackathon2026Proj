"""Integration layer: teammates' modules -> one result dict per case -> UI files.

Each teammate's module sits behind one small adapter (get_prediction, get_flag,
get_similar). When real code lands, or its output changes, only the matching
adapter changes; the UI keeps reading the same result format.

Currently builds a "semi-real" UI dataset: real CRC-VAL-HE-7K query tiles and
real QuiltNet similarities against a CRC-100K gallery (different patients, so
no retrieval leakage). Classifier / uncertainty / retrieval are STUBS
(src/stubs.py).

Also writes evidence.json for the Methods & evidence page (stats over the
query cases, a risk-coverage curve, and a model comparison table).

Example:
    python -m src.analyze --gallery embeddings/quiltnet_train_subset.csv \
        --queries embeddings/quiltnet_val_subset.csv --out ui/data_semi --compare plip
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.stubs import StubClassifier, StubRetriever, stub_flag

# Full names per Kather et al. 2019 (NCT-CRC-HE-100K / CRC-VAL-HE-7K).
CRC_CLASS_NAMES = {
    "ADI": "Adipose tissue",
    "BACK": "Background",
    "DEB": "Debris",
    "LYM": "Lymphocytes",
    "MUC": "Mucus",
    "MUS": "Smooth muscle",
    "NORM": "Normal colon mucosa",
    "STR": "Cancer-associated stroma",
    "TUM": "Colorectal adenocarcinoma epithelium",
}
MODEL_NAMES = {"quiltnet": "QuiltNet-B-32", "plip": "PLIP", "conch": "CONCH", "musk": "MUSK"}


def parse_embedding(s: str) -> np.ndarray:
    """Accept a JSON list or a space/comma-separated string."""
    s = s.strip()
    if s.startswith("["):
        return np.asarray(json.loads(s), dtype=np.float32)
    return np.asarray(s.replace(",", " ").split(), dtype=np.float32)


def load_embeddings(csv: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (image paths, labels from parent folder, embeddings [N, D])."""
    df = pd.read_csv(csv)
    paths = df["image_path"].tolist()
    labels = np.array([Path(p).parent.name for p in paths])
    emb = np.stack(df["embedding"].map(parse_embedding))
    return paths, labels, emb


# ---- Adapters: the only place that knows each teammate's interface ----

def get_prediction(clf: StubClassifier, emb: np.ndarray) -> tuple[list[str], np.ndarray]:
    classes, probs = clf.predict(emb[None, :])
    return classes, probs[0]


def get_flag(probs: np.ndarray, classes: list[str]) -> dict:
    return stub_flag(probs, classes)


def get_similar(retriever: StubRetriever, emb: np.ndarray, k: int, label: str | None = None) -> list[dict]:
    return retriever.retrieve(emb, k=k, label=label)


# ---- Packaging ----

def analyze_case(emb: np.ndarray, clf: StubClassifier, retriever: StubRetriever) -> dict:
    """One query embedding -> the result fields the UI reads (images still as source paths)."""
    classes, probs = get_prediction(clf, emb)
    flag = get_flag(probs, classes)
    candidates = flag["candidate_set"]
    order = np.argsort(probs)[::-1]
    return {
        "prediction": classes[order[0]],
        "probabilities": {classes[i]: round(float(probs[i]), 3) for i in order},
        **flag,
        "similar_cases": get_similar(retriever, emb, k=5),
        "contrast_cases": get_similar(retriever, emb, k=2, label=candidates[1]) if len(candidates) > 1 else [],
        # Side-by-side comparison in the UI: closest confirmed tiles of each candidate class.
        "candidate_examples": {label: get_similar(retriever, emb, k=3, label=label) for label in candidates},
    }


def localize_result(result: dict, img_dir: Path) -> dict:
    """Copy every referenced gallery tile next to the UI data and rewrite its path."""
    return {
        **result,
        "similar_cases": localize_images(result["similar_cases"], img_dir),
        "contrast_cases": localize_images(result["contrast_cases"], img_dir),
        "candidate_examples": {k: localize_images(v, img_dir) for k, v in result["candidate_examples"].items()},
    }


def export_png(src: str, dest: Path) -> None:
    """Browsers can't show TIFF, so the UI gets PNG copies."""
    if not dest.exists():
        Image.open(src).convert("RGB").save(dest)


def localize_images(refs: list[dict], img_dir: Path) -> list[dict]:
    """Copy referenced gallery tiles next to the UI data; rewrite paths relative to it."""
    out = []
    for r in refs:
        name = f"ref_{Path(r['image']).stem}.png"
        export_png(r["image"], img_dir / name)
        out.append({**r, "image": f"images/{name}", "score": round(r["score"], 3)})
    return out


# ---- Evidence (Methods & evidence page) ----

def case_stats(cases: list[dict]) -> dict:
    """Headline numbers over cases that have a true label."""
    n = len(cases)
    wrong = [c["prediction"] != c["true_label"] for c in cases]
    confident = [c["risk"] == "confident" for c in cases]
    n_conf = sum(confident)
    n_wrong_conf = sum(w and k for w, k in zip(wrong, confident))
    return {
        "n_cases": n,
        "accuracy": round(1 - sum(wrong) / n, 3),
        "review_rate": round(1 - n_conf / n, 3),
        "n_confident": n_conf,
        "n_wrong_confident": n_wrong_conf,
        "error_rate_confident": round(n_wrong_conf / n_conf, 3) if n_conf else None,
        "empirical_coverage": round(sum(c["true_label"] in c["candidate_set"] for c in cases) / n, 3),
        "mean_set_size": round(sum(len(c["candidate_set"]) for c in cases) / n, 2),
    }


def risk_coverage(cases: list[dict]) -> list[dict]:
    """Error rate (%) among the k most confident cases, for k = 1..n (coverage in %)."""
    ranked = sorted(cases, key=lambda c: -max(c["probabilities"].values()))
    points, n_wrong = [], 0
    for k, c in enumerate(ranked, start=1):
        n_wrong += c["prediction"] != c["true_label"]
        points.append({"coverage": round(100 * k / len(ranked), 1), "error": round(100 * n_wrong / k, 1)})
    return points


def model_key(csv: Path) -> str:
    """embed_local names files <model>_<dataset>.csv."""
    return csv.name.split("_", 1)[0]


def sibling_csv(csv: Path, model: str) -> Path:
    return csv.with_name(f"{model}_{csv.name.split('_', 1)[1]}")


def score_model(gallery_csv: Path, queries_csv: Path, seed: int) -> list[dict]:
    """Prediction + flag only (no retrieval/images), for the model comparison table."""
    _, g_labels, g_emb = load_embeddings(gallery_csv)
    _, q_labels, q_emb = load_embeddings(queries_csv)
    clf = StubClassifier(g_emb, g_labels, seed=seed)
    classes, probs = clf.predict(q_emb)
    return [
        {
            "prediction": classes[int(np.argmax(p))],
            "probabilities": dict(zip(classes, p.tolist())),
            "true_label": str(t),
            **get_flag(p, classes),
        }
        for p, t in zip(probs, q_labels)
    ]


def build_evidence(cases: list[dict], gallery_csv: Path, queries_csv: Path, compare: list[str], seed: int) -> dict:
    main = model_key(gallery_csv)
    models = [{"model": MODEL_NAMES.get(main, main), **case_stats(cases)}]
    for m in compare:
        g, q = sibling_csv(gallery_csv, m), sibling_csv(queries_csv, m)
        if g.exists() and q.exists():
            models.append({"model": MODEL_NAMES.get(m, m), **case_stats(score_model(g, q, seed))})
        else:
            print(f"skip {m}: missing {g} or {q}")
    n_gallery = len(pd.read_csv(gallery_csv, usecols=["image_path"]))
    return {
        "source": "Stub models. Placeholder numbers until teammates' results arrive; do not report.",
        "alpha": None,  # set by a conformal method; stubs give no coverage guarantee
        **case_stats(cases),
        "risk_coverage": risk_coverage(cases),
        "models": models,
        "split_note": (
            f"Test cases: {len(cases)} tiles from CRC-VAL-HE-7K. Reference gallery and classifier training: "
            f"{n_gallery} tiles from NCT-CRC-HE-100K. The two sets come from different patients, "
            "so similar-tile retrieval cannot return tiles from the test case's own patient."
        ),
    }


def export_examples(examples_dir: Path, queries_csv: Path, out: Path, seed: int) -> int:
    """One example tile per class for the Lookup tab, excluding tiles that are worklist cases."""
    taken = {Path(p).name for p in pd.read_csv(queries_csv, usecols=["image_path"])["image_path"]}
    picks = []
    for class_dir in sorted(p for p in examples_dir.iterdir() if p.is_dir()):
        tiles = sorted(t for t in class_dir.glob("*.tif") if t.name not in taken)
        picks += tiles[:1]
    random.Random(seed).shuffle(picks)
    ex_dir = out / "examples"
    ex_dir.mkdir(parents=True, exist_ok=True)
    for n, tile in enumerate(picks, start=1):
        export_png(str(tile), ex_dir / f"example_{n:02d}.png")
    return len(picks)


def build(gallery_csv: Path, queries_csv: Path, out: Path, seed: int, compare: list[str]) -> list[dict]:
    g_paths, g_labels, g_emb = load_embeddings(gallery_csv)
    q_paths, q_labels, q_emb = load_embeddings(queries_csv)
    clf = StubClassifier(g_emb, g_labels, seed=seed)
    retriever = StubRetriever(g_emb, g_labels, g_paths)

    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle so case IDs don't reveal the true class.
    order = list(range(len(q_paths)))
    random.Random(seed).shuffle(order)

    cases = []
    for n, i in enumerate(order, start=1):
        case_id = f"case_{n:03d}"
        export_png(q_paths[i], img_dir / f"{case_id}.png")
        result = localize_result(analyze_case(q_emb[i], clf, retriever), img_dir)
        cases.append({
            "case_id": case_id,
            "image": f"images/{case_id}.png",
            "true_label": str(q_labels[i]),
            **result,
        })

    meta = {
        "source": "semi-real data (stub models)",
        "dataset": "queries: CRC-VAL-HE-7K subset; gallery: NCT-CRC-HE-100K subset",
        "model": "QuiltNet-B-32 embeddings; classifier/uncertainty/retrieval are stubs",
        "classes": clf.classes,
        "class_names": {c: CRC_CLASS_NAMES.get(c, c) for c in clf.classes},
        # What one gallery item is, for UI wording: "tiles" for tissue-type labels,
        # "lesions" once labels are diagnostic lesion categories.
        "reference_noun": "tiles",
    }
    evidence = build_evidence(cases, gallery_csv, queries_csv, compare, seed)
    (out / "cases.json").write_text(json.dumps(cases, indent=2))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2))
    return cases


def summarize(cases: list[dict]) -> None:
    wrong = [c for c in cases if c["prediction"] != c["true_label"]]
    review = [c for c in cases if c["risk"] == "review"]
    print(f"{len(cases)} cases | stub accuracy {1 - len(wrong) / len(cases):.3f} | "
          f"review {len(review)} | wrong {len(wrong)} (flagged {sum(c['risk'] == 'review' for c in wrong)})")
    print("demo candidates:")
    print("  confident & correct:", [c["case_id"] for c in cases if c["risk"] == "confident" and c not in wrong][:5])
    print("  review & correct:   ", [c["case_id"] for c in review if c not in wrong][:5])
    print("  wrong but flagged:  ", [c["case_id"] for c in wrong if c["risk"] == "review"][:5])
    print("  wrong, NOT flagged: ", [c["case_id"] for c in wrong if c["risk"] == "confident"][:5])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gallery", type=Path, required=True, help="embedding CSV of confirmed reference tiles")
    p.add_argument("--queries", type=Path, required=True, help="embedding CSV of test cases")
    p.add_argument("--out", type=Path, default=Path("ui/data_semi"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", nargs="*", default=[], help="other models for the comparison table, e.g. plip")
    p.add_argument("--examples", type=Path, default=None, help="tile folder (one subfolder per class) for Lookup examples")
    args = p.parse_args()
    summarize(build(args.gallery, args.queries, args.out, args.seed, args.compare))
    if args.examples:
        print(f"exported {export_examples(args.examples, args.queries, args.out, args.seed)} lookup examples")


if __name__ == "__main__":
    main()
