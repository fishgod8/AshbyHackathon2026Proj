"""Run Mayumi's exact classifier on the unlabeled BACH challenge test photos."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis_pipeline import analyze_prediction
from bach.predict_image import predict_image
from model_identity import inspect_checkpoint


ENTROPY_THRESHOLDS = (.20, .30, .40, .50, .60, .70, .80)
MARGIN_THRESHOLDS = (.10, .20, .30, .40, .50, .60, .70, .80)
REVIEW_BUDGETS = (.10, .20, .30, .40)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(image_directory, output_directory, checkpoint, device="cpu"):
    images = sorted(Path(image_directory).glob("*.tif"), key=lambda p: int(p.stem.removeprefix("test")))
    if len(images) != 100:
        raise ValueError(f"expected 100 challenge photos, found {len(images)}")
    identity = inspect_checkpoint(checkpoint)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "image_predictions.csv"
    existing = {}
    if prediction_path.exists():
        with prediction_path.open(encoding="utf-8", newline="") as handle:
            existing = {row["image_id"]: row for row in csv.DictReader(handle)}

    rows = []
    for index, image in enumerate(images, 1):
        digest = file_hash(image)
        cached = existing.get(image.name)
        if cached and cached.get("embedding") and cached.get("image_sha256") == digest and cached.get("classifier_sha256") == identity["classifier_sha256"]:
            rows.append(cached)
            continue
        raw = predict_image(image, model_path=checkpoint, device=device)
        result = analyze_prediction(raw)
        row = {
            "image_id": image.name,
            "image_sha256": digest,
            "dataset": "ICIAR2018_BACH_Challenge_TestDataset/Photos",
            "ground_truth_status": "unavailable",
            "classifier_sha256": identity["classifier_sha256"],
            "predicted_label": result["predicted_class"],
            "confidence": result["confidence"],
            "max_probability_uncertainty": result["uncertainty"]["max_probability_uncertainty"],
            "normalized_entropy": result["uncertainty"]["normalized_entropy"],
            "top_two_margin": result["uncertainty"]["margin"],
            **{f"score_{name}": result["probabilities"][name] for name in identity["class_order"]},
            "embedding": json.dumps(raw["embedding"]),
        }
        rows.append(row)
        _write_csv(prediction_path, rows)
        print(f"[{index:03d}/100] {image.name}: {row['predicted_label']} confidence={row['confidence']:.4f}", flush=True)

    # Encoder-space duplicate audit against all 400 images used by Mayumi.
    with Path("bach/image_embeddings.csv").open(encoding="utf-8", newline="") as handle:
        reference_rows = list(csv.DictReader(handle))
    reference = np.asarray([ast.literal_eval(row["embedding"]) for row in reference_rows], dtype=np.float32)
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    for row in rows:
        vector = np.asarray(json.loads(row["embedding"]), dtype=np.float32)
        vector /= np.linalg.norm(vector)
        similarities = reference @ vector
        nearest = int(np.argmax(similarities))
        row["nearest_existing_image_id"] = reference_rows[nearest]["image_id"]
        row["nearest_existing_split"] = reference_rows[nearest]["split"]
        row["nearest_existing_cosine"] = float(similarities[nearest])
    _write_csv(prediction_path, rows)

    numeric = [{**row,
                "confidence": float(row["confidence"]),
                "normalized_entropy": float(row["normalized_entropy"]),
                "top_two_margin": float(row["top_two_margin"])} for row in rows]
    threshold_rows = []
    rules = [("confidence", .70, lambda r: r["confidence"] < .70)]
    rules += [("normalized_entropy", t, lambda r, t=t: r["normalized_entropy"] >= t) for t in ENTROPY_THRESHOLDS]
    rules += [("top_two_margin", t, lambda r, t=t: r["top_two_margin"] <= t) for t in MARGIN_THRESHOLDS]
    for metric, threshold, rule in rules:
        count = sum(rule(row) for row in numeric)
        threshold_rows.append({"rule": metric, "threshold": threshold, "flagged": count,
                               "review_rate": count / len(numeric), "labels_available": False})
    _write_csv(output / "threshold_review_rates.csv", threshold_rows)

    combination_rows = []
    confidence = lambda r: r["confidence"] < .70
    for entropy_threshold in ENTROPY_THRESHOLDS:
        entropy = lambda r, t=entropy_threshold: r["normalized_entropy"] >= t
        for margin_threshold in MARGIN_THRESHOLDS:
            margin = lambda r, t=margin_threshold: r["top_two_margin"] <= t
            named = {
                "A_confidence": confidence,
                "B_entropy": entropy,
                "C_margin": margin,
                "D_confidence_or_entropy": lambda r: confidence(r) or entropy(r),
                "E_confidence_or_margin": lambda r: confidence(r) or margin(r),
                "F_all_or": lambda r: confidence(r) or entropy(r) or margin(r),
            }
            for name, rule in named.items():
                count = sum(rule(row) for row in numeric)
                combination_rows.append({"rule": name, "entropy_threshold": entropy_threshold,
                                         "margin_threshold": margin_threshold, "flagged": count,
                                         "review_rate": count / len(numeric), "labels_available": False})
    _write_csv(output / "combination_review_rates.csv", combination_rows)

    budget_rows = []
    rankings = {
        "confidence": sorted(numeric, key=lambda r: (r["confidence"], r["image_id"])),
        "entropy": sorted(numeric, key=lambda r: (-r["normalized_entropy"], r["image_id"])),
        "margin": sorted(numeric, key=lambda r: (r["top_two_margin"], r["image_id"])),
    }
    for budget in REVIEW_BUDGETS:
        count = round(len(numeric) * budget)
        selected = {name: {row["image_id"] for row in ranking[:count]} for name, ranking in rankings.items()}
        for name, ids in selected.items():
            budget_rows.append({"budget": budget, "ranking": name, "selected": count,
                                "overlap_with_confidence": len(ids & selected["confidence"]),
                                "labels_available": False})
    _write_csv(output / "review_budget_overlap.csv", budget_rows)

    distributions = {}
    for key in ("confidence", "normalized_entropy", "top_two_margin"):
        values = np.asarray([row[key] for row in numeric])
        distributions[key] = {"min": float(values.min()), "p10": float(np.quantile(values, .1)),
                              "median": float(np.median(values)), "p90": float(np.quantile(values, .9)),
                              "max": float(values.max()), "mean": float(values.mean())}
    summary = {
        "status": "unlabeled_independent_challenge_test_images",
        "classifier_identity": identity,
        "image_count": len(numeric),
        "ground_truth_labels_available": False,
        "threshold_selection_status": "not_possible_without_ground_truth_labels",
        "error_detection_metrics": None,
        "overlap_audit": {
            "exact_filename_matches": 0,
            "exact_file_hash_matches": "not_testable_without_original_400_image_files",
            "encoder_nearest_cosine_max": max(float(row["nearest_existing_cosine"]) for row in rows),
            "encoder_nearest_cosine_at_least_0_999": sum(float(row["nearest_existing_cosine"]) >= .999 for row in rows),
        },
        "score_distributions": distributions,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for axis, key, title in zip(
            axes,
            ("confidence", "normalized_entropy", "top_two_margin"),
            ("Confidence", "Normalized entropy", "Top-two margin"),
        ):
            axis.hist([row[key] for row in numeric], bins=15)
            axis.set_title(title)
            axis.set_ylabel("Images")
        figure.suptitle("Unlabeled BACH challenge test photos (n=100)")
        figure.tight_layout()
        figure.savefig(output / "score_distributions.png", dpi=160)
        plt.close(figure)
    except ImportError:
        pass
    with (output / "REPORT.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# BACH challenge test uncertainty analysis\n\n"
            "The 100 microscopy images are an official challenge test set distinct from the 400 images used by "
            "Mayumi, but this download contains no image-wise ground-truth labels. Therefore error recall, review "
            "precision, unflagged error rate, and scientifically selected entropy/margin thresholds are unavailable.\n\n"
            f"Classifier SHA-256: `{identity['classifier_sha256']}`. All predictions use QuiltNet-B-32, Mayumi's "
            "12-patch mean aggregation, and her saved StandardScaler/LogisticRegression checkpoint.\n\n"
            f"Confidence < 0.70 flags {sum(row['confidence'] < .70 for row in numeric)}/100 images. "
            "Candidate threshold review rates are in `threshold_review_rates.csv`; same-budget ranking overlap is "
            "in `review_budget_overlap.csv`. These describe workload only, not error-detection effectiveness.\n\n"
            "No final entropy or margin threshold is recommended without trusted ground truth.\n"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("bach/classifier.pkl"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.image_directory, args.output_directory, args.checkpoint, args.device), indent=2))


if __name__ == "__main__":
    main()
