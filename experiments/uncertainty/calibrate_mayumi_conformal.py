"""Calibrate split conformal prediction for Mayumi's exact checkpoint.

Input rows must be predictions produced by that checkpoint on a genuinely
independent labeled calibration split. This command deliberately requires an
explicit independence attestation and records full provenance in the artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.uncertainty.conformal import calibrate_conformal, predict_set
from model_identity import inspect_checkpoint
from uncertainty import assess_probabilities


def read_prediction_csv(path, class_order):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        required = {"image_id", "true_label", *(f"score_{name}" for name in class_order)}
        if reader.fieldnames is None:
            raise ValueError("prediction CSV has no header")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError("prediction CSV is missing: " + ", ".join(missing))
        rows = []
        for number, row in enumerate(reader, 1):
            image_id = (row.get("image_id") or "").strip()
            label = (row.get("true_label") or "").strip()
            if not image_id or label not in class_order:
                raise ValueError(f"row {number}: invalid image_id or true_label")
            try:
                scores = {name: float(row[f"score_{name}"]) for name in class_order}
                assess_probabilities(scores)
            except ValueError as error:
                raise ValueError(f"row {number}: {error}") from error
            rows.append({"image_id": image_id, "true_label": label, "scores": scores})
    if not rows:
        raise ValueError("prediction CSV contains no records")
    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prediction CSV contains duplicate image_id values")
    return rows


def build_artifact(checkpoint, calibration_csv, *, alpha, dataset_name,
                   dataset_split, independence_note):
    identity = inspect_checkpoint(checkpoint)
    rows = read_prediction_csv(calibration_csv, identity["class_order"])
    artifact = calibrate_conformal(
        [[row["scores"][name] for name in identity["class_order"]] for row in rows],
        [row["true_label"] for row in rows], identity["class_order"], alpha,
    )
    artifact.update({
        "schema_version": 2,
        "calibration_status": "independent_labeled_calibration",
        "classifier_identity": identity,
        "dataset_provenance": {
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "prediction_csv": str(Path(calibration_csv).resolve()),
            "independence_note": independence_note,
        },
        "calibration_image_ids": sorted(row["image_id"] for row in rows),
    })
    return artifact


def evaluate_artifact(artifact, evaluation_csv):
    rows = read_prediction_csv(evaluation_csv, artifact["class_names"])
    calibration_ids = set(artifact["calibration_image_ids"])
    evaluation_ids = {row["image_id"] for row in rows}
    overlap = calibration_ids & evaluation_ids
    if overlap:
        raise ValueError(f"calibration/evaluation overlap detected ({len(overlap)} images)")
    sets = [predict_set([row["scores"][name] for name in artifact["class_names"]],
                        artifact["class_names"], artifact["quantile"])
            for row in rows]
    sizes = [len(values) for values in sets]
    return {
        "evaluation_sample_count": len(rows),
        "empirical_marginal_coverage": sum(row["true_label"] in values for row, values in zip(rows, sets)) / len(rows),
        "average_prediction_set_size": sum(sizes) / len(sizes),
        "multiple_label_rate": sum(size > 1 for size in sizes) / len(sizes),
        "empty_set_rate": sum(size == 0 for size in sizes) / len(sizes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_predictions", type=Path)
    parser.add_argument("output_artifact", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("bach/classifier.pkl"))
    parser.add_argument("--alpha", type=float, default=.10)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-split", required=True)
    parser.add_argument("--independence-note", required=True)
    parser.add_argument("--evaluation-predictions", type=Path)
    args = parser.parse_args()
    artifact = build_artifact(
        args.checkpoint, args.calibration_predictions, alpha=args.alpha,
        dataset_name=args.dataset_name, dataset_split=args.dataset_split,
        independence_note=args.independence_note,
    )
    artifact["coverage_evaluation"] = None
    if args.evaluation_predictions:
        artifact["coverage_evaluation"] = evaluate_artifact(artifact, args.evaluation_predictions)
    args.output_artifact.parent.mkdir(parents=True, exist_ok=True)
    with args.output_artifact.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved exact-checkpoint conformal artifact: {args.output_artifact}")


if __name__ == "__main__":
    main()
