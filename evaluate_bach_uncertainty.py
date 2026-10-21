"""Evaluate three uncertainty metrics on a real labeled prediction CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

from model_identity import inspect_checkpoint, sha256_file
from uncertainty import assess_probabilities


CANDIDATES = {
    "max_probability_uncertainty": (0.10, 0.20, 0.30, 0.40, 0.50),
    "normalized_entropy": (0.30, 0.50, 0.70, 0.80, 0.90),
    "inverse_margin": (0.50, 0.60, 0.70, 0.80, 0.90),
}


def _rate(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file, strict=True)
        if not reader.fieldnames:
            raise ValueError("prediction CSV has no header")
        score_columns = [name for name in reader.fieldnames if name.startswith("score_")]
        if len(score_columns) < 2:
            raise ValueError("prediction CSV must contain at least two score_* columns")
        required = {"image_id", "true_label", "predicted_label", "confidence"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError("prediction CSV is missing: " + ", ".join(missing))
        rows = []
        for number, raw in enumerate(reader, 1):
            try:
                scores = {name[6:]: float(raw[name]) for name in score_columns}
                metrics = assess_probabilities(scores, confidence_threshold=0.70)
                confidence = float(raw["confidence"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"row {number}: {error}") from error
            if not math.isclose(confidence, metrics["confidence"], abs_tol=1e-6):
                raise ValueError(f"row {number}: confidence does not match scores")
            if raw["predicted_label"] != metrics["predicted_class"]:
                raise ValueError(f"row {number}: predicted_label does not match scores")
            rows.append({
                **raw,
                "confidence": confidence,
                "is_error": raw["true_label"] != raw["predicted_label"],
                "max_probability_uncertainty": metrics["max_probability_uncertainty"],
                "entropy": metrics["entropy"],
                "normalized_entropy": metrics["normalized_entropy"],
                "margin": metrics["margin"],
                "inverse_margin": 1.0 - metrics["margin"],
            })
    if not rows:
        raise ValueError("prediction CSV contains no records")
    return rows


def metric_summary(rows):
    truth = [int(row["is_error"]) for row in rows]
    summaries = []
    for metric in CANDIDATES:
        scores = [float(row[metric]) for row in rows]
        if len(set(truth)) < 2:
            roc_auc = pr_auc = None
        else:
            roc_auc = float(roc_auc_score(truth, scores))
            pr_auc = float(average_precision_score(truth, scores))
        summaries.append({"metric": metric, "roc_auc": roc_auc, "pr_auc": pr_auc})
    return summaries


def threshold_rows(rows):
    output = []
    errors = sum(row["is_error"] for row in rows)
    for metric, thresholds in CANDIDATES.items():
        for threshold in thresholds:
            flags = [float(row[metric]) > threshold for row in rows]
            reviewed = sum(flags)
            captured = sum(flag and row["is_error"] for flag, row in zip(flags, rows))
            unreviewed = len(rows) - reviewed
            missed = errors - captured
            output.append({
                "metric": metric,
                "threshold": threshold,
                "review_count": reviewed,
                "review_rate": _rate(reviewed, len(rows)),
                "captured_errors": captured,
                "error_recall": _rate(captured, errors),
                "review_precision": _rate(captured, reviewed),
                "flagged_error_rate": _rate(captured, reviewed),
                "unflagged_error_rate": _rate(missed, unreviewed),
                "status": "exploratory_test_set_candidate",
            })
    return output


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(rows, output):
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay
    except ImportError:
        return []
    generated = []
    correct = [row["confidence"] for row in rows if not row["is_error"]]
    incorrect = [row["confidence"] for row in rows if row["is_error"]]
    plt.figure()
    plt.hist([correct, incorrect], bins=10, label=["correct", "incorrect"], alpha=.7)
    plt.xlabel("Maximum class probability")
    plt.ylabel("Images")
    plt.title(f"BACH test confidence (n={len(rows)})")
    plt.legend()
    path = output / "confidence_correct_vs_incorrect.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    generated.append(path.name)
    truth = [int(row["is_error"]) for row in rows]
    if len(set(truth)) == 2:
        _, axis = plt.subplots()
        for metric in CANDIDATES:
            RocCurveDisplay.from_predictions(
                truth, [row[metric] for row in rows], name=metric, ax=axis
            )
        axis.set_title(f"Error-detection ROC curves (n={len(rows)})")
        path = output / "error_detection_roc.png"
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        generated.append(path.name)
    return generated


def run(input_csv, output_directory, checkpoint=Path("bach/classifier.pkl")):
    rows = read_rows(input_csv)
    identity = inspect_checkpoint(checkpoint)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summaries = metric_summary(rows)
    thresholds = threshold_rows(rows)
    class_summary = []
    for label in sorted({row["true_label"] for row in rows}):
        selected = [row for row in rows if row["true_label"] == label]
        class_summary.append({"true_class": label, "count": len(selected),
                              "errors": sum(row["is_error"] for row in selected),
                              "error_rate": _rate(sum(row["is_error"] for row in selected), len(selected))})
    confusion = []
    labels = identity["class_order"]
    for true_label in labels:
        for predicted_label in labels:
            confusion.append({
                "true_label": true_label,
                "predicted_label": predicted_label,
                "count": sum(row["true_label"] == true_label and row["predicted_label"] == predicted_label for row in rows),
            })
    serializable_rows = [{key: value for key, value in row.items()} for row in rows]
    _write_csv(output / "image_uncertainty.csv", serializable_rows)
    _write_csv(output / "threshold_comparison.csv", thresholds)
    _write_csv(output / "metric_comparison.csv", summaries)
    _write_csv(output / "confusion_matrix.csv", confusion)
    plots = _plots(rows, output)
    errors = sum(row["is_error"] for row in rows)
    report = {
        "data_status": "real_supplied_classifier_test_predictions",
        "classifier_identity": identity,
        "prediction_csv_sha256": sha256_file(input_csv),
        "dataset_split": "BACH test",
        "evaluation_status": "exploratory_test_only_do_not_tune_thresholds",
        "sample_size": len(rows),
        "correct": len(rows) - errors,
        "incorrect": errors,
        "accuracy": (len(rows) - errors) / len(rows),
        "metric_comparison": summaries,
        "class_wise_errors": class_summary,
        "plots": plots,
        "threshold_recommendation": None,
        "threshold_note": "No independent threshold-selection data were available.",
    }
    with (output / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "REPORT.md").open("w", encoding="utf-8") as handle:
        class_lines = "\n".join(
            f"- {item['true_class']}: {item['count'] - item['errors']}/{item['count']} correct "
            f"({1 - item['error_rate']:.1%} accuracy)"
            for item in class_summary
        )
        handle.write(
            "# BACH uncertainty evaluation\n\n"
            f"Real Mayumi-checkpoint test predictions: n={len(rows)}, accuracy={report['accuracy']:.1%} "
            f"({len(rows)-errors} correct, {errors} incorrect).\n\n"
            f"Checkpoint SHA-256: `{identity['classifier_sha256']}`. Prediction CSV SHA-256: "
            f"`{report['prediction_csv_sha256']}`. Dataset split: BACH test.\n\n"
            "## Per-class accuracy\n\n" + class_lines + "\n\n"
            "These results are exploratory: the test set must not be used to select a final review threshold. "
            "ROC-AUC and PR-AUC measure error ranking, not probability calibration or clinical safety.\n\n"
            "No final review threshold is recommended because independent validation predictions for threshold "
            "selection are unavailable. See the CSV and JSON files for full metrics and class-wise counts.\n"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("bach/classifier.pkl"))
    args = parser.parse_args()
    report = run(args.input_csv, args.output_directory, args.checkpoint)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
