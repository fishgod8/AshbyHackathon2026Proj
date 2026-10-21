"""Run the reproducible BACH confidence/conformal experiment from embeddings.

This script never loads QuiltNet or images. It treats the supplied image-level
embeddings as immutable input and labels all test results exploratory because
the repository's 40-image test split has been inspected previously.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.uncertainty.conformal import assess_conformal_prediction, calibrate_conformal
from uncertainty import assess_prediction


CLASS_NAMES = ("Benign", "InSitu", "Invasive", "Normal")
ORIGINAL_SPLITS = ("train", "val", "test")
DEFAULT_C_VALUES = (0.01, 0.1, 1.0, 10.0)
DEFAULT_SPLIT_SALT = "bach-calibration-v1"
EXPECTED_EMBEDDING_DIMENSION = 512
EXPLORATORY_NOTICE = (
    "EXPLORATORY: the 40-image BACH test split was inspected in earlier work; "
    "these results are not a pristine final benchmark."
)


@dataclass(frozen=True)
class EmbeddingRecord:
    image_id: str
    label: str
    original_split: str
    embedding: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_ref_commit(reference: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", reference],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_embedding_records(
    path: Path, expected_dimension: int = EXPECTED_EMBEDDING_DIMENSION
) -> list[EmbeddingRecord]:
    """Load and validate one finite image embedding per unique BACH image ID."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file, strict=True)
        required = {"image_id", "label", "split", "embedding"}
        if reader.fieldnames is None:
            raise ValueError("embedding CSV is empty or has no header")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(
                "embedding CSV is missing required column(s): " + ", ".join(missing)
            )

        records = []
        seen_ids = set()
        for row_number, row in enumerate(reader, start=1):
            if None in row:
                raise ValueError(f"row {row_number}: too many CSV fields")
            image_id = (row.get("image_id") or "").strip()
            label = (row.get("label") or "").strip()
            split = (row.get("split") or "").strip()
            raw_embedding = (row.get("embedding") or "").strip()
            if not image_id:
                raise ValueError(f"row {row_number}: image_id must not be empty")
            if image_id in seen_ids:
                raise ValueError(f"row {row_number}: duplicate image_id {image_id}")
            seen_ids.add(image_id)
            if label not in CLASS_NAMES:
                raise ValueError(f"row {row_number}: unknown BACH label {label!r}")
            if split not in ORIGINAL_SPLITS:
                raise ValueError(f"row {row_number}: unknown split {split!r}")
            try:
                parsed = ast.literal_eval(raw_embedding)
            except (SyntaxError, ValueError) as error:
                raise ValueError(
                    f"row {row_number}: embedding is not a valid numeric list"
                ) from error
            try:
                # Match Mayumi's train_bach.py, which parses embeddings as float32.
                embedding = np.asarray(parsed, dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"row {row_number}: embedding must contain numeric values"
                ) from error
            if embedding.ndim != 1 or embedding.shape[0] != expected_dimension:
                raise ValueError(
                    f"row {row_number}: embedding must have exactly "
                    f"{expected_dimension} dimensions"
                )
            if not np.isfinite(embedding).all():
                raise ValueError(f"row {row_number}: embedding values must be finite")
            records.append(EmbeddingRecord(image_id, label, split, embedding))

    if not records:
        raise ValueError("embedding CSV must contain at least one record")
    return records


def validate_bach_layout(
    records: list[EmbeddingRecord],
    train_per_class: int = 80,
    validation_per_class: int = 10,
    test_per_class: int = 10,
) -> None:
    """Require the expected stratified original-image BACH split."""
    counts = Counter((record.label, record.original_split) for record in records)
    expected = {
        (label, split): count
        for label in CLASS_NAMES
        for split, count in (
            ("train", train_per_class),
            ("val", validation_per_class),
            ("test", test_per_class),
        )
    }
    if counts != expected:
        differences = [
            f"{label}/{split}: expected {count}, found {counts[(label, split)]}"
            for (label, split), count in expected.items()
            if counts[(label, split)] != count
        ]
        raise ValueError("unexpected BACH split counts: " + "; ".join(differences))


def select_calibration_ids(
    records: list[EmbeddingRecord],
    per_class: int = 10,
    salt: str = DEFAULT_SPLIT_SALT,
) -> set[str]:
    """Select calibration images deterministically from the original train split."""
    if isinstance(per_class, bool) or not isinstance(per_class, int) or per_class < 1:
        raise ValueError("calibration per class must be a positive integer")
    if not isinstance(salt, str) or not salt:
        raise ValueError("split salt must be a non-empty string")

    selected = set()
    for label in CLASS_NAMES:
        candidates = [
            record
            for record in records
            if record.label == label and record.original_split == "train"
        ]
        if len(candidates) <= per_class:
            raise ValueError(
                f"class {label} needs more than {per_class} original train images"
            )
        ranked = sorted(
            candidates,
            key=lambda record: (
                hashlib.sha256(f"{salt}|{record.image_id}".encode()).hexdigest(),
                record.image_id,
            ),
        )
        selected.update(record.image_id for record in ranked[:per_class])
    return selected


def final_split(record: EmbeddingRecord, calibration_ids: set[str]) -> str:
    if record.image_id in calibration_ids:
        if record.original_split != "train":
            raise ValueError("calibration images must come from the original train split")
        return "calibration"
    return record.original_split


def _unpack(records: list[EmbeddingRecord]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([record.embedding for record in records]),
        np.asarray([record.label for record in records]),
    )


def fit_classifier(
    records: list[EmbeddingRecord],
    calibration_ids: set[str],
    c_values: tuple[float, ...] = DEFAULT_C_VALUES,
) -> tuple[object, float, list[dict[str, float]]]:
    """Select C on train/validation, then refit on train plus validation."""
    train = [
        record
        for record in records
        if final_split(record, calibration_ids) == "train"
    ]
    validation = [record for record in records if record.original_split == "val"]
    if not c_values or any(not math.isfinite(c) or c <= 0 for c in c_values):
        raise ValueError("C values must be finite positive numbers")

    x_train, y_train = _unpack(train)
    x_validation, y_validation = _unpack(validation)
    best_c = None
    best_accuracy = -1.0
    validation_results = []
    for c_value in c_values:
        candidate = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, max_iter=3000),
        )
        candidate.fit(x_train, y_train)
        accuracy = float(accuracy_score(y_validation, candidate.predict(x_validation)))
        validation_results.append({"C": float(c_value), "accuracy": accuracy})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_c = float(c_value)

    fitting_records = train + validation
    x_fitting, y_fitting = _unpack(fitting_records)
    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=best_c, max_iter=3000),
    )
    final_model.fit(x_fitting, y_fitting)
    return final_model, best_c, validation_results


def prediction_rows(
    model, records: list[EmbeddingRecord], split_name: str
) -> list[dict[str, object]]:
    """Generate validated probability rows in the fitted model's class order."""
    if not records:
        raise ValueError(f"{split_name} split must not be empty")
    x_values, _ = _unpack(records)
    probabilities = model.predict_proba(x_values)
    predicted_labels = model.predict(x_values)
    class_names = [str(name) for name in model.classes_]
    rows = []
    for record, predicted_label, scores in zip(
        records, predicted_labels, probabilities
    ):
        score_map = {
            class_name: float(score)
            for class_name, score in zip(class_names, scores)
        }
        confidence = max(score_map.values())
        rows.append(
            {
                "image_id": record.image_id,
                "true_label": record.label,
                "predicted_label": str(predicted_label),
                "confidence": confidence,
                **{
                    f"score_{class_name}": score_map[class_name]
                    for class_name in class_names
                },
                "split": split_name,
            }
        )
    return rows


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def policy_metrics(
    policy_name: str,
    true_labels: list[str],
    predicted_labels: list[str],
    review_flags: list[bool],
) -> dict[str, object]:
    if not (len(true_labels) == len(predicted_labels) == len(review_flags)):
        raise ValueError("policy metric inputs must have matching lengths")
    if not true_labels:
        raise ValueError("policy metric inputs must not be empty")
    errors = [true != predicted for true, predicted in zip(true_labels, predicted_labels)]
    review_count = sum(review_flags)
    total_errors = sum(errors)
    captured_errors = sum(error and review for error, review in zip(errors, review_flags))
    unreviewed_errors = total_errors - captured_errors
    unreviewed_count = len(errors) - review_count
    return {
        "policy": policy_name,
        "review_count": review_count,
        "review_rate": _rate(review_count, len(errors)),
        "captured_errors": captured_errors,
        "error_capture_rate": _rate(captured_errors, total_errors),
        "reviewed_error_rate": _rate(captured_errors, review_count),
        "unreviewed_error_rate": _rate(unreviewed_errors, unreviewed_count),
    }


def evaluate_policies(
    test_rows: list[dict[str, object]],
    calibration_artifact: dict[str, object],
    confidence_threshold: float,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    """Evaluate three policies at settings fixed before inspecting test results."""
    class_names = list(calibration_artifact["class_names"])
    confidence_flags = []
    conformal_flags = []
    prediction_sets = []
    annotated_rows = []
    for row in test_rows:
        scores = {name: float(row[f"score_{name}"]) for name in class_names}
        prediction = {
            "predicted_class": str(row["predicted_label"]),
            "scores": scores,
        }
        confidence_result = assess_prediction(prediction, confidence_threshold)
        conformal_result = assess_conformal_prediction(
            prediction, calibration_artifact
        )
        covered = str(row["true_label"]) in conformal_result["prediction_set"]
        confidence_review = bool(confidence_result["needs_review"])
        conformal_review = bool(conformal_result["needs_review"])
        combined_review = confidence_review or conformal_review
        confidence_flags.append(confidence_review)
        conformal_flags.append(conformal_review)
        prediction_sets.append(conformal_result["prediction_set"])
        annotated_rows.append(
            {
                **row,
                "prediction_set": json.dumps(conformal_result["prediction_set"]),
                "set_size": conformal_result["set_size"],
                "covered": covered,
                "confidence_review": confidence_review,
                "conformal_review": conformal_review,
                "combined_review": combined_review,
                "is_error": row["true_label"] != row["predicted_label"],
            }
        )

    true_labels = [str(row["true_label"]) for row in test_rows]
    predicted_labels = [str(row["predicted_label"]) for row in test_rows]
    combined_flags = [
        confidence or conformal
        for confidence, conformal in zip(confidence_flags, conformal_flags)
    ]
    comparison = [
        policy_metrics("confidence_only", true_labels, predicted_labels, confidence_flags),
        policy_metrics("conformal_only", true_labels, predicted_labels, conformal_flags),
        policy_metrics("combined_or", true_labels, predicted_labels, combined_flags),
    ]
    set_sizes = [len(prediction_set) for prediction_set in prediction_sets]
    conformal_metrics = {
        "empirical_coverage": sum(
            label in prediction_set
            for label, prediction_set in zip(true_labels, prediction_sets)
        )
        / len(test_rows),
        "average_prediction_set_size": sum(set_sizes) / len(set_sizes),
        "singleton_rate": sum(size == 1 for size in set_sizes) / len(set_sizes),
        "empty_set_rate": sum(size == 0 for size in set_sizes) / len(set_sizes),
        "multi_class_set_rate": sum(size > 1 for size in set_sizes) / len(set_sizes),
    }
    for policy in comparison:
        policy.update(conformal_metrics)
    return comparison, conformal_metrics, annotated_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_safe_artifact(artifact: dict[str, object]) -> dict[str, object]:
    safe = dict(artifact)
    if math.isinf(float(safe["quantile"])):
        safe["quantile"] = "Infinity"
    return safe


def run_experiment(
    embedding_csv: Path,
    output_directory: Path,
    *,
    alpha: float = 0.10,
    confidence_threshold: float = 0.70,
    calibration_per_class: int = 10,
    split_salt: str = DEFAULT_SPLIT_SALT,
    expected_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
    train_per_class: int = 80,
    validation_per_class: int = 10,
    test_per_class: int = 10,
) -> dict[str, object]:
    """Run and save the complete experiment; intended for CLI and tests."""
    records = load_embedding_records(embedding_csv, expected_dimension)
    validate_bach_layout(
        records, train_per_class, validation_per_class, test_per_class
    )
    calibration_ids = select_calibration_ids(
        records, calibration_per_class, split_salt
    )
    model, selected_c, validation_results = fit_classifier(records, calibration_ids)
    class_names = [str(name) for name in model.classes_]
    if class_names != list(CLASS_NAMES):
        raise ValueError(
            f"unexpected fitted class order {class_names}; expected {list(CLASS_NAMES)}"
        )

    calibration_records = [
        record
        for record in records
        if final_split(record, calibration_ids) == "calibration"
    ]
    test_records = [record for record in records if record.original_split == "test"]
    calibration_rows = prediction_rows(model, calibration_records, "calibration")
    test_rows = prediction_rows(model, test_records, "test")
    calibration_probabilities = [
        [float(row[f"score_{name}"]) for name in class_names]
        for row in calibration_rows
    ]
    calibration_labels = [str(row["true_label"]) for row in calibration_rows]
    calibration_artifact = calibrate_conformal(
        calibration_probabilities,
        calibration_labels,
        class_names,
        alpha,
    )
    calibration_artifact.update(
        {
            "schema_version": 1,
            "calibration_status": "separate_refitted_classifier_experiment",
            "production_compatible": False,
            "calibration_image_ids": sorted(calibration_ids),
            "source_embedding_csv_sha256": _sha256(embedding_csv),
            "split_salt": split_salt,
            "model_type": "StandardScaler + LogisticRegression",
            "selected_C": selected_c,
            "code_commit": _git_commit(),
            "working_tree_dirty": _git_dirty(),
            "experiment_script_sha256": _sha256(Path(__file__)),
        }
    )
    comparison, conformal_metrics, annotated_test_rows = evaluate_policies(
        test_rows, calibration_artifact, confidence_threshold
    )
    scaler = model.named_steps["standardscaler"]
    logistic_regression = model.named_steps["logisticregression"]

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_rows = [
        {
            "image_id": record.image_id,
            "label": record.label,
            "original_split": record.original_split,
            "final_split": final_split(record, calibration_ids),
            "split_method": "sha256_rank_within_class_from_original_train",
            "split_salt": split_salt,
        }
        for record in sorted(records, key=lambda item: item.image_id)
    ]
    _write_csv(output_directory / "split_manifest.csv", manifest_rows)
    _write_csv(output_directory / "calibration_predictions.csv", calibration_rows)
    _write_csv(output_directory / "test_predictions.csv", annotated_test_rows)
    _write_csv(output_directory / "policy_comparison.csv", comparison)
    with (output_directory / "calibration_artifact.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(
            _json_safe_artifact(calibration_artifact),
            output_file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output_file.write("\n")

    split_counts = Counter(row["final_split"] for row in manifest_rows)
    summary = {
        "experiment_status": "exploratory",
        "notice": EXPLORATORY_NOTICE,
        "input": {
            "embedding_csv": str(embedding_csv.resolve()),
            "sha256": _sha256(embedding_csv),
            "record_count": len(records),
            "embedding_dimension": expected_dimension,
        },
        "split": {
            "method": "sha256_rank_within_class_from_original_train",
            "salt": split_salt,
            "calibration_per_class": calibration_per_class,
            "counts": dict(sorted(split_counts.items())),
        },
        "model": {
            "type": "sklearn.pipeline(StandardScaler, LogisticRegression)",
            "class_order": class_names,
            "candidate_C_values": list(DEFAULT_C_VALUES),
            "selected_C": selected_c,
            "selection_data": "final train for fitting; original validation for C selection",
            "final_fit_data": "final train plus original validation only",
            "logistic_regression_max_iter": 3000,
            "standard_scaler": {
                "with_mean": scaler.with_mean,
                "with_std": scaler.with_std,
            },
            "logistic_regression": {
                "C": float(logistic_regression.C),
                "solver": logistic_regression.solver,
                "penalty": logistic_regression.penalty,
                "class_weight": logistic_regression.class_weight,
                "max_iter": logistic_regression.max_iter,
                "random_state": logistic_regression.random_state,
            },
            "validation_results": validation_results,
        },
        "uncertainty": {
            "alpha": alpha,
            "confidence_threshold": confidence_threshold,
            "combined_rule": "confidence_review OR conformal_review",
            "conformal_metrics": conformal_metrics,
            "policy_comparison": comparison,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "code_commit": _git_commit(),
            "working_tree_dirty": _git_dirty(),
            "experiment_script_sha256": _sha256(Path(__file__)),
            "inspected_mayumi_commit": _git_ref_commit("origin/mayumi-model"),
        },
        "pipeline_alignment": {
            "matches_mayumi_components": (
                "StandardScaler + LogisticRegression; C grid 0.01, 0.1, 1.0, 10.0; "
                "max_iter=3000"
            ),
            "known_difference": (
                "this experiment removes calibration images from training and refits "
                "on final train plus validation; Mayumi's inspected pipeline trains on "
                "all original train images and does not perform that refit"
            ),
        },
    }
    with (output_directory / "experiment_summary.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embedding_csv", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--calibration-per-class", type=int, default=10)
    parser.add_argument("--split-salt", default=DEFAULT_SPLIT_SALT)
    arguments = parser.parse_args()
    try:
        summary = run_experiment(
            arguments.embedding_csv,
            arguments.output_directory,
            alpha=arguments.alpha,
            confidence_threshold=arguments.confidence_threshold,
            calibration_per_class=arguments.calibration_per_class,
            split_salt=arguments.split_salt,
        )
    except (OSError, csv.Error, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(EXPLORATORY_NOTICE)
    print(f"Selected C: {summary['model']['selected_C']}")
    print(f"Outputs: {arguments.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
