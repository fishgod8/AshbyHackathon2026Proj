"""Annotate prediction CSV rows with a confidence-based review flag.

Web integrations can pass CSV row mappings to ``annotate_predictions`` and
receive mappings with a boolean ``needs_review`` field. The command-line
interface reads and writes CSV files directly.
"""

import argparse
import csv
from itertools import chain
import math
import sys
from collections.abc import Iterable, Mapping
from numbers import Real
from pathlib import Path


REQUIRED_COLUMNS = ("image_id", "predicted_label", "confidence")


def _validate_threshold(threshold: float) -> None:
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("threshold must be a finite number between 0 and 1")


def annotate_predictions(
    records: Iterable[Mapping[str, object]], *, threshold: float
) -> list[dict[str, object]]:
    """Return CSV-like prediction records with a boolean ``needs_review``.

    Required fields are ``image_id``, ``predicted_label``, and ``confidence``.
    All original fields are retained. The review rule is strictly
    ``confidence < threshold``. This function is framework-independent and can
    be called by a web app after parsing uploaded CSV rows.
    """
    _validate_threshold(threshold)
    iterator = iter(records)
    try:
        first_record = next(iterator)
    except StopIteration as error:
        raise ValueError("input must contain at least one prediction row") from error

    columns = list(first_record.keys())
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing_columns:
        raise ValueError(
            "input is missing required column(s): " + ", ".join(missing_columns)
        )
    if None in columns:
        raise ValueError("input row contains extra fields without a column name")

    annotated = []
    for row_number, record in enumerate(chain((first_record,), iterator), start=1):
        if None in record:
            raise ValueError(f"row {row_number}: extra fields without a column name")

        missing_values = [
            column
            for column in REQUIRED_COLUMNS
            if record.get(column) is None or not str(record[column]).strip()
        ]
        if missing_values:
            raise ValueError(
                f"row {row_number}: missing value(s) in " + ", ".join(missing_values)
            )

        raw_confidence = record["confidence"]
        if isinstance(raw_confidence, bool):
            raise ValueError(f"row {row_number}: confidence must be a number")
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError) as error:
            raise ValueError(f"row {row_number}: confidence must be a number") from error
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"row {row_number}: confidence must be finite and between 0 and 1"
            )

        annotated_record = dict(record)
        annotated_record["needs_review"] = confidence < threshold
        annotated.append(annotated_record)

    return annotated


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        content = (line for line in csv_file if not line.lstrip().startswith("#"))
        reader = csv.DictReader(content, strict=True)
        if reader.fieldnames is None:
            raise ValueError("input CSV is empty or has no header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("input CSV has duplicate column names")
        rows = list(reader)
        return reader.fieldnames, rows


def annotate_csv(input_csv: Path, output_csv: Path, *, threshold: float) -> None:
    """Validate, annotate, and write a prediction CSV file."""
    fieldnames, rows = _read_csv(input_csv)
    annotated = annotate_predictions(rows, threshold=threshold)
    output_columns = list(fieldnames)
    if "needs_review" not in output_columns:
        output_columns.append("needs_review")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(annotated)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add needs_review flags to prediction rows using confidence."
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="input CSV path")
    parser.add_argument("--output-csv", type=Path, required=True, help="output CSV path")
    parser.add_argument(
        "--threshold", type=float, required=True, help="review threshold from 0 to 1"
    )
    arguments = parser.parse_args()

    try:
        annotate_csv(
            arguments.input_csv,
            arguments.output_csv,
            threshold=arguments.threshold,
        )
    except (OSError, csv.Error, UnicodeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Annotated CSV written to {arguments.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())