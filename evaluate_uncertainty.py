"""Compare confidence-threshold review policies using labeled CSV predictions.

The bundled synthetic CSV is fake data for checking this script only; it does
not represent real model performance. Run with:
    python evaluate_uncertainty.py synthetic_uncertainty_example.csv
"""

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
REQUIRED_COLUMNS = ("image_id", "true_label", "predicted_label", "confidence")


@dataclass(frozen=True)
class Prediction:
    image_id: str
    true_label: str
    predicted_label: str
    confidence: float

    @property
    def is_error(self) -> bool:
        return self.true_label != self.predicted_label


def read_predictions(path: Path) -> list[Prediction]:
    """Read and validate prediction rows from a CSV file."""
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        content = (line for line in csv_file if not line.lstrip().startswith("#"))
        reader = csv.DictReader(content, strict=True)
        if reader.fieldnames is None:
            raise ValueError("CSV is empty or has no header")

        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                "CSV is missing required column(s): " + ", ".join(missing_columns)
            )

        predictions = []
        for record_number, row in enumerate(reader, start=1):
            if None in row:
                raise ValueError(f"record {record_number}: too many CSV fields")

            values = {column: row.get(column) for column in REQUIRED_COLUMNS}
            missing_values = [
                column
                for column, value in values.items()
                if value is None or not value.strip()
            ]
            if missing_values:
                raise ValueError(
                    f"record {record_number}: missing value(s) in "
                    + ", ".join(missing_values)
                )

            try:
                confidence = float(values["confidence"])
            except ValueError as error:
                raise ValueError(
                    f"record {record_number}: confidence must be a number"
                ) from error

            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"record {record_number}: confidence must be finite and "
                    "between 0 and 1"
                )

            predictions.append(
                Prediction(
                    image_id=values["image_id"].strip(),
                    true_label=values["true_label"].strip(),
                    predicted_label=values["predicted_label"].strip(),
                    confidence=confidence,
                )
            )

    return predictions


def format_rate(numerator: int, denominator: int) -> str:
    """Format a ratio as a percentage, or N/A when its denominator is zero."""
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator:.1%}"


def build_comparison(predictions: list[Prediction]) -> list[dict[str, object]]:
    """Calculate review and error-capture metrics for each configured threshold."""
    total_count = len(predictions)
    total_errors = sum(prediction.is_error for prediction in predictions)
    results = []

    for threshold in THRESHOLDS:
        reviewed = [
            prediction
            for prediction in predictions
            if prediction.confidence < threshold
        ]
        unreviewed = [
            prediction
            for prediction in predictions
            if prediction.confidence >= threshold
        ]
        captured_errors = sum(prediction.is_error for prediction in reviewed)
        reviewed_errors = captured_errors
        unreviewed_errors = sum(prediction.is_error for prediction in unreviewed)

        results.append(
            {
                "threshold": threshold,
                "review_count": len(reviewed),
                "review_rate": format_rate(len(reviewed), total_count),
                "captured_errors": captured_errors,
                "captured_error_rate": format_rate(captured_errors, total_errors),
                "reviewed_error_rate": format_rate(
                    reviewed_errors, len(reviewed)
                ),
                "unreviewed_error_rate": format_rate(
                    unreviewed_errors, len(unreviewed)
                ),
            }
        )

    return results


def print_comparison(results: list[dict[str, object]], output: TextIO) -> None:
    """Print comparison rows as a compact, dependency-free text table."""
    columns = (
        ("Threshold", "threshold"),
        ("Review count", "review_count"),
        ("Review rate", "review_rate"),
        ("Errors captured", "captured_errors"),
        ("Captured / all errors", "captured_error_rate"),
        ("Reviewed error rate", "reviewed_error_rate"),
        ("Unreviewed error rate", "unreviewed_error_rate"),
    )
    rows = [
        [
            f"{result[key]:.2f}" if key == "threshold" else str(result[key])
            for _, key in columns
        ]
        for result in results
    ]
    headers = [title for title, _ in columns]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(" | ".join(value.ljust(width) for value, width in zip(headers, widths)), file=output)
    print("-+-".join("-" * width for width in widths), file=output)
    for row in rows:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)), file=output)


def write_comparison_csv(
    results: list[dict[str, object]], path: Path, simulated: bool
) -> None:
    """Write comparison metrics to CSV and label simulated output explicitly."""
    columns = (
        "threshold",
        "review_count",
        "review_rate",
        "captured_errors",
        "captured_error_rate",
        "reviewed_error_rate",
        "unreviewed_error_rate",
        "data_status",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    **result,
                    "threshold": f"{result['threshold']:.2f}",
                    "data_status": "SIMULATED" if simulated else "PROVIDED_DATA",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare confidence thresholds against labeled CSV predictions."
    )
    parser.add_argument("csv_path", type=Path, help="input CSV file")
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="path for the comparison results CSV",
    )
    parser.add_argument(
        "--simulated",
        action="store_true",
        help="mark exported results as simulated data",
    )
    arguments = parser.parse_args()

    try:
        predictions = read_predictions(arguments.csv_path)
    except (OSError, csv.Error, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    results = build_comparison(predictions)
    try:
        write_comparison_csv(results, arguments.output_csv, arguments.simulated)
    except OSError as error:
        print(f"Error writing result CSV: {error}", file=sys.stderr)
        return 2

    print_comparison(results, sys.stdout)
    print(
        f"Results CSV: {arguments.output_csv} "
        f"({'SIMULATED' if arguments.simulated else 'PROVIDED_DATA'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())