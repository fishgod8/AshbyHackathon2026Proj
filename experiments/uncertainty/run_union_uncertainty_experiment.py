"""Retrospective union-of-top-K uncertainty experiment on Mayumi's BACH test CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate_bach_uncertainty import read_rows
from model_identity import inspect_checkpoint, sha256_file


PERCENTAGES = tuple(value / 10 for value in range(50, 501, 25))
ERROR_IDS = {"is067.tif", "b100.tif", "iv062.tif", "is057.tif"}
TARGETS = (0.25, 0.50, 0.75, 1.00)


def rankings(rows):
    return {
        "confidence_only": sorted(rows, key=lambda row: (float(row["confidence"]), row["image_id"])),
        "entropy_only": sorted(rows, key=lambda row: (-float(row["normalized_entropy"]), row["image_id"])),
        "margin_only": sorted(rows, key=lambda row: (float(row["margin"]), row["image_id"])),
    }


def selected_for_base_k(ranked, base_count):
    sets = {name: {row["image_id"] for row in values[:base_count]} for name, values in ranked.items()}
    votes = {image_id: sum(image_id in values for values in sets.values())
             for image_id in {row["image_id"] for values in ranked.values() for row in values}}
    return {
        **sets,
        "union_all_three": {image_id for image_id, count in votes.items() if count >= 1},
        "two_of_three": {image_id for image_id, count in votes.items() if count >= 2},
        "intersection_all_three": {image_id for image_id, count in votes.items() if count == 3},
    }


def metrics(selected, errors, total):
    captured = selected & errors
    missed = errors - selected
    return {
        "selected_count": len(selected),
        "actual_review_rate": len(selected) / total,
        "correct_selected": len(selected) - len(captured),
        "incorrect_selected": len(captured),
        "error_recall": len(captured) / len(errors),
        "review_precision": len(captured) / len(selected) if selected else None,
        "captured_error_ids": sorted(captured),
        "missed_error_ids": sorted(missed),
    }


def minimum_workloads(curves):
    output = []
    for method, settings in curves.items():
        for target in TARGETS:
            eligible = [setting for setting in settings if setting["error_recall"] >= target]
            best = min(eligible, key=lambda row: (row["selected_count"], row["base_count"]))
            output.append({
                "method": method,
                "target_error_recall": target,
                "minimum_review_count": best["selected_count"],
                "minimum_review_rate": best["actual_review_rate"],
                "base_count_per_metric": best["base_count"],
                "captured_error_ids": json.dumps(best["captured_error_ids"]),
            })
    return output


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(prediction_csv, output_directory, checkpoint=Path("bach/classifier.pkl")):
    rows = read_rows(prediction_csv)
    if len(rows) != 40:
        raise ValueError(f"expected Mayumi's 40-image BACH test set, found {len(rows)}")
    errors = {row["image_id"] for row in rows if row["is_error"]}
    if errors != ERROR_IDS:
        raise ValueError(f"unexpected error identities: {sorted(errors)}")
    ranked = rankings(rows)
    total = len(rows)
    curves = {name: [] for name in (*ranked, "union_all_three", "two_of_three", "intersection_all_three")}
    flagged_rows = []
    for base_count in range(1, total + 1):
        selections = selected_for_base_k(ranked, base_count)
        for method, selected in selections.items():
            result = {"method": method, "base_count": base_count,
                      "nominal_k_percent": 100 * base_count / total,
                      **metrics(selected, errors, total)}
            curves[method].append(result)
            flagged_rows.append({
                **{key: result[key] for key in (
                    "method", "base_count", "nominal_k_percent", "selected_count",
                    "actual_review_rate", "incorrect_selected", "error_recall"
                )},
                "selected_image_ids": json.dumps(sorted(selected)),
                "captured_error_ids": json.dumps(result["captured_error_ids"]),
                "missed_error_ids": json.dumps(result["missed_error_ids"]),
            })

    candidate_rows = []
    for percentage in PERCENTAGES:
        base_count = round(total * percentage / 100)
        result = next(row for row in curves["union_all_three"] if row["base_count"] == base_count)
        candidate_rows.append({"candidate_k_percent": percentage, **result})

    comparison = minimum_workloads(curves)
    union_20 = next(row for row in candidate_rows if row["candidate_k_percent"] == 20.0)
    union_100 = min((row for row in candidate_rows if row["error_recall"] == 1.0),
                    key=lambda row: (row["selected_count"], row["candidate_k_percent"]))
    union_75 = min((row for row in candidate_rows if row["error_recall"] >= .75),
                   key=lambda row: (row["selected_count"], row["candidate_k_percent"]))

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    serializable_candidates = [{**row,
        "captured_error_ids": json.dumps(row["captured_error_ids"]),
        "missed_error_ids": json.dumps(row["missed_error_ids"])} for row in candidate_rows]
    _write_csv(output / "union_candidates.csv", serializable_candidates)
    _write_csv(output / "flagged_image_ids.csv", flagged_rows)
    _write_csv(output / "method_minimum_workloads.csv", comparison)

    # At every actual count produced by the union, compare with each simple top-N ranking.
    matched = []
    for union_row in candidate_rows:
        review_count = union_row["selected_count"]
        for method, values in ranked.items():
            result = metrics({row["image_id"] for row in values[:review_count]}, errors, total)
            matched.append({"union_k_percent": union_row["candidate_k_percent"],
                            "actual_review_count": review_count, "method": method,
                            "errors_captured": result["incorrect_selected"],
                            "error_recall": result["error_recall"]})
        matched.append({"union_k_percent": union_row["candidate_k_percent"],
                        "actual_review_count": review_count, "method": "union_all_three",
                        "errors_captured": union_row["incorrect_selected"],
                        "error_recall": union_row["error_recall"]})
    _write_csv(output / "matched_actual_review_counts.csv", matched)

    identity = inspect_checkpoint(checkpoint)
    summary = {
        "status": "retrospective_exploratory_test_set_optimization",
        "classifier_identity": identity,
        "prediction_csv_sha256": sha256_file(prediction_csv),
        "sample_count": total,
        "error_count": len(errors),
        "error_ids": sorted(errors),
        "union_top_20_percent": union_20,
        "minimum_union_for_75_percent_recall": union_75,
        "minimum_union_for_100_percent_recall": union_100,
        "production_threshold_changed": False,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "REPORT.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# Union-of-top-K uncertainty experiment\n\n"
            "Retrospective exploratory analysis of Mayumi's 40-image labeled BACH test set. "
            "No production threshold or UI behavior was changed.\n\n"
            f"- Union Top 20%: {union_20['selected_count']} images reviewed, "
            f"{union_20['incorrect_selected']}/4 errors captured.\n"
            f"- Minimum candidate union for 75% recall: K={union_75['candidate_k_percent']}%, "
            f"{union_75['selected_count']} images reviewed.\n"
            f"- Minimum candidate union for 100% recall: K={union_100['candidate_k_percent']}%, "
            f"{union_100['selected_count']} images reviewed.\n\n"
            "The result is optimized on previously examined test labels and is not independently validated. "
            "Unreviewed images must not be described as safe or disease-free.\n"
        )

    import matplotlib.pyplot as plt
    styles = {
        "confidence_only": "Confidence", "entropy_only": "Entropy",
        "margin_only": "Margin", "union_all_three": "Union",
        "two_of_three": "Two of three", "intersection_all_three": "Intersection",
    }
    figure, axis = plt.subplots()
    for method, settings in curves.items():
        points = sorted({(row["selected_count"], row["error_recall"]) for row in settings})
        axis.plot([100*x/total for x, _ in points], [y for _, y in points], marker=".", label=styles[method])
    axis.scatter([100*union_20["selected_count"]/total], [union_20["error_recall"]], s=80,
                 facecolors="none", edgecolors="black", label="Union K=20%")
    axis.set(xlabel="Actual review rate", ylabel="Error recall",
             title="Retrospective BACH test review workload (n=40, errors=4)")
    axis.legend(fontsize=8)
    figure.savefig(output / "review_rate_vs_error_recall.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots()
    for method, settings in curves.items():
        points = sorted({(row["selected_count"], row["incorrect_selected"]) for row in settings})
        axis.plot([x for x, _ in points], [y for _, y in points], marker=".", label=styles[method])
    axis.scatter([union_20["selected_count"]], [union_20["incorrect_selected"]], s=80,
                 facecolors="none", edgecolors="black", label="Union K=20%")
    axis.set(xlabel="Images reviewed", ylabel="Errors captured",
             title="Review count versus captured errors")
    axis.set_yticks(range(5))
    axis.legend(fontsize=8)
    figure.savefig(output / "review_count_vs_errors_captured.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_csv", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("bach/classifier.pkl"))
    args = parser.parse_args()
    print(json.dumps(run(args.prediction_csv, args.output_directory, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
