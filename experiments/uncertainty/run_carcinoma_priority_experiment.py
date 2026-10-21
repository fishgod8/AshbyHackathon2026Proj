"""Retrospective review-ranking study focused on carcinoma false negatives."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from evaluate_bach_uncertainty import read_rows
from model_identity import inspect_checkpoint, sha256_file


CARCINOMA = {"InSitu", "Invasive"}
NON_CARCINOMA = {"Normal", "Benign"}
BUDGETS = (.05, .10, .15, .20, .25, .30, .40, .50)
KNOWN_FALSE_NEGATIVES = {"is067.tif", "is057.tif"}


def enrich(rows):
    enriched = []
    for position, row in enumerate(rows):
        scores = {key[6:]: float(value) for key, value in row.items() if key.startswith("score_")}
        p_carcinoma = scores["InSitu"] + scores["Invasive"]
        p_noncarcinoma = scores["Normal"] + scores["Benign"]
        binary_entropy = -sum(value * math.log(value) for value in (p_carcinoma, p_noncarcinoma) if value > 0) / math.log(2)
        enriched.append({
            **row,
            "source_position": position,
            "p_carcinoma": p_carcinoma,
            "p_noncarcinoma": p_noncarcinoma,
            "binary_carcinoma_entropy": binary_entropy,
            "is_carcinoma_false_negative": row["true_label"] in CARCINOMA and row["predicted_label"] in NON_CARCINOMA,
            "is_true_carcinoma": row["true_label"] in CARCINOMA,
        })
    return enriched


def _probability_tiebreak(row):
    """Deterministic tie-break using model outputs only, never IDs or labels."""
    return tuple(float(row[f"score_{name}"]) for name in ("Benign", "InSitu", "Invasive", "Normal"))


def build_rankings(rows):
    base = {
        "confidence": sorted(rows, key=lambda r: (float(r["confidence"]), _probability_tiebreak(r))),
        "four_class_entropy": sorted(rows, key=lambda r: (-float(r["normalized_entropy"]), _probability_tiebreak(r))),
        "top_two_margin": sorted(rows, key=lambda r: (float(r["margin"]), _probability_tiebreak(r))),
    }
    binary_entropy = sorted(rows, key=lambda r: (-r["binary_carcinoma_entropy"], _probability_tiebreak(r)))
    carcinoma_suspicion = sorted(rows, key=lambda r: (-r["p_carcinoma"], _probability_tiebreak(r)))
    # Fixed clinical hypothesis, defined without labels: among predicted non-carcinoma
    # cases, review the highest model-estimated carcinoma mass first. Predicted
    # carcinoma cases follow, ordered by ordinary four-class uncertainty.
    hybrid = sorted(rows, key=lambda r: (
        0 if r["predicted_label"] in NON_CARCINOMA else 1,
        -r["p_carcinoma"] if r["predicted_label"] in NON_CARCINOMA else float(r["confidence"]),
        _probability_tiebreak(r),
    ))
    return {
        **base,
        "binary_carcinoma_entropy": binary_entropy,
        "carcinoma_suspicion": carcinoma_suspicion,
        "predicted_class_hybrid": hybrid,
    }


def voting_selection(base_rankings, base_count, minimum_votes):
    """Select the union (1 vote) or two-of-three set at an exact per-metric K."""
    votes = {}
    rows_by_id = {}
    for ranked in base_rankings.values():
        for row in ranked:
            rows_by_id[row["image_id"]] = row
        for row in ranked[:base_count]:
            votes[row["image_id"]] = votes.get(row["image_id"], 0) + 1
    selected_ids = {image_id for image_id, count in votes.items() if count >= minimum_votes}
    return [rows_by_id[image_id] for image_id in selected_ids]


def selection_curves(rows, rankings):
    """Return nested selections; set strategies retain their actual deduplicated sizes."""
    curves = {}
    for strategy, ranked in rankings.items():
        curves[strategy] = [
            {"setting_count": count, "selected": ranked[:count]}
            for count in range(1, len(rows) + 1)
        ]
    base = {name: rankings[name] for name in ("confidence", "four_class_entropy", "top_two_margin")}
    curves["union_top_k"] = [
        {"setting_count": count, "selected": voting_selection(base, count, 1)}
        for count in range(1, len(rows) + 1)
    ]
    curves["two_of_three"] = [
        {"setting_count": count, "selected": voting_selection(base, count, 2)}
        for count in range(1, len(rows) + 1)
    ]
    return curves


def selection_metrics(selected, all_rows):
    false_negatives = {row["image_id"] for row in all_rows if row["is_carcinoma_false_negative"]}
    all_errors = {row["image_id"] for row in all_rows if row["is_error"]}
    true_carcinoma = {row["image_id"] for row in all_rows if row["is_true_carcinoma"]}
    selected_ids = {row["image_id"] for row in selected}
    caught_fn = selected_ids & false_negatives
    return {
        "review_count": len(selected),
        "review_rate": len(selected) / len(all_rows),
        "carcinoma_false_negatives_captured": len(caught_fn),
        "carcinoma_false_negative_recall": len(caught_fn) / len(false_negatives),
        "overall_errors_captured": len(selected_ids & all_errors),
        "overall_error_recall": len(selected_ids & all_errors) / len(all_errors),
        "true_carcinoma_reviewed": len(selected_ids & true_carcinoma),
        "true_carcinoma_recall": len(selected_ids & true_carcinoma) / len(true_carcinoma),
        "non_carcinoma_images_reviewed": sum(row["true_label"] in NON_CARCINOMA for row in selected),
        "false_positive_review_workload": len(selected) - len(caught_fn),
        "is067_included": "is067.tif" in selected_ids,
        "is057_included": "is057.tif" in selected_ids,
        "selected_image_ids": sorted(selected_ids),
    }


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(prediction_csv, output_directory, checkpoint=Path("bach/classifier.pkl")):
    rows = enrich(read_rows(prediction_csv))
    if len(rows) != 40:
        raise ValueError(f"expected 40 rows, found {len(rows)}")
    observed_fn = {row["image_id"] for row in rows if row["is_carcinoma_false_negative"]}
    if observed_fn != KNOWN_FALSE_NEGATIVES:
        raise ValueError(f"unexpected carcinoma false negatives: {sorted(observed_fn)}")
    rankings = build_rankings(rows)
    curves = selection_curves(rows, rankings)

    budget_rows = []
    for strategy, settings in curves.items():
        for budget in BUDGETS:
            count = round(len(rows) * budget)
            selected = settings[count - 1]["selected"]
            result = selection_metrics(selected, rows)
            budget_rows.append({
                "strategy": strategy,
                "nominal_budget": budget,
                "base_count_per_metric": count,
                **{key: value for key, value in result.items() if key != "selected_image_ids"},
                "selected_image_ids": json.dumps(result["selected_image_ids"]),
            })

    rank_rows = []
    for strategy, ranked in rankings.items():
        positions = {row["image_id"]: rank for rank, row in enumerate(ranked, 1)}
        rank_rows.append({"strategy": strategy,
                          "is067_rank": positions["is067.tif"],
                          "is057_rank": positions["is057.tif"],
                          "reviews_to_capture_both": max(positions["is067.tif"], positions["is057.tif"])})
    base_positions = {
        name: {row["image_id"]: rank for rank, row in enumerate(rankings[name], 1)}
        for name in ("confidence", "four_class_entropy", "top_two_margin")
    }
    for strategy, vote_index in (("union_top_k", 0), ("two_of_three", 1)):
        entries = {
            image_id: sorted(positions[image_id] for positions in base_positions.values())[vote_index]
            for image_id in KNOWN_FALSE_NEGATIVES
        }
        cutoff = max(entries.values())
        selected = curves[strategy][cutoff - 1]["selected"]
        rank_rows.append({"strategy": strategy,
                          "is067_rank": entries["is067.tif"],
                          "is057_rank": entries["is057.tif"],
                          "reviews_to_capture_both": len(selected)})

    summary_rows = []
    for strategy, settings in curves.items():
        candidates = []
        for setting in settings:
            result = selection_metrics(setting["selected"], rows)
            if result["carcinoma_false_negatives_captured"] == 2:
                candidates.append((result["review_count"], setting["setting_count"], result))
        review_count, setting_count, result = min(candidates, key=lambda item: (item[0], item[1]))
        summary_rows.append({"strategy": strategy, "minimum_reviews_for_both": review_count,
                             "minimum_review_rate": review_count / len(rows),
                             "base_count_per_metric_at_minimum": setting_count,
                             "overall_errors_captured_at_minimum": result["overall_errors_captured"],
                             "true_carcinoma_reviewed_at_minimum": result["true_carcinoma_reviewed"],
                             "non_carcinoma_reviewed_at_minimum": result["non_carcinoma_images_reviewed"],
                             "false_positive_review_workload_at_minimum": result["false_positive_review_workload"]})
    # Random ordering expectation at each exact budget. Expected recall for either
    # the two false negatives or all true carcinoma cases equals reviewed fraction.
    random_rows = [{"strategy": "random_expected", "nominal_budget": budget,
                    "review_count": round(len(rows) * budget),
                    "review_rate": round(len(rows) * budget) / len(rows),
                    "expected_carcinoma_false_negative_recall": round(len(rows) * budget) / len(rows),
                    "probability_capture_both_false_negatives": (
                        round(len(rows) * budget) * (round(len(rows) * budget) - 1) / (len(rows) * (len(rows) - 1))
                    ),
                    "expected_true_carcinoma_recall": round(len(rows) * budget) / len(rows)} for budget in BUDGETS]

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "budget_comparison.csv", budget_rows)
    _write_csv(output / "false_negative_ranks.csv", rank_rows)
    _write_csv(output / "minimum_workload.csv", summary_rows)
    _write_csv(output / "random_baseline.csv", random_rows)

    # Compare each set-valued method against scalar rankings at exactly the same
    # number of actually selected images.
    matched_rows = []
    for set_strategy in ("union_top_k", "two_of_three"):
        for setting in curves[set_strategy]:
            set_result = selection_metrics(setting["selected"], rows)
            actual_count = set_result["review_count"]
            for strategy, ranked in rankings.items():
                result = selection_metrics(ranked[:actual_count], rows)
                matched_rows.append({"set_strategy": set_strategy,
                                     "base_count_per_metric": setting["setting_count"],
                                     "actual_review_count": actual_count,
                                     "comparison_strategy": strategy,
                                     "carcinoma_false_negative_recall": result["carcinoma_false_negative_recall"],
                                     "overall_error_recall": result["overall_error_recall"],
                                     "true_carcinoma_recall": result["true_carcinoma_recall"]})
            matched_rows.append({"set_strategy": set_strategy,
                                 "base_count_per_metric": setting["setting_count"],
                                 "actual_review_count": actual_count,
                                 "comparison_strategy": set_strategy,
                                 "carcinoma_false_negative_recall": set_result["carcinoma_false_negative_recall"],
                                 "overall_error_recall": set_result["overall_error_recall"],
                                 "true_carcinoma_recall": set_result["true_carcinoma_recall"]})
    _write_csv(output / "matched_actual_review_counts.csv", matched_rows)

    identity = inspect_checkpoint(checkpoint)
    summary = {
        "status": "retrospective_exploratory_known_test_labels",
        "classifier_identity": identity,
        "prediction_csv_sha256": sha256_file(prediction_csv),
        "sample_count": len(rows),
        "classification_error_count": sum(row["is_error"] for row in rows),
        "true_carcinoma_count": sum(row["is_true_carcinoma"] for row in rows),
        "carcinoma_false_negative_ids": sorted(observed_fn),
        "strategy_minimum_workloads": summary_rows,
        "production_policy_changed": False,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    best = min(summary_rows, key=lambda row: row["minimum_reviews_for_both"])
    with (output / "REPORT.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# Carcinoma false-negative review prioritization\n\n"
            "Retrospective exploratory analysis of Mayumi's saved 40-image BACH test predictions. "
            "No model was retrained, QuiltNet was not rerun, and no production policy or UI was changed.\n\n"
            f"- Two carcinoma false negatives: `{', '.join(sorted(observed_fn))}`.\n"
            f"- Best observed candidate: `{best['strategy']}`, requiring "
            f"{best['minimum_reviews_for_both']}/40 reviews ({best['minimum_review_rate']:.1%}) to capture both.\n"
            "- This outcome is hypothesis-generating only because these test labels and error identities were already examined.\n"
            "- `union_top_k` and `two_of_three` use true set selection at each per-metric K; their actual review counts can differ from K.\n\n"
            "The predicted-class hybrid first ranks model-predicted non-carcinoma cases by decreasing model-generated "
            "carcinoma mass. It uses no filenames or true labels, but its observed performance is not independent validation.\n"
        )

    import matplotlib.pyplot as plt
    plot_specs = (
        ("carcinoma_false_negative_recall", "Carcinoma false-negative recall", "carcinoma_fn_recall_vs_review_count.png"),
        ("true_carcinoma_recall", "True carcinoma images reviewed", "true_carcinoma_recall_vs_review_count.png"),
        ("overall_error_recall", "Overall classification-error recall", "overall_error_recall_vs_review_count.png"),
    )
    for metric, ylabel, filename in plot_specs:
        figure, axis = plt.subplots()
        for strategy, settings in curves.items():
            points = sorted({(len(setting["selected"]), selection_metrics(setting["selected"], rows)[metric])
                             for setting in settings})
            axis.plot([point[0] for point in points], [point[1] for point in points],
                      label=strategy.replace("_", " "))
        axis.plot(range(1, len(rows) + 1), [count / len(rows) for count in range(1, len(rows) + 1)],
                  linestyle="--", color="black", label="random expected recall")
        axis.set(xlabel="Images reviewed", ylabel=ylabel,
                 title="Retrospective BACH test ranking comparison")
        axis.legend(fontsize=7)
        figure.savefig(output / filename, dpi=160, bbox_inches="tight")
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
