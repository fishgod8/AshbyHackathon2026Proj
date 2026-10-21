"""Validated uncertainty metrics for probabilistic image classification."""

from math import fsum, isclose, isfinite, log
from numbers import Real
from typing import Iterable


BACH_CLASS_NAMES = ("Benign", "InSitu", "Invasive", "Normal")


def assess_uncertainty(
    probabilities: Iterable[float], threshold: float
) -> tuple[float, bool]:
    """Return the highest class probability and whether the image needs review."""
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise ValueError("threshold must be a finite number between 0 and 1")
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite number between 0 and 1")

    values = tuple(probabilities)
    if not values:
        raise ValueError("probabilities must not be empty")

    for probability in values:
        if isinstance(probability, bool) or not isinstance(probability, Real):
            raise ValueError("probabilities must be finite numbers between 0 and 1")
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probabilities must be finite numbers between 0 and 1")

    if not isclose(fsum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("probabilities must sum approximately to 1")

    confidence = max(values)
    return confidence, confidence < threshold


def assess_prediction(prediction: dict, threshold: float) -> dict[str, object]:
    """Assess a classifier prediction without depending on model inference.

    ``embedding`` and any other prediction fields are intentionally ignored.
    """
    if not isinstance(prediction, dict):
        raise ValueError("prediction must be a dictionary")

    predicted_class = prediction.get("predicted_class")
    if not isinstance(predicted_class, str) or not predicted_class.strip():
        raise ValueError("prediction must contain a non-empty predicted_class")

    scores = prediction.get("scores")
    if not isinstance(scores, dict) or not scores:
        raise ValueError("prediction must contain a non-empty scores dictionary")

    confidence, needs_review = assess_uncertainty(scores.values(), threshold)
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "needs_review": needs_review,
        "threshold": threshold,
    }


def assess_probabilities(
    probabilities: dict[str, float],
    confidence_threshold: float = 0.70,
    *,
    entropy_threshold: float | None = None,
    margin_threshold: float | None = None,
    expected_labels: Iterable[str] | None = None,
) -> dict[str, object]:
    """Return all supported uncertainty metrics and review reasons.

    Review thresholds are deliberately configurable.  The defaults are routing
    examples, not clinically validated cutoffs.  Entropy is normalized to
    ``[0, 1]`` by dividing by ``log(number of classes)``.
    """
    if not isinstance(probabilities, dict) or not probabilities:
        raise ValueError("probabilities must be a non-empty label-to-score dictionary")
    if any(not isinstance(label, str) or not label.strip() for label in probabilities):
        raise ValueError("probability labels must be non-empty strings")
    if expected_labels is not None:
        expected = tuple(expected_labels)
        if len(expected) != len(set(expected)) or any(
            not isinstance(label, str) or not label.strip() for label in expected
        ):
            raise ValueError("expected_labels must contain unique non-empty strings")
        if set(probabilities) != set(expected):
            raise ValueError(
                "probability labels do not match expected classes: "
                + ", ".join(expected)
            )

    values = tuple(probabilities.values())
    confidence, low_confidence = assess_uncertainty(values, confidence_threshold)
    if len(values) < 2:
        raise ValueError("at least two class probabilities are required")

    def optional_threshold(value, name):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite number between 0 and 1")
        value = float(value)
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number between 0 and 1")
        return value

    entropy_threshold = optional_threshold(entropy_threshold, "entropy_threshold")
    margin_threshold = optional_threshold(margin_threshold, "margin_threshold")
    entropy = -fsum(float(p) * log(float(p)) for p in values if p > 0)
    normalized_entropy = entropy / log(len(values))
    largest, second_largest = sorted((float(p) for p in values), reverse=True)[:2]
    margin = largest - second_largest
    predicted_label = max(probabilities, key=probabilities.get)

    reasons = []
    if low_confidence:
        reasons.append("Low model confidence")
    if entropy_threshold is not None and normalized_entropy >= entropy_threshold:
        reasons.append("High predictive entropy")
    if margin_threshold is not None and margin <= margin_threshold:
        reasons.append("Small top-two margin")

    return {
        "predicted_class": predicted_label,
        "confidence": confidence,
        "max_probability_uncertainty": 1.0 - confidence,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "margin": margin,
        "needs_review": bool(reasons),
        "review_reasons": reasons,
        "probabilities": {label: float(score) for label, score in probabilities.items()},
    }


if __name__ == "__main__":
    # This threshold is provisional; select it later using validation data.
    example_threshold = 0.7
    examples = (
        ("clear prediction", [0.82, 0.12, 0.06]),
        ("uncertain prediction", [0.45, 0.35, 0.20]),
    )

    print(f"Provisional threshold (tune with validation data): {example_threshold}")
    for label, probabilities in examples:
        confidence, needs_review = assess_uncertainty(
            probabilities, example_threshold
        )
        print(
            f"{label}: confidence={confidence:.2f}, "
            f"needs_review={needs_review}"
        )
