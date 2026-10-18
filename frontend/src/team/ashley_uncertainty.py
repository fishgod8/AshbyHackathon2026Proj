# Vendored verbatim from fishgod8/AshbyHackathon2026Proj, branch ashley-uncertainty @ 07d25b3 (Ashley).
# Do not edit here; update by re-copying from the team repo.

"""Confidence-based uncertainty checks for one image classification."""

from math import fsum, isclose, isfinite
from numbers import Real
from typing import Iterable


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