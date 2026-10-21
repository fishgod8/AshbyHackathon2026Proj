"""Split conformal prediction for fixed probabilistic classifiers.

This module implements the LAC nonconformity score and has no model, image,
embedding, or retrieval dependencies.
"""

from math import ceil, fsum, isclose, isfinite, isinf
from numbers import Real


METHOD_NAME = "lac_split_conformal"
PROBABILITY_TOLERANCE = 1e-6


def _validate_class_names(class_names) -> tuple[str, ...]:
    if isinstance(class_names, (str, bytes)):
        raise ValueError("class_names must be a non-empty sequence")
    try:
        names = tuple(class_names)
    except TypeError as error:
        raise ValueError("class_names must be a non-empty sequence") from error
    if not names:
        raise ValueError("class_names must not be empty")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("class_names must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("class_names must not contain duplicates")
    return names


def _validate_probability_vector(probabilities, expected_length: int) -> tuple[float, ...]:
    if isinstance(probabilities, (str, bytes)):
        raise ValueError("probabilities must be a one-dimensional sequence")
    try:
        values = tuple(probabilities)
    except TypeError as error:
        raise ValueError("probabilities must be a one-dimensional sequence") from error
    if len(values) != expected_length:
        raise ValueError(
            f"probabilities must contain exactly {expected_length} values"
        )

    validated = []
    for probability in values:
        if isinstance(probability, bool) or not isinstance(probability, Real):
            raise ValueError("probabilities must be finite numbers between 0 and 1")
        value = float(probability)
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("probabilities must be finite numbers between 0 and 1")
        validated.append(value)

    if not isclose(
        fsum(validated), 1.0, rel_tol=0.0, abs_tol=PROBABILITY_TOLERANCE
    ):
        raise ValueError("probabilities must sum approximately to 1")
    return tuple(validated)


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    value = float(alpha)
    if not isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
    return value


def _validate_quantile(quantile: float) -> float:
    if isinstance(quantile, bool) or not isinstance(quantile, Real):
        raise ValueError("quantile must be a number between 0 and 1 or positive infinity")
    value = float(quantile)
    if value < 0.0 or (not isfinite(value) and not (isinf(value) and value > 0.0)):
        raise ValueError("quantile must be a number between 0 and 1 or positive infinity")
    if isfinite(value) and value > 1.0:
        raise ValueError("quantile must be a number between 0 and 1 or positive infinity")
    return value


def calibrate_conformal(
    calibration_probabilities,
    calibration_labels,
    class_names,
    alpha: float = 0.10,
) -> dict[str, object]:
    """Calibrate an LAC split-conformal quantile on independent labeled data."""
    names = _validate_class_names(class_names)
    alpha_value = _validate_alpha(alpha)

    if isinstance(calibration_probabilities, (str, bytes)):
        raise ValueError(
            "calibration_probabilities must be a two-dimensional array"
        )
    try:
        rows = tuple(calibration_probabilities)
    except TypeError as error:
        raise ValueError(
            "calibration_probabilities must be a two-dimensional array"
        ) from error
    if not rows:
        raise ValueError("calibration data must not be empty")

    if isinstance(calibration_labels, (str, bytes)):
        raise ValueError("calibration_labels must be a one-dimensional sequence")
    try:
        labels = tuple(calibration_labels)
    except TypeError as error:
        raise ValueError("calibration_labels must be a one-dimensional sequence") from error
    if len(labels) != len(rows):
        raise ValueError(
            "calibration_labels length must match calibration_probabilities rows"
        )

    class_indices = {name: index for index, name in enumerate(names)}
    nonconformity_scores = []
    for row_number, (row, label) in enumerate(zip(rows, labels), start=1):
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"calibration label {row_number} must be a non-empty string"
            )
        if label not in class_indices:
            raise ValueError(
                f"calibration label {row_number} is not present in class_names: {label}"
            )
        try:
            probabilities = _validate_probability_vector(row, len(names))
        except ValueError as error:
            raise ValueError(
                f"calibration row {row_number}: {error}"
            ) from error
        true_probability = probabilities[class_indices[label]]
        nonconformity_scores.append(1.0 - true_probability)

    nonconformity_scores.sort()
    sample_count = len(nonconformity_scores)
    rank = ceil((sample_count + 1) * (1.0 - alpha_value))
    quantile = (
        float("inf")
        if rank == sample_count + 1
        else nonconformity_scores[rank - 1]
    )
    return {
        "method": METHOD_NAME,
        "alpha": alpha_value,
        "quantile": quantile,
        "class_names": list(names),
        "calibration_sample_count": sample_count,
    }


def predict_set(probabilities, class_names, quantile: float) -> list[str]:
    """Return labels whose LAC nonconformity score is at most the quantile."""
    names = _validate_class_names(class_names)
    values = _validate_probability_vector(probabilities, len(names))
    quantile_value = _validate_quantile(quantile)
    return [
        name
        for name, probability in zip(names, values)
        if 1.0 - probability <= quantile_value
    ]


def assess_conformal_prediction(
    prediction: dict, calibration_artifact: dict
) -> dict[str, object]:
    """Apply a calibrated prediction-set review policy to Mayumi-style output."""
    if not isinstance(prediction, dict):
        raise ValueError("prediction must be a dictionary")
    if not isinstance(calibration_artifact, dict):
        raise ValueError("calibration_artifact must be a dictionary")
    if calibration_artifact.get("method") != METHOD_NAME:
        raise ValueError(f"calibration_artifact method must be {METHOD_NAME}")

    names = _validate_class_names(calibration_artifact.get("class_names"))
    alpha = _validate_alpha(calibration_artifact.get("alpha"))
    quantile = _validate_quantile(calibration_artifact.get("quantile"))
    sample_count = calibration_artifact.get("calibration_sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("calibration_sample_count must be a positive integer")

    scores = prediction.get("scores")
    if not isinstance(scores, dict) or not scores:
        raise ValueError("prediction must contain a non-empty scores dictionary")
    if set(scores) != set(names):
        missing = [name for name in names if name not in scores]
        unexpected = [name for name in scores if name not in names]
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(
            "prediction score keys must match calibration class_names: "
            + "; ".join(details)
        )

    predicted_class = prediction.get("predicted_class")
    if predicted_class is not None:
        if not isinstance(predicted_class, str) or not predicted_class.strip():
            raise ValueError("predicted_class must be a non-empty string when provided")
        if predicted_class not in names:
            raise ValueError("predicted_class must be present in calibration class_names")

    prediction_set = predict_set(
        [scores[name] for name in names], names, quantile
    )
    set_size = len(prediction_set)
    return {
        "predicted_class": predicted_class,
        "prediction_set": prediction_set,
        "set_size": set_size,
        "alpha": alpha,
        "quantile": quantile,
        "needs_review": set_size != 1,
    }
