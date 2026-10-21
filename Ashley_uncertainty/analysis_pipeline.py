"""Stable classifier uncertainty and review-queue interface."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .model_identity import EXPECTED_CLASSES
from .uncertainty import assess_probabilities


def analyze_prediction(
    prediction,
    *,
    image_id=None,
    confidence_threshold=0.70,
    entropy_threshold=None,
    margin_threshold=None,
):
    """Enrich Mayumi's prediction dictionary without running image inference."""
    if not isinstance(prediction, dict):
        raise ValueError("prediction must be a dictionary")
    scores = prediction.get("scores")
    metrics = assess_probabilities(
        scores,
        confidence_threshold,
        entropy_threshold=entropy_threshold,
        margin_threshold=margin_threshold,
        expected_labels=EXPECTED_CLASSES,
    )
    reported = prediction.get("predicted_class")
    if reported != metrics["predicted_class"]:
        raise ValueError("predicted_class does not match the maximum probability")

    # No independently calibrated conformal artifact exists for this exact
    # checkpoint. Keep an explicit status for UI backward compatibility.
    conformal_output = {
        "available": False,
        "prediction_set": None,
        "set_size": None,
        "alpha": 0.10,
        "reason": "Independent labeled calibration data required",
    }
    reasons = list(metrics["review_reasons"])

    return {
        "image_id": image_id if image_id is not None else prediction.get("image_id"),
        "predicted_class": metrics["predicted_class"],
        "confidence": metrics["confidence"],
        "probabilities": metrics["probabilities"],
        "uncertainty": {
            "max_probability_uncertainty": metrics["max_probability_uncertainty"],
            "entropy": metrics["entropy"],
            "normalized_entropy": metrics["normalized_entropy"],
            "margin": metrics["margin"],
        },
        "conformal": conformal_output,
        # Backward-compatible aliases for Bowen's current UI.
        "prediction_set": conformal_output["prediction_set"],
        "conformal_status": "available" if conformal_output["available"] else "unavailable",
        "needs_review": bool(reasons),
        "review_reasons": reasons,
        "embedding": (
            None if prediction.get("embedding") is None
            else [float(value) for value in prediction["embedding"]]
        ),
    }


def analyze_image(
    image_path,
    *,
    model_path=None,
    device=None,
    confidence_threshold=0.70,
    entropy_threshold=None,
    margin_threshold=None,
    predictor=None,
):
    """Run Mayumi's classifier and return UI/retrieval-ready analysis.

    ``predictor`` is injectable so tests and deployments can provide an already
    initialized inference service.  The built-in predictor caches both the
    classifier and QuiltNet adapter.
    """
    using_builtin_predictor = predictor is None
    if using_builtin_predictor:
        from bach.predict_image import predict_image

        predictor = predict_image
    canonical_model_path = (
        Path(model_path)
        if model_path is not None
        else Path(__file__).resolve().parent.parent / "bach" / "classifier.pkl"
    )
    kwargs = {"model_path": canonical_model_path} if using_builtin_predictor or model_path is not None else {}
    if device is not None:
        kwargs["device"] = device
    prediction = predictor(image_path, **kwargs)
    return analyze_prediction(
        prediction,
        image_id=Path(image_path).name,
        confidence_threshold=confidence_threshold,
        entropy_threshold=entropy_threshold,
        margin_threshold=margin_threshold,
    )


def rank_for_review(results: Iterable[dict], *, sort_by="confidence"):
    """Return every result in a deterministic uncertainty-first review queue.

    The review flag remains whatever was calculated by ``analyze_image`` or
    ``analyze_prediction``.  ``sort_by`` changes ordering within the flagged
    and unflagged groups; it does not change ``needs_review``.
    """
    allowed = {"confidence", "normalized_entropy", "margin"}
    if sort_by not in allowed:
        raise ValueError(f"sort_by must be one of: {', '.join(sorted(allowed))}")
    if isinstance(results, (str, bytes, dict)):
        raise ValueError("results must be an iterable of analysis dictionaries")
    items = list(results)

    def sort_key(result):
        if not isinstance(result, dict):
            raise ValueError("each review result must be a dictionary")
        image_id = result.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("each review result must have a non-empty image_id")
        if not isinstance(result.get("needs_review"), bool):
            raise ValueError("each review result must have a boolean needs_review")
        uncertainty = result.get("uncertainty")
        if not isinstance(uncertainty, dict):
            raise ValueError("each review result must have uncertainty metrics")
        try:
            confidence = float(result["confidence"])
            normalized_entropy = float(uncertainty["normalized_entropy"])
            margin = float(uncertainty["margin"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("review result contains invalid uncertainty metrics") from error
        metric_key = {
            "confidence": confidence,
            "normalized_entropy": -normalized_entropy,
            "margin": margin,
        }[sort_by]
        return (not result["needs_review"], metric_key, image_id)

    ranked = sorted(items, key=sort_key)
    output = []
    for rank, result in enumerate(ranked, 1):
        item = dict(result)
        item["review_rank"] = rank
        output.append(item)
    return output
