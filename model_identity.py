"""Identity and metadata validation for Mayumi's canonical BACH checkpoint."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import joblib


EXPECTED_CLASSES = ("Benign", "InSitu", "Invasive", "Normal")
EXPECTED_CLASSIFIER_SHA256 = "a3d3115fc3e1a88b4720ba85d083602202b86e315fcfd8f0abd381cc62e15885"
EXPECTED_EMBEDDING_MODEL = "openclip:wisdomik/QuiltNet-B-32"
EXPECTED_AGGREGATION = "mean of 12 patches"
EXPECTED_FEATURE_DIMENSION = 512


def sha256_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _load_checkpoint(resolved_path):
    checkpoint = Path(resolved_path)
    digest = sha256_file(checkpoint)
    if digest != EXPECTED_CLASSIFIER_SHA256:
        raise ValueError(
            "Unexpected classifier checkpoint SHA-256: "
            f"{digest}; expected {EXPECTED_CLASSIFIER_SHA256}"
        )
    saved = joblib.load(checkpoint)
    required = {"model", "classes", "embedding_model", "aggregation"}
    missing = sorted(required - set(saved)) if isinstance(saved, dict) else sorted(required)
    if missing:
        raise ValueError("Classifier checkpoint is missing: " + ", ".join(missing))
    model = saved["model"]
    classes = tuple(str(value) for value in model.classes_)
    saved_classes = tuple(str(value) for value in saved["classes"])
    if classes != saved_classes:
        raise ValueError("Saved classes do not match classifier.classes_")
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected classifier class order: {classes}")
    if saved["embedding_model"] != EXPECTED_EMBEDDING_MODEL:
        raise ValueError(f"Unexpected embedding model: {saved['embedding_model']!r}")
    if saved["aggregation"] != EXPECTED_AGGREGATION:
        raise ValueError(f"Unexpected aggregation: {saved['aggregation']!r}")
    if int(model.n_features_in_) != EXPECTED_FEATURE_DIMENSION:
        raise ValueError(f"Unexpected feature dimension: {model.n_features_in_}")
    steps = list(getattr(model, "named_steps", {}))
    if steps != ["standardscaler", "logisticregression"]:
        raise ValueError(f"Unexpected classifier preprocessing steps: {steps}")
    identity = {
        "classifier_sha256": digest,
        "class_order": list(classes),
        "embedding_model": saved["embedding_model"],
        "aggregation": saved["aggregation"],
        "feature_dimension": int(model.n_features_in_),
        "pipeline_steps": steps,
    }
    return saved, identity


def load_checkpoint(path):
    """Load and validate Mayumi's canonical checkpoint once per resolved path."""
    saved, identity = _load_checkpoint(str(Path(path).resolve()))
    return saved, deepcopy(identity)


def inspect_checkpoint(path):
    """Return the validated canonical checkpoint identity."""
    _, identity = load_checkpoint(path)
    return identity


def verify_calibration_binding(artifact, checkpoint_path):
    """Reject conformal artifacts not calibrated for this exact checkpoint."""
    identity = inspect_checkpoint(checkpoint_path)
    binding = artifact.get("classifier_identity") if isinstance(artifact, dict) else None
    if not isinstance(binding, dict):
        raise ValueError("Conformal artifact has no classifier identity binding")
    if binding.get("classifier_sha256") != identity["classifier_sha256"]:
        raise ValueError("Conformal artifact checkpoint hash does not match active classifier")
    for key in ("class_order", "embedding_model", "aggregation", "feature_dimension"):
        if binding.get(key) != identity[key]:
            raise ValueError(f"Conformal artifact classifier metadata mismatch: {key}")
    if artifact.get("calibration_status") != "independent_labeled_calibration":
        raise ValueError("Conformal artifact is not marked as independent calibration data")
    return identity
