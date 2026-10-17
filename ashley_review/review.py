"""Website-callable review gate using a fixed Ashley reference-distance rule."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ashley_review.reference_support import (
    EMBEDDING_DIMENSION,
    K_NEIGHBORS,
    ReferenceSupportIndex,
    score_queries,
)


LOW_SUPPORT_REASON = "LOW_REFERENCE_SUPPORT"
FLAGGED_MESSAGE = "该图片与当前参考图库的特征距离较大，建议人工复核。"
CLEAR_MESSAGE = "未触发参考距离提醒；这不代表分类一定正确。"
_FEATURE_CONFIG_KEYS = (
    "model",
    "checkpoint",
    "checkpoint_sha256",
    "preprocessing",
    "extraction",
    "software_versions",
)


def _validated_threshold(
    reference_index: ReferenceSupportIndex,
    gate_config: Mapping[str, Any],
) -> float:
    if not isinstance(gate_config, Mapping):
        raise TypeError("reference_gate_config must be a mapping")
    if gate_config.get("k") != K_NEIGHBORS:
        raise ValueError(f"gate config k must be fixed at {K_NEIGHBORS}")
    if gate_config.get("distance") != "cosine distance = 1 - cosine similarity after L2 normalization":
        raise ValueError("gate config uses an unsupported distance definition")
    if gate_config.get("threshold_rule") != "flag if reference_distance > validation 95th percentile":
        raise ValueError("gate config uses an unsupported threshold rule")
    if gate_config.get("reference_split") != "reference":
        raise ValueError("gate config must identify the reference split")

    feature_settings = gate_config.get("feature_settings")
    if not isinstance(feature_settings, Mapping):
        raise ValueError("gate config is missing feature_settings")
    feature_shape = feature_settings.get("shape")
    if (
        not isinstance(feature_shape, (list, tuple))
        or not feature_shape
        or feature_shape[-1] != EMBEDDING_DIMENSION
    ):
        raise ValueError(
            f"gate config must specify {EMBEDDING_DIMENSION}-dimensional embeddings"
        )
    if feature_settings.get("dtype") != reference_index.embedding_dtype:
        raise ValueError("gate config embedding dtype does not match reference index")
    if len(reference_index.image_ids) < K_NEIGHBORS:
        raise ValueError(f"reference index must contain at least {K_NEIGHBORS} images")
    if len(reference_index.image_ids) != len(reference_index.labels):
        raise ValueError("reference index image IDs and labels have different lengths")
    if len(set(reference_index.image_ids)) != len(reference_index.image_ids):
        raise ValueError("reference index contains duplicate image IDs")
    reference_vectors = np.asarray(reference_index.normalized_embeddings)
    if reference_vectors.shape != (len(reference_index.image_ids), EMBEDDING_DIMENSION):
        raise ValueError("reference index has an invalid embedding matrix shape")
    if not np.issubdtype(reference_vectors.dtype, np.number) or not np.isfinite(
        reference_vectors
    ).all():
        raise ValueError("reference index contains non-finite or non-numeric vectors")
    reference_norms = np.linalg.norm(reference_vectors, axis=1)
    if not np.allclose(reference_norms, 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError("reference index vectors must already be L2-normalized")

    for key in _FEATURE_CONFIG_KEYS:
        if key not in feature_settings:
            raise ValueError(f"gate config feature_settings is missing {key}")
        if key not in reference_index.extractor_settings:
            raise ValueError(f"reference index extractor settings are missing {key}")
        if feature_settings[key] != reference_index.extractor_settings[key]:
            raise ValueError(
                f"gate config {key} does not match the reference index settings"
            )

    threshold = gate_config.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("gate config threshold must be a finite number")
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("gate config threshold must be a finite number")
    return threshold


def _prepare_queries(
    embeddings: NDArray[Any] | Sequence[Sequence[float]],
    image_ids: Sequence[str | None] | None,
    reference_image_ids: Sequence[str],
) -> tuple[NDArray[Any], list[str | None], list[str]]:
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"query embeddings must have shape (n, {EMBEDDING_DIMENSION}); got {matrix.shape}"
        )
    if matrix.shape[0] == 0:
        raise ValueError("at least one query embedding is required")
    if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(
        matrix.dtype, np.complexfloating
    ):
        raise ValueError("query embeddings must have a real numeric dtype")
    if not np.isfinite(matrix).all():
        raise ValueError("query embeddings contain non-finite values")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    invalid_rows = np.flatnonzero(~np.isfinite(norms) | (norms == 0))
    if invalid_rows.size:
        raise ValueError(
            "query embeddings contain zero or invalid-norm vectors at rows: "
            + ", ".join(str(int(row)) for row in invalid_rows[:10])
        )

    if image_ids is None:
        result_ids: list[str | None] = [None] * matrix.shape[0]
    else:
        result_ids = list(image_ids)
        if len(result_ids) != matrix.shape[0]:
            raise ValueError("image_ids count must match query embedding row count")
        if any(value is not None and (not isinstance(value, str) or not value) for value in result_ids):
            raise ValueError("image_id values must be non-empty strings or None")
        known_ids = [value for value in result_ids if value is not None]
        if len(set(known_ids)) != len(known_ids):
            raise ValueError("non-null image_id values must be unique within a batch")

    occupied_ids = set(reference_image_ids)
    occupied_ids.update(value for value in result_ids if value is not None)
    internal_ids: list[str] = []
    for row_id in result_ids:
        if row_id is not None:
            internal_ids.append(row_id)
            continue
        candidate = f"__ashley_query_without_image_id_{len(internal_ids)}__"
        while candidate in occupied_ids:
            candidate += "_"
        internal_ids.append(candidate)
        occupied_ids.add(candidate)
    return matrix, result_ids, internal_ids


def _format_result(
    image_id: str | None,
    reference_distance: float,
    distance_threshold: float,
    neighbors: Sequence[Any],
) -> dict[str, Any]:
    needs_review = reference_distance > distance_threshold
    return {
        "image_id": image_id,
        "needs_review": needs_review,
        "reference_distance": float(reference_distance),
        "distance_threshold": float(distance_threshold),
        "reasons": [LOW_SUPPORT_REASON] if needs_review else [],
        "message": FLAGGED_MESSAGE if needs_review else CLEAR_MESSAGE,
        "neighbors": [
            {
                "image_id": neighbor.image_id,
                "label": neighbor.label,
                "distance": float(neighbor.distance),
            }
            for neighbor in neighbors
        ],
    }


def review_embeddings(
    embeddings: NDArray[Any] | Sequence[Sequence[float]],
    reference_index: ReferenceSupportIndex,
    reference_gate_config: Mapping[str, Any],
    image_ids: Sequence[str | None] | None = None,
) -> list[dict[str, Any]]:
    """Review a batch against a fixed reference index and fixed gate threshold.

    ``None`` IDs are accepted and disable same-ID self-exclusion for those rows.
    The caller must provide embeddings from the same configured extractor as the
    reference index; the gate configuration is checked against that index.
    """
    threshold = _validated_threshold(reference_index, reference_gate_config)
    matrix, result_ids, internal_ids = _prepare_queries(
        embeddings, image_ids, reference_index.image_ids
    )
    if str(matrix.dtype) != reference_index.embedding_dtype:
        raise ValueError(
            "query and reference embedding dtypes differ: "
            f"{matrix.dtype} != {reference_index.embedding_dtype}"
        )
    scores = score_queries(
        reference_index,
        internal_ids,
        matrix,
        reference_index.extractor_settings,
    )
    return [
        _format_result(image_id, score.reference_distance, threshold, score.neighbors)
        for image_id, score in zip(result_ids, scores)
    ]


def review_embedding(
    embedding: NDArray[Any] | Sequence[float],
    reference_index: ReferenceSupportIndex,
    reference_gate_config: Mapping[str, Any],
    image_id: str | None = None,
) -> dict[str, Any]:
    """Review a single embedding and return a JSON-serializable result."""
    vector = np.asarray(embedding)
    if vector.ndim != 1:
        raise ValueError(f"single query embedding must be one-dimensional; got {vector.shape}")
    return review_embeddings(
        vector[np.newaxis, :], reference_index, reference_gate_config, [image_id]
    )[0]