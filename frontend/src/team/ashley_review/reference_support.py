# Vendored verbatim from fishgod8/AshbyHackathon2026Proj, branch main @ 1fa48f5 (Ashley, ashley_review/reference_support.py).
# Do not edit here; update by re-copying from the team repo.

"""Cosine-distance support scoring against Ashley's reference split only."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


K_NEIGHBORS = 5
EMBEDDING_DIMENSION = 1024
_CONFIG_KEYS = (
    "model",
    "checkpoint",
    "checkpoint_sha256",
    "preprocessing",
    "extraction",
    "software_versions",
)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Neighbor:
    image_id: str
    label: str
    distance: float


@dataclass(frozen=True)
class SupportScore:
    query_image_id: str
    reference_distance: float
    neighbors: tuple[Neighbor, ...]


@dataclass(frozen=True)
class ReferenceSupportIndex:
    image_ids: tuple[str, ...]
    labels: tuple[str, ...]
    normalized_embeddings: NDArray[np.float64]
    embedding_dtype: str
    extractor_settings: Mapping[str, Any]


def _extractor_settings(config: Mapping[str, Any], source: str) -> dict[str, Any]:
    missing = [key for key in _CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"{source} config is missing required fields: {', '.join(missing)}")
    return {key: config[key] for key in _CONFIG_KEYS}


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file_handle:
        config = json.load(file_handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must contain a JSON object: {path}")
    return config


def _as_embedding_matrix(
    embeddings: NDArray[Any] | Sequence[Sequence[float]], *, source: str
) -> NDArray[Any]:
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"{source} embeddings must have shape (n, {EMBEDDING_DIMENSION}); "
            f"got {matrix.shape}"
        )
    if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(
        matrix.dtype, np.complexfloating
    ):
        raise ValueError(f"{source} embeddings must have a real numeric dtype")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{source} embeddings contain non-finite values")
    return matrix


def _normalize_embeddings(
    embeddings: NDArray[Any], *, source: str
) -> NDArray[np.float64]:
    matrix = _as_embedding_matrix(embeddings, source=source).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    invalid = np.flatnonzero(~np.isfinite(norms) | (norms == 0))
    if invalid.size:
        preview = ", ".join(str(int(index)) for index in invalid[:10])
        raise ValueError(f"{source} embeddings contain zero or invalid-norm vectors at rows: {preview}")
    return matrix / norms[:, np.newaxis]


def build_reference_index(
    embeddings: NDArray[Any] | Sequence[Sequence[float]],
    manifest_rows: Iterable[Mapping[str, str]],
    reference_config: Mapping[str, Any],
) -> ReferenceSupportIndex:
    """Build a normalized index from rows marked reference in the split manifest."""
    matrix = _as_embedding_matrix(embeddings, source="reference")
    rows = list(manifest_rows)
    if matrix.shape[0] != len(rows):
        raise ValueError(
            f"embedding/manifest row count mismatch: {matrix.shape[0]} != {len(rows)}"
        )

    image_ids: list[str] = []
    labels: list[str] = []
    row_indices: list[int] = []
    seen_ids: set[str] = set()
    for position, row in enumerate(rows):
        try:
            row_index = int(row["row_index"])
            image_id = row["image_id"]
            label = row["label"]
            split = row["split"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid split manifest row at position {position}") from error
        if row_index != position:
            raise ValueError(
                f"manifest row_index mismatch at position {position}: {row_index}"
            )
        if not image_id or image_id in seen_ids:
            raise ValueError(f"missing or duplicate image_id in manifest: {image_id!r}")
        seen_ids.add(image_id)
        if split == "reference":
            image_ids.append(image_id)
            labels.append(label)
            row_indices.append(row_index)

    if len(row_indices) < K_NEIGHBORS:
        raise ValueError(
            f"reference split has {len(row_indices)} images; at least {K_NEIGHBORS} are required"
        )
    reference_vectors = _normalize_embeddings(
        matrix[np.asarray(row_indices, dtype=np.intp)], source="reference"
    )
    return ReferenceSupportIndex(
        image_ids=tuple(image_ids),
        labels=tuple(labels),
        normalized_embeddings=reference_vectors,
        embedding_dtype=str(matrix.dtype),
        extractor_settings=_extractor_settings(reference_config, "reference"),
    )


def load_reference_index(
    embeddings_path: str | Path,
    manifest_path: str | Path,
    reference_config_path: str | Path,
) -> ReferenceSupportIndex:
    """Load source embeddings and construct an index from reference rows only."""
    with Path(manifest_path).open(newline="", encoding="utf-8") as file_handle:
        manifest_rows = list(csv.DictReader(file_handle))
    required_columns = {"image_id", "row_index", "label", "split"}
    if not manifest_rows or not required_columns.issubset(manifest_rows[0]):
        raise ValueError("split manifest must contain image_id,row_index,label,split")
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    return build_reference_index(
        embeddings,
        manifest_rows,
        _load_config(Path(reference_config_path)),
    )


def score_queries(
    index: ReferenceSupportIndex,
    query_image_ids: Sequence[str],
    query_embeddings: NDArray[Any] | Sequence[Sequence[float]],
    query_config: Mapping[str, Any],
) -> list[SupportScore]:
    """Score queries by fifth-neighbor cosine distance, without query labels."""
    query_matrix = _as_embedding_matrix(query_embeddings, source="query")
    if len(query_image_ids) != query_matrix.shape[0]:
        raise ValueError("query ID count must match query embedding row count")
    if len(set(query_image_ids)) != len(query_image_ids):
        raise ValueError("query image_id values must be unique")
    if str(query_matrix.dtype) != index.embedding_dtype:
        raise ValueError(
            "query and reference embedding dtypes differ: "
            f"{query_matrix.dtype} != {index.embedding_dtype}"
        )
    query_settings = _extractor_settings(query_config, "query")
    if query_settings != index.extractor_settings:
        raise ValueError("query and reference feature extraction settings do not match")
    normalized_queries = _normalize_embeddings(query_matrix, source="query")

    results = []
    reference_ids = np.asarray(index.image_ids, dtype=str)
    for query_id, query_vector in zip(query_image_ids, normalized_queries):
        similarities = index.normalized_embeddings @ query_vector
        distances = np.clip(1.0 - similarities, 0.0, 2.0)
        eligible = reference_ids != query_id
        eligible_indices = np.flatnonzero(eligible)
        if eligible_indices.size < K_NEIGHBORS:
            raise ValueError(
                f"query {query_id!r} has only {eligible_indices.size} eligible reference "
                f"images after self-exclusion; {K_NEIGHBORS} are required"
            )
        ordered = eligible_indices[
            np.lexsort((reference_ids[eligible_indices], distances[eligible_indices]))
        ][:K_NEIGHBORS]
        neighbors = tuple(
            Neighbor(
                image_id=index.image_ids[int(position)],
                label=index.labels[int(position)],
                distance=float(distances[int(position)]),
            )
            for position in ordered
        )
        results.append(
            SupportScore(
                query_image_id=query_id,
                reference_distance=neighbors[-1].distance,
                neighbors=neighbors,
            )
        )
    return results


def _read_query_ids(path: Path, expected_rows: int) -> list[str]:
    with path.open(newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        if not {"image_id", "row_index"}.issubset(reader.fieldnames or ()):
            raise ValueError("query IDs CSV must contain image_id,row_index (labels are not used)")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise ValueError(
            f"query ID row count {len(rows)} does not match query embeddings {expected_rows}"
        )
    for position, row in enumerate(rows):
        if int(row["row_index"]) != position:
            raise ValueError(f"query row_index must match query array position {position}")
    return [row["image_id"] for row in rows]


def _write_scores(path: Path, scores: Sequence[SupportScore]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=(
                "query_image_id",
                "reference_distance",
                "neighbor_rank",
                "neighbor_image_id",
                "neighbor_label",
                "neighbor_distance",
            ),
        )
        writer.writeheader()
        for score in scores:
            for rank, neighbor in enumerate(score.neighbors, start=1):
                writer.writerow(
                    {
                        "query_image_id": score.query_image_id,
                        "reference_distance": score.reference_distance,
                        "neighbor_rank": rank,
                        "neighbor_image_id": neighbor.image_id,
                        "neighbor_label": neighbor.label,
                        "neighbor_distance": neighbor.distance,
                    }
                )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score query embeddings against Ashley's reference split using cosine distance."
    )
    parser.add_argument("--query-embeddings", required=True, type=Path)
    parser.add_argument("--query-ids", required=True, type=Path)
    parser.add_argument("--query-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--reference-embeddings",
        type=Path,
        default=_PROJECT_ROOT / "features/crc_val_musk/embeddings.npy",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=_PROJECT_ROOT / "experiments/ashley_review/split_manifest.csv",
    )
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=_PROJECT_ROOT / "features/crc_val_musk/extraction_config.json",
    )
    args = parser.parse_args(argv)
    try:
        index = load_reference_index(
            args.reference_embeddings, args.split_manifest, args.reference_config
        )
        queries = np.load(args.query_embeddings, allow_pickle=False)
        query_ids = _read_query_ids(args.query_ids, len(queries))
        scores = score_queries(
            index,
            query_ids,
            queries,
            _load_config(args.query_config),
        )
        _write_scores(args.output, scores)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(f"Wrote {len(scores)} query scores to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())