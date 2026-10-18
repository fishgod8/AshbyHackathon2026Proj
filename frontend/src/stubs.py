"""TEMPORARY stand-ins for teammates' modules. Delete when real code lands.

- StubClassifier -> Mayumi's predict()
- stub_flag      -> Person 2's flag()
- StubRetriever  -> Felix's retrieve()

They exist only so the UI can be developed against real tiles and real
embeddings. Do not report any number produced by these stubs.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

REVIEW_THRESHOLD = 0.85


class StubClassifier:
    """Plain logistic regression on the gallery embeddings.

    C=10: with unit-norm embeddings the default C=1 is over-regularized and
    gives flat probabilities (median top-p 0.72), so nearly every case gets flagged.
    """

    def __init__(self, emb: np.ndarray, labels: np.ndarray, seed: int = 0, C: float = 10.0) -> None:
        self.model = LogisticRegression(C=C, max_iter=5000, random_state=seed).fit(emb, labels)
        self.classes = [str(c) for c in self.model.classes_]

    def predict(self, emb: np.ndarray) -> tuple[list[str], np.ndarray]:
        return self.classes, self.model.predict_proba(emb)


def stub_flag(probs: np.ndarray, classes: list[str], threshold: float = REVIEW_THRESHOLD) -> dict:
    """Flag for review when the top probability is below a fixed threshold."""
    first, second = np.argsort(probs)[::-1][:2]
    top, runner_up = classes[first], classes[second]
    if probs[first] >= threshold:
        return {
            "risk": "confident",
            "candidate_set": [top],
            "reason": f"{top} scores {probs[first]:.0%}, above the {threshold:.0%} review threshold.",
        }
    return {
        "risk": "review",
        "candidate_set": [top, runner_up],
        "reason": (
            f"{top} ({probs[first]:.0%}) is below the {threshold:.0%} review threshold; "
            f"runner-up {runner_up} ({probs[second]:.0%})."
        ),
    }


class StubRetriever:
    """Cosine top-k over L2-normalized gallery embeddings (dot product)."""

    def __init__(
        self, emb: np.ndarray, labels: np.ndarray, paths: list[str], groups: np.ndarray | None = None
    ) -> None:
        self.emb, self.labels, self.paths, self.groups = emb, labels, paths, groups

    def retrieve(
        self, query_emb: np.ndarray, k: int = 5, label: str | None = None, exclude_group: str | None = None
    ) -> list[dict]:
        mask = np.ones(len(self.labels), dtype=bool)
        if label is not None:
            mask &= self.labels == label
        if exclude_group is not None and self.groups is not None:
            mask &= self.groups != exclude_group
        idx = np.flatnonzero(mask)
        scores = self.emb[idx] @ query_emb
        order = np.argsort(scores)[::-1][:k]
        return [
            {"image": self.paths[idx[i]], "label": str(self.labels[idx[i]]), "score": float(scores[i])}
            for i in order
        ]
