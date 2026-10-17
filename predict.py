"""Public interface for single-image classification and embedding extraction."""

from typing import TypedDict


class PredictionResult(TypedDict):
    """Prediction output; probabilities align positionally with class_names."""

    predicted_label: str
    class_names: list[str]
    probabilities: list[float]
    embedding: list[float]


def predict_image(image: object) -> PredictionResult:
    """Classify one image and return its label, class scores, and embedding.

    The probability at index ``i`` corresponds to ``class_names[i]``. This
    function defines the integration contract only; real model inference and
    embedding extraction have not been implemented.
    """
    raise NotImplementedError("Image inference and embeddings are not implemented")