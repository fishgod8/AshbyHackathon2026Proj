"""Public interface for retrieving images by embedding similarity."""

from typing import TypedDict


class SimilarImage(TypedDict):
    """One similar-image result."""

    image_path: str
    label: str
    similarity: float


def find_similar(embedding: object, k: int = 5) -> list[SimilarImage]:
    """Return up to ``k`` image records similar to the supplied embedding.

    Real embedding search has not been implemented.
    """
    raise NotImplementedError("Similar-image retrieval is not implemented")