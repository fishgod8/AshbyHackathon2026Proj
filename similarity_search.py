import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

CSV = "./old_embedding_attempts/crc_embeddings.csv"

def load_embeddings(csv_path=CSV):
    df = pd.read_csv(csv_path)

    # CSV stores each embedding as a string representation of a list.
    X = np.stack(
        df["embedding"].apply(
            lambda value: np.asarray(ast.literal_eval(value), dtype=np.float32)
        ).to_numpy()
    )

    labels = df["label"].to_numpy()
    paths = df["image_path"].to_numpy()

    # Normalize each row to unit length.
    X = normalize(X, norm="l2")

    return df, X, labels, paths


def cosine_search(X, labels, paths, query_index, k=5):
    """
    Find the k most similar images to an image already in X.
    Excludes the query image itself.
    """
    query = X[query_index]

    # Because vectors are normalized, dot products are cosine similarities.
    similarities = X @ query

    # Exclude the query itself.
    similarities[query_index] = -np.inf

    # Retrieve the k highest similarity scores.
    k = min(k, len(X) - 1)
    indices = np.argpartition(-similarities, k - 1)[:k]
    indices = indices[np.argsort(-similarities[indices])]

    print(f"Query image: {paths[query_index]}")
    print(f"Query label: {labels[query_index]}\n")

    for rank, idx in enumerate(indices, start=1):
        print(
            f"{rank}. Similarity: {similarities[idx]:.4f} | "
            f"Label: {labels[idx]} | "
            f"Image: {paths[idx]}"
        )


if __name__ == "__main__":
    df, X, labels, paths = load_embeddings()

    print(f"Loaded {len(X)} embeddings.")
    print(f"Embedding dimensions: {X.shape[1]}")

    # Example: retrieve the 5 nearest neighbors of image 0.
    cosine_search(X, labels, paths, query_index=0, k=5)

