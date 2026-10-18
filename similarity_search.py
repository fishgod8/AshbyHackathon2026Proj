import ast
import numpy as np
import pandas as pd
from old_embedding_attempts.get_embeddings import embed, load_your_own_image
from sklearn.preprocessing import normalize
import sys

def load_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    column_names = df.columns.tolist()
    # CSV stores each embedding as a string representation of a list.
    X = np.stack(
        df["embedding"].apply(
            lambda value: np.asarray(ast.literal_eval(value), dtype=np.float32)
        ).to_numpy()
    )
    # info = np.full(df.shape, fill_value="", dtype="U20")
    # for i in range(df.shape[1]):
    #     info[:, i] = df.iloc[1:, i]


    # labels = df["label"].to_numpy()
    # paths = df["image_path"].to_numpy()

    # Normalize each row to unit length.
    X = normalize(X, norm="l2")

    return df, X


def cosine_search(X, df, new_embedding, k=5):
    """
    Find the k most similar images to an image already in X.
    Excludes the query image itself.
    """

    # Because vectors are normalized, dot products are cosine similarities.
    similarities = X @ new_embedding

    # Retrieve the k highest similarity scores.
    k = min(k, len(similarities))
    indices = np.argpartition(-similarities, k - 1)[:k]
    indices = indices[np.argsort(-similarities[indices])]
    
    for rank, idx in enumerate(indices, start=1):
        print(
            f"{rank}. Similarity: {similarities[idx]:.4f} | "
        )
        # print(f"Type: {type(idx)} | val: {idx}")
        for num, col in enumerate(df.columns.tolist()):
            if col != "embedding":
                print(f"{df.columns[num]}: {df.iat[idx, num]} | ")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <image_path> <embeddings_path>")
        sys.exit(1)
    new_embedding = embed([load_your_own_image(sys.argv[1])])
    df, X = load_embeddings(sys.argv[2])
    new_embedding = np.asarray(new_embedding, dtype=np.float32)

    # Normalize the query using the same method as the database.
    new_embedding = normalize(new_embedding.reshape(1, -1), norm="l2")[0]


    print(f"Loaded {len(X)} embeddings.")
    print(f"Embedding dimensions: {X.shape[1]}")

    # Example: retrieve the 5 nearest neighbors of image 0.
    cosine_search(X, df, new_embedding, k=5)

