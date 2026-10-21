import argparse
import ast
import csv
from pathlib import Path

import joblib
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(Path(__file__).with_name("classifier.pkl")))
    args = parser.parse_args()

    saved = joblib.load(args.model)
    model = saved["model"]

    with open(args.embeddings) as f:
        rows = list(csv.DictReader(f))

    vectors = np.array(
        [ast.literal_eval(row["embedding"]) for row in rows],
        dtype=np.float32
    )

    probabilities = model.predict_proba(vectors)
    predictions = model.predict(vectors)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id", "predicted_label", "confidence",
            *[f"score_{c}" for c in model.classes_]
        ])

        for row, label, probs in zip(rows, predictions, probabilities):
            writer.writerow([
                row["image_id"],
                label,
                float(max(probs)),
                *[float(p) for p in probs]
            ])

    print(f"Saved predictions for {len(rows)} images to {args.output}")

if __name__ == "__main__":
    main()
