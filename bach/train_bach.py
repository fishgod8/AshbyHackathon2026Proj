import ast
import csv
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent

# Match embeddings to labels and train/validation/test splits.
with open(root / "manifest.csv") as f:
    manifest = {r["image_path"]: r for r in csv.DictReader(f)}

grouped = defaultdict(list)
metadata = {}

with open(root / "bach_embeddings.csv") as f:
    for row in csv.DictReader(f):
        info = manifest[row["image_path"]]
        vector = np.array(ast.literal_eval(row["embedding"]), dtype=np.float32)

        image_id = info["original_image"]
        grouped[image_id].append(vector)
        metadata[image_id] = (info["label"], info["split"])

assert len(grouped) == 400
assert all(len(v) == 12 for v in grouped.values())

# Average 12 patch embeddings per original image.
data = {"train": [], "val": [], "test": []}

with open(root / "image_embeddings.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "label", "split", "embedding"])

    for image_id in sorted(grouped):
        label, split = metadata[image_id]
        vector = np.mean(grouped[image_id], axis=0)
        data[split].append((image_id, label, vector))
        writer.writerow([image_id, label, split, vector.tolist()])

def unpack(split):
    rows = data[split]
    return (
        [r[0] for r in rows],
        np.stack([r[2] for r in rows]),
        np.array([r[1] for r in rows])
    )

_, X_train, y_train = unpack("train")
_, X_val, y_val = unpack("val")
test_ids, X_test, y_test = unpack("test")

print("Training images:", len(y_train))
print("Validation images:", len(y_val))
print("Testing images:", len(y_test))
print("Embedding dimensions:", X_train.shape[1])

best_score = -1
best_model = None

# Select model regularization on validation images only.
for C in [0.01, 0.1, 1.0, 10.0]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=3000)
    )
    model.fit(X_train, y_train)

    score = accuracy_score(y_val, model.predict(X_val))
    print(f"C={C}: validation accuracy={score:.3f}")

    if score > best_score:
        best_score = score
        best_model = model

# Evaluate on untouched test images.
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)

print("Test accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump({
    "model": best_model,
    "classes": list(best_model.classes_),
    "embedding_model": "openclip:wisdomik/QuiltNet-B-32",
    "aggregation": "mean of 12 patches"
}, root / "classifier.pkl")

with open(root / "evaluation.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_id", "true_label", "predicted_label",
        "confidence", *[f"score_{c}" for c in best_model.classes_]
    ])

    for image_id, true, pred, probs in zip(
        test_ids, y_test, predictions, probabilities
    ):
        writer.writerow([
            image_id, true, pred, float(max(probs)),
            *[float(p) for p in probs]
        ])

print("Saved classifier.pkl")
print("Saved evaluation.csv")
print("Saved image_embeddings.csv")
