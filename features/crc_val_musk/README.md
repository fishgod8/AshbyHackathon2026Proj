# CRC-VAL-HE-7K MUSK Features

These are 1024-dimensional embeddings from the MUSK **image encoder**'s global
image representation. They are image-only features, not combined or fused
image-text representations. The classifier and uncertainty-analysis stages are
separate from this feature extraction.

`CRC-VAL-HE-7K` is originally external validation data. This handoff only
extracts features: it does not split data, train a classifier, select a
threshold, or report model performance. If the team uses any of this dataset for
classifier training or tuning, it is no longer an untouched independent test
set. Keep separate, unused data for final evaluation. Patient identifiers were
not inferred from image filenames.

## Files and Alignment

- `embeddings.npy`: float32 array with shape `(7156, 1024)`.
- `metadata.csv`: row index, image ID, path relative to the dataset root, and class label.
- `extraction_config.json`: model/checkpoint fingerprint, preprocessing, options, versions, and class counts.
- `failures.csv`: failed-image report; zero data rows means no failures.

Embedding row `i` corresponds exactly to metadata row with `row_index == i`.
The final output is published only when all images have finite feature values.

## Loading

Dependencies used in this example are NumPy and Python's standard-library
`csv` module. Run it from the project root:

```python
import csv
import numpy as np

embeddings = np.load("features/crc_val_musk/embeddings.npy")
with open("features/crc_val_musk/metadata.csv", newline="", encoding="utf-8") as f:
    metadata = list(csv.DictReader(f))
assert embeddings.shape == (len(metadata), 1024)
assert all(int(row["row_index"]) == i for i, row in enumerate(metadata))
assert len({row["image_id"] for row in metadata}) == len(metadata)
```

## Classifier Evaluation Protocol

Create training, validation, and test sets from an appropriate data source
before fitting. Do not automatically split this feature handoff. Fit both the
scaler and logistic regression using training data only. Apply the fitted
scaler to validation data, and use validation predictions to choose any
uncertainty threshold. Once the pipeline and threshold are fixed, use the test
set only with `transform` and `predict` (or `predict_proba`); do not fit or tune
the scaler, classifier, or threshold on test data.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train_embeddings)
train_features = scaler.transform(train_embeddings)
validation_features = scaler.transform(validation_embeddings)

classifier = LogisticRegression(max_iter=1000).fit(train_features, train_labels)
validation_probabilities = classifier.predict_proba(validation_features)
# Select and freeze any review threshold using validation data only.

test_features = scaler.transform(test_embeddings)
test_predictions = classifier.predict(test_features)
# Apply the already-frozen threshold to test predictions; do not retune it here.
```

No classifier fitting, data split, threshold selection, or performance reporting
has been performed as part of this feature extraction.
