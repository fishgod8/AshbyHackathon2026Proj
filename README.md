# AshbyHackathon2026Proj

## Team Responsibilities

- **Mayumi:** image embeddings and tissue classification (`predict.py`).
- **Ashley:** uncertainty assessment and evaluation (`uncertainty.py`).
- **Felix:** similar-image retrieval (`retrieval.py`).
- **Bowen:** Streamlit UI and integration (`app.py`).

## Interfaces

- The classifier's `predict_image(image)` output consumed by this module contains `predicted_class`, a `scores` mapping, and an optional `embedding`.
- `find_similar(embedding, k=5)` is the planned retrieval interface. It returns records containing `image_path`, `label`, and `similarity`. Real retrieval is not implemented yet and raises `NotImplementedError`.
- `assess_uncertainty(probabilities, threshold)` is implemented in `uncertainty.py`. It returns `(confidence, needs_review)`, where confidence is the largest class probability. It validates the threshold and probabilities, including that their sum is approximately 1.
- `assess_prediction(prediction, threshold)` accepts the dictionary returned by the classifier's single-image interface. It returns `predicted_class`, `confidence`, `needs_review`, and `threshold`; an optional embedding is ignored.

## Uncertainty Detection

Uncertainty detection uses the largest class score as confidence and flags an
image for review when `confidence < threshold`. It is a review-routing signal:
low confidence does not guarantee that a prediction is wrong, and high
confidence does not guarantee that it is correct.

```python
from uncertainty import assess_prediction

result = assess_prediction(
    {
        "predicted_class": "InSitu",
        "scores": {
            "Benign": 0.05,
            "InSitu": 0.85,
            "Invasive": 0.07,
            "Normal": 0.03,
        },
    },
    threshold=0.70,
)
```

The threshold remains configurable. `0.70` is only a provisional example, not
a selected final threshold. Compare thresholds on labeled predictions with:

```powershell
python evaluate_uncertainty.py "evaluation.csv" --output-csv exploratory_results.csv
```

The evaluator requires `image_id`, `true_label`, `predicted_label`, and
`confidence`; it also accepts and ignores the per-class `score_*` columns.
Comparisons made with test-set `evaluation.csv` are exploratory and must not be
used to select a final threshold. The final threshold will be selected using
validation-set predictions and then evaluated once on the held-out test set.

## Mock Demo

Install Streamlit in the project's selected Python environment if it is not already available, then run:

```powershell
python -m streamlit run app.py
```

The demo accepts an image upload, but its clearly labeled mock mode uses fixed synthetic probabilities and does **not** analyze the uploaded image. The reference-image area is only a placeholder; this repository does not include reference images or real cases. The mock threshold (`0.7`) is provisional and must be selected using validation data before real use.

## Implementation Status

- **Implemented:** input validation and confidence-based review flag in `uncertainty.py`; a synthetic mock demo in `app.py`.
- **Placeholders:** real image inference and embeddings in `predict.py`; real similar-image search in `retrieval.py`; these interfaces raise `NotImplementedError`.
- **Not included:** model weights, datasets, reference cases, or evaluation/validation data.
