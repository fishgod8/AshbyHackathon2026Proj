# Ashley uncertainty

This package adds validated uncertainty metrics and deterministic human-review
prioritization to Mayumi's unchanged BACH classifier.

## Public API

```python
from Ashley_uncertainty import analyze_image, rank_for_review

result = analyze_image(
    "path/to/sample.tif",
    model_path="bach/classifier.pkl",
    confidence_threshold=0.70,
)

queue = rank_for_review(results)
entropy_queue = rank_for_review(results, sort_by="normalized_entropy")
margin_queue = rank_for_review(results, sort_by="margin")
```

The metrics are:

- `confidence = max(probabilities)`
- `max_probability_uncertainty = 1 - confidence`
- `entropy = -sum(p * log(p))` for `p > 0`
- `normalized_entropy = entropy / log(4)`
- `margin = highest_probability - second_highest_probability`

The default and provisional review rule is only
`needs_review = confidence < 0.70`. Entropy and margin thresholds are disabled
unless explicitly supplied. Optional queue sorting never changes the review
flag. Flagged images appear first; each group is sorted by the selected metric,
then by `image_id`. No images are discarded.

The result is JSON serializable and retains the 512D `embedding` for Felix's
retrieval system. `conformal_status` remains explicitly `unavailable` because
there is no independent calibration set for Mayumi's exact checkpoint.

Run tests and reproduce the saved BACH evaluation from the repository root:

```powershell
python -m unittest discover -v
python -m Ashley_uncertainty.evaluate_bach_uncertainty `
  bach/evaluation.csv outputs/bach_uncertainty `
  --checkpoint bach/classifier.pkl
```

The 0.70 threshold is not statistically optimal or clinically validated.
Unflagged images must not be described as safe or disease-free.
