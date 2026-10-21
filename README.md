# AshbyHackathon2026Proj

## Ashley uncertainty component

`analysis_pipeline.py` provides the stable bridge from Mayumi's BACH image
classifier to Bowen's UI and Felix's retrieval system. It preserves Mayumi's
QuiltNet-B-32 encoder, twelve spatial patches, mean 512D embedding,
`StandardScaler + LogisticRegression`, and four-class output. The canonical
checkpoint is `bach/classifier.pkl` with SHA-256
`a3d3115fc3e1a88b4720ba85d083602202b86e315fcfd8f0abd381cc62e15885`.

For class probabilities `p`:

- confidence: `max(p)`
- maximum-probability uncertainty: `1 - max(p)`
- entropy: `-sum(p * log(p))` for positive probabilities
- normalized entropy: `entropy / log(4)`
- top-two margin: largest probability minus second-largest probability

The default review rule is only:

```python
needs_review = confidence < 0.70
```

The threshold is configurable and provisional. It is not clinically validated
or statistically optimal. An unflagged image is merely not flagged by this
rule; it must not be described as safe or disease-free. Entropy and margin are
reported for display and optional ordering, but their thresholds are disabled
by default.

### Analyze one image

```python
from analysis_pipeline import analyze_image

result = analyze_image(
    "path/to/sample.tif",
    model_path="bach/classifier.pkl",
    confidence_threshold=0.70,
)
```

The JSON-serializable result contains:

```json
{
  "image_id": "sample.tif",
  "predicted_class": "InSitu",
  "confidence": 0.62,
  "probabilities": {
    "Benign": 0.10,
    "InSitu": 0.62,
    "Invasive": 0.20,
    "Normal": 0.08
  },
  "uncertainty": {
    "max_probability_uncertainty": 0.38,
    "entropy": 1.073,
    "normalized_entropy": 0.774,
    "margin": 0.42
  },
  "needs_review": true,
  "review_reasons": ["Low model confidence"],
  "embedding": []
}
```

The values above illustrate the schema. `analyze_image()` calculates the real
values. Felix can consume the returned 512-value `embedding` directly; the
uncertainty component does not modify it.

### Build the review queue

```python
from analysis_pipeline import rank_for_review

queue = rank_for_review(results)  # confidence ascending within each group
entropy_queue = rank_for_review(results, sort_by="normalized_entropy")
margin_queue = rank_for_review(results, sort_by="margin")
```

Flagged images always appear first, followed by unflagged images. Optional
sorting changes only the order inside those groups and does not change the
review flag. `image_id` is the deterministic tie-breaker, all images are
retained, and each returned row receives a one-based `review_rank`.

### Validation

```powershell
python -m unittest discover -v
python evaluate_bach_uncertainty.py bach/evaluation.csv outputs/bach_uncertainty `
  --checkpoint bach/classifier.pkl
```

The saved 40-image evaluation is a reproducibility check, not a source for
tuning the production threshold. Research-only conformal, union, carcinoma,
and downloaded-demo analyses are retained under `experiments/uncertainty/` and
are not enabled by the default inference or review path. Exact-checkpoint
conformal prediction remains unavailable until genuinely independent labeled
calibration data are supplied.
