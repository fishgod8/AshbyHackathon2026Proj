# Uncertainty Evaluation CSV

Input CSV must contain these columns:

```text
image_id, true_label, predicted_label, confidence
```

`confidence` is the highest predicted class probability, in the range 0 to 1.
Review marking remains `confidence < threshold`.

Demo command (the input and exported results are simulated; they do not
represent real model performance):

```powershell
python evaluate_uncertainty.py synthetic_uncertainty_example.csv --output-csv outputs/synthetic_uncertainty_results.csv --purpose simulated
```

Use `--purpose validation`, `--purpose test`, or `--purpose simulated`. The
output CSV records this value in `evaluation_purpose`; `--simulated` remains a
supported alias for `--purpose simulated`. Test-set results must not be used to
select a threshold.

## Split Conformal Prediction

`experiments/uncertainty/conformal.py` implements split conformal prediction with the LAC (Least
Ambiguous set-valued Classifier) score. For each independent calibration
example, its nonconformity score is:

```text
1 - probability assigned to the true class
```

The finite-sample corrected calibration rank is
`ceil((n + 1) * (1 - alpha))`. A new class is included in the prediction set
when `1 - p(class | image) <= quantile`. With `alpha=0.10`, the nominal target
is 90% **marginal** coverage under exchangeability. This is a repeated-sample
population guarantee, not a correctness guarantee for an individual image or
a separate coverage guarantee for every class. A singleton prediction set can
still be wrong.

```python
from experiments.uncertainty.conformal import assess_conformal_prediction, calibrate_conformal

artifact = calibrate_conformal(
    calibration_probabilities,
    calibration_labels,
    class_names=["Benign", "InSitu", "Invasive", "Normal"],
    alpha=0.10,
)
result = assess_conformal_prediction(classifier_prediction, artifact)
```

Calibration requires labeled examples that are independent of classifier
fitting and model selection. The classifier, preprocessing, class order, and
all hyperparameters must be fixed before calibration probabilities are
computed. The existing BACH validation set cannot automatically serve as a
standard split-conformal calibration set if its labels were already used to
select the classifier or its hyperparameters. No independent BACH calibration
artifact or real conformal coverage result is currently included here.

Coverage also depends on calibration and future examples being exchangeable.
Calibration on BACH does not establish coverage for CRC images or other shifted
populations. Felix's top-5 retrieval similarity is a separate evidence tool:
nearest-neighbor similarity is neither a class probability nor a conformal
coverage guarantee and is not used by this implementation.

## Reproducible BACH Embedding Experiment

> **Isolation warning:** this is a research experiment using a separately
> refitted classifier. Its artifact is marked production-incompatible and is
> rejected by the canonical `analysis_pipeline`. It does not provide conformal
> sets for Mayumi's saved `bach/classifier.pkl`.

`experiments/uncertainty/run_bach_conformal_experiment.py` runs the fixed confidence and split-
conformal workflow using the existing image-level QuiltNet embedding CSV. It
does not load QuiltNet, read images, or extract embeddings.

```powershell
python -m experiments.uncertainty.run_bach_conformal_experiment `
  "C:\path\to\image_embeddings.csv" `
  "outputs\bach_conformal" `
  --alpha 0.10 `
  --confidence-threshold 0.70
```

The script validates the expected 400-image BACH layout and 512-dimensional
finite embeddings. Within each class, it ranks original-training image IDs by
`SHA256("bach-calibration-v1|" + image_id)` and reserves the first 10 for
calibration. The remaining allocation is 70 train, 10 validation, 10
calibration, and 10 test images per class. Logistic-regression `C` is selected
on train/validation only; the final `StandardScaler + LogisticRegression`
pipeline is refitted on train plus validation. Calibration and test images are
never used for fitting or model selection.

Outputs are:

- `split_manifest.csv`
- `calibration_predictions.csv`
- `calibration_artifact.json`
- `test_predictions.csv`
- `policy_comparison.csv`
- `experiment_summary.json`

The three policies are fixed before test evaluation: confidence below the
configured threshold, conformal set size other than one, and their logical OR.
The summary records empirical marginal coverage, prediction-set size,
singleton rate, review rate, error capture, and reviewed/unreviewed error
rates. `alpha=0.10` and confidence threshold `0.70` are configurable defaults,
not values selected from this test set.

This experiment uses the same estimator components, candidate `C` grid, and
`max_iter` as the inspected Mayumi pipeline. It is not the same fitted model:
it removes 40 calibration images from the original training split and refits
the selected pipeline on the remaining train plus validation images. The
40-image test split was inspected in earlier exploratory work, so these results
must be described as exploratory rather than a pristine final benchmark. The
model artifacts, split, preprocessing provenance, and fitting procedure must
be aligned with Mayumi's final pipeline before attributing these metrics to the
team's final classifier.

BACH calibration establishes no coverage guarantee for BreakHis. BreakHis is
a different population and requires its own distribution assessment and,
where conformal guarantees are desired, suitable independent calibration.
