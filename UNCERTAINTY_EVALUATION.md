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
python evaluate_uncertainty.py synthetic_uncertainty_example.csv --output-csv outputs/synthetic_uncertainty_results.csv --simulated
```

The output CSV includes a `data_status` column set to `SIMULATED` when the
`--simulated` option is supplied.