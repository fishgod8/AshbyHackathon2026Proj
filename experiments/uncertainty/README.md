# Uncertainty research experiments

This directory retains Ashley's retrospective experiments for reproducibility.
They are not imported by the default review rule except for the optional,
explicit conformal artifact path in `analysis_pipeline.py`.

- `run_bach_conformal_experiment.py` refits a separate experimental model.
- `calibrate_mayumi_conformal.py` requires genuinely independent labeled
  calibration predictions bound to Mayumi's exact checkpoint.
- `evaluate_bach_demo_uncertainty.py` analyzes the unlabeled challenge images.
- `run_union_uncertainty_experiment.py` and
  `run_carcinoma_priority_experiment.py` are retrospective ranking studies.

Run scripts as modules from the repository root, for example:

```powershell
python -m experiments.uncertainty.run_bach_conformal_experiment `
  bach/image_embeddings.csv outputs/bach_conformal
```

These experiments do not change the production confidence threshold or review
queue. Their reported test-set comparisons are exploratory, not independently
validated clinical performance.
