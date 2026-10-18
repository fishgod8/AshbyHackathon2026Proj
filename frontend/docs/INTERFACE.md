# Integration interface proposal (draft — please reply 👍 or suggest changes)

Hi all — I'm wiring everything into the demo UI. The UI reads only one result format, so if we agree on the function signatures below, we can all work in parallel and swap real code in later. Until then I build against mock data and stubs.

## What I need from each of you

**1. Mayumi — embeddings + classifier**
```python
embed(images: list[PIL.Image]) -> np.ndarray               # [N, D], L2-normalized (same as the official run_embeddings.py CSV); needed later for upload
predict(emb: np.ndarray) -> tuple[list[str], np.ndarray]   # (classes, probs [N, C]); same class order on every call
```
Plus the gallery embeddings CSV: `image_path, embedding, label` (+ `patient_id` / `slide_id` if available).

**2. [Person 2] — uncertainty**
```python
flag(probs: np.ndarray, classes: list[str]) -> dict
# {"risk": "confident" | "review", "candidate_set": ["TUM", "STR"], "reason": "one short sentence"}
```
Plus, later, `evidence.json` for the methods page (format below). A 3-level risk (green/yellow/red) is fine too; the UI can handle either.

**3. Felix — retrieval**
```python
retrieve(query_emb: np.ndarray, k: int = 5, label: str | None = None,
         exclude_group: str | None = None) -> list[dict]
# [{"image": "<path>", "label": "TUM", "score": 0.93}, ...] sorted by score, descending
```
`label` lets me pull "most similar tiles of the runner-up class" for a side-by-side comparison ("why A and not B"). `exclude_group` skips tiles from the query's own patient/slide.

## What I build on top (you don't need to produce this)

One dict per case; `cases.json` is a list of them (values below are examples):
```json
{
  "case_id": "case_04",
  "image": "images/case_04.png",
  "true_label": "STR",
  "prediction": "TUM",
  "probabilities": {"TUM": 0.48, "STR": 0.416, "MUS": 0.029},
  "risk": "review",
  "candidate_set": ["TUM", "STR"],
  "reason": "TUM and STR score closely; manual review recommended.",
  "similar_cases": [{"image": "images/ref_TUM_2.png", "label": "TUM", "score": 0.937}],
  "contrast_cases": [{"image": "images/ref_STR_1.png", "label": "STR", "score": 0.902}]
}
```
Plus `meta.json` (dataset name, class list + full names, model name), so the UI isn't hard-coded to any dataset.

## 3 things to agree on today

1. **Class order**: `predict()` always returns the same class list; I copy it into `meta.json`.
2. **Same embeddings everywhere**: classification and retrieval both use the L2-normalized embeddings, so cosine similarity = dot product.
3. **Image paths**: gallery tiles must be readable by the UI — paths relative to one shared `images/` root (or I copy the files into the app).

## 2 pitfalls that would show up in the demo

- **Split train/test by patient or slide, not by tile** — otherwise accuracy is inflated.
- **Retrieval must exclude the query's own patient/slide** — otherwise top-5 returns adjacent patches at ~0.99 similarity, and judges will notice. (For CRC, using CRC-100K as the gallery and CRC-VAL-HE-7K as queries avoids this, since they come from different patients.)

## evidence.json (methods page, draft — example values)

```json
{
  "alpha": 0.1,
  "empirical_coverage": 0.91,
  "error_rate_confident": 0.02,
  "risk_coverage": [{"coverage": 0.5, "error": 0.01}, {"coverage": 1.0, "error": 0.06}],
  "models": [{"model": "QuiltNet-B-32", "accuracy": 0.93, "mean_set_size": 1.3}]
}
```

FYI: I have QuiltNet + PLIP embeddings for a 1,800-tile CRC-100K subset (official script, same CSV format) if anyone wants them for testing.
